from __future__ import annotations

import json
import math
import pathlib
import random
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from negmas import SAOResponse, ResponseType, Outcome, SAOState
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE
from scml.oneshot.agent import OneShotAgent
from scml.oneshot.agents import GreedySyncAgent, RandDistOneShotAgent, EqualDistOneShotAgent, SyncRandomOneShotAgent
from scml.oneshot.ufun import OneShotUFun

ACCEPT = ("ACCEPT", None) 

def build_action_set(
    max_q: int,
    p_min: int,
    p_max: int,
    price_buckets: int = 3
) -> List[Tuple[int, int]]:
    """
    Return list of (quantity, price) actions.
      - Quantities: 1 … max_q
      - Prices: price_buckets evenly spaced points in [p_min, p_max]
    """
    if price_buckets < 2:
        raise ValueError("price_buckets must be >= 2")
    # generate evenly spaced prices (rounded to int)
    prices = np.linspace(p_min, p_max, price_buckets, dtype=int).tolist()
    actions = [(q, p) for q in range(1, max_q + 1) for p in prices]
    return [ACCEPT] + actions


def delta_to_real(d: int) -> int:
    """Convert delta price {-1,0,+1} to an actual unit price."""
    return int(self.mid_price + d * self.price_step)

def info_key(
        role: str, #[0 = buyer, 1 = seller]
        phase: int, #[0 = initial, 1 = early, 2 = middle, 3 = late]
        qty_needed: int, #[0-10]
        qty_cmp: int, #[-1 = lower, 0 = same, 1 = higher]
        price_cmp: int, #[-1 = lower, 0 = same, 1 = higher]
        low_cash: int, #[0 = false,1 = true]
    ) -> str:
    """Serialize infoset into a compact string key."""
    return f"{role}|{phase:+d}|{qty_needed:+d}|{qty_cmp:+d}|{price_cmp:+d}|{low_cash:+d}"

def opponent_policy(kind: str, rnd: int, need: int,
                    price_list: List[int],
                    last: Tuple[int,int]|None) -> Tuple[int,int]:
    if kind == "hardheaded":
        q = need;            
        p = price_list[-1]
    elif kind == "conceder":
        # total_rounds defines the round after which full concession is reached
        total_rounds = 10
        # Clamp rnd to total_rounds to avoid overshooting the concession interpolation
        t = min(rnd, total_rounds) / total_rounds  # normalized progression: 0 at start, 1 at total_rounds
        # Linear interpolation between the lowest and highest price
        p = int(round(price_list[0] * (1 - t) + price_list[-1] * t))
        q = need
    elif kind == "tft" and last:
        q = last[0]
        p = max(price_list[0], min(last[1]-1, price_list[-1]))
    else:  # random
        q = random.randint(1, need)
        p = random.choice(price_list)
    return q, p

ARCHES  = ["hardheaded", "conceder", "random", "tft"]
WEIGHTS = [0.25,  0.5, 0.125, 0.125]

#  CFR trainer (tabular, two‑player zero‑sum)
class CFRTrainer:
    """Self-play CFR trainer for *one* negotiation tree (20 rounds)."""

    def __init__(
        self,
        max_q: int = 10,
        price_buckets: int = 3,
        alpha: float = 0.5, 
        save_interval: int = 100_000,
        out_pattern: str = "cfr_util_iter_{}.policy.json",
        # add the following:
        mu_price: float   = 0,
        mu_prod_c: float  = 3,
        mu_disp_c: float  = 0.15,
        mu_store_c: float = 0.0,
        mu_short: float   = 0.7,
        n_partners: int   = 3,
    ):
        self.max_q = max_q
        self.price_buckets = price_buckets
        self.alpha = alpha  # weight on price term in utility

        # add the following:
        self.mu_price   = mu_price
        self.mu_prod_c  = mu_prod_c
        self.mu_disp_c  = mu_disp_c
        self.mu_store_c = mu_store_c
        self.mu_short   = mu_short
        self.n_partners = n_partners

        self.action_set = build_action_set(max_q, -1, +1, price_buckets)
        self.n_actions = len(self.action_set)

        # infoset -> regret & strategy accumulators
        self.regret_sum: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )
        self.strategy_sum: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )

        self.regret_history = []
        self.PLOT_EXPLOITABILITY = True
        self.LOW_CASH_PROBABILITY = 0.2

        self.save_interval = save_interval
        self.out_pattern = out_pattern

    def train(self, iters: int = 100_000):
        for t in range(iters):
            if t % 10000 == 0:
                print(f"{t} iterations done")
            for role in ("S", "B"): # S = seller, B = buyer
                self._traverse(role)

            if self.PLOT_EXPLOITABILITY:
                avg_regret = np.mean([np.max(r) for r in self.regret_sum.values()])
                self.regret_history.append(avg_regret)

            if (t + 1) % self.save_interval == 0:
                pol = self.average_strategy()
                out_file = self.out_pattern.format(t + 1)
                pathlib.Path(out_file).write_text(json.dumps(pol))
                print(f"[CFR] Saved intermediate policy to {out_file}")

    def average_strategy(self) -> Dict[str, List[float]]:
        return {
            I: self._normalise(s)
            for I, s in self.strategy_sum.items()
            if s.sum() > 0
        }

    def _traverse(self, role: str):
        """
        One pass of external-sampling CFR that evaluates each (q,p)
        action with a *real* SCML OneShot utility function.

        We approximate the missing market parameters by their means.
        """

        # --- 0) pre-compute means that stand in for the true parameters ----
        μ_price   = self.mu_price                    # mean of (-1 , +1) during offline training
        μ_prod_c  = self.mu_prod_c                  # ← CHANGE LATER when AWI gives real cost
        μ_disp_c  = self.mu_disp_c                # "
        μ_store_c = self.mu_store_c                  # "
        μ_short   = self.mu_short                # "

        n_partners = self.n_partners                   # you said “three concurrent negotiations”
        n_lines    = self.max_q          # capacity == max train Q

        # sample current need
        need = random.randint(1, self.max_q)
        opp    = random.choices(ARCHES, WEIGHTS)[0]

        last_offer = None

        # pre-build price discrete set once
        price_vals = np.linspace(-1, 1, self.price_buckets, dtype=int).tolist()
        # ------------------------------------------------------------------
        for rnd in range(10):

            # ------- 1) fictitious last offer from opponent ----------------
            last_q, last_p = opponent_policy(opp, rnd, need, price_vals, last_offer)
            last_offer = (last_q, last_p)

            qty_cmp   = 0 if last_q == need else int(math.copysign(1, last_q - need))
            price_cmp = 0 if last_p == 0   else int(math.copysign(1, last_p))
            
            phase = 0
            if rnd == 0:
                phase = 0
            elif rnd <= 1:
                phase = 1
            elif rnd <= 2:
                phase = 2
            else:
                phase = 3
            
            infoset   = info_key(role, phase, need, qty_cmp, price_cmp, 0)

            σ = self._get_strategy(infoset)
            a_idx = np.random.choice(self.n_actions, p=σ)
            action = self.action_set[a_idx]          # (q,p) **or** ACCEPT sentinel

            # ------- 2) create a *fresh* ufun object with mean params -------
            ufun = OneShotUFun(
                # exogenous contracts → assume none in training
                ex_pin=0, ex_qin=0, ex_pout=0, ex_qout=0,
                input_product=0,
                input_agent=(role == "B"),       # buyer negotiates input
                output_agent=(role == "S"),      # seller negotiates output
                production_cost=μ_prod_c,
                disposal_cost=μ_disp_c,
                storage_cost=μ_store_c,
                shortfall_penalty=μ_short,
                input_penalty_scale=None,
                output_penalty_scale=None,
                storage_penalty_scale=None,
                n_input_negs=n_partners if role == "B" else 0,
                n_output_negs=n_partners if role == "S" else 0,
                current_step=0,
                agent_id=None,
                time_range=(0, 0),
                input_qrange=(0, self.max_q),
                input_prange=(-1, 1),            # train with δ-prices
                output_qrange=(0, self.max_q),
                output_prange=(-1, 1),
                force_exogenous=True,
                n_lines=n_lines,
                normalized=False,
            )

            # ------- 3) build the hypothetical contract set ---------------
            # “Quota” assumption: other partners fulfil exactly need − act_q
            if action == ACCEPT:
                sel_q, sel_p = last_q, last_p      # accepting opponent’s last offer
            else:
                sel_q, sel_p = action

            # partner we’re playing with is ALWAYS first in tuple
            offers  = (
                (sel_q, 0, sel_p),                # our decision now
                (need - sel_q, 0, μ_price),       # quota from others
            )
            outputs = (
                role == "S",                      # True for sell
                role == "S",                      # quota side same product
            )

            util = ufun.from_offers(offers, outputs)   # scalar profit

            # ------- 4) build utility vector for ALL actions ---------------
            U = np.zeros(self.n_actions)
            for i, act in enumerate(self.action_set):
                if act == ACCEPT:
                    sel_q, sel_p = last_q, last_p
                else:
                    sel_q, sel_p = act
                offers = (
                    (sel_q, 0, sel_p),
                    (max(0, need - sel_q), 0, μ_price),
                )
                outputs = (role == "S", role == "S")
                U[i] = ufun.from_offers(offers, outputs)

            u_ref = U[a_idx]
            self.regret_sum[infoset] += U - u_ref
            self.strategy_sum[infoset] += σ

    def _get_strategy(self, I: str) -> np.ndarray:
        r   = self.regret_sum[I]
        pos = np.maximum(r, 0.0)
        if pos.sum() > 0:
            return pos / pos.sum()
        return np.full(self.n_actions, 1.0 / self.n_actions)

    def _normalise(self, x: np.ndarray) -> List[float]:
        s = x.sum()
        if s == 0:
            return (np.ones_like(x) / len(x)).tolist()
        return (x / s).tolist()

class CFROneShotAgent(OneShotAgent):
    """Agent that negotiates using a pre‑trained CFR policy."""

    def init(self):
        policy_path = pathlib.Path(__file__).with_name("cfr_util_iter_100000.policy.json")
        if not policy_path.exists():
            # fallback to embedded empty dict (all infosets unseen)
            print("Policy not found.")
            self.policy: Dict[str, List[float]] = {}
        else:
            self.policy = json.loads(policy_path.read_text())
        
        issues = self.awi.current_input_issues or self.awi.current_output_issues
        self.price_min = issues[UNIT_PRICE].min_value
        self.price_max = issues[UNIT_PRICE].max_value
        print(f"Price_min: {self.price_min}", f"Price_max: {self.price_max}")
        self.mid_price = (self.price_min + self.price_max) // 2

        self.price_step = (self.price_max - self.price_min) // 2

        self.action_set = build_action_set(
            max_q=self.awi.n_lines,
            p_min=self.price_min,
            p_max=self.price_max,
            price_buckets=3
        )

        # 1) compute the true SCML parameters
        price_min, price_max = (
            self.awi.current_input_issues[UNIT_PRICE].min_value,
            self.awi.current_input_issues[UNIT_PRICE].max_value,
        )
        mu_price   = (price_min + price_max) / 2
        mu_prod_c  = self.ufun.production_cost
        mu_disp_c  = self.ufun.disposal_cost
        mu_store_c = self.ufun.storage_cost
        mu_short   = self.ufun.shortfall_penalty
        # assume three concurrent negs on your side:
        n_partners = 3

        print(mu_price, mu_prod_c, mu_disp_c, mu_store_c, mu_short)

        # seeding 
        random.seed(hash(self.id) & 0xFFFF)
        np.random.seed(hash(self.id) & 0xFFFF)

    def _needed(self, n_id: str | None) -> int:
        return (self.awi.needed_sales if n_id in self.awi.my_consumers else self.awi.needed_supplies)

    def _infoset(self, role: str, state: SAOState, needed: int) -> str:
        offer = state.current_offer
        if offer is None:
            qty_cmp = price_cmp = 0
        else:
            qty_cmp = int(math.copysign(1, offer[QUANTITY] - needed)) if offer[QUANTITY] != needed else 0
            price_cmp = int(math.copysign(1, offer[UNIT_PRICE] - self.mid_price)) if offer[UNIT_PRICE] != self.mid_price else 0
        
        step = state.step
        if step == 0:
            phase = 0  # initial
        elif step <= 1:
            phase = 1  # early
        elif step <= 2:
            phase = 2  # mid
        else:
            phase = 3  # late

        
        #TODO: set low cash flag according to balance
        low_cash = int(self.awi.current_balance < 4000)  # or any threshold you define

        return info_key(role, phase, needed, qty_cmp, price_cmp, low_cash)    
    

    def _sample_action(self, infoset: str, role: str, need : int):
        pmf = self.policy.get(infoset) # this gets a distribution we can sample an action from
        if pmf is None: # if there is no policy, we fall back
            if role == "B":                          # we are BUYER (need inputs)
                q   = max(1, need)                   # ask for the full remaining need
                p   = self.price_max                 # be willing to pay the max band price
            else:                                    # we are SELLER
                q   = max(1, need)
                p   = self.price_min                 # entice consumers with lowest price
            fallback_idx = len(self.action_set) - 1
            return fallback_idx, (q,p)
        idx = np.random.choice(len(self.action_set), p=pmf) #sample from distribution
        return idx, self.action_set[idx]

    def _role(self, partner_id: str) -> str:
        #is [partner_id] a seller or a buyer?
        return "S" if partner_id in self.awi.my_consumers else "B"

def _train_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="Run CFR self‑play training")
    ap.add_argument("--iters", type=int, default=200_000)
    ap.add_argument("--save_interval", type=int, default=10_000, help="Save intermediate policy every N iterations")
    ap.add_argument("--out_pattern", type=str, default="cfr_oneshot_agent_iter_{}.policy.json", help="Output file pattern for intermediate policies")
    ap.add_argument("--out", type=str, default="cfr_oneshot_agent2.policy.json")
    args = ap.parse_args()

    if args.train:
        trainer = CFRTrainer()
        print(f"[CFR] Training for {args.iters:,} iterations…")
        trainer.train(args.iters)
        pol = trainer.average_strategy()
        pathlib.Path(args.out).write_text(json.dumps(pol))
        print(f"[CFR] Saved average strategy to {args.out}")

        import matplotlib.pyplot as plt

        plt.plot(trainer.regret_history)
        plt.xlabel("Iterations (in blocks)")
        plt.ylabel("Average Max Regret")
        plt.title("CFR Convergence")
        plt.grid()
        plt.show()
        plt.savefig('img/last_training.png')


if __name__ == "__main__":
    _train_cli()

# python3 cfr_builtin_util.py --train --iters 300000 --save_interval 300000 --out "cfr_builtin_util.policy.json"
# python3 cfr_builtin_util.py --train --iters 200000 --save_interval 100000 --out "cfr_builtin_util.policy.json"