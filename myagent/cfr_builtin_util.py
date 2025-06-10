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
from scml.oneshot.agents import GreedySyncAgent
from MatchingPennies import MyAgent
from scml.oneshot.ufun import OneShotUFun

# class SyncWrapper:
#     def __init__(self, agent_cls, max_q, price_buckets):
#         self.agent = agent_cls()
#         # --- minimal AWI stub the agent might inspect
#         from types import SimpleNamespace
#         awi_stub = SimpleNamespace(
#             current_step = 0,
#             n_lines      = max_q,
#             level        = 0,               # 0=L0, 1=L1 (change if needed)
#             my_suppliers = ["ENV", MyAgent],
#             my_consumers = ["ENV", MyAgent],
#             current_input_issues  = {UNIT_PRICE: SimpleNamespace(min_value=-1, max_value=1, rand=lambda :0)},
#             current_output_issues = {UNIT_PRICE: SimpleNamespace(min_value=-1, max_value=1, rand=lambda :0)},
#             is_first_level = False, 
#             needed_supplies=5,
#             needed_sales=5,
#         )
#         self.agent._awi = awi_stub          # attach stub
#         # prepare first proposals cache so we don’t recompute
#         self.fp_cache  = self.agent.first_proposals()   # {partner: (q,s,p)|None}
#         self.state_by_partner = {}         # keep SAOState objects
#         self.price_vals = np.linspace(-1,1,price_buckets,dtype=int).tolist()

#     # -------------------------------------------------------------
#     def next_offer(self, partner, rnd, need, last_offer):
#         """
#         Return (q,p) for this round, or ("ACCEPT", None).
#         partner is always "ENV" in our single-partner setting.
#         """

#         # -- build/update SAOState stub the Sync agent expects ----
#         s = self.state_by_partner.get(partner)
#         if s is None:
#             s = SAOState(step=rnd, relative_time=rnd/20, current_offer=last_offer)
#             self.state_by_partner[partner] = s
#         else:
#             s.step = rnd
#             s.relative_time = rnd / 20
#             s.current_offer = last_offer

#         # -- round 0 => first_proposals ---------------------------
#         if rnd == 0:
#             offer = self.fp_cache.get(partner)
#             if offer is None:
#                 return ("ACCEPT", None)
#             return offer[:2]               # (q,p)

#         # -- later rounds => counter_all --------------------------
#         answer = self.agent.counter_all({partner: last_offer}, {partner: s})[partner]

#         if answer.response_type == ResponseType.ACCEPT_OFFER:
#             return ("ACCEPT", None)

#         if answer.outcome is None:
#             # agent ended negotiation => treat as accept nothing
#             return ("ACCEPT", None)

#         q, _, p = answer.outcome
#         return (q, p)


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
    elif kind == "random":  # random
        q = random.randint(1, need)
        p = random.choice(price_list)
    return q, p

ARCHES  = ["hardheaded", "conceder", "random"]
WEIGHTS = [0.3,        0.4,        0.3]

#  CFR trainer (tabular, two‑player zero‑sum)
class CFRTrainer:
    """Outcome-sampling CFR with OneShotUFun payoffs."""

    def __init__(self,
                 max_q: int = 10,
                 price_vals: List[int] = (-1, 1),
                 save_interval: int = 100_000,
                 out_pattern: str = "cfr_util_iter_{:06d}.policy.json"):
        self.max_q       = max_q
        self.price_vals  = list(price_vals)      # e.g. [-1, 0, +1]
        self.action_set  = build_action_set(max_q, -1, 1, 3)
        self.nA          = len(self.action_set)

        # cost parameters (your new numbers)
        self.C_PROD  = 3.0
        self.C_SHORT = 0.6
        self.C_DISP  = 0.08
        self.C_STORE = 0.0

        # regret / strategy per role
        self.regret_sum   = {r: defaultdict(lambda: np.zeros(self.nA))
                             for r in ("B", "S")}
        self.strategy_sum = {r: defaultdict(lambda: np.zeros(self.nA))
                             for r in ("B", "S")}

        # misc
        self.save_every   = save_interval
        self.out_pattern  = out_pattern
        self.rng          = random.Random(7)
        self.nprng        = np.random.RandomState(7)

        price_buckets = 3

        # self.sync_wrappers = {
        #     "sync_mp":     SyncWrapper(MyAgent,          max_q, price_buckets),
        #     "sync_greedy": SyncWrapper(GreedySyncAgent,  max_q, price_buckets),
        # }

    # =============================================================
    def train(self, iters=200_000):
        for t in range(1, iters + 1):
            self._traverse("B")
            self._traverse("S")

            if self.save_every and t % self.save_every == 0:
                self._save(self.out_pattern.format(t))

    def average_strategy(self) -> Dict[str, List[float]]:
        out = {}
        for role, table in self.strategy_sum.items():
            for I, freq in table.items():
                s = freq.sum()
                if s > 0:
                    out[I] = (freq / s).tolist()
        return out

    # =============================================================
    def _save(self, fname):
        pathlib.Path(fname).write_text(json.dumps(self.average_strategy()))
        print(f"[CFR] saved -> {fname}")

    # ------------------------------------------------------------
    def _traverse(self, to_update: str):
        need  = self.rng.randint(1, self.max_q)
        opp   = self.rng.choices(ARCHES, WEIGHTS)[0]
        last_offer = None
        role_next  = "B"                       # buyer opens

        # ------- single OneShotUFun for entire traversal ----------
        ufun = OneShotUFun(
            ex_pin=0, ex_qin=0, ex_pout=0, ex_qout=0,
            input_product=0,
            input_penalty_scale=None,
            output_penalty_scale=None,
            storage_penalty_scale=None,
            agent_id=None,
            input_agent=(to_update == "B"),
            output_agent=(to_update == "S"),
            production_cost=self.C_PROD,
            disposal_cost=self.C_DISP,
            storage_cost=self.C_STORE,
            shortfall_penalty=self.C_SHORT,
            n_input_negs=1 if to_update == "B" else 0,
            n_output_negs=1 if to_update == "S" else 0,
            current_step=0,
            time_range=(0, 0),
            input_qrange=(1, self.max_q),
            input_prange=(-1, 1),
            output_qrange=(1, self.max_q),
            output_prange=(-1, 1),
            force_exogenous=True,
            n_lines=self.max_q,
            normalized=False,
        )

        # ------------ 20 alternating rounds -----------------------
        for rnd in range(20):
            # fictitious opponent offer
            lq, lp = opponent_policy(opp, rnd, need, self.price_vals, last_offer)
            last_offer = (lq, lp)

            # infoset key
            phase = 0 if rnd == 0 else 1 if rnd <= 2 else 2 if rnd <= 2 else 3
            qty_cmp   = 0 if lq == need else int(math.copysign(1, lq - need))
            price_cmp = 0 if lp == 0   else int(math.copysign(1, lp))
            I = info_key(role_next, phase, need, qty_cmp, price_cmp, 0)

            # σ and sampled action
            sigma = self._get_sigma(role_next, I)
            a_idx = self.nprng.choice(self.nA, p=sigma)
            act   = self.action_set[a_idx]

            # payoff of chosen action
            util_chosen = self._payoff(act, last_offer, role_next, need, ufun)

            # regret update for traversing player
            if role_next == to_update:
                a_prime = self.rng.randrange(self.nA)
                util_prime = self._payoff(self.action_set[a_prime],
                                          last_offer, role_next, need, ufun)
                self.regret_sum[role_next][I][a_prime] += (util_prime - util_chosen)
                self.strategy_sum[role_next][I][a_idx] += 1.0

            if act != ACCEPT:
                last_offer = act
            role_next = "S" if role_next == "B" else "B"

    # ------------------------------------------------------------
    def _payoff(self, action, last_offer, role, need, ufun):
        if action == ACCEPT:
            if last_offer is None:
                return 0.0
            q, p = last_offer
        else:
            q, p = action

        # naive “others fulfil rest at p=0” assumption
        q_other = max(0, need - q)
        offers  = [(q, 0, p),
                   (q_other, 0, 0)]
        outputs = [role == "S", role == "S"]
        return ufun.from_offers(offers, outputs)

    # ------------------------------------------------------------
    def _get_sigma(self, role, I):
        r = self.regret_sum[role][I]
        pos = np.maximum(r, 0.0)
        return pos/pos.sum() if pos.sum() else np.full(self.nA, 1/self.nA)

class CFROneShotAgent(OneShotAgent):
    """Agent that negotiates using a pre‑trained CFR policy."""

    def init(self):
        policy_path = pathlib.Path(__file__).with_name("cfr_builtin_utilv2.policy.json")
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

        n_partners = 1

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
        low_cash = int(self.awi.current_balance < 2000)  # or any threshold you define

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

# python3 cfr_builtin_util.py --train --iters 300000 --save_interval 300000 --out "cfr_builtin_utilv2.policy.json"