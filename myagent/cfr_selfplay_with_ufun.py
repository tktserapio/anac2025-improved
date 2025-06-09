# ------------------------------------------------------------------
#  Two-player self-play CFR trainer for SCML-OneShot (20-round tree)
# ------------------------------------------------------------------

import json
import math
import pathlib
import random
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import math, random, json, pathlib
from scml.runner import WorldRunner
from collections import defaultdict
from typing import Dict, List, Tuple

from scml.oneshot import ANACOneShotContext
from negmas import SAOResponse, ResponseType, Outcome, SAOState
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE
from scml.oneshot.agent import OneShotAgent
from scml.oneshot.agents import OneshotDoNothingAgent

from scml.oneshot.world import SCML2024OneShotWorld
from scml.oneshot import OneShotUFun

import numpy as np

# ------------------------------------------------------------------
ACCEPT = ("ACCEPT", None)

def build_action_set(max_q: int,
                     p_min: int,
                     p_max: int,
                     price_buckets: int = 3):
    """index 0 = ACCEPT ;   1… = (q, p) tuples"""
    prices  = np.linspace(p_min, p_max,
                          price_buckets, dtype=int).tolist()
    acts    = [(q, p) for q in range(1, max_q + 1) for p in prices]
    return [ACCEPT] + acts

def info_key(role, phase, need, qty_cmp, price_cmp, low_cash):
    return f"{role}|{phase}|{need}|{qty_cmp}|{price_cmp}|{low_cash}"

# ---------------------------------------------------------------
class CFRTrainer:
    """Outcome-sampling external CFR for one buyer–seller negotiation."""

    def __init__(self,
                 max_q: int         = 10,
                 price_buckets: int = 3):
        self.max_q         = max_q
        self.price_buckets = price_buckets
        self.action_set    = build_action_set(max_q, -1, +1, price_buckets)
        self.nA            = len(self.action_set)

        # one regret & strategy table PER ROLE
        self.regret_sum   = {r: defaultdict(lambda: np.zeros(self.nA))
                             for r in ("B", "S")}
        self.strategy_sum = {r: defaultdict(lambda: np.zeros(self.nA))
                             for r in ("B", "S")}

        self.rng    = random.Random(1)
        self.nprng  = np.random.RandomState(1)

        # cost parameters (same scale as ±1 price band)
        self.C_PROD   = 0.0
        self.C_SHORT  = 10.0
        self.C_DISP   = 10.0
        self.C_STORE  = 0.0

    # =============================================================
    # public API
    # =============================================================
    def train(self, iters=200_000,
              save_every=0,
              out_pattern="cfr_iter_{:06d}.json"):
        for t in range(1, iters + 1):
            self._traverse("B")
            self._traverse("S")
            if save_every and t % save_every == 0:
                self._save(out_pattern.format(t))

    def average_strategy(self) -> Dict[str, List[float]]:
        avg = {}
        for role, table in self.strategy_sum.items():
            for I, freq in table.items():
                s = freq.sum()
                if s > 0:
                    avg[I] = (freq / s).tolist()
        return avg

    # =============================================================
    # internal
    # =============================================================
    def _save(self, fname):
        pathlib.Path(fname).write_text(json.dumps(self.average_strategy()))
        print(f"[CFR] saved -> {fname}")

    # ------------------------------------------------------------
    def _traverse(self, to_update: str):
        need = self.rng.randint(1, self.max_q)

        # --------- create lightweight OneShotUFun -----------------
        n_partners = 1                       # one negotiator per side
        ufun = OneShotUFun(
            ex_pin=0, ex_qin=0, ex_pout=0, ex_qout=0,
            input_product=0,
            input_agent=(to_update == "B"),
            output_agent=(to_update == "S"),
            production_cost=self.C_PROD,
            disposal_cost=self.C_DISP,
            storage_cost=self.C_STORE,
            shortfall_penalty=self.C_SHORT,
            input_penalty_scale=None,
            output_penalty_scale=None,
            storage_penalty_scale=None,
            n_input_negs=n_partners if to_update == "B" else 0,
            n_output_negs=n_partners if to_update == "S" else 0,
            current_step=0,
            agent_id=None,
            time_range=(0, 0),
            input_qrange=(0, self.max_q),
            input_prange=(-1, 1),
            output_qrange=(0, self.max_q),
            output_prange=(-1, 1),
            force_exogenous=True,
            n_lines=self.max_q,
            normalized=False,
        )

        self._cfr_round(0, need, None, to_update, "B", ufun)

    # ------------------------------------------------------------
    def _cfr_round(self, rnd, need, last_offer,
                   to_update, next_role, ufun) -> float:

        # ---------- terminal? ------------------------------------
        if rnd == 10 or last_offer == ACCEPT:
            return self._payoff(last_offer, next_role, need, ufun)

        # ---------- infoset key ----------------------------------
        phase = 0 if rnd==0 else 1 if rnd<=2 else 2 if rnd<=3 else 3
        if last_offer is None or last_offer == ACCEPT:
            qty_cmp = price_cmp = 0
        else:
            ql, pl = last_offer
            qty_cmp   = 0 if ql==need else int(math.copysign(1, ql-need))
            price_cmp = 0 if pl==0  else int(math.copysign(1, pl))
        I = info_key(next_role, phase, need, qty_cmp, price_cmp, 0)

        # ---------- σ via regret-matching ------------------------
        rvec = self.regret_sum[next_role][I]
        pos  = np.maximum(rvec, 0.0)
        sigma= pos/pos.sum() if pos.sum()>0 else np.full(self.nA, 1/self.nA)

        # ---------- sample chosen action -------------------------
        a_idx = self.nprng.choice(self.nA, p=sigma)
        chosen = self.action_set[a_idx]

        if a_idx == 0:       # ACCEPT chosen
            util_chosen = self._payoff(last_offer, next_role, need, ufun)
        else:
            util_chosen = self._cfr_round(
                rnd+1, need, chosen, to_update,
                "S" if next_role=="B" else "B", ufun)

        # ---------- regret update for traversing player ----------
        if next_role == to_update:
            a_prime = self.rng.randrange(self.nA)
            if a_prime == 0:
                u_prime = self._payoff(last_offer, next_role, need, ufun)
            else:
                alt = self.action_set[a_prime]
                u_prime = self._cfr_round(
                    rnd+1, need, alt, to_update,
                    "S" if next_role=="B" else "B", ufun)

            self.regret_sum[next_role][I][a_prime] += (u_prime - util_chosen)
            self.strategy_sum[next_role][I][a_idx] += 1.0

        return util_chosen

    # ------------------------------------------------------------
    def _payoff(self, last_offer, role, need, ufun) -> float:
        """Utility via OneShotUFun.from_offers()."""
        if last_offer is None or last_offer == ACCEPT:
            return 0.0
        q, p = last_offer
        outcome  = (q, 0, p)
        is_sell  = (role == "S")
        return ufun.from_offers([outcome], [is_sell])

class CFROneShotAgent(OneShotAgent):
    """Agent that negotiates using a pre‑trained CFR policy."""

    def init(self):
        policy_path = pathlib.Path(__file__).with_name("cfr_oneshot_selfplay_ufun.policy.json")
        if not policy_path.exists():
            # fallback to embedded empty dict (all infosets unseen)
            print("Policy not found.")
            self.policy: Dict[str, List[float]] = {}
        else:
            self.policy = json.loads(policy_path.read_text())

        issues = self.awi.current_input_issues or self.awi.current_output_issues
        self.price_min = issues[UNIT_PRICE].min_value
        self.price_max = issues[UNIT_PRICE].max_value
        self.mid_price = (self.price_min + self.price_max) // 2
        self.action_set = build_action_set(
            max_q=self.awi.n_lines,
            p_min=self.price_min,
            p_max=self.price_max,
            price_buckets=3
        )

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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300_000)
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--out", type=str,
                    default="cfr_oneshot_selfplay_ufun.policy.json")
    args = ap.parse_args()

    trainer = CFRTrainer(max_q=10, price_buckets=3)
    trainer.train(args.iters, save_every=args.save_every)
    pathlib.Path(args.out).write_text(json.dumps(trainer.average_strategy()))
    print(f"Final strategy saved → {args.out}")

# python3 cfr_selfplay_with_ufun.py --iters 300000 --save_every 150000 --out cfr_oneshot_selfplay_ufun.policy.json