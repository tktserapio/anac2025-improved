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
from collections import defaultdict
from typing import Dict, List, Tuple

from negmas import SAOResponse, ResponseType, Outcome, SAOState
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE
from scml.oneshot.agent import OneShotAgent

import numpy as np

# ------------------------------------------------------------------
ACCEPT = ("ACCEPT", None)

def build_action_set(max_q: int,
                     p_min: int,
                     p_max: int,
                     price_buckets: int = 2
) -> List[Tuple[int, int]]:
    prices  = np.linspace(p_min, p_max, price_buckets, dtype=int).tolist()
    actions = [(q, p) for q in range(1, max_q + 1) for p in prices]
    return [ACCEPT] + actions

def info_key(role: str, phase: int,
             need: int, qty_cmp: int,
             price_cmp: int, low_cash: int) -> str:
    return f"{role}|{phase}|{need}|{qty_cmp}|{price_cmp}|{low_cash}"

# ------------------------------------------------------------------
class CFRTrainer:
    """Outcome‐sampling CFR for a 20‐round buy/sell negotiation."""

    def __init__(self,
                 max_q: int         = 10,
                 price_buckets: int = 3,
                 beta_mult: float   = 5.0):
        self.max_q         = max_q
        self.price_buckets = price_buckets
        self.action_set    = build_action_set(max_q, -1, +1, price_buckets)
        self.nA            = len(self.action_set)

        # one regret & one strategy table *per role*
        self.regret_sum   = {r: defaultdict(lambda: np.zeros(self.nA))
                             for r in ("B","S")}
        self.strategy_sum = {r: defaultdict(lambda: np.zeros(self.nA))
                             for r in ("B","S")}

        # β so one‐unit mismatch ≫ price spread of 2
        self.beta = beta_mult * (1 - (-1))  # = 10 if beta_mult=5

        # reproducible streams
        self.rng = random.Random(42)
        self.np_rng = np.random.RandomState(42)
        self.LOW_CASH_PROBABILITY = 0.2

    def train(self, iters: int = 200_000,
              save_every: int = 0,
              out_pattern: str = "cfr_iter_{:06d}.json"):
        for t in range(1, iters + 1):
            if t % 10000 == 0:
                print(f"Iterations: {t}")
            # buyer update
            self._traverse("B")
            # seller update
            self._traverse("S")
            if save_every and t % save_every == 0:
                self._save(out_pattern.format(t))

    def average_strategy(self) -> Dict[str, List[float]]:
        out = {}
        for role, table in self.strategy_sum.items():
            for I, sigma_sum in table.items():
                s = sigma_sum.sum()
                if s > 0:
                    out[I] = (sigma_sum / s).tolist()
        return out

    def _save(self, fname: str):
        pathlib.Path(fname).write_text(json.dumps(self.average_strategy()))
        print(f"[CFR] saved → {fname}")

    # --------------------------------------------------------------
    # 1) small wrapper to pick a random need and role to update
    # --------------------------------------------------------------
    def _traverse(self, to_update: str):
        need = self.rng.randint(1, self.max_q)
        self._cfr_round(
            rnd        = 0,
            need       = need,
            last_offer = None,
            to_update  = to_update,
            next_role  = "B"       # buyer always starts at round 0
        )

    # --------------------------------------------------------------
    # 2) outcome-sampling CFR recursion
    # --------------------------------------------------------------
    def _cfr_round(self,
                   rnd: int,
                   need: int,
                   last_offer: Tuple[int,int]|None,
                   to_update: str,
                   next_role: str) -> float:
        # Terminal if 20 rounds done or if someone chose ACCEPT
        if rnd == 10 or last_offer == ACCEPT:
            return self._payoff_accept(last_offer, next_role, need)

        # Infoset features
        phase = 0 
        if rnd == 0:
                phase = 0
        elif rnd <= 1:
            phase = 1
        elif rnd <= 2:
            phase = 2
        else:
            phase = 3
        
        if last_offer is None or last_offer == ACCEPT:
            qty_cmp = price_cmp = 0
        else:
            ql, pl = last_offer
            qty_cmp   = 0 if ql == need else int(math.copysign(1, ql - need))
            price_cmp = 0 if pl == 0   else int(math.copysign(1, pl))

        low_cash = int(random.random() < self.LOW_CASH_PROBABILITY) #simlated probability of having a low cash balance
        
        I = info_key(next_role, phase, need, qty_cmp, price_cmp, low_cash)

        # Regret-matching policy σ(I)
        rvec  = self.regret_sum[next_role][I]
        pos   = np.maximum(rvec, 0.0)
        sigma = pos/pos.sum() if pos.sum()>0 else np.full(self.nA, 1/self.nA)

        # Sample the actual action a ~ σ(I)
        a_idx = self.np_rng.choice(self.nA, p=sigma)
        action = self.action_set[a_idx]

        # Compute utility of the chosen action
        if a_idx == 0:
            util_chosen = self._payoff_accept(last_offer, next_role, need)
        else:
            util_chosen = self._cfr_round(
                rnd+1, need, action, to_update,
                "S" if next_role=="B" else "B"
            )

        # If this is the player to update, sample one alt action for regret
        if next_role == to_update:
            a_prime = self.rng.randrange(self.nA)
            if a_prime == 0:
                u_prime = self._payoff_accept(last_offer, next_role, need)
            else:
                alt = self.action_set[a_prime]
                u_prime = self._cfr_round(
                    rnd+1, need, alt, to_update,
                    "S" if next_role=="B" else "B"
                )
            # only update regret for the sampled alt
            self.regret_sum[next_role][I][a_prime] += (u_prime - util_chosen)
            # accumulate frequency of the actual chosen a_idx
            self.strategy_sum[next_role][I][a_idx] += 1.0

        return util_chosen

    # --------------------------------------------------------------
    # 3) unified accept-payoff helper
    # --------------------------------------------------------------
    def _payoff_accept(self,
                       last_offer: Tuple[int,int]|None,
                       role: str,
                       need: int) -> float:
        if last_offer is None or last_offer == ACCEPT:
            return 0.0
        ql, pl = last_offer
        base = (-ql*pl) if role=="B" else (+ql*pl)
        return base - self.beta * abs(ql - need)

class CFROneShotAgent(OneShotAgent):
    """Agent that negotiates using a pre‑trained CFR policy."""

    def init(self):
        policy_path = pathlib.Path(__file__).with_name("cfr_oneshot_selfplay.policy.json")
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
    trainer = CFRTrainer(max_q=10, price_buckets=3)
    trainer.train(iters=300_000, save_every=100_000)
    # final policy
    pathlib.Path("cfr_oneshot_selfplay.policy.json")\
            .write_text(json.dumps(trainer.average_strategy()))