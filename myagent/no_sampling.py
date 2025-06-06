"""
fulltree_cfr_trainer.py
-----------------------
Tabular CFR that enumerates *every* joint action at each infoset.

  • depth              – how many offer rounds (≤ 20 is safe)
  • price_buckets      – 2 or 3 discrete price points
  • alpha              – weight on price term in toy utility
  • eps_converge       – stop when max |σ_new - σ_old| < eps   for *all* infosets
  • max_iters          – hard safety cap
"""

from __future__ import annotations
import math, random, json, pathlib
from collections import defaultdict
from typing      import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
def build_action_set(max_q:int, p_min:int, p_max:int, buckets:int) -> List[Tuple[int,int]]:
    prices = np.linspace(p_min, p_max, buckets, dtype=int).tolist()
    prices = sorted(set(prices))
    return [(q, p) for q in range(1, max_q + 1) for p in prices]

def info_key(role:str, rnd:int, qty_cmp:int, price_cmp:int)->str:
    return f'{role}|{rnd}|{qty_cmp:+d}|{price_cmp:+d}'

# ----------------------------------------------------------------------
class FullTreeCFRTrainer:
    """
    Full-tree Counterfactual-Regret-Minimisation.
    """

    def __init__(
        self,
        max_q:int = 3,
        price_buckets:int = 2,
        depth:int = 5,
        alpha:float = .5,
        eps_converge:float = 1e-4,
        max_iters:int = 100_000,
    ):
        self.max_q = max_q
        self.price_buckets = price_buckets
        self.depth = depth
        self.alpha = alpha
        self.eps_converge = eps_converge
        self.max_iters = max_iters

        self.action_set = build_action_set(max_q, -1, +1, price_buckets)
        self.nA = len(self.action_set)

        self.regret_sum  : Dict[str,np.ndarray] = defaultdict(
            lambda: np.zeros(self.nA, dtype=np.float64))
        self.strategy_sum: Dict[str,np.ndarray] = defaultdict(
            lambda: np.zeros(self.nA, dtype=np.float64))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train_until_converged(self) -> int:
        """
        Runs iterations until   max_infoset ||σ_new - σ_old||_∞  < eps_converge
        or max_iters reached.  Returns number of iterations executed.
        """
        old_sigma = defaultdict(lambda: np.full(self.nA, 1/self.nA))

        for it in range(1, self.max_iters+1):
            print(f"Iteration: {it}")
            for role in ("S","B"):
                needed = random.randint(1, self.max_q)
                self._cfr(role, rnd=0, needed=needed,
                          pi_self=1.0, pi_opp=1.0,
                          last_q=0, last_p=0)

            # --- check convergence every 100 iterations --------------
            if it % 100 == 0:
                max_diff = 0.0
                for I, s_sum in self.strategy_sum.items():
                    sigma_new = self._normalise(s_sum)
                    diff = np.max(np.abs(sigma_new - old_sigma[I]))
                    if diff > max_diff: max_diff = diff
                    old_sigma[I] = sigma_new         # update snapshot
                print(f"[CFR] iter={it:,}  max σ-change={max_diff:.3e}")
                if max_diff < self.eps_converge:
                    print(f"[CFR] converged after {it:,} iterations")
                    return it
        print(f"[CFR] reached hard cap {self.max_iters:,} iterations")
        return self.max_iters

    def average_strategy(self) -> Dict[str,List[float]]:
        return {I: self._normalise(s) for I,s in self.strategy_sum.items() if s.sum()>0}

    # ------------------------------------------------------------------
    # Internal recursion
    # ------------------------------------------------------------------
    def _cfr(self, role:str, rnd:int, needed:int,
             pi_self:float, pi_opp:float,
             last_q:int, last_p:int) -> float:
        """returns util to caller; updates regrets & strategy accumulators."""

        if rnd == self.depth:                   # terminal utility = 0
            return 0.0

        # --- infoset key ---------------------------------------------
        qty_cmp   = 0 if last_q==needed else int(math.copysign(1,last_q-needed))
        price_cmp = 0 if last_p==0      else int(math.copysign(1,last_p))
        I = info_key(role, rnd, qty_cmp, price_cmp)

        sigma = self._get_strategy(I)
        util  = np.zeros(self.nA)
        node_value = 0.0

        # loop over OUR actions --------------------------------------
        for a in range(self.nA):
            q_a, p_a = self.action_set[a]
            value_a = 0.0

            # loop over OPPONENT actions (assume uniform) ------------
            opp_prob = 1.0 / self.nA
            for o in range(self.nA):
                q_o, p_o = self.action_set[o]

                # recurse
                value_child = self._cfr(
                    role=role, rnd=rnd+1, needed=needed,
                    pi_self=pi_self * sigma[a],
                    pi_opp=pi_opp * opp_prob,
                    last_q=q_o, last_p=p_o
                )
                value_a += opp_prob * value_child

            util[a] = value_a
            node_value += sigma[a] * value_a

        # regret & strategy updates ----------------------------------
        regret = util - node_value
        self.regret_sum[I]   += pi_opp * regret
        self.strategy_sum[I] += pi_self * sigma

        return node_value

    # ----------------------------------------------------------------
    # Helper functions
    # ----------------------------------------------------------------
    def _get_strategy(self, I:str) -> np.ndarray:
        r_plus = np.maximum(self.regret_sum[I], 0.0)
        if r_plus.sum() > 0:
            return r_plus / r_plus.sum()
        return np.full(self.nA, 1.0 / self.nA)

    def _normalise(self, s:np.ndarray) -> np.ndarray:
        tot = s.sum()
        return (s / tot) if tot>0 else np.full_like(s, 1/len(s))
