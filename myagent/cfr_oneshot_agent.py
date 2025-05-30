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
    # cartesian product
    return [(q, p) for q in range(1, max_q + 1) for p in prices]


def info_key(role: str, round_no: int, qty_cmp: int, price_cmp: int) -> str:
    """Serialize infoset into a compact string key."""

    return f"{role}|{round_no}|{qty_cmp:+d}|{price_cmp:+d}"


#  CFR trainer (tabular, two‑player zero‑sum)
class CFRTrainer:
    """Self-play CFR trainer for *one* negotiation tree (20 rounds)."""

    def __init__(
        self,
        max_q: int = 10,
        price_buckets: int = 3,
        alpha: float = 0.5
    ):
        self.max_q = max_q
        self.price_buckets = price_buckets
        self.alpha = alpha  # weight on price term in utility

        # Use delta‐price encoding {-1 … +1} only for relative comparisons
        # but actual training now uses real price buckets
        self.action_set = build_action_set(max_q, -1, +1, price_buckets)
        self.n_actions = len(self.action_set)

        # infoset -> regret & strategy accumulators
        self.regret_sum: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )
        self.strategy_sum: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )

    def train(self, iters: int = 200_000):
        for t in range(iters):
            if t % 10000 == 0:
                print(f"{t} iterations done")
            for role in ("S", "B"):        # S = seller, B = buyer
                self._traverse(role)

    def average_strategy(self) -> Dict[str, List[float]]:
        return {
            I: self._normalise(s)
            for I, s in self.strategy_sum.items()
            if s.sum() > 0
        }

    def _traverse(self, role: str):
        """
        One pass of external-sampling CFR with a price-aware utility:
          U(q,p) = base_match   ±  α*(p - mid_price)
        where base_match = +1 if q==need else -|q-need|/need.
        """

        nA = self.n_actions
        needed = random.randint(1, self.max_q)
        price_samples = np.linspace(-1, 1, self.price_buckets, dtype=int).tolist()
        
        for round_no in range(20):
            # fictitious “last offer” from opponent
            last_q = random.randint(1, self.max_q)
            last_p = random.choice(price_samples)

            qty_cmp   = 0 if last_q == needed else int(math.copysign(1, last_q - needed))
            price_cmp = 0 if last_p == 0     else int(math.copysign(1, last_p))

            I = info_key(role, round_no, qty_cmp, price_cmp)

            # ------------- 3) current policy via regret matching
            sigma = self._get_strategy(I)

            # ------------- 4) sample ACTION to “play” this pass
            a_idx = np.random.choice(nA, p=sigma)
            q_sel, p_sel = self.action_set[a_idx]     # (unused except for clarity)

            # ------------- 5) compute utility for *all* actions
            U = np.empty(nA, dtype=np.float64)
            for i, (q, p) in enumerate(self.action_set):
                # quantity component
                if q == needed:
                    base = 1.0
                else:
                    base = -abs(q - needed) / needed
                # price component (seller likes +p, buyer likes –p)
                price_term = self.alpha * (p if role == "S" else -p)
                U[i] = base + price_term

            # ------------- 6) regret & strategy accumulation
            u_ref = U[a_idx]                 # utility of the sampled action
            self.regret_sum[I]   += U - u_ref  # vectorised counterfactual regrets
            self.strategy_sum[I] += sigma      # reach-prob = 1 in ext-sampling

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

# Runtime agent
class CFROneShotAgent(OneShotAgent):
    """Agent that negotiates using a pre‑trained CFR policy."""

    def init(self):
        # Load policy once per factory
        policy_path = pathlib.Path(__file__).with_suffix(".policy.json")
        if not policy_path.exists():
            # Fallback to embedded empty dict (all infosets unseen)
            print("Policy not found.")
            self.policy: Dict[str, List[float]] = {}
        else:
            self.policy = json.loads(policy_path.read_text())

        # For price comparisons we use mid‐price of input issues (they are same as output)
        issues = self.awi.current_input_issues or self.awi.current_output_issues
        self.price_min = issues[UNIT_PRICE].min_value
        self.price_max = issues[UNIT_PRICE].max_value
        self.mid_price = (self.price_min + self.price_max) // 2
        print(f"Midprice: {self.mid_price}")
        self.action_set = build_action_set(
            max_q=self.awi.n_lines,
            p_min=self.price_min,
            p_max=self.price_max,
            price_buckets=3
        )

        # Seed RNG reproducibly but unique per factory
        random.seed(hash(self.id) & 0xFFFF)
        np.random.seed(hash(self.id) & 0xFFFF)

    # -------------------- HELPERS ----------------------------
    def _needed(self, n_id: str | None) -> int:
        if n_id in self.awi.current_negotiation_details.get("sell", {}):
            return self.awi.needed_sales
        return self.awi.needed_supplies

    def _infoset(self, role: str, state: SAOState, needed: int) -> str:
        offer = state.current_offer
        if offer is None:
            # dummy values (treat as neutral)
            qty_cmp = price_cmp = 0
        else:
            qty_cmp = int(math.copysign(1, offer[QUANTITY] - needed)) if offer[QUANTITY] != needed else 0
            price_cmp = int(math.copysign(1, offer[UNIT_PRICE] - self.mid_price)) if offer[UNIT_PRICE] != self.mid_price else 0
        return info_key(role, state.step, qty_cmp, price_cmp)

    def _sample_action(self, infoset: str, role: str) -> Tuple[int, int]:
        pmf = self.policy.get(infoset)
        if pmf is None:
            # unseen: heuristic fallback – extreme price, needed quantity 1
            idx = 0 if role == "B" else len(self.action_set) - 1
        else:
            idx = np.random.choice(len(self.action_set), p=pmf)
        return self.action_set[idx]

    def _role(self, partner_id: str) -> str:
        return "S" if partner_id in self.awi.my_consumers else "B"

    # -------------------- NEGOTIATION ------------------------
    def propose(self, negotiator_id: str, state: SAOState) -> Outcome | None:
        role = self._role(negotiator_id)
        needed = self._needed(negotiator_id)
        if needed <= 0:
            return None
        I    = self._infoset(role, state, needed)
        q, price = self._sample_action(I, role)
        q = min(max(1, q), needed)  # never offer more than remaining need
        return (q, self.awi.current_step, price)

    def respond(self, negotiator_id: str, state: SAOState, source: str = "") -> ResponseType | SAOResponse:
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        needed = self._needed(negotiator_id)
        if needed <= 0:
            return ResponseType.END_NEGOTIATION

        role = self._role(negotiator_id)
        I    = self._infoset(role, state, needed)

        # Simple accept heuristic: if quantity <= need and price not terrible
        price_ok = (
            offer[UNIT_PRICE] <= self.mid_price if role == "B" else offer[UNIT_PRICE] >= self.mid_price
        )
        qty_ok = offer[QUANTITY] <= needed
        # Accept if quantity fits need and price on our good side
        if price_ok and qty_ok:
             return ResponseType.ACCEPT_OFFER

        # Otherwise counter with CFR sample
        q, price = self._sample_action(I, role)
        q = min(max(1, q), needed)
        return SAOResponse(ResponseType.REJECT_OFFER, (q, self.awi.current_step, price))

def _train_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="Run CFR self‑play training")
    ap.add_argument("--iters", type=int, default=200_000)
    ap.add_argument("--out", type=str, default="cfr_oneshot_agent.policy.json")
    args = ap.parse_args()

    if args.train:
        trainer = CFRTrainer()
        print(f"[CFR] Training for {args.iters:,} iterations…")
        trainer.train(args.iters)
        pol = trainer.average_strategy()
        pathlib.Path(args.out).write_text(json.dumps(pol))
        print(f"[CFR] Saved average strategy to {args.out}")


if __name__ == "__main__":
    _train_cli()

# python3 cfr_oneshot_agent.py --train --iters 300000 --out cfr_oneshot_agent.policy.json