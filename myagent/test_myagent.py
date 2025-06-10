"""
**Submitted to ANAC 2024 SCML (OneShot track)**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML competition.
"""
import pickle
import json
import math
import pathlib
import random
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
# from cfr_oneshot_agent import CFROneShotAgent
# from cfr_oneshot_agent_acceptinc import CFROneShotAgent as cfr_acceptinc
# from cfr_oneshot_agent_acceptinc_util import CFROneShotAgent as cfr_acceptinc_util
from cfr_builtin_util import CFROneShotAgent as cfr_builtin_util
from cfr_builtin_util_sync import CFROneShotAgent as cfr_builtin_util_sync
# from cfr_selfplay import CFROneShotAgent as cfr_selfplay
from cfr_selfplay_with_ufun import CFROneShotAgent as cfr_selfplay_with_ufun
from collections import defaultdict

from negmas import SAOResponse, ResponseType, Outcome, SAOState
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE
from scml.oneshot.agent import OneShotAgent
# from MatchingPennies import MyAgent as mp

class CFRAgentMain(cfr_builtin_util):
    """CFR-policy agent with trust-weighted quantity distribution."""

    _tau   = 0.4    # EMA learning rate
    _floor = 0.1    # minimum trust

    # so the one we trust the most can have a lot
    # the one we trust the least can just have nothing

    # we should drop a negotiation when it's not going well
    # - we need to reassess how much our quantity mismatch
    # before every action, make a check on the 
    # trust update end of neg, but the quantities you ask for each negotiation should be updating very frequently

    # --------------- Offer-quality metric -------------------------
    def _assess_offer_quality(self, offer, role: str, needed: int) -> float:
        if needed <= 0 or offer is None:
            return 0.0
        quantity_score = min(1.0, offer[QUANTITY] / needed)
        price_range = max(1, self.price_max - self.price_min)
        if role == "B":
            price_score = 1.0 - ((offer[UNIT_PRICE] - self.price_min) / price_range)
        else:
            price_score = (offer[UNIT_PRICE] - self.price_min) / price_range
        price_score = min(max(price_score, 0.0), 1.0)
        return 0.6 * quantity_score + 0.4 * price_score

    # --------------- initialisation ------------------------------
    def init(self):
        super().init()
        self.trust = defaultdict(lambda: 0.5)
        self._daily_reset()

    def before_step(self):
        super().before_step()
        self._daily_reset()

    def _daily_reset(self):
        # slow trust decay ★
        self.rem_buy  = self.awi.needed_supplies
        self.rem_sell = self.awi.needed_sales
        self.outstanding = defaultdict(int)

    # --------------- trust-weighted quota ------------------------
    def _quota(self, partner_id: str, role: str) -> int:
        partner_ids = (self.awi.my_suppliers if role == "B" else self.awi.my_consumers)
        need = self.rem_buy if role == "B" else self.rem_sell
        need -= self.outstanding[partner_id]
        if need <= 0:
            return 0                                    # ★ no forced 1-unit
        weights = [max(self.trust[p], self._floor) for p in partner_ids]
        total_w = sum(weights) or len(partner_ids) * self._floor
        my_w    = max(self.trust[partner_id], self._floor)
        return int(round(need * my_w / total_w))

    # --------------- needed after subtracting offers -------------
    def _needed(self, pid: str) -> int:
        base = self.awi.needed_supplies if pid in self.awi.my_suppliers else self.awi.needed_sales
        return max(0, base - self.outstanding[pid])

    def _role(self, partner_id: str) -> str:
        #is [partner_id] a seller or a buyer?
        return "S" if partner_id in self.awi.my_consumers else "B"

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

    # --------------- PROPOSE -------------------------------------
    def propose(self, pid, state):
        role  = self._role(pid)
        need  = self._needed(pid)
        if need == 0:
            return None
        cap   = self._quota(pid, role)
        if cap == 0:
            return None
        need  = min(need, cap)

        I = self._infoset(role, state, need)
        idx, (q, price) = self._sample_action(I, role, need)
        q = min(max(1, q), need)

        self.outstanding[pid] += q
        return (q, self.awi.current_step, price)

    # if we can wait to get the offers first, then just do resopnd instead of proposing!!
    # base it off of matching quantities
    # --------------- RESPOND -------------------------------------
    def respond(self, negotiator_id, state, source=""):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        needed = self._needed(negotiator_id)
        if needed == 0:
            return ResponseType.END_NEGOTIATION

        role   = self._role(negotiator_id)
        infoset = self._infoset(role, state, needed)
        idx, action = self._sample_action(infoset, role, needed)

        # ---------- accept paths ----------
        if idx == 0 and offer[QUANTITY] <= needed:
            self._record_accept(negotiator_id, offer, role)
            return ResponseType.ACCEPT_OFFER

        price_ok = offer[UNIT_PRICE] <= self.price_max if role == "B" else offer[UNIT_PRICE] >= self.price_min
        if offer[QUANTITY] <= needed and price_ok:
            self._record_accept(negotiator_id, offer, role)
            return ResponseType.ACCEPT_OFFER

        # ---------- reject / counter ----------
        q, price = action
        q = min(max(1, q), needed)
        self._update_trust(negotiator_id, self._assess_offer_quality(offer, role, needed))
        return SAOResponse(ResponseType.REJECT_OFFER, (q, self.awi.current_step, price))

    def _record_accept(self, pid, offer, role):
        q = offer[QUANTITY]
        if role == "B":
            self.rem_buy  = max(0, self.rem_buy  - q)
        else:
            self.rem_sell = max(0, self.rem_sell - q)
        self.outstanding[pid] = max(0, self.outstanding[pid] - q)
        self._update_trust(pid, self._assess_offer_quality(offer, role, q))

    # --------------- TRUST UPDATE --------------------------------
    def _update_trust(self, pid: str, reward: float):
        self.trust[pid] = (1 - self._tau) * self.trust[pid] + self._tau * reward

    # --------------- SUCCESS / FAILURE ---------------------------
    def on_negotiation_success(self, contract, mechanism):
        pid = next(p for p in contract.partners if p != self.id)
        q   = contract.agreement["quantity"]
        if pid in self.awi.my_suppliers:
            self.rem_buy  = max(0, self.rem_buy  - q)
        else:
            self.rem_sell = max(0, self.rem_sell - q)
        self.outstanding[pid] = 0
        self._update_trust(pid, 1.0)

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        for pid in partners:
            if pid != self.id:
                self._update_trust(pid, 0.0)
                self.outstanding[pid] = 0

if __name__ == "__main__":
    import sys
    from helpers.runner import run

    run([CFRAgentMain], sys.argv[1] if len(sys.argv) > 1 else "oneshot")


# print(
        #     f"ex_pin: {self.ufun.ex_pin}, ex_qin: {self.ufun.ex_qin}, "
        #     f"ex_pout: {self.ufun.ex_pout}, ex_qout: {self.ufun.ex_qout}, "
        #     f"input_product: {self.ufun.input_product}, input_agent: {self.ufun.input_agent}, "
        #     f"output_agent: {self.ufun.output_agent}, production_cost: {self.ufun.production_cost}, "
        #     f"disposal_cost: {self.ufun.disposal_cost}, storage_cost: {self.ufun.storage_cost}, "
        #     f"shortfall_penalty: {self.ufun.shortfall_penalty}, "
        #     f"n_input_negs: {self.ufun.n_input_negs}, n_output_negs: {self.ufun.n_output_negs}, "
        #     f"current_step: {self.ufun.current_step}, agent_id: {self.ufun.agent_id}, "
        #     f"time_range: {self.ufun.time_range}, input_qrange: {self.ufun.input_qrange}, "
        #     f"input_prange: {self.ufun.input_prange}, output_qrange: {self.ufun.output_qrange}, "
        #     f"output_prange: {self.ufun.output_prange}, force_exogenous: {self.ufun.force_exogenous}, "
        #     f"n_lines: {self.ufun.n_lines}, normalized: {self.ufun.normalized}"
        # )