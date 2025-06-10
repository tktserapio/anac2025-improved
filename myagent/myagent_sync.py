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
# from cfr_selfplay import CFROneShotAgent as cfr_selfplay
from cfr_selfplay_with_ufun import CFROneShotAgent as cfr_selfplay_with_ufun
from collections import defaultdict

from negmas import SAOResponse, ResponseType, Outcome, SAOState
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE, OneShotSyncAgent
from scml.oneshot.agent import OneShotAgent
# from MatchingPennies import MyAgent as mp

class CFRAgentMain(cfr_builtin_util, OneShotSyncAgent):
    """CFR-policy agent with trust-weighted quantity distribution."""
    _tau   = 0.4
    _floor = 0.1

    # ---------------- initialisation -------------------------
    def init(self):
        cfr_builtin_util.init(self)           # loads CFR policy, sets prices
        self.trust = defaultdict(lambda: 0.5)
        self._daily_reset()

    def before_step(self):
        OneShotSyncAgent.before_step(self)
        self._daily_reset()

    def _daily_reset(self):
        self.rem_buy  = self.awi.needed_supplies
        self.rem_sell = self.awi.needed_sales
        self.outstanding = defaultdict(int)    # qty proposed but not yet accepted

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
        return 0.9 * quantity_score + 0.1 * price_score

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

    # ---------------- trust helpers --------------------------
    def _distribute(self, partners, need) -> dict:
        """Return {partner: qty} summing to need, proportional to trust."""
        if need <= 0:
            return {p: 0 for p in partners}
        weights = [max(self.trust[p], self._floor) for p in partners]
        total_w = sum(weights)
        alloc = {p: int(round(need * w / total_w)) for p, w in zip(partners, weights)}
        # fix rounding
        diff = need - sum(alloc.values())
        for p in partners:
            if diff == 0:
                break
            alloc[p] += 1
            diff -= 1
        return alloc

    # =========================================================
    #  SyncAgent callbacks
    # =========================================================
    def first_proposals(self):
        """One offer per partner at the start of each negotiation day."""
        offers = {}

        # ---------- suppliers (we are buyer) -----------------
        alloc_in = self._distribute(self.awi.my_suppliers, self.rem_buy)
        for pid, qty in alloc_in.items():
            offers[pid] = self._make_offer(pid, "B", qty)

        # ---------- consumers (we are seller) ----------------
        alloc_out = self._distribute(self.awi.my_consumers, self.rem_sell)
        for pid, qty in alloc_out.items():
            offers[pid] = self._make_offer(pid, "S", qty)

        return offers

    def counter_all(self, offers, states):
        """Respond to *all* partner offers in one shot."""
        responses = {}

        for pid, offer in offers.items():
            role   = self._role(pid)
            needed = self._needed(pid)

            # trust update on every received offer
            qscore = self._assess_offer_quality(offer, role, needed)
            self._update_trust(pid, qscore)

            # nothing left to trade with this partner
            if needed <= 0:
                continue

            # CFR‐based decision
            infoset              = self._infoset(role, states[pid], needed)
            idx, (q, price)      = self._sample_action(infoset, role, needed)

            # --- accept branch -----------------------------------
            if idx == 0 and offer[QUANTITY] <= needed:
                self._book_accept(pid, role, offer[QUANTITY])
                responses[pid] = ResponseType.ACCEPT_OFFER
                self._update_trust(pid, 1.0)
                continue

            # --- pragmatic accept --------------------------------
            price_ok = (
                offer[UNIT_PRICE] <= self.price_max if role == "B"
                else offer[UNIT_PRICE] >= self.price_min
            )
            if offer[QUANTITY] <= needed and price_ok:
                self._book_accept(pid, role, offer[QUANTITY])
                responses[pid] = ResponseType.ACCEPT_OFFER
                self._update_trust(pid, 1.0)
                continue

            # --- counter ----------------------------------------
            q = min(max(1, q), needed)
            responses[pid] = SAOResponse(
                ResponseType.REJECT_OFFER,
                (q, self.awi.current_step, price)
            )
            self.outstanding[pid] += q

        return responses

    # =========================================================
    #  Helpers for offers / accounting
    # =========================================================
    def _make_offer(self, pid: str, role: str, qty: int):
        """Return SAOResponse or None for a first proposal."""
        if qty == 0:
            return None
        state   = self.awi.current_states[pid]
        infoset = self._infoset(role, state, qty)
        idx, (q, price) = self._sample_action(infoset, role, qty)

        if idx == 0:                               # unlikely, but handle
            return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)

        q = min(max(1, q), qty)
        self.outstanding[pid] += q
        return SAOResponse(ResponseType.REJECT_OFFER,
                           (q, self.awi.current_step, price))

    def _book_accept(self, pid: str, role: str, qty: int):
        """Adjust remaining needs and outstanding counters after accept."""
        if role == "B":
            self.rem_buy  = max(0, self.rem_buy  - qty)
        else:
            self.rem_sell = max(0, self.rem_sell - qty)
        self.outstanding[pid] = max(0, self.outstanding[pid] - qty)

    # =========================================================
    #  Success / failure hooks (unchanged)
    # =========================================================
    def on_negotiation_success(self, contract, mechanism):
        partner = next(p for p in contract.partners if p != self.id)
        qty     = contract.agreement["quantity"]
        role    = "B" if partner in self.awi.my_suppliers else "S"
        self._book_accept(partner, role, qty)
        self._update_trust(partner, 1.0)

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