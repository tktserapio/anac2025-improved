"""
**Submitted to ANAC 2024 SCML (OneShot track)**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML competition.
"""
import pickle
import torch
# from cfr_oneshot_agent import CFROneShotAgent
from cfr_oneshot_agent_acceptinc import CFROneShotAgent as cfr_acceptinc
from collections import defaultdict

from negmas import SAOResponse, ResponseType, Outcome, SAOState
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE
from scml.oneshot.agent import OneShotAgent
# from MatchingPennies import MyAgent as mp

class CFRAgent(cfr_acceptinc):
    """
    This is the only class you *need* to implement. The current skeleton simply loads a single model
    that is supposed to be saved in MODEL_PATH (train.py can be used to train such a model).
    """

    _tau   = 0.4     # learning rate for trust EMA
    _floor = 0.1     # never drive trust to zero

    def init(self):
        super().init()
        self.trust = defaultdict(lambda: 0.5)
        self._daily_reset()

    def before_step(self):
        super().before_step()
        self._daily_reset()

    def _daily_reset(self):
        self.rem_buy  = self.awi.needed_supplies
        self.rem_sell = self.awi.needed_sales
        self.outstanding = defaultdict(int)   # partner -> qty proposed today

    def _quota(self, partner_id: str, role: str) -> int:
        """Allocate portion of remaining need proportional to trust."""
        partner_ids = (self.awi.my_suppliers if role=="B" else self.awi.my_consumers)
        weights = [max(self.trust[p], self._floor) for p in partner_ids]
        total_w = sum(weights)
        if total_w == 0:                      # shouldn’t happen
            total_w = len(partner_ids)*self._floor
        my_w   = max(self.trust[partner_id], self._floor)
        need   = self.rem_buy if role=="B" else self.rem_sell
        return max(1, int(round(need * my_w / total_w)))

    # PROPOSE WITH NEW TRAINING
    def propose(self, negotiator_id, state):
        role   = self._role(negotiator_id)
        
        need   = self._needed(negotiator_id) - self.outstanding[negotiator_id]
        if need <= 0:
            return None

        offer_cap = self._quota(negotiator_id, role)
        need      = min(need, offer_cap)

        I = self._infoset(role, state, need)
        idx, (q, price) = self._sample_action(I, role, need)
        q = min(max(1, q), need)
        return (q, self.awi.current_step, price)

    # def propose(self, negotiator_id, state):
    #     role   = self._role(negotiator_id)
        
    #     need   = self._needed(negotiator_id) - self.outstanding[negotiator_id]
    #     if need <= 0:
    #         return None

    #     offer_cap = self._quota(negotiator_id, role)
    #     need      = min(need, offer_cap)

    #     I = self._infoset(role, state, need)
    #     q, price = self._sample_action(I, role, need)
    #     q = min(q, need)
    #     self.outstanding[negotiator_id] += q
    #     return (q, self.awi.current_step, price)

    # RESPOND WITH NEW TRAINING (includes ACCEPT in CFR):
    def respond(
        self,
        negotiator_id: str,
        state,
        source: str = ""
    ) -> ResponseType | SAOResponse:
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        # how many units we still need for THIS negotiation role
        needed = self._needed(negotiator_id)
        if needed <= 0:
            return ResponseType.END_NEGOTIATION

        role = self._role(negotiator_id)                 # "B" or "S"
        infoset = self._infoset(role, state, needed)

        # ───────────────── 1) sample from CFR policy  ────────────
        idx, action = self._sample_action(infoset, role, needed)
        #   • idx == 0  → policy says “ACCEPT”
        #   • action is (q, price) for idx > 0

        # ───────────────── 2) obey policy if it says ACCEPT ─────
        if idx == 0 and offer[QUANTITY] <= needed:
            return ResponseType.ACCEPT_OFFER

        # ───────────────── 3) quick pragmatic accept fallback ───
        price_ok = (
            offer[UNIT_PRICE] <= self.price_max if role == "B"
            else offer[UNIT_PRICE] >= self.price_min
        )
        if offer[QUANTITY] <= needed and price_ok:
            return ResponseType.ACCEPT_OFFER

        # ───────────────── 4) otherwise counter with CFR offer ──
        q, price = action                      # action unpack
        q = min(max(1, q), needed)             # clamp to remaining need
        return SAOResponse(
            ResponseType.REJECT_OFFER,
            (q, self.awi.current_step, price)
        )

    # def respond(self, negotiator_id, state, src=""):
    #     offer = state.current_offer
    #     if offer is None:
    #         return ResponseType.REJECT_OFFER

    #     role   = self._role(negotiator_id)
    #     need   = self._needed(negotiator_id)
    #     if need <= 0:
    #         return ResponseType.END_NEGOTIATION

    #     # price/qty filter identical to your previous version
    #     price_ok = (offer[UNIT_PRICE] <= self.price_max if role == "B"
    #                 else offer[UNIT_PRICE] >= self.price_min)
    #     qty_ok   = offer[QUANTITY] <= need

    #     if price_ok and qty_ok:
    #         if role == "B":
    #             self.rem_buy  -= offer[QUANTITY]
    #         else:
    #             self.rem_sell -= offer[QUANTITY]
    #         return ResponseType.ACCEPT_OFFER

    #     # counter-offer via CFR
    #     I = self._infoset(role, state, need)
    #     q, price = self._sample_action(I, role, need)
    #     q = min(q, need)
    #     return SAOResponse(ResponseType.REJECT_OFFER,
    #                        (q, self.awi.current_step, price))

    def _update_trust(self, pid: str, accepted: bool):
        prev = self.trust[pid]
        self.trust[pid] = (1 - self._tau) * prev + self._tau * (1.0 if accepted else 0.0)

    def on_negotiation_success(self, contract, mechanism):
        partner = next(p for p in contract.partners if p != self.id)
        self._update_trust(partner, accepted=True)
        self.outstanding[partner] = 0           # clear any pending qty

    def on_negotiation_failure(self,
                               partners,        # list of IDs
                               annotation,      # dict or None
                               mechanism,       # SAO mechanism object
                               state):          # final SAOState
        """
        Called by NegMAS when a bilateral negotiation ends without agreement.
        We lower trust for the partner(s) and clear any outstanding quantity.
        """
        for pid in partners:
            if pid == self.id:
                continue
            self._update_trust(pid, accepted=False)
            self.outstanding[pid] = 0


if __name__ == "__main__":
    import sys
    from helpers.runner import run

    # run([CFRAgent], sys.argv[1] if len(sys.argv) > 1 else "oneshot")
    run([CFRAgent], sys.argv[1] if len(sys.argv) > 1 else "oneshot")
