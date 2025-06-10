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
# from cfr_oneshot_agent_acceptinc import CFROneShotAgent as cfr_acceptinc
# from cfr_oneshot_agent_acceptinc_util import CFROneShotAgent as cfr_acceptinc_util
from cfr_builtin_util import CFROneShotAgent as cfr_builtin_util
# from cfr_selfplay import CFROneShotAgent as cfr_selfplay
from cfr_selfplay_with_ufun import CFROneShotAgent as cfr_selfplay_with_ufun
from collections import defaultdict

from negmas import SAOResponse, ResponseType, Outcome, SAOState
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE
from scml.oneshot.agent import OneShotAgent
# from MatchingPennies import MyAgent as mp

class CFRAgentMain(cfr_builtin_util):
    """
    This is the only class you *need* to implement. The current skeleton simply loads a single model
    that is supposed to be saved in MODEL_PATH (train.py can be used to train such a model).
    """

    _tau   = 0.4     # learning rate for trust EMA
    _floor = 0.1     # never drive trust to zero

    def _assess_offer_quality(self, offer, role: str, needed: int) -> float:
        """Rate offer quality from 0 (terrible) to 1 (excellent)"""
        if needed <= 0 or offer is None:
            return 0.0
        quantity_score = min(1.0, offer[QUANTITY] / needed)

        price_range = max(1, self.price_max - self.price_min)
        if role == "B":                      # buyer prefers lower prices
            price_score = 1.0 - ((offer[UNIT_PRICE] - self.price_min) / price_range)
        else:                                # seller prefers higher prices
            price_score = (offer[UNIT_PRICE] - self.price_min) / price_range
        price_score = min(max(price_score, 0.0), 1.0)
        return 0.5 * (quantity_score + price_score)

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

        self.outstanding[negotiator_id] += q
        
        return (q, self.awi.current_step, price)

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
            quality = self._assess_offer_quality(offer, role, needed)
            self._update_trust(negotiator_id, quality)
            return ResponseType.ACCEPT_OFFER

        # ───────────────── 3) quick pragmatic accept fallback ───
        price_ok = (
            offer[UNIT_PRICE] <= self.price_max if role == "B"
            else offer[UNIT_PRICE] >= self.price_min
        )
        if not price_ok and role=="B":
            # if we really need the supply but partner is low‐trust,
            # be *more* reluctant to accept from them
            threshold = self.trust[negotiator_id]
            if random.random() > threshold:
                quality = self._assess_offer_quality(offer, role, needed)
                self._update_trust(negotiator_id, quality)
                return ResponseType.REJECT_OFFER
        
        if offer[QUANTITY] <= needed and price_ok:
            quality = self._assess_offer_quality(offer, role, needed)
            self._update_trust(negotiator_id, quality)
            return ResponseType.ACCEPT_OFFER

        # ───────────────── 4) otherwise counter with CFR offer ──
        q, price = action                      # action unpack
        q = min(max(1, q), needed)             # clamp to remaining need
        quality = self._assess_offer_quality(offer, role, needed)
        self._update_trust(negotiator_id, quality)
        return SAOResponse(
            ResponseType.REJECT_OFFER,
            (q, self.awi.current_step, price)
        )

    def _update_trust(self, pid: str, reward: float):
        """EMA update: τ·reward  +  (1−τ)·old"""
        self.trust[pid] = (1 - self._tau) * self.trust[pid] + self._tau * reward

    def on_negotiation_success(self, contract, mechanism):
        partner = next(p for p in contract.partners if p != self.id)
        # reward = 1.0 because we got a contract
        self._update_trust(partner, 1.0)
        self.outstanding[partner] = 0

        if partner in self.awi.my_suppliers:
            self.rem_buy = max(0, self.rem_buy - contract.agreement["quantity"])
        else:
            self.rem_sell = max(0, self.rem_sell - contract.agreement["quantity"])

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