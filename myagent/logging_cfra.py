# logging_cfra.py

from myagent import CFRAgent
import matplotlib.pyplot as plt
from negmas import SAOResponse, ResponseType, Outcome, SAOState

class LoggingCFRAgent(CFRAgent):
    """
    Same as your CFRAgent, but records
    how many rounds each bilateral negotiation took.
    """

    def init(self):
        super().init()
        # list of completed negotiation lengths
        self.neg_lengths: list[int] = []
        # remember the last seen round number per partner
        self._last_step: dict[str,int] = {}

    def propose(self, negotiator_id: str, state: SAOState) -> Outcome | None:
        # record that at this round we are in state.step
        self._last_step[negotiator_id] = state.step
        return super().propose(negotiator_id, state)

    def respond(self, negotiator_id: str, state: SAOState, source: str = "") -> SAOResponse | ResponseType:
        self._last_step[negotiator_id] = state.step
        return super().respond(negotiator_id, state, source)

    def on_negotiation_success(self, contract, mechanism):
        # figure out which partner just succeeded
        partner = next(p for p in contract.partners if p != self.id)
        # pull out the last step we saw for that partner
        last = self._last_step.pop(partner, None)
        if last is not None:
            # +1 because step is zero-indexed
            self.neg_lengths.append(last + 1)
        # chain up to original handling
        super().on_negotiation_success(contract, mechanism)

    def on_negotiation_failure(self,
                               partners: list[str],
                               annotation,
                               mechanism,
                               state: SAOState):
        # same idea on failure
        for partner in partners:
            last = self._last_step.pop(partner, None)
            if last is not None:
                self.neg_lengths.append(last + 1)
        super().on_negotiation_failure(partners, annotation, mechanism, state)