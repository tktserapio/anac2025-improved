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
#from cfr_builtin_util import CFROneShotAgent as cfr_builtin_util
# from cfr_selfplay import CFROneShotAgent as cfr_selfplay
# from myagent.old.cfr_selfplay_with_ufun import CFROneShotAgent as cfr_selfplay_with_ufun
from collections import defaultdict

from negmas import SAOResponse, ResponseType, Outcome, SAOState, Contract
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE
from scml.oneshot import OneShotAgent, OneShotSyncAgent, OneShotAWI
# from MatchingPennies import MyAgent as mp





import math
import json
import random
import pathlib
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

from negmas import SAOResponse, ResponseType, Outcome, SAOState
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE, OneShotAgent
from typing import Any

from itertools import chain, combinations
from collections import Counter
from numpy.random import choice

import logging, csv, os, uuid, datetime

import csv, os, atexit, datetime

def powerset(iterable):
    """Returns the powerset of an iterable."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def distribute_goods(goods, capacities):
    """Distributes 'goods' into 'capacities' fairly (without exceeding)."""
    bins = [0] * len(capacities)
    while goods > 0:
        distributed = False
        for i, capacity in sorted(enumerate(capacities), key=lambda x: bins[x[0]]):
            if bins[i] < capacity:
                bins[i] += 1
                goods -= 1
                distributed = True
                if goods == 0:
                    break
        if not distributed:
            break
    return bins


# ---- single shared writer + file handle -------------------------
_LOG_DIR          = "logs"
_SINGLE_CSV_PATH  = os.path.join(_LOG_DIR, "run.csv")
_SINGLE_CSV_FILE  = None      # will hold the open file object
_SINGLE_CSV_WRITER = None     # will hold the csv.writer


def _get_shared_writer() -> csv.writer:
    """Open logs/run.csv once per Python process and return the same writer."""
    global _SINGLE_CSV_FILE, _SINGLE_CSV_WRITER

    # First call?  -> make directory, open file, write header
    if _SINGLE_CSV_WRITER is None:
        os.makedirs(_LOG_DIR, exist_ok=True)
        new_file = not os.path.exists(_SINGLE_CSV_PATH)

        _SINGLE_CSV_FILE   = open(_SINGLE_CSV_PATH, "a", newline="")
        _SINGLE_CSV_WRITER = csv.writer(_SINGLE_CSV_FILE)

        if new_file:                        # header only once
            _SINGLE_CSV_WRITER.writerow([
                "day",
                "step",
                "partner",
                "phase",
                "action",
                "q_off",
                "p_off",
                "my_need_pre",
                "my_need_post",
                "decision",
                "reason"
            ])

        # make sure file closes on normal interpreter exit
        atexit.register(_SINGLE_CSV_FILE.close)

    return _SINGLE_CSV_WRITER










import pandas as pd
import numpy as np
from pathlib import Path

EXPECTED_COLS = [
    "day", "step", "partner", "phase", "action",
    "q_off", "p_off", "my_need_pre", "my_need_post",
    "decision", "reason",
]

def _load_run(path: str | Path) -> pd.DataFrame:
    """Read `run.csv`, coping gracefully if the header row is missing/garbled."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    # If the header is missing (or overwritten by a data row), reload with names
    if list(df.columns) != EXPECTED_COLS:
        df = pd.read_csv(path, header=None, names=EXPECTED_COLS)

    # strip any duplicated header rows that got appended accidentally
    df = df[df["day"] != "day"]

    # numeric coercions we’ll need later
    num = ["day", "step", "q_off", "p_off", "my_need_pre", "my_need_post"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")

    return df


def analyze_run_csv(path: str | Path = "logs/run.csv") -> None:
    """
    High‑level diagnostics for a single tournament run.

    Prints:
      • overall volume of log rows
      • contract‑level win/loss counts & acceptance rate
      • average quantity/price of accepted contracts
      • detailed counts of every `decision` value the agent produced
      • quick sanity‑check of offer behaviour (mean q/p for propose/response)
    """
    df = _load_run(path)

    # ─── contract outcomes ──────────────────────────────────────────────
    successes = df.query("phase == 'cb' and action == 'success'")
    failures  = df.query("phase == 'cb' and action == 'failure'")
    n_success, n_failure = len(successes), len(failures)
    n_negs               = n_success + n_failure

    print("\n================  RUN SUMMARY  ================\n")
    print(f"rows logged           : {len(df):>6}")
    print(f"distinct days         : {df['day'].nunique():>6}")
    print(f"negotiations recorded : {n_negs:>6}")
    print(f"  ├─ successes        : {n_success:>6}")
    print(f"  └─ failures         : {n_failure:>6}")
    if n_negs:
        print(f"accept‑rate           : {n_success / n_negs:>6.2%}")
    if n_success:
        print(
            f"mean qty accepted     : {successes['q_off'].mean():>6.2f}\n"
            f"mean price accepted   : {successes['p_off'].mean():>6.2f}"
        )

    # ─── decision breakdown (from the respond loop) ─────────────────────
    print("\n-------------  Decision Counts  --------------")
    print(
        df["decision"]
        .fillna("NA")
        .value_counts()
        .to_string()
    )

    # ─── offer behaviour ────────────────────────────────────────────────
    propose = df.query("action == 'propose'")
    respond = df.query("action == 'respond'")
    if len(propose) or len(respond):
        print("\n-------------  Offer Behaviour  --------------")
    if len(propose):
        print(
            f"mean propose q / p    : "
            f"{propose['q_off'].mean():.2f}  /  {propose['p_off'].mean():.2f}"
        )
    if len(respond):
        print(
            f"mean respond q / p    : "
            f"{respond['q_off'].mean():.2f}  /  {respond['p_off'].mean():.2f}"
        )
    print("\n==============================================\n")
























class MATAgent(OneShotSyncAgent):





    def _setup_logger(self,*, to_console: bool = False):
        """Attach the shared CSV writer + a simple console logger."""
        self._csv = _get_shared_writer()          # <– keeps the single CSV
        self._log_file = _SINGLE_CSV_FILE         # <-- keep the actual file handle
        # 2) Python logger  ─────────────────────────────────────────────────
        name          = self.id                   # one logger per agent
        self.logger   = logging.getLogger(name)

        # wipe any previous handlers to avoid duplicates
        self.logger.handlers.clear()
        self.logger.propagate = False             # do NOT bubble to root logger

        if to_console and self.verbose:           # only attach if explicitly asked
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            self.logger.addHandler(h)
        else:
            # a do‑nothing handler so `logger.debug(...)` is still safe to call
            self.logger.addHandler(logging.NullHandler())

        # choose whatever default level you like for *internal* logging calls
        self.logger.setLevel(logging.INFO)



    def init(self):
        """Initialize agent-specific tracking variables and configuration"""
        self.verbose = False
        self._setup_logger(to_console=False)
        self.first = True

        # Determine negotiation role and partner list
        if self.awi.level == 0:
            self.partners = self.awi.my_consumers
        else:
            self.partners = self.awi.my_suppliers

        # Track partner pricing history and negotiation outcomes
        self.partner_price_history = defaultdict(list)
        self.secured = 0  # total quantity secured so far in this round
        self.rejection = 0  # total number of rejections so far
        self.SOFT_THRESHOLD = 2.0
        self.HARD_THRESHOLD = 4.0



        self.PARAMS = {
            'soft_threshold': False,
            'delay_early_accept': False
        }







    def _current_threshold(self, r: float) -> float:
        """Threshold of acceptable quantity mismatch based on time."""
        mn, mx = 0, self.awi.n_lines // 2
        if self.PARAMS["soft_threshold"]:
            return mn + (mx - mn) * (r ** self.SOFT_THRESHOLD)
        return mn + (mx - mn) * (r ** self.HARD_THRESHOLD)


    def best_subset(self, needs, partners, offers, quantity_cost_tradeoff=0.9):
        """Finds the best subset of offers that minimize a loss function."""
        best_loss = float("inf")
        best_index = -1
        partner_list = list(powerset(partners))

        for i, subset in enumerate(partner_list):
            offered_qty = sum(offers[p][QUANTITY] for p in subset)
            qty_diff = abs(offered_qty - needs)

            if offered_qty > needs and (offered_qty - needs) / needs <= 0.20:
                qty_diff = 0
                
            cost = sum(offers[p][UNIT_PRICE] for p in subset)

            if offered_qty < needs:
                cost += (needs - offered_qty) * self.awi.current_shortfall_penalty
            elif offered_qty > needs:
                cost += (offered_qty - needs) * self.awi.current_disposal_cost

            loss = quantity_cost_tradeoff * qty_diff + (1 - quantity_cost_tradeoff) * cost

            if loss < best_loss:
                best_loss = loss
                best_index = i

        return partner_list[best_index]










    def counter_all(self, offers: dict[str, Outcome], states: dict[str, SAOState]):
        """Decide which offers to accept, reject, or counter (multi-agent negotiation)."""
        response = {}

        for needs, partners, issues in [
            (self.awi.needed_supplies, self.awi.my_suppliers, self.awi.current_input_issues),
            (self.awi.needed_sales, self.awi.my_consumers, self.awi.current_output_issues),
        ]:
            active = set(partners).intersection(offers.keys())
            if not active or needs <= 0:
                continue

            price = issues[UNIT_PRICE].rand()

            subset = self.best_subset(needs, active, offers)
            others = list(active.difference(subset))
            offered_qty = sum(offers[p][QUANTITY] for p in subset)

            # If good enough, accept best subset
            min_rel_time = min([s.relative_time for s in states.values()])
            threshold = self._current_threshold(min_rel_time)

            if abs(offered_qty - needs) <= threshold:
                capacity = [offers[o][QUANTITY] for o in others]
                redistrib = distribute_goods(abs(needs - offered_qty), capacity)

                # Accept best offers
                response.update({
                    p: SAOResponse(ResponseType.ACCEPT_OFFER, offers[p])
                    for p in subset
                })

                # Reject or counter the rest
                for i, o in enumerate(others):
                    q = redistrib[i]
                    if q == 0:
                        response[o] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                    else:
                        response[o] = SAOResponse(
                            ResponseType.REJECT_OFFER,
                            (q, self.awi.current_step, offers[o][UNIT_PRICE])
                        )
            else:
                # Not good enough — counter with redistributed needs
                capacity = [offers[p][QUANTITY] for p in active]
                redistrib = distribute_goods(needs, capacity)
                for i, p in enumerate(active):
                    q = redistrib[i]
                    if q == 0:
                        response[p] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                    else:
                        response[p] = SAOResponse(
                            ResponseType.REJECT_OFFER,
                            (q, self.awi.current_step, offers[p][UNIT_PRICE])
                        )


        for pid, res in response.items():
            kind = "accept" if res.response == ResponseType.ACCEPT_OFFER else "counter"
            self._csv.writerow([
                self.awi.current_step,
                states[pid].step,
                pid,
                "multi",
                kind,
                res.outcome[QUANTITY] if res.outcome else "",
                res.outcome[UNIT_PRICE] if res.outcome else "",
                self._needed(pid),
                "",          # pre‑need; post‑need tricky here
                kind.upper(),
                ""                # decision / reason
            ])


        return response









    def on_negotiation_success(self, contract: Contract, mechanism: OneShotAWI) -> None:
        """Called when a negotiation ends with agreement."""
        quantity = contract.agreement.get("quantity", 0)
        self.secured += quantity

        self._csv.writerow([
            self.awi.current_step,
            "",
            contract.partners[0],
            "cb",
            "success",
            contract.agreement["quantity"],
            contract.agreement["unit_price"],
            "",
            "",
            "ACCEPTED",
            ""
        ])
        self.logger.info(f"✔ success q={quantity}")


        if self.verbose:
            print(f"[SUCCESS] Got contract for {quantity} units. Total secured: {self.secured}")


    def on_negotiation_failure(
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: OneShotAWI,
        state: SAOState,
    ) -> None:
        """Log a negotiation failure and update counters."""
        self.rejection += 1

        # find the counterpart(s) – exclude own id, keep the rest
        others = [p for p in partners if p != self.id]
        partner_field = ",".join(others) if others else partners[0]  # fallback

        #log – keep exactly 11 columns to match the header
        self._csv.writerow([
            self.awi.current_step,                     # day
            state.step if state else "",               # step inside SAO
            partner_field,                             # partner(s)
            "cb", "failure",                           # phase, action
            "", "", "", "",                            # q_off … my_need_post
            "FAIL",                                    # decision
            annotation.get("reason", "")               # reason
        ])
        if self._log_file and not self._log_file.closed:
            self._log_file.flush()                     # crash‑safe

        self.logger.info(f"✖ fail with {partner_field}")
        if self.verbose:
            print(f"[FAILURE] Negotiation with {partner_field} failed "
                f"(total: {self.rejection})")







    def _is_selling(self, ami) -> bool:
        """Returns True if we're acting as a seller in this negotiation."""
        return ami.annotation["product"] == self.awi.my_output_product


    def propose(self, negotiator_id: str, state: SAOState):
        """Propose an offer to a given partner at a specific negotiation state."""
        step = state.step
        my_needs = self._needed(negotiator_id)

        if my_needs <= 0:
            return None  # No need to propose

        ami = self.get_nmi(negotiator_id)
        if not ami:
            return None

        # Quantity logic: split early, go all-in late
        if step <= 3:
            q = min(my_needs, math.ceil(my_needs / 2))  # cautious
        else:
            q = my_needs  # offer all that we need

        t = self.awi.current_step

        # Price logic based on role and phase
        if step <= 17:  # aggressive
            price = (
                ami.issues[UNIT_PRICE].max_value
                if self._is_selling(ami)
                else ami.issues[UNIT_PRICE].min_value
            )
        else:  # concession
            price = (
                ami.issues[UNIT_PRICE].min_value
                if self._is_selling(ami)
                else ami.issues[UNIT_PRICE].max_value
            )

        self._csv.writerow([
            self.awi.current_step,
            state.step,
            negotiator_id,
            "loop",
            "propose",
            q,
            price,
            my_needs,
            my_needs - q,
            "",
            ""
        ])
        self.logger.debug(
            f"→ propose to {negotiator_id}: q={q} p={price} "
            f"step={state.step} need={my_needs}"
        )


        return (q, t, price)






    def _needed(self, negotiator_id=None) -> int:
        """Compute remaining needed quantity (supplies or sales)."""
        base = (
            self.awi.current_exogenous_input_quantity
            if self.awi.level == 0         # L0 buys inputs
            else self.awi.current_exogenous_output_quantity  # L1 sells outputs
        )
        return max(0, base - self.secured)










    def respond(self, negotiator_id: str, state: SAOState, source: str = "") -> ResponseType:
        """Respond to an offer during ongoing negotiation and log everything."""
        offer = state.current_offer
        step  = state.step

        # default decision
        decision = ResponseType.REJECT_OFFER
        reason   = "no_offer"

        if not offer:
            return decision

        my_needs = self._needed(negotiator_id)
        offered_qty, offered_price = offer[QUANTITY], offer[UNIT_PRICE]

        # 1) already done
        if my_needs <= 0:
            decision, reason = ResponseType.END_NEGOTIATION, "need_met"

        # 2) perfect match
        elif offered_qty == my_needs:
            decision, reason = ResponseType.ACCEPT_OFFER, "exact_qty"

        # 3) helpful partial
        elif offered_qty < my_needs:
            ratio = offered_qty / my_needs
            if ratio >= 0.25:          # accept any offer that covers ≥25 %
                decision, reason = ResponseType.ACCEPT_OFFER, "partial_ok"
            else:
                # counter with *something* meaningful, not 1‑unit
                counter_q = max(1, math.ceil(my_needs * 0.25))
                return SAOResponse(ResponseType.REJECT_OFFER, (counter_q, self.awi.current_step, offered_price))

        # 4) almost good in late rounds
        elif abs(offered_qty - my_needs) < my_needs and step >= 18:
            decision, reason = ResponseType.ACCEPT_OFFER, "late_close"

        # else: keep default reject
        # ----------------------------------------------------------------

        self._csv.writerow([
            self.awi.current_step,
            step,
            negotiator_id,
            "loop",
            "respond",
            offered_qty,
            offered_price,
            my_needs,                                  # pre
            my_needs - (offered_qty if decision==ResponseType.ACCEPT_OFFER else 0),  # post
            decision.name,
            reason
        ])

        self.logger.debug(
            f"← respond {negotiator_id}: q={offered_qty} p={offered_price} "
            f"need={my_needs} step={step} ⇒ {decision.name}({reason})"
        )

        return decision






    def first_proposals(self) -> dict[str, Outcome | None]:
        """Generate first proposals to all active partners."""
        



        # Step and pricing based on role
        s = self.awi.current_step
        seller = self.awi.is_first_level
        issues = (
            self.awi.current_output_issues if seller else self.awi.current_input_issues
        )
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value
        price = pmax if seller else pmin  # Offer best price for counterpart

        # Distribute needs over partners still negotiating
        distribution = dict()
        for needs, partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            active = [pid for pid in partners if pid in self.negotiators.keys()]
            n = len(active)

            if needs <= 0 or n == 0:
                distribution.update({pid: 0 for pid in active})
            else:
                # Simple even distribution
                base = needs // n
                remainder = needs % n
                allocation = [base + 1 if i < remainder else base for i in range(n)]
                distribution.update(dict(zip(active, allocation)))

        # Format proposals
        proposals = {
            pid: (q, s, price) if q > 0 else None
            for pid, q in distribution.items()
        }

        if self.verbose:
            print(f"[First Proposals] Step={s}, Price={price}, Proposals={proposals}")


        for pid, q in distribution.items():
            self._csv.writerow([
                self.awi.current_step,     # day
                0,                         # step 0 inside SAO
                pid,
                "first",
                "propose",
                q,
                price,
                self.q,
                self.q - q,
                "",
                ""                    # accepted, reason
            ])




        return proposals






    def before_step(self):
        self.secured = 0
        self.rejection = 0

        # compute need & price range first  ─────────
        if self.awi.level == 0:
            self.q = self.awi.current_exogenous_input_quantity
            self.min_price = self.awi.current_output_issues[UNIT_PRICE].min_value
            self.max_price = self.awi.current_output_issues[UNIT_PRICE].max_value
            self.best_price = self.max_price
        else:
            self.q = self.awi.current_exogenous_output_quantity
            self.min_price = self.awi.current_input_issues[UNIT_PRICE].min_value
            self.max_price = self.awi.current_input_issues[UNIT_PRICE].max_value
            self.best_price = self.min_price
        # ───────────────────────────────────────────

        # now you can log safely
        self.logger.info(
            f"D{self.awi.current_step:03d} need={self.q} "
            f"price_range=({self.min_price},{self.max_price})"
        )





    def after_step(self):
        """Close the CSV on the very last production day."""
        if (
            self.awi.current_step == self.awi.n_steps_world - 1
            and hasattr(self, "_log_file")
            and self._log_file
            and not self._log_file.closed
        ):
            self._log_file.close()





















if __name__ == "__main__":
    import sys
    from helpers.runner import run

    run([MATAgent], sys.argv[1] if len(sys.argv) > 1 else "oneshot")
    analyze_run_csv()

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