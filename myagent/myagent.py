"""
**Submitted to ANAC 2024 SCML (OneShot track)**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML competition.
"""
import pickle
import torch
from cfr_oneshot_agent import CFROneShotAgent

class MyAgent(CFROneShotAgent):
    """
    This is the only class you *need* to implement. The current skeleton simply loads a single model
    that is supposed to be saved in MODEL_PATH (train.py can be used to train such a model).
    """
    def init(self):
        super().init()


if __name__ == "__main__":
    import sys
    from helpers.runner import run

    run([MyAgent], sys.argv[1] if len(sys.argv) > 1 else "oneshot")
