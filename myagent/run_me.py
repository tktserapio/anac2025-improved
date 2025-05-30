from negmas import SAOResponse, ResponseType, Outcome, SAOState

from scml.oneshot.world import SCML2024OneShotWorld as W
from scml.oneshot import *
from scml.runner import WorldRunner
import pandas as pd
from rich.jupyter import print

from myagent import MyAgent
context = ANACOneShotContext(n_steps=50)
runner = WorldRunner(context, n_configs=10, n_repetitions=1)

# original CFR agent
runner(MyAgent)
orig = runner.score_summary()

print(orig)