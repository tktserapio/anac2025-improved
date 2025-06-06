#!/usr/bin/env python3
# run_with_logging.py

import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from negmas import save_stats
from scml.oneshot.world import SCML2024OneShotWorld
from scml.oneshot.agents import SyncRandomOneShotAgent, GreedyOneShotAgent, RandDistOneShotAgent
from myagent import CFRAgent
from MatchingPennies import MyAgent as mp
import numpy as np

agent_types = [
    SyncRandomOneShotAgent,
    CFRAgent,
    mp
]

world = SCML2024OneShotWorld(
    **SCML2024OneShotWorld.generate(
        agent_types=agent_types, n_steps=50
    ),
    construct_graphs=True,
)
world.run()
world.plot_stats(pertype=True)
# world.plot_stats("bankrupt", ylegend=1.25)
world_agent_scores = world.scores()
print(world_agent_scores)

cfr_keys = [agent for agent in world_agent_scores if "CFR" in agent]
print("CFR Agent Keys:", cfr_keys)
print({k: (v[0], v[-1]) for k, v in world.stats.items() if "CFR" in k})

plt.show()

improved_shortfalls = {'shortfall_penalty_04CFR@1': (np.float64(0.0), np.float64(0.0)), 
'shortfall_penalty_06CFR@1': (np.float64(0.0), np.float64(0.0)), 
'shortfall_penalty_07CFR@1': (np.float64(0.0), np.float64(0.0)), 
'shortfall_penalty_08CFR@1': (np.float64(0.0), np.float64(45.70455694764701)),
'shortfall_penalty_09CFR@1': (np.float64(14.128381751607597), np.float64(0.0))}