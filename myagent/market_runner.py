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
    RandDistOneShotAgent,
    mp
]

for i in range(10):
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

    cfr_keys = [agent for agent in world_agent_scores if "CFR" in agent]
    print("CFR Agent Keys:", cfr_keys)
    for cfr in cfr_keys:
        shortfall_stats = {k: (v[0], v[-1]) for k, v in world.stats.items() if "CFR" in k and "shortfall_penalty" in k}
        score = world_agent_scores.get(cfr, None)
        print(f"{cfr}: score = {score}, shortfall penalties = {shortfall_stats}")