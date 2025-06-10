#!/usr/bin/env python3
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from negmas import save_stats
from scml.oneshot.world import SCML2024OneShotWorld
from scml.oneshot.agents import (
    SyncRandomOneShotAgent,
    GreedyOneShotAgent,
    RandDistOneShotAgent,
)
from logging_cfra import LoggingCFRAgent
from MatchingPennies import MyAgent as mp

import numpy as np

agent_types = [
    SyncRandomOneShotAgent,
    LoggingCFRAgent,
    RandDistOneShotAgent,
    mp,
]

n_runs  = 10
n_steps = 50  # must match the `n_steps` you generate

# Accumulators
aggregated_neg_lengths    = {}
aggregated_shortfall_days = {}
aggregated_disposal_days  = {}
aggregated_scores         = {}

for run_idx in range(2):
    # build & run the world
    config = SCML2024OneShotWorld.generate(agent_types=agent_types, n_steps=n_steps)
    world  = SCML2024OneShotWorld(**config, construct_graphs=True)
    world.run()
    world.plot_stats(pertype=True)

    # 1) negotiation‐length logging (unchanged)
    for aid, agent in world.agents.items():
        if hasattr(agent, "neg_lengths"):
            hist = np.bincount(agent.neg_lengths, minlength=21)[1:]
            aggregated_neg_lengths.setdefault(aid, np.zeros_like(hist)).__iadd__(hist)

    # 2) score logging
    scores = world.scores()
    for aid, sc in scores.items():
        aggregated_scores.setdefault(aid, []).append(sc)

    # 3) shortfall/disposal logging
    # world.stats is a dict: stat_name → list of length n_steps
    for aid in scores.keys():
        # count days with shortfall > 0
        sf_keys = [k for k in world.stats if "shortfall_penalty" in k and aid in k]
        sf_days = sum(
            1
            for k in sf_keys
            for val in world.stats[k]
            if val > 0
        )
        aggregated_shortfall_days[aid] = aggregated_shortfall_days.get(aid, 0) + sf_days

        # count days with disposal > 0
        dp_keys = [k for k in world.stats if "disposal_cost" in k and aid in k]
        dp_days = sum(
            1
            for k in dp_keys
            for val in world.stats[k]
            if val > 0
        )
        aggregated_disposal_days[aid] = aggregated_disposal_days.get(aid, 0) + dp_days

# === After all runs: print summary ===
total_days = n_runs * n_steps

print(f"\n=== Shortfall vs Disposal over {n_runs} runs ({total_days} days each) ===\n")
for aid in aggregated_scores.keys():
    sf = aggregated_shortfall_days.get(aid, 0)
    dp = aggregated_disposal_days.get(aid, 0)
    score_list = aggregated_scores[aid]
    avg_score   = np.mean(score_list)
    print(f"{aid}:")
    print(f"  • avg score       = {avg_score:.3f}  (runs={len(score_list)})")
    print(f"  • shortfall days  = {sf}/{total_days} ({100*sf/total_days:.1f}%)")
    print(f"  • disposal days   = {dp}/{total_days} ({100*dp/total_days:.1f}%)\n")