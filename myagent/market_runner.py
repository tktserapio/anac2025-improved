#!/usr/bin/env python3
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from negmas import save_stats
from scml.oneshot.world import SCML2024OneShotWorld
from scml.oneshot.agents import SyncRandomOneShotAgent, GreedyOneShotAgent, RandDistOneShotAgent
# from myagent_builtin_util import CFRAgent as builtin_util
from myagent_builtin_util import CFRAgent as builtin
from logging_cfra import LoggingCFRAgent # this uses the main agent
from MatchingPennies import MyAgent as mp
import numpy as np

agent_types = [
    SyncRandomOneShotAgent,
    LoggingCFRAgent,      # ← use this now
    RandDistOneShotAgent,
    mp, 
    builtin
]
# Accumulators for aggregated shortfall_penalty stats and scores across runs
aggregated_neg_lengths = {}  # accumulator for negotiation lengths
aggregated_shortfall = {}
aggregated_scores = {}

for run_idx in range(1):
    world = SCML2024OneShotWorld(
        **SCML2024OneShotWorld.generate(
            agent_types=agent_types,
            n_steps=50
        ),
        construct_graphs=True,
    )

    world.run()
    world.plot_stats(pertype=True)

    # accumulate negotiation‐length stats
    for aid, agent in world.agents.items():
        if hasattr(agent, "neg_lengths"):
            # compute histogram: note that index 0 is skipped
            neg_hist = np.bincount(agent.neg_lengths, minlength=21)[1:]
            if aid not in aggregated_neg_lengths:
                aggregated_neg_lengths[aid] = neg_hist.copy()
            else:
                aggregated_neg_lengths[aid] += neg_hist

            # print per-run negotiation lengths (optional)
            print(f"\n── {aid} ──")
            print(f"  {len(agent.neg_lengths)} negotiations recorded")
            print("  lengths histogram:")
            for r, c in enumerate(neg_hist, 1):
                if c > 0:
                    print(f"    rounds={r:2d}: {c}")

    # Get world scores and filter CFR and LCF keys
    world_agent_scores = world.scores()
    print("Scores:", world_agent_scores)
    cfr_keys = [agent for agent in world_agent_scores if "CFR" in agent]
    lcf_keys = [agent for agent in world_agent_scores if "LCF" in agent]

    # Process CFR agents
    for cfr in cfr_keys:
        # Filter out only the shortfall_penalty stats for this CFR agent
        shortfall_stats = {k: (v[0], v[-1])
                        for k, v in world.stats.items()
                        if cfr in k and "shortfall_penalty" in k}
        # Accumulate the shortfall stats for CFR
        if cfr not in aggregated_shortfall:
            aggregated_shortfall[cfr] = {}
        for key, (first, last) in shortfall_stats.items():
            if key not in aggregated_shortfall[cfr]:
                aggregated_shortfall[cfr][key] = [0, 0]
            aggregated_shortfall[cfr][key][0] += first
            aggregated_shortfall[cfr][key][1] += last

        # Accumulate the score for this run for CFR
        if cfr not in aggregated_scores:
            aggregated_scores[cfr] = []
        aggregated_scores[cfr].append(world_agent_scores[cfr])

    # Process LCF agents
    for lcf in lcf_keys:
        # Filter out only the shortfall_penalty stats for this LCF agent
        shortfall_stats = {k: (v[0], v[-1])
                        for k, v in world.stats.items()
                        if lcf in k and "shortfall_penalty" in k}
        # Accumulate the shortfall stats for LCF
        if lcf not in aggregated_shortfall:
            aggregated_shortfall[lcf] = {}
        for key, (first, last) in shortfall_stats.items():
            if key not in aggregated_shortfall[lcf]:
                aggregated_shortfall[lcf][key] = [0, 0]
            aggregated_shortfall[lcf][key][0] += first
            aggregated_shortfall[lcf][key][1] += last

        # Accumulate the score for this run for LCF
        if lcf not in aggregated_scores:
            aggregated_scores[lcf] = []
        aggregated_scores[lcf].append(world_agent_scores[lcf])

# After running all iterations, print the aggregated negotiation lengths alongside scores and shortfall stats
print("\nAggregated Negotiation Length Stats over 5 runs:")
for aid, hist in aggregated_neg_lengths.items():
    print(f"{aid}:")
    for r, count in enumerate(hist, 1):
        if count > 0:
            print(f"  rounds={r:2d}: {count}")

print("\nAggregated Shortfall Penalty Stats over 5 runs:")
for cfr, stats in aggregated_shortfall.items():
    print(f"{cfr}:")
    for stat_key, (sum_first, sum_last) in stats.items():
        print(f"  {stat_key}: (sum_first={sum_first}, sum_last={sum_last})")
    print(f"  Scores over runs: {aggregated_scores.get(cfr, [])}")

# #!/usr/bin/env python3
# # run_with_logging.py

# import os
# import pandas as pd
# from pathlib import Path
# import matplotlib.pyplot as plt

# from negmas import save_stats
# from scml.oneshot.world import SCML2024OneShotWorld
# from scml.oneshot.agents import SyncRandomOneShotAgent, GreedyOneShotAgent, RandDistOneShotAgent
# from myagent import CFRAgent
# from logging_cfra import LoggingCFRAgent
# from MatchingPennies import MyAgent as mp
# import numpy as np

# agent_types = [
#     SyncRandomOneShotAgent,
#     CFRAgent,
#     RandDistOneShotAgent,
#     mp
# ]

# for i in range(1):
#     world = SCML2024OneShotWorld(
#         **SCML2024OneShotWorld.generate(
#             agent_types=agent_types, n_steps=50
#         ),
#         construct_graphs=True,
#     )

#     world.run()
#     world.plot_stats(pertype=True)
#     world_agent_scores = world.scores()

#     cfr_keys = [agent for agent in world_agent_scores if "CFR" in agent]
#     print("CFR Agent Keys:", cfr_keys)
#     for cfr in cfr_keys:
#         disposal_stats =  {k: (v[0], v[-1]) for k, v in world.stats.items() if ("CFR" in k) and ("@0" in k) and "disposal_cost" in k}
#         shortfall_p_stats = {k: (v[0], v[-1]) for k, v in world.stats.items() if ("CFR" in k) and ("@1" in k) and "shortfall_penalty" in k}
#         shortfall_q_stats = {k: (v[0], v[-1]) for k, v in world.stats.items() if ("CFR" in k) and ("@1" in k) and "shortfall_quantity" in k}
#         production_costs = {k: (v[0], v[-1]) for k, v in world.stats.items() if "CFR" in k and "production_costs" in k}
#         score = world_agent_scores.get(cfr, None)
#         print(f"{cfr}: score = {score}, disposal_cost = {disposal_stats}, shortfall_penalty = {shortfall_p_stats}, production_costs = {production_costs}")