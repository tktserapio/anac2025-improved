import numpy as np

# collect per‐step penalties
records = []
for world in single_agent_runner.worlds_of():
    for day in world.daily_stats:
        records.append((day['shortfall_penalty'], day['disposal_cost']))
S, D = zip(*records)
print("Shortfall days: ", np.count_nonzero(np.array(S) > 0), "/", len(S))
print("Overbuy days:   ", np.count_nonzero(np.array(D) > 0), "/", len(D))