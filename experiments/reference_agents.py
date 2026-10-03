"""Ball-tracking rule, random moves and a still paddle on the 100 test matches -> results/reference_agents.csv"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from experiments.train import make_net, CFG, TEST_BASE

net = make_net(CFG)
seeds = np.arange(TEST_BASE, TEST_BASE + CFG["test_games"])
rows = []
for kind, name in [(1, "track_ball_rule"), (2, "random_moves"), (3, "paddle_still")]:
    r = net.evaluate(np.zeros(net.D), seeds, kind)[0]
    rows.append(dict(agent=name, points_for=r[:, 0].mean(), points_against=r[:, 1].mean(), hits=r[:, 2].mean(),
                     match_win_rate=(r[:, 0] > r[:, 1]).mean(), match_loss_rate=(r[:, 0] < r[:, 1]).mean(),
                     return_rate=r[:, 2].sum() / (r[:, 2] + r[:, 1]).sum()))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "reference_agents.csv"), index=False)
print(df.round(3).to_string())
