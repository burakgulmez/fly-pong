"""Silence groups of neurons in the trained networks -> results/lesion.csv"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from experiments.train import make_net, CFG, TEST_BASE
from src import connectome as C

_, z = C.load()
role, types, side = z["role"], z["types"], z["side"]
net = make_net(CFG)
seeds = np.arange(TEST_BASE, TEST_BASE + CFG["test_games"])
conds = {"intact": np.zeros(net.n, bool), "all intermediates (hDelta)": role == 1,
         "all PFN inputs": role == 0, "PFL3 left": (types == "PFL3") & (side == "left"),
         "PFL3 right": (types == "PFL3") & (side == "right"), "DNa02 (both)": types == "DNa02"}
rows = []
for s in range(5):
    theta = np.load(os.path.join(ROOT, "results", "runs", f"seed{s}", "theta.npy"))
    for name, silenced in conds.items():
        net.mask[:] = 1.0
        net.mask[silenced] = 0.0
        r = net.evaluate(theta, seeds)[0]
        rows.append(dict(seed=s, condition=name, n_silenced=int(silenced.sum()), points_for=r[:, 0].mean(),
                         points_against=r[:, 1].mean(), hits=r[:, 2].mean(), match_win_rate=(r[:, 0] > r[:, 1]).mean()))
    net.mask[:] = 1.0
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "lesion.csv"), index=False)
print(df.pivot(index="condition", columns="seed", values="points_for").round(1).to_string())
print(df.pivot(index="condition", columns="seed", values="points_against").round(1).to_string())
