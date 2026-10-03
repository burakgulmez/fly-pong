"""Train the central-complex steering network to play Pong with CMA-ES.

usage: python experiments/train.py --seed S [--budget FRAMES]
-> results/runs/seed{S}/{config.json, train_log.csv, theta.npy, test_games.csv, summary.json}
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import cma
import numpy as np
import pandas as pd

from src import connectome as C
from src.brain import Net

CFG = dict(rho=0.9, k_in=16, T=4, readout="per_output_neuron", popsize=16, games_per_eval=4, sigma0=0.5, budget_frames=4_000_000,
           val_every=10, val_games=16, test_games=100, w_points=2.0, w_hits=0.05)
TRAIN_MAX, VAL_BASE, TEST_BASE = 100_000_000, 500_000_000, 1_000_000_000


def make_net(cfg):
    W, z = C.load(cfg["rho"])
    # one readout weight per output neuron (the two-group left/right readout did not learn: results/diagnose.csv)
    labels = np.array([f"{str(z['types'][i])}_{str(z['side'][i])}_{i}" for i in range(int(z["n"]))])
    return Net(W, z["role"], k_in=cfg["k_in"], T=cfg["T"], out_labels=labels)


def fitness(res, cfg):
    """res[..., 0] agent points, 1 opponent points, 2 agent hits."""
    return (cfg["w_points"] * (res[:, :, 0] - res[:, :, 1]) + cfg["w_hits"] * res[:, :, 2]).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=int, default=CFG["budget_frames"])
    a = ap.parse_args()
    cfg = dict(CFG, budget_frames=a.budget)
    out = os.path.join(ROOT, "results", "runs", f"seed{a.seed}")
    os.makedirs(out, exist_ok=True)
    net = make_net(cfg)
    cfg["n_params"] = net.D
    cfg["out_groups"] = [str(x) for x in net.out_names]
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    rng = np.random.default_rng(a.seed)
    es = cma.CMAEvolutionStrategy(np.zeros(net.D), cfg["sigma0"],
                                  {"seed": a.seed + 1, "popsize": cfg["popsize"], "verbose": -9})
    val_seeds = np.arange(VAL_BASE, VAL_BASE + cfg["val_games"])
    best, best_theta, rows, frames, gen, t0 = -1e9, np.zeros(net.D), [], 0, 0, time.time()
    while frames < cfg["budget_frames"]:
        X = np.array(es.ask())
        res = net.evaluate(X, rng.integers(0, TRAIN_MAX, cfg["games_per_eval"]))
        f = fitness(res, cfg)
        es.tell(list(X), list(-f))
        frames += int(res[:, :, 3].sum())
        row = dict(gen=gen, frames=frames, time_s=time.time() - t0, fit_mean=f.mean(), fit_best=f.max())
        if gen % cfg["val_every"] == 0 or frames >= cfg["budget_frames"]:
            cand = np.stack([es.mean, X[np.argmax(f)]])
            vr = net.evaluate(cand, val_seeds)
            vf = fitness(vr, cfg)
            j = int(np.argmax(vf))
            row.update(val_points_for=vr[j, :, 0].mean(), val_points_against=vr[j, :, 1].mean(),
                       val_hits=vr[j, :, 2].mean())
            if vf[j] > best:
                best, best_theta = vf[j], cand[j].copy()
        rows.append(row)
        gen += 1
    train_time = time.time() - t0
    pd.DataFrame(rows).to_csv(os.path.join(out, "train_log.csv"), index=False)
    np.save(os.path.join(out, "theta.npy"), best_theta)
    seeds = np.arange(TEST_BASE, TEST_BASE + cfg["test_games"])
    r = net.evaluate(best_theta, seeds)[0]
    pd.DataFrame(dict(game_seed=seeds, points_for=r[:, 0], points_against=r[:, 1], hits=r[:, 2], frames=r[:, 3])).to_csv(
        os.path.join(out, "test_games.csv"), index=False)
    summ = dict(seed=a.seed, n_params=net.D, generations=gen, train_frames=frames, train_time_s=train_time,
                test_points_for=float(r[:, 0].mean()), test_points_against=float(r[:, 1].mean()),
                test_hits=float(r[:, 2].mean()), test_match_win_rate=float((r[:, 0] > r[:, 1]).mean()),
                test_match_loss_rate=float((r[:, 0] < r[:, 1]).mean()),
                test_return_rate=float(r[:, 2].sum() / max((r[:, 2] + r[:, 1]).sum(), 1)))
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
