"""Quick learnability check of interface variants (1.5M frames, seed 0). Output: results/diagnose.csv"""
import os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import cma, numpy as np, pandas as pd
from src import connectome as C
from src.brain import Net
from experiments.train import fitness, CFG, VAL_BASE

def run(name, rho, T, per_neuron_out, k_in, budget=1_500_000):
    W, z = C.load(rho)
    labels = np.array([f"{s}_{i}" for i, s in enumerate(z["side"])]) if per_neuron_out else z["side"]
    net = Net(W, z["role"], k_in=k_in, T=T, out_labels=labels)
    es = cma.CMAEvolutionStrategy(np.zeros(net.D), 0.5, {"seed": 1, "popsize": 16, "verbose": -9})
    rng = np.random.default_rng(0); frames = 0; t0 = time.time()
    while frames < budget:
        X = np.array(es.ask()); res = net.evaluate(X, rng.integers(0, 10**8, 4))
        es.tell(list(X), list(-fitness(res, CFG))); frames += int(res[:, :, 3].sum())
    v = net.evaluate(es.mean, np.arange(VAL_BASE, VAL_BASE + 32))[0]
    # how different are the output readouts? std of left-right activity difference over a game
    return dict(variant=name, rho=rho, T=T, per_neuron_out=per_neuron_out, k_in=k_in, n_params=net.D,
                val_points_for=v[:, 0].mean(), val_points_against=v[:, 1].mean(), val_hits=v[:, 2].mean(),
                time_s=time.time() - t0)

rows = [run("base", 0.9, 4, False, 8), run("rho0.5", 0.5, 4, False, 8), run("T8", 0.9, 8, False, 8),
        run("per-neuron readout", 0.9, 4, True, 8), run("per-neuron readout, k_in 16", 0.9, 4, True, 16)]
d = pd.DataFrame(rows); d.to_csv(os.path.join(ROOT, "results", "diagnose.csv"), index=False)
print(d.round(2).to_string())
