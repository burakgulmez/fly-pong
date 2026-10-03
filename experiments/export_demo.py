"""Export demo/weights.js (and weights.json) from a trained network (default: seed 0)."""
import json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import numpy as np
from experiments.train import make_net, CFG
from src import connectome as C


def export(seed=0):
    net = make_net(CFG)
    _, z = C.load(CFG["rho"])
    run = os.path.join(ROOT, "results", "runs", f"seed{seed}")
    theta = np.load(os.path.join(run, "theta.npy"))
    summ = json.load(open(os.path.join(run, "summary.json")))
    return dict(seed=seed, n=int(net.n), indptr=net.indptr.tolist(), indices=net.indices.tolist(), data=net.wdata.tolist(),
                in_grp=net.in_grp.tolist(), out_grp=net.out_grp.tolist(), out_size=net.out_size.tolist(),
                k_in=int(net.k_in), k_out=int(net.k_out), T=int(net.T), theta=theta.tolist(), role=z["role"].tolist(),
                types=[str(t) for t in z["types"]], side=[str(t) for t in z["side"]], summary=summ)


if __name__ == "__main__":
    m = export(0)
    js = json.dumps(m, separators=(",", ":"))
    open(os.path.join(ROOT, "demo", "weights.json"), "w").write(js)
    open(os.path.join(ROOT, "demo", "weights.js"), "w").write("window.PONG_WEIGHTS=" + js + ";\n")
    print("exported seed 0:", m["n"], "neurons")
