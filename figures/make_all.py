"""Figures for the Pong study (PDF + PNG) from results/."""
import glob
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.train import make_net, CFG, TEST_BASE
from src import connectome as C
from src import game as G

R = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
BLUE, ORANGE, RED, GRAY, INK2 = "#2a78d6", "#eb6834", "#e34948", "#8a8983", "#52514e"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_circuit():
    W, z = C.load()
    role, types, side = z["role"], z["types"], z["side"]
    order = np.lexsort((side, types, role))
    M = W.toarray()[order][:, order]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(np.sign(M), cmap=matplotlib.colors.ListedColormap([RED, "#ffffff", BLUE]), vmin=-1, vmax=1,
              interpolation="nearest")
    for c in np.cumsum([np.sum(role == r) for r in range(3)])[:-1]:
        ax.axhline(c - 0.5, color=INK2, lw=0.6)
        ax.axvline(c - 0.5, color=INK2, lw=0.6)
    ax.set_xlabel("presynaptic (PFN inputs | hΔ | PFL3 + DNa02)")
    ax.set_ylabel("postsynaptic")
    ax.set_title(f"Central-complex steering circuit:\n{W.shape[0]} neurons, {W.nnz} connections (all excitatory)")
    save(fig, "fig1_circuit")


def fig_training():
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for i, f in enumerate(sorted(glob.glob(os.path.join(R, "runs", "seed*", "train_log.csv")))):
        d = pd.read_csv(f).dropna(subset=["val_points_for"])
        ax.plot(d.frames / 1e6, d.val_points_for - d.val_points_against, "-o", ms=2.5, lw=1.4, color=BLUE,
                alpha=0.35 + 0.13 * i, label="training seeds (5)" if i == 0 else None)
    ax.axhline(0, color=GRAY, lw=1, ls="--")
    ax.set_xlabel("training frames (millions)")
    ax.set_ylabel("validation point difference\n(network − opponent, mean of 16)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig2_training")


def fig_rally():
    net = make_net(CFG)
    theta = np.load(os.path.join(R, "runs", "seed0", "theta.npy"))
    seed = TEST_BASE
    _, acts, _ = net.trace(theta, seed, max_frames=900)
    st = G.new_game(seed)
    by, py, bx = [], [], []
    for a in acts:
        G.step(st, a)
        by.append(st[G.BY]); py.append(st[G.PY]); bx.append(st[G.BX])
    t = np.arange(1, len(by) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(7, 3.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.fill_between(t, np.array(py) - G.PAD_H, np.array(py) + G.PAD_H, color=BLUE, alpha=0.25, lw=0, label="network paddle")
    ax.plot(t, by, color=ORANGE, lw=1.4, label="ball height")
    ax.set_ylim(0, 1)
    ax.set_ylabel("height")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.set_title(f"First 900 frames of test match {seed} (seed-0 network)")
    axes[1].plot(t, bx, color=GRAY, lw=1.2)
    axes[1].set_ylabel("ball x")
    axes[1].set_xlabel("frame")
    save(fig, "fig3_rally")


def fig_lesion():
    les = pd.read_csv(os.path.join(R, "lesion.csv"))
    conds = list(dict.fromkeys(les.condition))
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    for i, c in enumerate(conds):
        v = (les[les.condition == c].points_for - les[les.condition == c].points_against).values
        ax.bar(i, v.mean(), color=BLUE if c == "intact" else GRAY, width=0.6, edgecolor="white")
        ax.scatter(i + np.linspace(-0.15, 0.15, len(v)), v, s=10, color="#0b0b0b", zorder=3, lw=0)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("points for − against\n(mean of 100 test matches)")
    ax.set_title("Silencing parts of the circuit (bars = mean of 5 networks, dots = networks)")
    save(fig, "fig4_lesion")


if __name__ == "__main__":
    for f in [fig_circuit, fig_training, fig_rally, fig_lesion]:
        f()
        print("ok", f.__name__)
