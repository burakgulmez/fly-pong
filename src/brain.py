"""Rate network with fixed connectome W and a trained interface; batched rollouts (numba).

x <- x + a(-x + g W tanh(x) + I),  I_i = W_in[grp(i)] . u + b_in[grp(i)] for input neurons.
Each game frame runs T inner steps. Readout: r_k = mean tanh(x) over output group k
(k = left / right hemisphere), paddle move = tanh(w_out . r + b_out) in [-1, 1].
theta layout: W_in (K_in x 6, row-major), b_in (K_in), w_out (K_out), b_out, leak a, gain g.
"""
import numpy as np
from numba import njit, prange

from src import game as F  # Pong


def n_params(k_in, k_out):
    return k_in * F.N_FEAT + k_in + k_out + 1 + 2


@njit(cache=True)
def policy_step(theta, u, x, phi, rec, drive, rvec, indptr, indices, wdata, in_grp, out_grp, out_size,
                k_in, k_out, T, mask):
    nf = F.N_FEAT
    p = k_in * nf
    for k in range(k_in):
        s = theta[p + k]
        for j in range(nf):
            s += theta[k * nf + j] * u[j]
        drive[k] = s
    p += k_in
    a = 0.05 + 0.95 / (1.0 + np.exp(-theta[p + k_out + 1]))
    g = 2.0 / (1.0 + np.exp(-theta[p + k_out + 2]))
    N = x.shape[0]
    for t in range(T):
        for i in range(N):
            s = 0.0
            for q in range(indptr[i], indptr[i + 1]):
                s += wdata[q] * phi[indices[q]]
            rec[i] = s
        for i in range(N):
            inp = 0.0
            if in_grp[i] >= 0:
                inp = drive[in_grp[i]]
            x[i] += a * (-x[i] + g * rec[i] + inp)
            phi[i] = np.tanh(x[i]) * mask[i]
    for k in range(k_out):
        rvec[k] = 0.0
    for i in range(N):
        if out_grp[i] >= 0:
            rvec[out_grp[i]] += phi[i]
    z = theta[p + k_out]
    for k in range(k_out):
        z += theta[p + k] * rvec[k] / out_size[k]
    return z


@njit(cache=True)
def play(theta, seed, kind, indptr, indices, wdata, in_grp, out_grp, out_size, k_in, k_out, T, mask,
         rec_act, rec_phi):
    """kind 0 = network, 1 = heuristic (track ball height), 2 = random move each frame, 3 = paddle still.
    Returns (agent points, opponent points, agent hits, frames)."""
    st = F.new_game(seed)
    N = indptr.shape[0] - 1
    x = np.zeros(N)
    phi = np.zeros(N)
    rec = np.zeros(N)
    drive = np.zeros(max(k_in, 1))
    rvec = np.zeros(max(k_out, 1))
    u = np.zeros(F.N_FEAT)
    r = F.rng_next((seed * 7 + 1) & 0xFFFFFFFF)
    t = 0
    while st[F.DONE] == 0:
        if t < rec_phi.shape[0]:
            for i in range(N):
                rec_phi[t, i] = phi[i]
        if kind == 0:
            F.features(st, u)
            z = policy_step(theta, u, x, phi, rec, drive, rvec, indptr, indices, wdata, in_grp, out_grp,
                            out_size, k_in, k_out, T, mask)
            act = np.tanh(z)
        elif kind == 1:
            act = F.heuristic_move(st)
        elif kind == 2:
            r = F.rng_next(r)
            act = ((r % 2001) - 1000) / 1000.0
        else:
            act = 0.0
        if t < rec_act.shape[0]:
            rec_act[t] = act
        F.step(st, act)
        t += 1
    return st[F.SA], st[F.SO], st[F.HITS], st[F.FRAME]


@njit(parallel=True, cache=True)
def play_batch(thetas, seeds, kind, indptr, indices, wdata, in_grp, out_grp, out_size, k_in, k_out, T, mask):
    P, S = thetas.shape[0], seeds.shape[0]
    out = np.zeros((P, S, 4))
    ea = np.zeros(0)
    ep = np.zeros((0, 0))
    for job in prange(P * S):
        p = job // S
        s = job % S
        a_, o_, h_, f_ = play(thetas[p], seeds[s], kind, indptr, indices, wdata, in_grp, out_grp, out_size,
                              k_in, k_out, T, mask, ea, ep)
        out[p, s, 0] = a_
        out[p, s, 1] = o_
        out[p, s, 2] = h_
        out[p, s, 3] = f_
    return out


class Net:
    def __init__(self, W, role, k_in=8, k_out=2, T=4, out_labels=None):
        n = W.shape[0]
        self.indptr = W.indptr.astype(np.int64)
        self.indices = W.indices.astype(np.int64)
        self.wdata = W.data.astype(float)
        self.in_grp = -np.ones(n, np.int64)
        self.out_grp = -np.ones(n, np.int64)
        for gi, ch in enumerate(np.array_split(np.where(role == 0)[0], k_in)):
            self.in_grp[ch] = gi
        outs = np.where(role == 2)[0]
        if out_labels is not None:  # one readout group per label (here: hemisphere side)
            names = sorted(set(out_labels[outs]))
            k_out = len(names)
            for i in outs:
                self.out_grp[i] = names.index(out_labels[i])
            self.out_names = names
        else:
            for gi, ch in enumerate(np.array_split(outs, k_out)):
                self.out_grp[ch] = gi
        self.out_size = np.bincount(self.out_grp[self.out_grp >= 0], minlength=k_out).astype(float)
        self.k_in, self.k_out, self.T, self.n = k_in, k_out, T, n
        self.mask = np.ones(n)
        self.D = n_params(k_in, k_out)

    def args(self):
        return (self.indptr, self.indices, self.wdata, self.in_grp, self.out_grp, self.out_size,
                self.k_in, self.k_out, self.T, self.mask)

    def evaluate(self, thetas, seeds, kind=0):
        return play_batch(np.atleast_2d(thetas).astype(float), np.asarray(seeds, np.int64), kind, *self.args())

    def trace(self, theta, seed, max_frames=4000, record_phi=False):
        acts = np.zeros(max_frames)
        ph = np.zeros((max_frames, self.n)) if record_phi else np.zeros((0, 0))
        res = play(np.asarray(theta, float), seed, 0, *self.args(), acts, ph)
        n = int(min(res[3], max_frames))
        return res, acts[:n], (ph[:n] if record_phi else None)
