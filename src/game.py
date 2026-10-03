"""Deterministic Pong (numba). Court is 1x1. The agent paddle is on the left, a simple
built-in opponent on the right. A match ends when one side reaches WIN_POINTS or after
MAX_FRAMES frames. Serve angles come from the same xorshift32 as the JS demo.
Every serve goes toward the agent.
"""
import numpy as np
from numba import njit

MASK32 = 0xFFFFFFFF
PAD_H = 0.10          # paddle half-height
AGENT_X = 0.06        # agent paddle face x
OPP_X = 0.94          # opponent paddle face x
BALL_R = 0.015
SERVE_SPEED = 0.014
MAX_SPEED = 0.028
SPEEDUP = 1.04
MAX_BOUNCE = 0.9      # radians at the paddle edge
AGENT_V = 0.020       # max agent paddle speed per frame
OPP_V = 0.0070        # opponent paddle speed per frame
REACT_X = 0.5         # opponent only tracks the ball once it passes this x
WIN_POINTS = 11
MAX_FRAMES = 6000
N_FEAT = 6
FEATURE_NAMES = ["ball_minus_paddle", "ball_x", "ball_vx", "ball_vy", "paddle_y", "ball_y"]

# state indices (float64 array)
BX, BY, VX, VY, PY, OY, SA, SO, HITS, FRAME, DONE, RNG = range(12)


@njit(cache=True)
def rng_next(x):
    x ^= (x << 13) & MASK32
    x ^= x >> 17
    x ^= (x << 5) & MASK32
    return x & MASK32


@njit(cache=True)
def serve(st):
    x = rng_next(int(st[RNG]))
    st[RNG] = x
    ang = -0.6 + 1.2 * ((x % 10000) / 9999.0)
    st[BX] = 0.5
    st[BY] = 0.5
    st[VX] = -SERVE_SPEED * np.cos(ang)
    st[VY] = SERVE_SPEED * np.sin(ang)


@njit(cache=True)
def new_game(seed):
    x = (seed * 2654435761 + 12345) & MASK32
    if x == 0:
        x = 1
    for _ in range(4):
        x = rng_next(x)
    st = np.zeros(12)
    st[PY] = 0.5
    st[OY] = 0.5
    st[RNG] = x
    serve(st)
    return st


@njit(cache=True)
def _bounce(st, face_x, paddle_y, direction):
    off = (st[BY] - paddle_y) / PAD_H
    if off > 1.0:
        off = 1.0
    if off < -1.0:
        off = -1.0
    speed = np.sqrt(st[VX] * st[VX] + st[VY] * st[VY]) * SPEEDUP
    if speed > MAX_SPEED:
        speed = MAX_SPEED
    ang = off * MAX_BOUNCE
    st[VX] = direction * speed * np.cos(ang)
    st[VY] = speed * np.sin(ang)
    st[BX] = face_x + direction * BALL_R


@njit(cache=True)
def step(st, move):
    """move in [-1, 1]: agent paddle velocity as a fraction of AGENT_V (+ = up)."""
    if st[DONE] > 0:
        return
    if move > 1.0:
        move = 1.0
    if move < -1.0:
        move = -1.0
    st[PY] = st[PY] + AGENT_V * move
    if st[PY] < PAD_H:
        st[PY] = PAD_H
    if st[PY] > 1.0 - PAD_H:
        st[PY] = 1.0 - PAD_H
    # opponent follows the ball once it crosses REACT_X toward it, otherwise drifts to the centre
    target = st[BY] if (st[VX] > 0.0 and st[BX] > REACT_X) else 0.5
    d = target - st[OY]
    if d > OPP_V:
        d = OPP_V
    if d < -OPP_V:
        d = -OPP_V
    st[OY] = st[OY] + d
    if st[OY] < PAD_H:
        st[OY] = PAD_H
    if st[OY] > 1.0 - PAD_H:
        st[OY] = 1.0 - PAD_H
    px = st[BX]
    st[BX] = st[BX] + st[VX]
    st[BY] = st[BY] + st[VY]
    if st[BY] < BALL_R:
        st[BY] = 2.0 * BALL_R - st[BY]
        st[VY] = -st[VY]
    if st[BY] > 1.0 - BALL_R:
        st[BY] = 2.0 * (1.0 - BALL_R) - st[BY]
        st[VY] = -st[VY]
    if st[VX] < 0.0 and px - BALL_R >= AGENT_X and st[BX] - BALL_R < AGENT_X:
        if abs(st[BY] - st[PY]) <= PAD_H + BALL_R:
            _bounce(st, AGENT_X, st[PY], 1.0)
            st[HITS] = st[HITS] + 1.0
    if st[VX] > 0.0 and px + BALL_R <= OPP_X and st[BX] + BALL_R > OPP_X:
        if abs(st[BY] - st[OY]) <= PAD_H + BALL_R:
            _bounce(st, OPP_X, st[OY], -1.0)
    st[FRAME] = st[FRAME] + 1.0
    if st[BX] < 0.0:
        st[SO] = st[SO] + 1.0
        serve(st)
    elif st[BX] > 1.0:
        st[SA] = st[SA] + 1.0
        serve(st)
    if st[SA] >= WIN_POINTS or st[SO] >= WIN_POINTS or st[FRAME] >= MAX_FRAMES:
        st[DONE] = 1.0


@njit(cache=True)
def features(st, u):
    u[0] = (st[BY] - st[PY]) * 2.0
    u[1] = st[BX] * 2.0 - 1.0
    u[2] = st[VX] / MAX_SPEED
    u[3] = st[VY] / MAX_SPEED
    u[4] = st[PY] * 2.0 - 1.0
    u[5] = st[BY] * 2.0 - 1.0


@njit(cache=True)
def heuristic_move(st):
    """Reference rule: move toward the ball's height."""
    return np.tanh((st[BY] - st[PY]) * 30.0)
