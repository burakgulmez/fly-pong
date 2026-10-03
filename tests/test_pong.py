import json, os, shutil, subprocess, sys
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src import game as G


def test_serve_goes_toward_agent_and_is_deterministic():
    a, b, c = G.new_game(3), G.new_game(3), G.new_game(4)
    assert a[G.VX] < 0 and np.array_equal(a, b) and not np.array_equal(a, c)


def test_wall_bounce_keeps_ball_inside():
    st = G.new_game(1)
    for _ in range(3000):
        G.step(st, 0.0)
        assert G.BALL_R - 1e-12 <= st[G.BY] <= 1 - G.BALL_R + 1e-12
        if st[G.DONE]:
            break


def test_still_paddle_loses_and_tracker_returns():
    st = G.new_game(5)
    while st[G.DONE] == 0:
        G.step(st, 0.0)
    assert st[G.SO] == G.WIN_POINTS
    st = G.new_game(5)
    for _ in range(2000):
        G.step(st, G.heuristic_move(st))
    assert st[G.HITS] > 5 and st[G.SO] == 0


def test_paddle_limits():
    st = G.new_game(1)
    for _ in range(200):
        G.step(st, 1.0)
    assert st[G.PY] == pytest.approx(1 - G.PAD_H)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_python_js_identical(tmp_path):
    if not os.path.exists(os.path.join(ROOT, "results", "runs", "seed0", "theta.npy")):
        pytest.skip("no trained run")
    from experiments.export_demo import export
    from experiments.train import make_net, CFG
    m = export(0)
    w = tmp_path / "w.json"
    w.write_text(json.dumps(m))
    seeds = list(range(2_000_000_000, 2_000_000_010))
    js = f"""const S=require({json.dumps(os.path.join(ROOT, 'demo', 'sim.js'))});
const m=JSON.parse(require('fs').readFileSync({json.dumps(str(w))}));
const net=new S.Net(m); const out=[];
for (const s of {json.dumps(seeds)}) {{ const r=S.playGame(net,s); out.push([r.actions.slice(0,2000), r.pointsFor, r.pointsAgainst, r.hits, r.frames]); }}
console.log(JSON.stringify(out));"""
    res = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True)
    out = json.loads(res.stdout)
    net = make_net(CFG)
    theta = np.array(m["theta"])
    # Continuous paddle moves use sin/cos/tanh, whose last bits can differ between numba (libm) and
    # V8, so we require near-identical actions (< 1e-9) and identical game outcomes.
    for s, (ja, jf, jag, jh, jfr) in zip(seeds, out):
        res, pa, _ = net.trace(theta, s, max_frames=2000)
        ja = np.array(ja)
        assert len(pa) == len(ja)
        assert np.abs(pa - ja).max() < 1e-9, f"seed {s}: max diff {np.abs(pa - ja).max()}"
        assert (res[0], res[1], res[2], res[3]) == (jf, jag, jh, jfr)
