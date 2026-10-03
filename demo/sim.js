// Pong engine + central-complex steering rate network: line-by-line port of src/game.py and
// src/brain.py (same float64 operation order, same xorshift32), for the browser and Node.
(function (root) {
  "use strict";
  const P = {
    PAD_H: 0.10, AGENT_X: 0.06, OPP_X: 0.94, BALL_R: 0.015, SERVE_SPEED: 0.014, MAX_SPEED: 0.028,
    SPEEDUP: 1.04, MAX_BOUNCE: 0.9, AGENT_V: 0.020, OPP_V: 0.0070, REACT_X: 0.5, WIN_POINTS: 11, MAX_FRAMES: 6000,
  };

  function rngNext(x) {
    x = (x ^ (x << 13)) >>> 0;
    x = (x ^ (x >>> 17)) >>> 0;
    x = (x ^ (x << 5)) >>> 0;
    return x >>> 0;
  }

  class Game {
    constructor(seed) {
      let x = Number((BigInt(seed) * 2654435761n + 12345n) & 0xFFFFFFFFn);
      if (x === 0) x = 1;
      for (let i = 0; i < 4; i++) x = rngNext(x);
      this.bx = 0; this.by = 0; this.vx = 0; this.vy = 0; this.py = 0.5; this.oy = 0.5;
      this.sa = 0; this.so = 0; this.hits = 0; this.frame = 0; this.done = false; this.rng = x;
      this.serve();
    }
    serve() {
      this.rng = rngNext(this.rng);
      const ang = -0.6 + 1.2 * ((this.rng % 10000) / 9999.0);
      this.bx = 0.5; this.by = 0.5;
      this.vx = -P.SERVE_SPEED * Math.cos(ang);
      this.vy = P.SERVE_SPEED * Math.sin(ang);
    }
    bounce(faceX, paddleY, dir) {
      let off = (this.by - paddleY) / P.PAD_H;
      if (off > 1.0) off = 1.0;
      if (off < -1.0) off = -1.0;
      let speed = Math.sqrt(this.vx * this.vx + this.vy * this.vy) * P.SPEEDUP;
      if (speed > P.MAX_SPEED) speed = P.MAX_SPEED;
      const ang = off * P.MAX_BOUNCE;
      this.vx = dir * speed * Math.cos(ang);
      this.vy = speed * Math.sin(ang);
      this.bx = faceX + dir * P.BALL_R;
    }
    step(move) {
      if (this.done) return;
      if (move > 1.0) move = 1.0;
      if (move < -1.0) move = -1.0;
      this.py = this.py + P.AGENT_V * move;
      if (this.py < P.PAD_H) this.py = P.PAD_H;
      if (this.py > 1.0 - P.PAD_H) this.py = 1.0 - P.PAD_H;
      const target = (this.vx > 0.0 && this.bx > P.REACT_X) ? this.by : 0.5;
      let d = target - this.oy;
      if (d > P.OPP_V) d = P.OPP_V;
      if (d < -P.OPP_V) d = -P.OPP_V;
      this.oy = this.oy + d;
      if (this.oy < P.PAD_H) this.oy = P.PAD_H;
      if (this.oy > 1.0 - P.PAD_H) this.oy = 1.0 - P.PAD_H;
      const px = this.bx;
      this.bx = this.bx + this.vx;
      this.by = this.by + this.vy;
      if (this.by < P.BALL_R) { this.by = 2.0 * P.BALL_R - this.by; this.vy = -this.vy; }
      if (this.by > 1.0 - P.BALL_R) { this.by = 2.0 * (1.0 - P.BALL_R) - this.by; this.vy = -this.vy; }
      if (this.vx < 0.0 && px - P.BALL_R >= P.AGENT_X && this.bx - P.BALL_R < P.AGENT_X) {
        if (Math.abs(this.by - this.py) <= P.PAD_H + P.BALL_R) { this.bounce(P.AGENT_X, this.py, 1.0); this.hits = this.hits + 1.0; }
      }
      if (this.vx > 0.0 && px + P.BALL_R <= P.OPP_X && this.bx + P.BALL_R > P.OPP_X) {
        if (Math.abs(this.by - this.oy) <= P.PAD_H + P.BALL_R) this.bounce(P.OPP_X, this.oy, -1.0);
      }
      this.frame = this.frame + 1.0;
      if (this.bx < 0.0) { this.so = this.so + 1.0; this.serve(); }
      else if (this.bx > 1.0) { this.sa = this.sa + 1.0; this.serve(); }
      if (this.sa >= P.WIN_POINTS || this.so >= P.WIN_POINTS || this.frame >= P.MAX_FRAMES) this.done = true;
    }
    features(u) {
      u[0] = (this.by - this.py) * 2.0;
      u[1] = this.bx * 2.0 - 1.0;
      u[2] = this.vx / P.MAX_SPEED;
      u[3] = this.vy / P.MAX_SPEED;
      u[4] = this.py * 2.0 - 1.0;
      u[5] = this.by * 2.0 - 1.0;
    }
  }

  class Net {
    constructor(m) { this.m = m; this.n = m.n; this.mask = new Float64Array(m.n).fill(1); this.reset(); }
    reset() {
      const n = this.n;
      this.x = new Float64Array(n); this.phi = new Float64Array(n); this.rec = new Float64Array(n);
      this.drive = new Float64Array(Math.max(this.m.k_in, 1)); this.rvec = new Float64Array(Math.max(this.m.k_out, 1));
      this.z = 0;
    }
    step(u) {
      const m = this.m, th = m.theta, nf = 6, ki = m.k_in, ko = m.k_out, N = this.n;
      let p = ki * nf;
      for (let k = 0; k < ki; k++) {
        let s = th[p + k];
        for (let j = 0; j < nf; j++) s += th[k * nf + j] * u[j];
        this.drive[k] = s;
      }
      p += ki;
      const a = 0.05 + 0.95 / (1.0 + Math.exp(-th[p + ko + 1]));
      const g = 2.0 / (1.0 + Math.exp(-th[p + ko + 2]));
      const x = this.x, phi = this.phi, rec = this.rec, ip = m.indptr, ix = m.indices, wd = m.data;
      for (let t = 0; t < m.T; t++) {
        for (let i = 0; i < N; i++) {
          let s = 0.0;
          for (let q = ip[i]; q < ip[i + 1]; q++) s += wd[q] * phi[ix[q]];
          rec[i] = s;
        }
        for (let i = 0; i < N; i++) {
          let inp = 0.0;
          if (m.in_grp[i] >= 0) inp = this.drive[m.in_grp[i]];
          x[i] += a * (-x[i] + g * rec[i] + inp);
          phi[i] = Math.tanh(x[i]) * this.mask[i];
        }
      }
      this.rvec.fill(0);
      for (let i = 0; i < N; i++) if (m.out_grp[i] >= 0) this.rvec[m.out_grp[i]] += phi[i];
      let z = th[p + ko];
      for (let k = 0; k < ko; k++) z += th[p + k] * this.rvec[k] / m.out_size[k];
      this.z = z;
      return Math.tanh(z);
    }
  }

  function playGame(net, seed, maxFrames) {
    const g = new Game(seed), u = new Float64Array(6), actions = [];
    net.reset();
    while (!g.done && actions.length < (maxFrames || 1e9)) {
      g.features(u);
      const a = net.step(u);
      actions.push(a);
      g.step(a);
    }
    return { actions, pointsFor: g.sa, pointsAgainst: g.so, hits: g.hits, frames: g.frame };
  }

  const api = { P, Game, Net, playGame };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.FlyPong = api;
})(this);
