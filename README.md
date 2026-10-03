# Pong with a fly central-complex steering circuit

One of three small hobby studies in which a circuit from the fruit fly connectome plays a game. The other two: [Snake](https://github.com/burakgulmez/fly-snake), [Flappy Bird](https://github.com/burakgulmez/fly-flappybird).

A fixed network wired like the fruit fly's **central-complex steering pathway** moves the left paddle:
- **Inputs:** 116 PFN neurons (PFNa, PFNd, PFNv)
- **In between:** 14 hΔ neurons
- **Outputs:** 24 PFL3 steering neurons and the DNa02 pair

That is 156 neurons and 2,190 fixed connections, all excitatory, with no direct input-to-output
connections. Only 141 numbers are trained.

**[Open the demo](demo/index.html)**: watch a match, see the neurons, or silence the hΔ neurons or one side of PFL3.

## Results (100 unseen test matches to 11 points)

| Training run | Points (network – opponent) | Matches won |
|---|---|---|
| 0 | 11.00 – 0.95 | 100% |
| 1 | 11.00 – 0.55 | 100% |
| 2 | 10.84 – 0.19 | 100% |
| 3 | 5.55 – 10.89 | 4% |
| 4 | 10.99 – 0.15 | 100% |

- **Reference agents:** a paddle that simply follows the ball height almost never misses but rarely
  scores (1.74 – 0.02). A paddle that stands still loses 3.12 – 10.99.
- **Silencing the 14 hΔ neurons** makes every network lose all its matches, so the signal really has to
  pass through the circuit.
- **Readout change:** the first readout (left vs right hemisphere, two groups) did not learn. Reading
  each output neuron separately did; see `results/diagnose.csv`.

Details are in [`paper/main.tex`](paper/main.tex).

## How it works

- **Game** (`src/game.py`): numba; a simple opponent that tracks the ball only after it crosses the
  middle; serves toward the network with seeded angles; match to 11 points or 6,000 frames.
- **Circuit** (`src/connectome.py`): PFNa/PFNd/PFNv → neurons with ≥ 20 synapses from the inputs and to
  the outputs → PFL3 + DNa02.
- **Network and training** (`src/brain.py`, `experiments/train.py`): rate model with 4 updates per frame,
  6 readings → 16 input groups; paddle speed = tanh(weighted sum of output neurons). CMA-ES, 4M frames
  per run.

## Reproduce

```bash
# FlyWire tables in data/raw (see data/README.md)
bash reproduce.sh
python -m pytest tests
```

The browser demo reproduces the Python matches: paddle moves agree to within 1e-9 (tiny `sin`/`cos`/`tanh`
differences between numba and JavaScript), and the match outcomes are identical.

## Layout

```
src/            game engine (numba), circuit extraction, network
experiments/    training and analysis scripts
results/        logged results (CSV / JSON) used by the paper
figures/        make_all.py -> figures
paper/          main.tex + make_numbers.py (every number is generated from results/)
arxiv/          self-contained arXiv submission package
demo/           static HTML/JS demo (index.html, sim.js, weights.js) and a short video
tests/          pytest (game rules, Python/JS equivalence)
tools/          reference fetcher and verifier (OpenAlex, Crossref), arXiv packaging, video recorder
data/           where the FlyWire tables go (see data/README.md)
```

Install with `pip install -r requirements.txt` (Python 3.10+). Node.js is only needed for the
Python/JS equivalence test and for recording the demo video.

## Data and credit

The connectome data come from the FlyWire project (FAFB v783) via the
[FlyWire Codex](https://codex.flywire.ai) download page and are not included here.
Please follow the FlyWire citation guidelines when using the data.

This is not a model of how flies think, and flies do not play Pong. It is a small,
reproducible experiment with real connectome data.

## License

Code: MIT (see `LICENSE`). The FlyWire data are subject to their own terms.
