# Weight-Space Invaders

An independent **MinAtar Space Invaders extension** inspired by Experiment 5 of [*Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*](https://rikiyat.github.io/blog/weight-space-programming/) by **Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab** (November 2026).

The original experiment asks a language model to write policy-network weights from scratch for **Breakout, Seaquest, and Asterix**. This repository adds **Space Invaders** using the same `600 → 32 → 32 → 6` network shape, builds an independent evaluation path, and trains its own PPO baselines for comparison. Space Invaders results in this repository are an **extension, not a reported result of the original authors**. The original [experiment logs, prompts, and weights](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp5_write_from_scratch) remain in the authors' repository; `get_reference.sh` fetches a pinned revision for local verification.

The implementation and conclusions here are independent of the original authors. See the [experiment guide](experiments/weight_space_exp5/README.md) for the protocol, limitations, and full commands. MinAtar is described in [Young and Tian, *MinAtar: An Atari-Inspired Testbed*](https://arxiv.org/abs/1903.03176).

## Contents

| File | Purpose |
| --- | --- |
| `exp5.py` | Create, edit, and evaluate `600 → 32 → 32 → 6` policy weights |
| `prepare_space_invaders.py` | Build the new-game prompt from installed MinAtar 1.0.15 source |
| `run_agent.py` | Run a single tool-free Codex weight-writing attempt |
| `ppo_baseline.py` | PPO baseline and short CPU/GPU benchmarking |
| `ppo_live_training.py` | CPU PPO baseline with real 64-game observation snapshots |
| `run_ppo_overnight.sh` | Resume-capable, three-seed 500M-step training supervisor |
| `watch_ppo.py` | Local browser viewer for actual training games and a separate policy demo |

All files are under `experiments/weight_space_exp5/`. The PPO trainer saves full state every 500 updates and exports actor weights at the 100M and 500M milestones. Each update collects 64 environments × 128 steps. The 100M actor is a milestone within a 500M learning-rate schedule, not an independent 100M training run.

## Set up

Run commands from the repository root. Install [uv](https://docs.astral.sh/uv/) and Python 3.14 for the PPO environment; Python 3.12 also works for the evaluation-only environment. The optional one-shot model attempt requires the [Codex CLI](https://developers.openai.com/codex/cli) and an authenticated account.

```sh
uv venv --python 3.14 .venv-ppo
uv pip install --python .venv-ppo/bin/python -r experiments/weight_space_exp5/requirements-ppo.txt
.venv-ppo/bin/python experiments/weight_space_exp5/prepare_space_invaders.py
```

Smoke-test a PPO run:

```sh
.venv-ppo/bin/python experiments/weight_space_exp5/ppo_live_training.py \
  --game space_invaders --seed 0 --updates 10 --target-steps 500000000 \
  --live-frame-every 32 --output-dir experiments/weight_space_exp5/results/smoke
```

For the full three-seed run on macOS, keep the machine awake and the lid open:

```sh
PPO_PYTHON="$PWD/.venv-ppo/bin/python" caffeinate -im \
  zsh experiments/weight_space_exp5/run_ppo_overnight.sh
```

Training continues from each seed's `latest.pt` if restarted. The supervisor runs seeds 0, 1, and 2 in sequence by default; pass seed numbers to select a subset. To view the actual 64 training environments for a running seed:

```sh
.venv-ppo/bin/python experiments/weight_space_exp5/watch_ppo.py \
  --run-root experiments/weight_space_exp5/results/space_invaders/ppo_500m_3seeds
```

Open <http://127.0.0.1:8765/training>. The viewer samples the real training observations; it is not a complete recording of every game frame. The `/` page plays a separate demonstration game with the latest saved policy. The viewer currently offers seed 1 and seed 2; seed 0 was completed before live snapshots were added.

For original-game verification, run `get_reference.sh` and follow the [experiment guide](experiments/weight_space_exp5/README.md). Generated prompts, upstream reference files, checkpoints, and results are intentionally excluded from Git. No completed Space Invaders weight-writing result is claimed here.
