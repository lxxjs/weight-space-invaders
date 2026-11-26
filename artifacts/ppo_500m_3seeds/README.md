# Space Invaders PPO baseline: three 500M-step runs

These are **independent local baselines** for the Space Invaders extension of Experiment 5 in [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/). The original authors evaluated Breakout, Seaquest, and Asterix, not Space Invaders. Their PPO implementation and Space Invaders results are not represented here.

## Training protocol

- Date and machine: 2026-11-25 KST, Apple M1 Pro (8 performance + 2 efficiency CPU cores), 32 GB RAM. Final training used CPU, one PyTorch thread per seed. Seed 0 ran first; seeds 1 and 2 later ran as separate concurrent processes.
- Environment: MinAtar 1.0.15 `space_invaders`, sticky action probability 0.1, maximum 10,000 steps per episode. Each training seed creates 64 independent environments, seeded as `training_seed * 1000 + environment_index`.
- Actor: 600 inputs (10×10×6 native channels), 32 and 32 tanh hidden units, 6 action logits. The critic uses the same hidden dimensions and one output.
- PPO: 64 environments × 128 steps = 8,192 environment steps per rollout; 4 optimization epochs and 8 minibatches per epoch; Adam with initial learning rate 5e-4 annealed linearly over **500M** steps; discount 0.99; GAE lambda 0.95; policy clip 0.2; value coefficient 0.5; entropy coefficient 0.01; gradient norm clip 0.5.
- Each run completed 61,036 updates = **500,006,912 steps**. Full training state was saved every 500 updates and at the 100M/500M milestones. Restarts restored the actor, critic, optimizer, all environments, and RNG states. `train.log` records one row per 100 updates, plus resume and milestone events; `timing.json.gz` includes all 61,036 per-update timing and loss rows.
- The 100M actor is a **milestone in the 500M learning-rate schedule**, not a separately trained 100M baseline.

The supervisor logs record seed 0 completing at 07:22 KST, seed 1 at 14:20 KST, and seed 2 at 16:15 KST. Seed 0 resumed once at update 1,000; seed 1 at updates 19,000 and 37,500; seed 2 at update 18,500. Training times stored in the final checkpoints were approximately 5.69, 6.21, and 6.28 hours respectively; the overlapping jobs make these **per-seed durations**, not a sum of wall-clock time.

The exact CPU trainer used for seed 0, seed 1 through update 37,500, and seed 2 through update 18,500 is [`ppo_baseline_cpu.py`](../../experiments/weight_space_exp5/ppo_baseline_cpu.py). The remaining updates of seeds 1 and 2 used [`ppo_live_training.py`](../../experiments/weight_space_exp5/ppo_live_training.py), which adds observation snapshots for the browser viewer. In a paired one-update resume check from the same checkpoint, the two versions produced byte-identical actor tensors. `manifest.json` includes SHA-256 hashes of these source files, the runner, and the evaluator. The separate `ppo_baseline.py` in the repository contains later CPU/GPU benchmarking options and was not the code launched for these final runs.

## Evaluation protocol and results

Each actor was evaluated on **1,000 episodes** by `experiments/weight_space_exp5/exp5.py`, using MinAtar 1.0.15, sticky actions 0.1, an episode cap of 10,000 steps, and evaluation seed 20261125. Environment seed is `20261125 + episode_index`; the policy-action sampler uses `21261125 + episode_index`. The same episode seeds are used for every actor. The score is the sum of game rewards per episode.

| Training seed | 100M-step actor: mean ± episode SE | 500M-step actor: mean ± episode SE |
| --- | ---: | ---: |
| 0 | 362.057 ± 11.024 | 1052.551 ± 22.035 |
| 1 | 538.095 ± 15.746 | 1158.526 ± 21.971 |
| 2 | 499.757 ± 15.627 | 1148.469 ± 22.606 |
| **Mean across 3 training seeds** | **466.636 ± 53.448 seed SE** | **1119.849 ± 33.774 seed SE** |

The per-policy SE describes variation over 1,000 episodes. The final row instead reports the mean and SE **across three training seeds**. All six evaluations retained their 1,000 individual scores and episode lengths. The 100M actors were saved at **100,007,936** steps, and the 500M actors at **500,006,912** steps. The 500M mean is about 2.40 times the 100M milestone mean, within this protocol.

## Files and verification

For each `seed_N/`:

- `actor_100m.npz`, `actor_500m.npz`: policy tensors in the format consumed by `exp5.py`. The final `actor.npz` was byte-identical to `actor_500m.npz`, so only one copy is stored.
- `actor_100m.json`, `actor_500m.json`: exact milestone step counts.
- `eval_100m_1000.json`, `eval_500m_1000.json`: summary, all scores and lengths, evaluation settings, and policy SHA-256.
- `train.log`: console training records and resume/milestone events.
- `timing.json.gz`: full per-update training metrics and timing.
- `latest.pt.gz`: **final** full PyTorch checkpoint, including actor, critic, optimizer, environments, RNG states, and log. Earlier full checkpoints are not archived.

`manifest.json` lists sizes and SHA-256 values for the archive files, plus the uncompressed hashes of the two gzip files and hashes of the relevant source files. Every evaluation JSON's policy hash was checked against its archived actor file. To restore a full checkpoint for `--resume`, decompress `seed_N/latest.pt.gz` to `seed_N/latest.pt` in an output directory with the same seed and `--target-steps 500000000`. The run is already complete; the checkpoint is retained for inspection and reproducibility.

Generated prompts and original-author artifacts are not copied into this archive. The source code and dependency pins are in `experiments/weight_space_exp5/`.
