"""Train a same-actor MinAtar PPO baseline and measure real update throughput.

This follows the published CleanRL-style hyperparameters. The actor weights are
exported in the same NPZ format consumed by exp5.py. Run on CPU; small network
updates and sequential MinAtar environments make GPU dispatch unhelpful here.
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from exp5 import CHANNELS, NPZ_NAMES, observation_indices


class MLP(nn.Module):
    def __init__(self, output, final_std):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(600, 32), nn.Tanh(),
                                 nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, output))
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2) if layer.out_features == 32 else final_std)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.net(x)


def observations(envs, game):
    batch = np.zeros((len(envs), 600), dtype=np.float32)
    for i, env in enumerate(envs):
        active = observation_indices(env.state(), env.env.channels, CHANNELS[game])
        batch[i, active] = 1
    return torch.from_numpy(batch)


def save_actor(actor, path):
    layers = (actor.net[0], actor.net[2], actor.net[4])
    arrays = {}
    for index, layer in enumerate(layers, 1):
        arrays[NPZ_NAMES[f"W{index}"]] = layer.weight.detach().numpy().T.copy()
        arrays[NPZ_NAMES[f"b{index}"]] = layer.bias.detach().numpy().copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **arrays)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def atomic_json(path, content):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(content, output, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def save_live_frame(path, seed, update, t, obs, actions, rewards, done, episode_steps):
    """Publish the actual 64 next-action observations without touching training RNG."""
    masks = np.packbits(obs.numpy().astype(np.uint8).reshape(64, 10, 10, 6),
                        axis=3, bitorder="little").reshape(64, 100)
    frame = {
        "seed": seed, "update": update + 1,
        "steps": update * 8192 + (t + 1) * 64,
        "captured_at": time.time(),
        "states": [base64.b64encode(row.tobytes()).decode("ascii") for row in masks],
        "actions": actions.tolist(), "rewards": rewards.tolist(),
        "done": done.tolist(), "episode_steps": episode_steps.tolist(),
    }
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(frame, output, separators=(",", ":"))
    os.replace(temporary, path)


def save_checkpoint(path, args, completed_updates, actor, critic, optimizer,
                    envs, current_obs, current_done, episode_steps,
                    rollout_seconds, update_seconds, training_seconds, log):
    state = {
        "schema": 1,
        "config": {"game": args.game, "seed": args.seed, "threads": args.threads,
                   "target_steps": args.target_steps},
        "completed_updates": completed_updates,
        "actor": actor.state_dict(), "critic": critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "envs": envs, "current_obs": current_obs, "current_done": current_done,
        "episode_steps": episode_steps,
        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
        "rollout_seconds": rollout_seconds, "update_seconds": update_seconds,
        "training_seconds": training_seconds, "log": log,
    }
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        torch.save(state, output)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    atomic_json(path.parent / "progress.json", {
        "game": args.game, "seed": args.seed,
        "completed_updates": completed_updates,
        "steps": completed_updates * 8192,
        "target_steps": args.target_steps,
        "training_seconds": training_seconds,
        "checkpoint": str(path),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=CHANNELS, default="space_invaders")
    parser.add_argument("--updates", type=int, default=10,
                        help="Each update collects 64*128=8192 environment steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-steps", type=int, default=100_000_000,
                        help="Total training budget for linear learning-rate annealing")
    parser.add_argument("--checkpoint-every", type=int, default=500,
                        help="Save full training state this many updates apart")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true",
                        help="Continue from output-dir/latest.pt with exact RNG and environment state")
    parser.add_argument("--live-frame-every", type=int, default=0,
                        help="Publish real 64-game observation snapshot every N rollout ticks; 0 disables")
    args = parser.parse_args()
    if (args.updates < 1 or args.threads < 1 or args.checkpoint_every < 1
            or args.log_every < 1 or args.target_steps < 1 or args.live_frame_every < 0):
        parser.error("Require positive updates, threads, checkpoint-every, log-every and target-steps")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.output_dir / "latest.pt"
    if args.resume and not latest_path.exists():
        parser.error("--resume requires output-dir/latest.pt")
    if not args.resume and latest_path.exists():
        parser.error("A checkpoint exists; pass --resume or choose a fresh output directory")

    cache = Path(__file__).resolve().parent / "data/cache"
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    from minatar import Environment

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    actor, critic = MLP(6, 0.01), MLP(1, 1.0)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()),
                                 lr=5e-4, eps=1e-5)
    if args.resume:
        state = torch.load(latest_path, map_location="cpu", weights_only=False)
        expected = {"game": args.game, "seed": args.seed, "threads": args.threads,
                    "target_steps": args.target_steps}
        if state.get("schema") != 1 or state.get("config") != expected:
            parser.error("Checkpoint schema or game/seed/threads/target-steps mismatch")
        if args.updates < state["completed_updates"]:
            parser.error("--updates is below the completed checkpoint update count")
        actor.load_state_dict(state["actor"])
        critic.load_state_dict(state["critic"])
        optimizer.load_state_dict(state["optimizer"])
        envs = state["envs"]
        current_obs = state["current_obs"]
        current_done = state["current_done"]
        episode_steps = state["episode_steps"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        first_update = state["completed_updates"]
        rollout_seconds = state["rollout_seconds"]
        update_seconds = state["update_seconds"]
        prior_training_seconds = state["training_seconds"]
        log = state["log"]
        print(json.dumps({"resumed_update": first_update,
                          "resumed_steps": first_update * 8192}), flush=True)
    else:
        envs = [Environment(args.game, sticky_action_prob=0.1) for _ in range(64)]
        for i, env in enumerate(envs):
            env.seed(args.seed * 1000 + i)
            env.reset()
        current_obs = observations(envs, args.game)
        current_done = torch.zeros(64)
        episode_steps = np.zeros(64, dtype=np.int32)
        first_update = 0
        rollout_seconds = 0.0
        update_seconds = 0.0
        prior_training_seconds = 0.0
        log = []
    batch_size = 64 * 128
    minibatch_size = batch_size // 8
    began = time.perf_counter()

    for update in range(first_update, args.updates):
        lr = 5e-4 * max(0.0, 1 - update * batch_size / args.target_steps)
        optimizer.param_groups[0]["lr"] = lr
        obs_buf = torch.empty((128, 64, 600), dtype=torch.float32)
        actions = torch.empty((128, 64), dtype=torch.long)
        logprobs = torch.empty((128, 64))
        rewards = torch.empty((128, 64))
        dones = torch.empty((128, 64))
        values = torch.empty((128, 64))

        start = time.perf_counter()
        for t in range(128):
            obs_buf[t] = current_obs
            dones[t] = current_done
            with torch.no_grad():
                dist = Categorical(logits=actor(current_obs))
                action = dist.sample()
                logprobs[t] = dist.log_prob(action)
                values[t] = critic(current_obs).flatten()
            actions[t] = action
            done_next = np.zeros(64, dtype=np.float32)
            reward_next = np.zeros(64, dtype=np.float32)
            for i, env in enumerate(envs):
                reward, done = env.act(int(action[i]))
                episode_steps[i] += 1
                reward_next[i] = reward
                done_next[i] = done or episode_steps[i] >= 10000
                if done_next[i]:
                    env.reset()
                    episode_steps[i] = 0
            rewards[t] = torch.from_numpy(reward_next)
            current_done = torch.from_numpy(done_next)
            current_obs = observations(envs, args.game)
            if args.live_frame_every and (t + 1) % args.live_frame_every == 0:
                save_live_frame(args.output_dir / "live_frame.json", args.seed, update, t,
                                current_obs, action, reward_next, done_next, episode_steps)
        rollout = time.perf_counter() - start
        rollout_seconds += rollout

        start = time.perf_counter()
        with torch.no_grad():
            next_value = critic(current_obs).flatten()
            advantages = torch.zeros_like(rewards)
            last_gae = torch.zeros(64)
            for t in reversed(range(128)):
                if t == 127:
                    next_nonterminal, next_values = 1 - current_done, next_value
                else:
                    next_nonterminal, next_values = 1 - dones[t + 1], values[t + 1]
                delta = rewards[t] + 0.99 * next_values * next_nonterminal - values[t]
                last_gae = delta + 0.99 * 0.95 * next_nonterminal * last_gae
                advantages[t] = last_gae
            returns = advantages + values

        flat_obs = obs_buf.reshape(-1, 600)
        flat_actions = actions.reshape(-1)
        flat_logprobs = logprobs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_values = values.reshape(-1)
        indices = np.arange(batch_size)
        for _ in range(4):
            np.random.shuffle(indices)
            for offset in range(0, batch_size, minibatch_size):
                idx = indices[offset:offset + minibatch_size]
                dist = Categorical(logits=actor(flat_obs[idx]))
                new_logprob = dist.log_prob(flat_actions[idx])
                ratio = (new_logprob - flat_logprobs[idx]).exp()
                adv = flat_advantages[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                policy_loss = torch.max(-adv * ratio,
                                        -adv * torch.clamp(ratio, 0.8, 1.2)).mean()
                new_value = critic(flat_obs[idx]).flatten()
                unclipped_value_loss = (new_value - flat_returns[idx]).square()
                clipped_value = flat_values[idx] + (new_value - flat_values[idx]).clamp(-0.2, 0.2)
                clipped_value_loss = (clipped_value - flat_returns[idx]).square()
                value_loss = 0.5 * torch.max(unclipped_value_loss, clipped_value_loss).mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * dist.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
                optimizer.step()
        optimize = time.perf_counter() - start
        update_seconds += optimize
        row = {"update": update + 1, "steps": (update + 1) * batch_size,
               "rollout_seconds": rollout, "optimize_seconds": optimize,
               "loss": float(loss.detach()), "mean_reward_per_step": float(rewards.mean())}
        log.append(row)
        completed = update + 1
        if completed % args.log_every == 0 or completed == args.updates:
            print(json.dumps(row), flush=True)

        milestone = None
        for threshold in (100_000_000, 500_000_000):
            if update * batch_size < threshold <= completed * batch_size:
                milestone = threshold
                label = f"actor_{threshold // 1_000_000}m.npz"
                save_actor(actor, args.output_dir / label)
                atomic_json(args.output_dir / f"actor_{threshold // 1_000_000}m.json", {
                    "game": args.game, "seed": args.seed,
                    "requested_steps": threshold, "actual_steps": completed * batch_size,
                    "completed_updates": completed, "actor": label,
                })
                print(json.dumps({"milestone": threshold,
                                  "actual_steps": completed * batch_size,
                                  "actor": label}), flush=True)
        if (completed % args.checkpoint_every == 0 or milestone is not None
                or completed == args.updates):
            save_checkpoint(latest_path, args, completed, actor, critic, optimizer,
                            envs, current_obs, current_done, episode_steps,
                            rollout_seconds, update_seconds,
                            prior_training_seconds + time.perf_counter() - began, log)

    elapsed = prior_training_seconds + time.perf_counter() - began
    actor_path = args.output_dir / "actor.npz"
    save_actor(actor, actor_path)
    # Exclude first update, which includes warm-up and backend initialization.
    steady = log[1:] if len(log) > 1 else log
    seconds_per_update = float(np.median([x["rollout_seconds"] + x["optimize_seconds"] for x in steady]))
    summary = {
        "game": args.game, "seed": args.seed, "updates": args.updates,
        "steps": args.updates * batch_size, "threads": args.threads,
        "target_steps": args.target_steps, "elapsed_seconds": elapsed,
        "rollout_seconds": rollout_seconds, "optimize_seconds": update_seconds,
        "steady_median_seconds_per_8192_steps": seconds_per_update,
        "steady_steps_per_second": batch_size / seconds_per_update,
        "estimated_hours_100m": 100_000_000 / batch_size * seconds_per_update / 3600,
        "estimated_hours_500m": 500_000_000 / batch_size * seconds_per_update / 3600,
        "actor_path": str(actor_path), "per_update": log,
        "notes": "Linear extrapolation from short training; full-run speed and returns may differ."
    }
    atomic_json(args.output_dir / "timing.json", summary)
    atomic_json(args.output_dir / "complete.json", {
        "game": args.game, "seed": args.seed,
        "completed_updates": args.updates, "steps": args.updates * batch_size,
        "target_steps": args.target_steps, "actor": str(actor_path),
        "training_seconds": elapsed,
    })
    print(json.dumps({k: v for k, v in summary.items() if k != "per_update"}, indent=2))


if __name__ == "__main__":
    main()
