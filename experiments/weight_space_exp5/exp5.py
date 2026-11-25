"""Apply a one-shot weight edit and evaluate a MinAtar policy on the CPU."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SHAPES = {"W1": (600, 32), "b1": (32,), "W2": (32, 32), "b2": (32,), "W3": (32, 6), "b3": (6,)}
NPZ_NAMES = {
    "W1": "actor/dense1/kernel", "b1": "actor/dense1/bias",
    "W2": "actor/dense2/kernel", "b2": "actor/dense2/bias",
    "W3": "actor/logits/kernel", "b3": "actor/logits/bias",
}
CHANNELS = {
    "breakout": (("paddle",), (), ("brick",), ("ball",), ("trail",), ()),
    "seaquest": (
        ("sub_front", "sub_back"), ("enemy_fish", "enemy_sub", "enemy_bullet"),
        ("enemy_fish", "enemy_sub", "diver"), ("friendly_bullet",),
        ("trail", "sub_back"), ("oxygen_guage", "diver_guage"),
    ),
    "asterix": (("player",), ("enemy",), ("gold",), (), ("trail",), ()),
    "space_invaders": (
        ("cannon",), ("alien",), ("alien_left",), ("alien_right",),
        ("friendly_bullet",), ("enemy_bullet",),
    ),
}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get_last_edits(text):
    blocks = re.findall(r"```edits\s*\n(.*?)\n```", text, flags=re.DOTALL | re.IGNORECASE)
    if not blocks:
        raise ValueError("No fenced ```edits block found in the answer")
    return blocks[-1]


def index_value(node):
    if isinstance(node, ast.Tuple):
        return tuple(index_value(x) for x in node.elts)
    if isinstance(node, ast.Slice):
        def part(value):
            if value is None:
                return None
            result = ast.literal_eval(value)
            if type(result) is not int:
                raise ValueError("Slice boundaries must be integers")
            return result
        return slice(part(node.lower), part(node.upper), part(node.step))
    result = ast.literal_eval(node)
    if type(result) is not int:
        raise ValueError("Indices must be integers")
    return result


def numeric_value(node):
    value = ast.literal_eval(node)
    array = np.asarray(value)
    if array.dtype.kind not in "fi" or not np.all(np.isfinite(array)) or np.any(np.abs(array) > 1e6):
        raise ValueError("Edit values must be finite numbers with absolute value <= 1e6")
    return array.astype(np.float32)


def edit_target(node, weights):
    if isinstance(node, ast.Name) and node.id in weights:
        return weights[node.id], (...,)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in weights:
        return weights[node.value.id], index_value(node.slice)
    raise ValueError("Edit target must be W1, b1, W2, b2, W3 or b3 (optionally indexed)")


def apply_edits(answer):
    weights = {name: np.zeros(shape, dtype=np.float32) for name, shape in SHAPES.items()}
    statements = ast.parse(get_last_edits(answer)).body
    if not statements:
        raise ValueError("The edits block is empty")
    for statement in statements:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            tensor, index = edit_target(statement.targets[0], weights)
            tensor[index] = numeric_value(statement.value)
        elif isinstance(statement, ast.AugAssign):
            tensor, index = edit_target(statement.target, weights)
            value = numeric_value(statement.value)
            if isinstance(statement.op, ast.Add):
                tensor[index] += value
            elif isinstance(statement.op, ast.Sub):
                tensor[index] -= value
            elif isinstance(statement.op, ast.Mult):
                tensor[index] *= value
            else:
                raise ValueError(f"Unsupported edit operation on line {statement.lineno}")
        else:
            raise ValueError(f"Unsupported edit statement on line {statement.lineno}")
    return weights, len(statements)


def load_weights(path):
    with np.load(path, allow_pickle=False) as source:
        weights = {}
        for short, full in NPZ_NAMES.items():
            item = np.asarray(source[full], dtype=np.float32)
            if item.shape != SHAPES[short] or not np.all(np.isfinite(item)):
                raise ValueError(f"Bad {full}: expected finite {SHAPES[short]}, got {item.shape}")
            weights[short] = item
    return weights


def observation_indices(state, game_channels, mapping):
    observation = np.zeros((10, 10, 6), dtype=np.bool_)
    for output_channel, source_names in enumerate(mapping):
        for name in source_names:
            observation[:, :, output_channel] |= state[:, :, game_channels[name]]
    return np.flatnonzero(observation.reshape(-1))


def policy_action(weights, active, rng):
    h1 = np.tanh(weights["W1"][active].sum(axis=0) + weights["b1"])
    h2 = np.tanh(h1 @ weights["W2"] + weights["b2"])
    logits = h2 @ weights["W3"] + weights["b3"]
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    return min(5, int(np.searchsorted(np.cumsum(probabilities), rng.random(), side="right")))


def evaluate_episode(game, weights, episode, seed, max_steps):
    cache = HERE / "data/cache"
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    from minatar import Environment

    environment = Environment(game, sticky_action_prob=0.1)
    environment.seed(seed + episode)
    environment.reset()
    rng = np.random.default_rng(seed + 1_000_000 + episode)
    score = 0.0
    for step in range(max_steps):
        active = observation_indices(environment.state(), environment.env.channels, CHANNELS[game])
        action = policy_action(weights, active, rng)
        reward, done = environment.act(action)
        score += float(reward)
        if done:
            break
    return score, step + 1


def evaluate(game, weights, episodes, seed, max_steps, workers):
    if workers == 1:
        outcomes = [evaluate_episode(game, weights, episode, seed, max_steps)
                    for episode in range(episodes)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            outcomes = list(executor.map(evaluate_episode, repeat(game), repeat(weights),
                                         range(episodes), repeat(seed), repeat(max_steps), chunksize=8))
    scores, lengths = zip(*outcomes)
    standard_error = float(np.std(scores, ddof=1) / math.sqrt(episodes)) if episodes > 1 else None
    return {
        "game": game, "episodes": episodes, "seed": seed, "max_steps": max_steps,
        "workers": workers,
        "sticky_action_prob": 0.1, "mean": float(np.mean(scores)),
        "standard_error": standard_error, "mean_length": float(np.mean(lengths)),
        "scores": scores, "lengths": lengths,
        "minatar_version": importlib.metadata.version("minatar"),
        "numpy_version": np.__version__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    zero = commands.add_parser("init-zero", help="Create an all-zero policy (uniform-action baseline)")
    zero.add_argument("--output", type=Path, required=True)
    apply = commands.add_parser("apply", help="Apply the final edits block to a zero policy")
    apply.add_argument("--answer", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("evaluate", help="Evaluate a policy with fresh MinAtar episodes")
    score.add_argument("--game", choices=CHANNELS, required=True)
    score.add_argument("--weights", type=Path, required=True)
    score.add_argument("--episodes", type=int, default=1000)
    score.add_argument("--seed", type=int, default=20261125)
    score.add_argument("--max-steps", type=int, default=10000)
    score.add_argument("--workers", type=int, default=1)
    score.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "init-zero":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **{
            NPZ_NAMES[name]: np.zeros(shape, dtype=np.float32) for name, shape in SHAPES.items()
        })
        print(json.dumps({"output": str(args.output), "nonzero_weights": 0}, indent=2))
        return
    if args.command == "apply":
        weights, count = apply_edits(args.answer.read_text(encoding="utf-8"))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **{NPZ_NAMES[name]: array for name, array in weights.items()})
        print(json.dumps({"output": str(args.output), "statements": count,
                          "nonzero_weights": int(sum(np.count_nonzero(x) for x in weights.values())),
                          "answer_sha256": file_sha256(args.answer)}, ensure_ascii=False, indent=2))
        return
    if args.episodes < 1 or args.max_steps < 1 or args.workers < 1:
        parser.error("--episodes, --max-steps and --workers must be positive")
    weights = load_weights(args.weights)
    result = evaluate(args.game, weights, args.episodes, args.seed, args.max_steps, args.workers)
    result["weights_sha256"] = file_sha256(args.weights)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("scores", "lengths")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
