"""Build a new one-shot prompt from the installed MinAtar 1.0.15 game source."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    distribution = importlib.metadata.distribution("minatar")
    if distribution.version != "1.0.15":
        raise SystemExit(f"Expected MinAtar 1.0.15, found {distribution.version}")
    source_path = Path(distribution.locate_file("minatar/environments/space_invaders.py"))
    source = source_path.read_text(encoding="utf-8")
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    prompt_path = HERE / "data/prompts/space_invaders.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    before = """YOUR TASK: Write from scratch the weights of a policy network that plays this game as well as possible. There is no training, gameplay data, loss, gradients, or fine-tuning. You choose every number yourself. You have no tools: no code execution, calculator, simulator, or forward pass. This is one response with no feedback. Briefly describe your design, then write one final edits block.

THE GAME: Space Invaders from MinAtar 1.0.15 on a 10 x 10 grid. A cannon at the bottom moves left and right and shoots upward. Aliens move across the screen, descend at a wall, and shoot at the cannon. Shooting an alien gives +1. The episode ends if an alien or enemy bullet touches the cannon. Left and right move the cannon; fire shoots when its cooldown permits. Up and down have no effect.

The exact implementation from the installed MinAtar 1.0.15 package, file minatar/environments/space_invaders.py, follows. The environment wrapper calls act(a) with a = 0..5 meaning no-op, left, up, right, down, fire. With probability 0.1 it repeats the previous action instead of the new one (sticky actions).

```python
"""
    after = """```

THE INPUT: 600 binary numbers, a 10 x 10 grid with 6 channels. Rows run 0 (top) to 9 (bottom); columns run 0 (left) to 9 (right). Input index i = (row * 10 + column) * 6 + channel. The channels are the game's native state() channels, in their original order:
  channel 0: cannon (player)
  channel 1: alien
  channel 2: alien_left (aliens currently moving left)
  channel 3: alien_right (aliens currently moving right)
  channel 4: friendly_bullet
  channel 5: enemy_bullet
The same alien can appear in channels 1 and 2 or in channels 1 and 3. Unlike the original paper's three games, these six channels are not remapped to the common player/danger/reward/projectile/trail/gauge scheme; this is a new-game extension.

THE OUTPUT: six action logits, indices 0 no-op, 1 left, 2 up, 3 right, 4 down, 5 fire. An action is sampled from their softmax probabilities; sticky actions have probability 0.1.

THE NETWORK: a 600 -> 32 -> 32 -> 6 multilayer perceptron, tanh on both hidden layers, 20,486 parameters. Every weight and bias starts at zero:
  h1 = tanh(x @ W1 + b1)          W1: 600 x 32, b1: 32
  h2 = tanh(h1 @ W2 + b2)         W2: 32 x 32,  b2: 32
  logits = h2 @ W3 + b3           W3: 32 x 6,  b3: 6
Row i of W1 connects input i to first-layer units. Column j of each matrix feeds unit j of the next layer.

SCORING: mean score over 1,000 independent episodes of MinAtar 1.0.15. Actions are sampled as described above. An episode is also stopped after 10,000 steps. You will not see any episode or score before your answer is final.

HOW TO WRITE WEIGHTS: finish your reply with a code block labeled edits. Use one Python-style numeric assignment per statement. Tensors are W1, b1, W2, b2, W3, b3. Indexing accepts integers, slices, and two-dimensional blocks. Plain decimal numbers and lists are allowed; values must be finite with absolute value <= 1e6. Statements apply in order. Examples:

```edits
W1[12, 0] = 2.5
W1[0:6, 3] = [1, 0, -1, 0, 2, 0]
b1[0:4] = -1
W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]
W3[:, 2] *= 2
```

The last edits block is read. Malformed statements make the submission invalid. Do not run the network or ask for feedback.
"""
    prompt = before + source.rstrip() + "\n" + after
    prompt_path.write_text(prompt, encoding="utf-8")
    metadata = {
        "game": "space_invaders", "minatar_version": distribution.version,
        "source_path": "minatar/environments/space_invaders.py",
        "source_sha256": source_sha256,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "status": "new extension; no published Astra or PPO score for this protocol",
    }
    (prompt_path.parent / "space_invaders.meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"prompt": str(prompt_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
