"""One tool-free Codex turn using a prepared MinAtar game prompt."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
GAMES = ("breakout", "seaquest", "asterix", "space_invaders")
DISABLED_FEATURES = (
    "code_mode", "collaboration_modes", "shell_tool", "unified_exec",
    "unified_exec_tty", "shell_snapshot", "view_image", "image_generation",
    "multi_agent", "apps", "plugins", "remote_plugin", "browser_use",
    "browser_use_external", "browser_use_full_cdp_access", "in_app_browser",
    "computer_use", "code_mode_host", "skill_search",
    "skill_mcp_dependency_install", "sleep_tool", "tool_suggest", "goals",
    "hooks", "worktrees", "workspace_dependencies",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game", choices=GAMES)
    parser.add_argument("--reference", type=Path, default=HERE / "data/upstream/exp5_write_from_scratch")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.game == "space_invaders":
        prompt_path = HERE / "data/prompts/space_invaders.txt"
    else:
        prompt_path = args.reference / "runs" / args.game / "message.txt"
    if not prompt_path.is_file():
        preparation = "prepare_space_invaders.py" if args.game == "space_invaders" else "get_reference.sh"
        parser.error(f"Prompt missing: {prompt_path}. Run {preparation} first.")
    output = args.output or HERE / "results" / args.game
    output.mkdir(parents=True, exist_ok=True)
    prompt = prompt_path.read_text(encoding="utf-8")
    (output / "prompt.txt").write_text(prompt, encoding="utf-8")

    command = [
        "codex", "exec", "--json", "--strict-config", "--skip-git-repo-check",
        "-m", "gpt-6-astra", "-c", 'model_reasoning_effort="xhigh"',
        "-c", 'model_reasoning_summary="detailed"',
        "-c", "model_context_window=872000",
        "-c", "model_auto_compact_token_limit=1000000",
        "-c", 'approval_policy="never"', "-c", 'sandbox_mode="read-only"',
        "-c", 'web_search="disabled"',
        "-c", f"model_instructions_file={json.dumps(str(HERE / 'instructions.md'))}",
        "-c", "agents.enabled=false",
    ]
    for feature in DISABLED_FEATURES:
        command += ["-c", f"features.{feature}=false"]
    command += ["-o", str(output / "answer.md"), "-"]

    if args.dry_run:
        print("Prompt:", prompt_path)
        print("Output:", output)
        print("Command:", json.dumps(command, ensure_ascii=False))
        return

    # An isolated cwd avoids loading this repository's unrelated reading instructions.
    with tempfile.TemporaryDirectory(prefix="weight-space-exp5-") as workdir, \
         (output / "events.jsonl").open("w", encoding="utf-8") as events, \
         (output / "stderr.log").open("w", encoding="utf-8") as errors:
        completed = subprocess.run(command, input=prompt, text=True, cwd=workdir,
                                   stdout=events, stderr=errors, check=False)
    print(f"Codex exit={completed.returncode}; answer={output / 'answer.md'}")
    if completed.returncode:
        raise SystemExit(f"Codex failed. Read {output / 'stderr.log'}")
    observed = []
    for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("type") == "item.completed":
            observed.append(event.get("item", {}).get("type"))
    forbidden = [kind for kind in observed if kind not in ("agent_message", "reasoning", "error")]
    (output / "tool_audit.json").write_text(
        json.dumps({"tool_free": not forbidden, "completed_item_types": observed}, indent=2) + "\n",
        encoding="utf-8",
    )
    if forbidden:
        raise SystemExit(f"Unexpected Codex tool items: {forbidden}")


if __name__ == "__main__":
    main()
