#!/bin/zsh
# Run three Space Invaders PPO seeds in sequence. Start under caffeinate.
set -u

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${PPO_REPO_ROOT:-$(realpath "$SCRIPT_DIR/../..")}" 
PYTHON="${PPO_PYTHON:-${REPO_ROOT}/.venv-ppo/bin/python}"
RUN_ROOT="${PPO_RUN_ROOT:-${SCRIPT_DIR}/results/space_invaders/ppo_500m_3seeds}"
TAG="${PPO_TAG:-all}"
if (( $# == 0 )); then
    set -- 0 1 2
fi
mkdir -p "$RUN_ROOT"
cd "$REPO_ROOT" || exit 1
if [[ -f "$RUN_ROOT/fatal_${TAG}.txt" ]]; then
    print -r -- "Previous run failed repeatedly; inspect fatal_${TAG}.txt and seed logs"
    exit 0
fi
print -r -- "$$" > "$RUN_ROOT/supervisor_${TAG}.pid"
print -r -- "$(date '+%Y-%m-%d %H:%M:%S %Z') supervisor started" 

for seed in "$@"; do
    output="$RUN_ROOT/seed_$seed"
    mkdir -p "$output"
    if [[ -f "$output/complete.json" ]]; then
        print -r -- "$(date '+%Y-%m-%d %H:%M:%S %Z') seed $seed already complete"
        continue
    fi
    attempts=0
    while [[ ! -f "$output/complete.json" ]]; do
        resume=()
        if [[ -f "$output/latest.pt" ]]; then
            resume=(--resume)
        fi
        print -r -- "$(date '+%Y-%m-%d %H:%M:%S %Z') starting seed $seed ${resume[*]}"
        if "$PYTHON" "$SCRIPT_DIR/ppo_live_training.py" \
            --game space_invaders --seed "$seed" \
            --updates 61036 --target-steps 500000000 \
            --checkpoint-every 500 --log-every 100 --live-frame-every 32 \
            --output-dir "$output" "${resume[@]}" >> "$output/train.log" 2>&1; then
            if [[ ! -f "$output/complete.json" ]]; then
                print -r -- "seed $seed returned success without complete.json"
                exit 1
            fi
            print -r -- "$(date '+%Y-%m-%d %H:%M:%S %Z') seed $seed complete"
        else
            code=$?
            attempts=$((attempts + 1))
            print -r -- "$(date '+%Y-%m-%d %H:%M:%S %Z') seed $seed exit $code, retry $attempts/5"
            if (( attempts >= 5 )); then
                print -r -- "seed $seed stopped after five failures"
                print -r -- "$(date '+%Y-%m-%d %H:%M:%S %Z') seed $seed exit $code" > "$RUN_ROOT/fatal_${TAG}.txt"
                exit "$code"
            fi
            sleep 15
        fi
    done
done

print -r -- "$(date '+%Y-%m-%d %H:%M:%S %Z') requested seeds complete: $*"
print -r -- "$(date '+%Y-%m-%d %H:%M:%S %Z')" > "$RUN_ROOT/all_complete_${TAG}.txt"
