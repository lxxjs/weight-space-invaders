#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
target="$script_dir/data/upstream"
revision=4d09ba6d6a7e891dd8063974dfe81b3d78e3494f

if [ ! -d "$target/.git" ]; then
    mkdir -p "$script_dir/data"
    git clone --filter=blob:none --sparse https://github.com/RikiyaT/lm_understand-logs.git "$target"
fi
git -C "$target" fetch origin "$revision"
git -C "$target" checkout --detach "$revision"
git -C "$target" sparse-checkout set exp5_write_from_scratch
printf 'Reference ready: %s\n' "$target/exp5_write_from_scratch"
