#!/bin/bash
# Run train.py under a fresh run name.
# Usage: scripts/run.sh <name> [overrides...]
# Refuses if <runs>/checkpoints/<name> exists. <runs> is the paths.runs override, else `runs` in
# config/paths/local.yaml, else the library's cache root.
set -eu
name=$1
shift
cd "$(dirname "$0")/.."
git rev-parse HEAD
runs=
for arg in "$@"; do
  case $arg in paths.runs=*) runs=${arg#paths.runs=} ;; esac
done
if [ -z "$runs" ] && [ -f config/paths/local.yaml ]; then
  runs=$(uv run python -c "from omegaconf import OmegaConf; print(OmegaConf.load('config/paths/local.yaml').get('runs') or '')")
fi
if [ -z "$runs" ]; then
  runs=$(uv run python -c "from stable_worldmodel.data.utils import get_cache_dir; print(get_cache_dir())")
fi
out=$runs/checkpoints/$name
if [ -e "$out" ]; then
  echo "refusing: $out exists" >&2
  exit 1
fi
exec uv run train.py "$@" subdir="$name" output_model_name="$name"
