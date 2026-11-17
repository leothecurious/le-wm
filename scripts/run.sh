#!/bin/bash
# Run one entry point from a clean output directory.
# Usage: scripts/run.sh <name> <script.py or "-m module"> [overrides...]
# Output goes to $STABLEWM_HOME/checkpoints/<name>, which must not exist yet, and metrics.csv to
# $STABLEWM_HOME/spt/<name>/runs. STABLEWM_HOME defaults to the library's cache root, so measure.py finds the same checkpoints.
set -eu
name=$1 script=$2
shift 2
cd "$(dirname "$0")/.."
git rev-parse HEAD
STABLEWM_HOME=$(uv run python -c "from stable_worldmodel.data.utils import get_cache_dir; print(get_cache_dir())")
export STABLEWM_HOME
# stable-pretraining writes metrics.csv under SPT_CACHE_DIR/runs; one cache per run keeps each curve findable by name
export SPT_CACHE_DIR=$STABLEWM_HOME/spt/$name
out=$STABLEWM_HOME/checkpoints/$name
if [ -e "$out" ]; then
  echo "refusing: $out exists" >&2
  exit 1
fi
exec uv run $script "$@" subdir="$name" output_model_name="$name"
