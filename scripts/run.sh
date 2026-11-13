#!/bin/bash
# Run one entry point from a clean output directory.
# Usage: scripts/run.sh <name> <script.py or "-m module"> [overrides...]
# The run writes to $STABLEWM_HOME/checkpoints/<name>; the script refuses if that exists, so a rerun never mixes
# with an old run; its metrics.csv is under $STABLEWM_HOME/spt/<name>/runs. Default storage is runs/ in the repo, with datasets under runs/datasets.
set -eu
name=$1 script=$2
shift 2
cd "$(dirname "$0")/.."
git rev-parse HEAD
export STABLEWM_HOME=${STABLEWM_HOME:-$PWD/runs}
# stable-pretraining writes metrics.csv under SPT_CACHE_DIR/runs; one cache per run keeps each curve findable by name
export SPT_CACHE_DIR=$STABLEWM_HOME/spt/$name
out=$STABLEWM_HOME/checkpoints/$name
if [ -e "$out" ]; then
  echo "refusing: $out exists" >&2
  exit 1
fi
exec "${PYTHON:-.venv/bin/python}" $script "$@" subdir="$name" output_model_name="$name"
