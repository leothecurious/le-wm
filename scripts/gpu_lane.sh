#!/bin/bash
# Run the commands in a job file one after another on one GPU.
# Usage: scripts/gpu_lane.sh <gpu> <jobfile> <logdir>
# Each non-empty, non-comment line of <jobfile> is "<name> <command...>"; its output goes to
# <logdir>/<name>.log. A job starts only once the GPU has more than 14 GB free and the machine
# more than 12 GB of available RAM, since the GPUs and RAM are shared with other users.
set -u
gpu=$1 jobs=$2 logdir=$3
mkdir -p "$logdir"
say(){ echo "$(date "+%F %T") gpu$gpu $*" >> "$logdir/lanes.log"; }
ready(){
  [ "$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)" -gt 14000 ] &&
  [ "$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)" -gt 12000 ]
}
grep -v '^\s*\(#\|$\)' "$jobs" | while read -r name cmd; do
  until ready; do sleep 60; done
  say "start $name"
  if CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd" > "$logdir/$name.log" 2>&1 < /dev/null; then
    say "ok $name"
  else
    say "FAILED $name (exit $?)"
  fi
done
say "lane finished"
