#!/usr/bin/env bash
# Energy-per-atom-update measurement: polls GPU power (nvidia-smi or rocm-smi)
# while the benchmark runs, then reports mean power, energy, and nJ/atom-update.
#
# Usage: tests/bench_gpu_energy.sh <binary> [binary args...]
#   e.g. tests/bench_gpu_energy.sh ./bench-bin/single+sort -n 1000 -nx 64 -ny 64 -nz 64
#
# Notes:
#  - Power includes everything on the GPU (idle floor, other jobs) - use
#    exclusive nodes and runs of >= 30 s so the idle ramp is amortized.
#  - For the clock-cap sweep (find the J/update minimum, not the time minimum):
#      for c in 1980 1600 1200 900; do sudo nvidia-smi -lgc 0,$c; <this script>; done
#      sudo nvidia-smi -rgc   # reset (needs admin; ask the cluster operators)
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 <binary> [args...]"; exit 1; }

POWERLOG=$(mktemp)
trap 'kill $POLLER 2>/dev/null || true; rm -f "$POWERLOG"' EXIT

if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --loop-ms=100 \
        > "$POWERLOG" &
    POLLER=$!
elif command -v rocm-smi >/dev/null; then
    ( while :; do
        rocm-smi --showpower --csv 2>/dev/null | awk -F, '/^card/{print $2}'
        sleep 0.1
    done ) > "$POWERLOG" &
    POLLER=$!
else
    echo "neither nvidia-smi nor rocm-smi found" >&2
    exit 1
fi

sleep 1 # let the poller settle
OUT=$("$@")
kill $POLLER 2>/dev/null || true

echo "$OUT" | grep Performance
PERF_LINE=$(echo "$OUT" | grep Performance)
WALL=$(echo "$PERF_LINE" | grep -oP '[0-9.]+(?=s total)')
AUPS=$(echo "$PERF_LINE" | grep -oP '[0-9.]+(?= atom updates)')
MEANW=$(awk '{s+=$1; n++} END{printf "%.1f", s/n}' "$POWERLOG")

awk -v w="$MEANW" -v t="$WALL" -v a="$AUPS" 'BEGIN{
    joules = w * t
    updates = a * t * 1e6
    printf "mean power: %s W | energy: %.1f J | %.3f nJ/atom-update\n",
        w, joules, joules / updates * 1e9
}'
