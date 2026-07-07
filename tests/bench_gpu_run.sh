#!/usr/bin/env bash
# Run phase: benchmark the binaries in bench-bin/ (built beforehand with
# tests/bench_gpu_build.sh on the target system). No compilation here, so it
# can run inside a batch job / on a GPU node without the build toolchain.
# All configs run interleaved (round-robin) to average out thermal/clock drift;
# prints per-config medians for every block size in NT_LIST.
#
# Usage (from the repo root):
#   tests/bench_gpu_run.sh
#   NX=128 REPEATS=5 NT_LIST="64 128 256 512 1024" tests/bench_gpu_run.sh
#
# Tunables (env):
#   NX         box size per dim, atoms = 4*NX^3    (default 64 -> 1M atoms)
#   STEPS      timesteps per run                   (default 200)
#   REPEATS    interleaved repetitions             (default 3)
#   NT_LIST    NUM_THREADS block sizes to sweep    (default "128 512")
#   SLEEP      seconds between runs                (default 10)
#   CONFIGS    which bench-bin/ binaries to run    (default: all present)
set -euo pipefail

BINDIR=${BINDIR:-bench-bin}
NX=${NX:-64}
STEPS=${STEPS:-200}
REPEATS=${REPEATS:-3}
NT_LIST=${NT_LIST:-"128 512"}
SLEEP=${SLEEP:-10}
CONFIGS=${CONFIGS:-$(cd "$BINDIR" 2>/dev/null && ls | grep -v build-info || true)}
[ -n "$CONFIGS" ] || { echo "no binaries in $BINDIR - run tests/bench_gpu_build.sh first"; exit 1; }

[ -f "$BINDIR/build-info.txt" ] && cat "$BINDIR/build-info.txt"

runs=$BINDIR/runs.$$.txt
: > "$runs"
echo "== Running: NX=$NX ($((4 * NX * NX * NX)) atoms), $STEPS steps, $REPEATS repeats, NT in {$NT_LIST}"
for rep in $(seq "$REPEATS"); do
    for nt in $NT_LIST; do
        for cfg in $CONFIGS; do
            sleep "$SLEEP"
            perf=$(NUM_THREADS=$nt "$BINDIR/$cfg" -n "$STEPS" -nx "$NX" -ny "$NX" -nz "$NX" |
                grep -oP '[0-9.]+(?= atom updates)')
            echo "$cfg NT=$nt rep$rep: $perf au/us"
            echo "$cfg $nt $perf" >> "$runs"
        done
    done
done

echo
echo "== Medians (atom updates/us)"
printf '%-14s %8s %10s\n' config NT median
for cfg in $CONFIGS; do
    for nt in $NT_LIST; do
        median=$(awk -v c="$cfg" -v n="$nt" '$1==c && $2==n {print $3}' "$runs" |
            sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
        printf '%-14s %8s %10s\n' "$cfg" "$nt" "$median"
    done
done
rm -f "$runs"
