#!/usr/bin/env bash
# GPU config-matrix benchmark: builds the LJ_COMB_RULE x SORT_ATOMS matrix,
# runs all configs interleaved (round-robin) to average out thermal drift,
# and prints per-config medians for every block size in NT_LIST.
#
# Usage (from the repo root):
#   NVIDIA H200:  TOOLCHAIN=NVCC MAKE_ARGS="CUDA_ARCH=sm_90" tests/bench_gpu_matrix.sh
#   AMD MI300X:   TOOLCHAIN=HIPCC MAKE_ARGS="GPU_ARCH=gfx942" tests/bench_gpu_matrix.sh
#
# Tunables (env):
#   TOOLCHAIN  NVCC | HIPCC                        (required)
#   MAKE_ARGS  extra make vars, e.g. "CUDA_ARCH=sm_90"
#   NX         box size per dim, atoms = 4*NX^3    (default 64 -> 1M atoms)
#   STEPS      timesteps per run                   (default 200)
#   REPEATS    interleaved repetitions             (default 3)
#   NT_LIST    NUM_THREADS block sizes to sweep    (default "128 512")
#   SLEEP      seconds between runs                (default 10)
#   CONFIGS    subset of: base single sort single+sort   (default all)
set -euo pipefail

: "${TOOLCHAIN:?set TOOLCHAIN=NVCC or TOOLCHAIN=HIPCC}"
MAKE_ARGS=${MAKE_ARGS:-}
NX=${NX:-64}
STEPS=${STEPS:-200}
REPEATS=${REPEATS:-3}
NT_LIST=${NT_LIST:-"128 512"}
SLEEP=${SLEEP:-10}
CONFIGS=${CONFIGS:-"base single sort single+sort"}

BINDIR=bench-bin
mkdir -p "$BINDIR"

build_flags() { # config name -> extra make flags
    case $1 in
    base) echo "" ;;
    single) echo "LJ_COMB_RULE=single" ;;
    sort) echo "SORT_ATOMS=true" ;;
    single+sort) echo "LJ_COMB_RULE=single SORT_ATOMS=true" ;;
    *) echo "unknown config: $1" >&2 && exit 1 ;;
    esac
}

echo "== Building configs: $CONFIGS (TOOLCHAIN=$TOOLCHAIN $MAKE_ARGS)"
for cfg in $CONFIGS; do
    flags=$(build_flags "$cfg")
    # LJ_COMB_RULE/SORT_ATOMS don't change the binary tag -> clean + copy away
    make clean TOOLCHAIN="$TOOLCHAIN" $MAKE_ARGS >/dev/null
    bin=$(make TOOLCHAIN="$TOOLCHAIN" $MAKE_ARGS $flags 2>&1 | awk '/LINKING/{print $NF}')
    [ -x "$bin" ] || { echo "build failed for $cfg"; exit 1; }
    cp "$bin" "$BINDIR/$cfg"
    echo "  $cfg -> $BINDIR/$cfg"
done

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
