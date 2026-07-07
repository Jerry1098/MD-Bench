#!/usr/bin/env bash
# Build phase: compile the LJ_COMB_RULE x SORT_ATOMS config matrix into bench-bin/.
# Compile ON the target system (the H200 nodes are ARM/Grace - an x86 build
# won't run there), then benchmark with tests/bench_gpu_run.sh.
#
# Usage (from the repo root):
#   H200 (ARM host): TOOLCHAIN=NVCC  MAKE_ARGS="CUDA_ARCH=sm_90 ISA=ARM" tests/bench_gpu_build.sh
#   MI300X:          TOOLCHAIN=HIPCC MAKE_ARGS="GPU_ARCH=gfx942"         tests/bench_gpu_build.sh
#
# Tunables (env):
#   TOOLCHAIN  NVCC | HIPCC                               (required)
#   MAKE_ARGS  extra make vars: CUDA_ARCH/GPU_ARCH/ISA/DATA_TYPE...
#   CONFIGS    subset of: base single sort single+sort    (default all)
set -euo pipefail

: "${TOOLCHAIN:?set TOOLCHAIN=NVCC or TOOLCHAIN=HIPCC}"
MAKE_ARGS=${MAKE_ARGS:-}
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
    [ -n "$bin" ] && [ -x "$bin" ] || { echo "build failed for $cfg"; exit 1; }
    cp "$bin" "$BINDIR/$cfg"
    echo "  $cfg -> $BINDIR/$cfg"
done

# Provenance for the report
{
    echo "date:      $(date -Is)"
    echo "host:      $(uname -n) ($(uname -m))"
    echo "branch:    $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
    echo "toolchain: $TOOLCHAIN $MAKE_ARGS"
    echo "configs:   $CONFIGS"
    { command -v nvcc >/dev/null && nvcc --version | tail -1; } || true
    { command -v hipcc >/dev/null && hipcc --version | head -1; } || true
} | tee "$BINDIR/build-info.txt"
