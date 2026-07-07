# TODO: remote GPU benchmark runs (H200 / MI300X)

MuCoSim Phase 2. Local RTX 4060 / Radeon 780M results and full analysis live in
`Phase2/docs/01..04-*.md` (outside this repo). This file says **what to run on the
remote systems and how**.

## Branches

| Branch | Contents | Purpose |
|---|---|---|
| `main` | upstream + `CUDA_ARCH` make knob + `reallocateGPUKeep` fix + these scripts | **baseline** measurements |
| `gpu-opt` | main + `__ldg` gathers (force/neighbor kernels) + skip redundant reneigh H2D pushes | **optimized** measurements |
| `gpu-opt-float4` | gpu-opt + packed-float4 position gathers (SP only) | one-shot retry on H200 (was −4% locally; HBM3/larger L2 may flip it) |

The scripts are identical on all branches; what you benchmark is the checkout.

## Scripts

```bash
# Config matrix (LJ_COMB_RULE x SORT_ATOMS) + block-size sweep, interleaved, medians:
TOOLCHAIN=NVCC  MAKE_ARGS="CUDA_ARCH=sm_90"  tests/bench_gpu_matrix.sh   # H200
TOOLCHAIN=HIPCC MAKE_ARGS="GPU_ARCH=gfx942"  tests/bench_gpu_matrix.sh   # MI300X
# Knobs: NX (64 -> 1M atoms), STEPS, REPEATS, NT_LIST, CONFIGS, SLEEP (see header)

# Energy per atom-update (poll power while running; use runs >= 30 s, exclusive node):
tests/bench_gpu_energy.sh ./bench-bin/single+sort -n 2000 -nx 64 -ny 64 -nz 64
```

## H200 (CUDA, `CUDA_ARCH=sm_90`)

1. `main`: full matrix — `TOOLCHAIN=NVCC MAKE_ARGS="CUDA_ARCH=sm_90" NT_LIST="64 128 256 512 1024" tests/bench_gpu_matrix.sh`
2. `gpu-opt`: same command. Headline = best `gpu-opt` config vs `main` base config.
3. Size scaling: repeat the best config with `NX=64` (1M) and `NX=128` (8M);
   131k (default `NX=32`) likely underutilizes 132 SMs — measure it anyway for the report.
4. `gpu-opt-float4`: single run of the best config, same NX — keep only if it beats `gpu-opt`.
5. Energy (`tests/bench_gpu_energy.sh`, best config, >= 30 s runs):
   - baseline vs optimized nJ/atom-update
   - clock-cap sweep if permitted (`nvidia-smi -lgc`, see script header) — the force
     kernel is DRAM-bound, expect a J/update minimum below max clocks.
6. Record with every number: driver/CUDA version, clocks, power limit
   (`nvidia-smi -q -d CLOCK,POWER | head`), and whether the node was exclusive.

## MI300X (HIP, `GPU_ARCH=gfx942`)

1. `main` + `gpu-opt`: full matrix, `NT_LIST="64 128 256 512 1024"`.
   Note: locally `SORT_ATOMS=true` *hurt* on the AMD iGPU (−11%) while `single` gave
   +35% — the matrix decides per-GPU, don't assume the CUDA config.
2. The reneigh push-skip in `gpu-opt` was neutral on the local APU (no PCIe);
   on MI300X it should behave like CUDA (+6% locally) — this is the number to check.
3. Energy: `tests/bench_gpu_energy.sh` (uses rocm-smi automatically).
4. Profile the force kernel once with rocprof/omniperf to confirm it is DRAM-bound
   there too (drives whether byte-reduction ideas transfer).

## Do NOT re-test (closed locally, see Phase2/docs/04)

- Half neighbor lists (`-half 1`): 1.8x slower on GPU (atomics).
- Clusterpair GPU scheme: 9-18x slower, host-bound cluster building.
- Skin 0.3 / reneigh_every 30: physically invalid, crashes by design.

## Reminders

- Runs shorter than ~1000 steps are too noisy/short to validate reneighboring
  changes (docs/03); the matrix default STEPS=200 is fine for config ranking only.
- `tests/test_lj_comb_rules.sh` rebuilds configs and deletes existing binaries —
  rerun the matrix script (it rebuilds into `bench-bin/`) after it.
- Compare "atom updates/us"; energies at steps 0/100/200 must match across configs
  (bit-identical except SORT_ATOMS, which changes summation order in the last digits).
