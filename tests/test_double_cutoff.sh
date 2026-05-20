#!/usr/bin/env bash
set -euo pipefail

# Regression test for the double-cutoff scheme (--outer-skin).
#
# Runs three configurations of the FCC lattice testcase and verifies that
# temperature and pressure remain physically consistent:
#
#   Baseline   : single-cutoff, skin=0.3, reneigh-every 20
#   Config A   : double-cutoff, same outer cutoff (0.1+0.2=0.3), same rebuild schedule
#                → near-identical dynamics, tight tolerance
#   Config B   : double-cutoff, extended outer cutoff (0.3+0.4=0.7), fewer rebuilds
#                → slight divergence from different rebuild times, loose tolerance
#
# Usage:
#   ./tests/test_double_cutoff.sh /path/to/MDBench-<TAG>
# or set MDBENCH_BIN in the environment.

BIN="${1:-${MDBENCH_BIN:-}}"

if [[ -z "${BIN}" ]]; then
    echo "Usage: $0 /path/to/MDBench-<TAG>  (or set MDBENCH_BIN)" >&2
    exit 1
fi

if [[ ! -x "${BIN}" ]]; then
    echo "Binary '${BIN}' is not executable" >&2
    exit 1
fi

# Skip gracefully when the binary was not built with double-cutoff support
# (i.e. it is not a clusterpair build).
if ! "${BIN}" --help 2>&1 | grep -q "outer-skin"; then
    echo "Binary does not support --outer-skin (not a clusterpair build), skipping."
    exit 0
fi

COMMON_ARGS="-n 200 -r 2.5"

LOG_BASE="$(mktemp "${TMPDIR:-/tmp}/mdbench_dcut_base.XXXXXX")"
LOG_A="$(mktemp    "${TMPDIR:-/tmp}/mdbench_dcut_A.XXXXXX")"
LOG_B="$(mktemp    "${TMPDIR:-/tmp}/mdbench_dcut_B.XXXXXX")"

cleanup() { rm -f "${LOG_BASE}" "${LOG_A}" "${LOG_B}"; }

extract_tp() {
    local file="$1"
    grep -E '^[[:space:]]*[0-9]+[[:space:]]+[0-9.eE+-]+' "${file}" | tail -n 1 || true
}

echo "Running baseline (skin=0.3, reneigh-every 20)..."
"${BIN}" ${COMMON_ARGS} -s 0.3 --reneigh-every 20 >"${LOG_BASE}"

echo "Running Config A (skin=0.1, outer-skin=0.2, prune-every 4, reneigh-every 20)..."
"${BIN}" ${COMMON_ARGS} -s 0.1 --outer-skin 0.2 --prune-every 4 --reneigh-every 20 >"${LOG_A}"

echo "Running Config B (skin=0.3, outer-skin=0.4, prune-every 10, reneigh-every 40)..."
"${BIN}" ${COMMON_ARGS} -s 0.3 --outer-skin 0.4 --prune-every 10 --reneigh-every 40 >"${LOG_B}"

base_line="$(extract_tp "${LOG_BASE}")"
a_line="$(extract_tp    "${LOG_A}")"
b_line="$(extract_tp    "${LOG_B}")"

if [[ -z "${base_line}" || -z "${a_line}" || -z "${b_line}" ]]; then
    echo "Could not extract thermo lines from one or more outputs." >&2
    echo "Logs: ${LOG_BASE}  ${LOG_A}  ${LOG_B}" >&2
    exit 1
fi

base_T=$(echo "${base_line}" | awk '{print $2}')
base_P=$(echo "${base_line}" | awk '{print $3}')
a_T=$(echo    "${a_line}"    | awk '{print $2}')
a_P=$(echo    "${a_line}"    | awk '{print $3}')
b_T=$(echo    "${b_line}"    | awk '{print $2}')
b_P=$(echo    "${b_line}"    | awk '{print $3}')

echo "Baseline  : T=${base_T}  P=${base_P}"
echo "Config A  : T=${a_T}     P=${a_P}"
echo "Config B  : T=${b_T}     P=${b_P}"

# --- Compare two values against a tolerance (relative error) -----------------
# Usage: check_rel <label> <got> <ref> <tol>
check_rel() {
    local label="$1" got="$2" ref="$3" tol="$4"
    local ok
    ok=$(awk -v a="${got}" -v b="${ref}" -v t="${tol}" \
        'BEGIN { d = a - b; if (d < 0) d = -d; print (d / (b < 0 ? -b : b) <= t) ? "1" : "0" }')
    if [[ "${ok}" != "1" ]]; then
        local diff
        diff=$(awk -v a="${got}" -v b="${ref}" \
            'BEGIN { d = a - b; if (d < 0) d = -d; print d / (b < 0 ? -b : b) }')
        echo "FAIL: ${label}: got ${got}, ref ${ref}, rel_err=${diff} > tol=${tol}" >&2
        return 1
    fi
}

# Config A: same outer cutoff (0.1+0.2=0.3) and same rebuild schedule as baseline
# → prune does not alter any non-zero force pair; expect near-identical dynamics.
TIGHT_T="1e-4"
TIGHT_P="1e-3"

# Config B: wider outer cutoff, different rebuild interval (40 vs 20 steps)
# → physical divergence from different rebuild times; loose tolerance.
LOOSE_T="2e-2"
LOOSE_P="5e-2"

pass=1
check_rel "Config A temperature" "${a_T}" "${base_T}" "${TIGHT_T}" || pass=0
check_rel "Config A pressure"    "${a_P}" "${base_P}" "${TIGHT_P}" || pass=0
check_rel "Config B temperature" "${b_T}" "${base_T}" "${LOOSE_T}" || pass=0
check_rel "Config B pressure"    "${b_P}" "${base_P}" "${LOOSE_P}" || pass=0

if [[ "${pass}" != "1" ]]; then
    echo "Logs: ${LOG_BASE}  ${LOG_A}  ${LOG_B}" >&2
    exit 1
fi

cleanup
echo "Double-cutoff regression PASSED."
