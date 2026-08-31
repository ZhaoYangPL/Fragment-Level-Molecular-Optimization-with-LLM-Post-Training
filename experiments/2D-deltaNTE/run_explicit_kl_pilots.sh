#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/../../tinker_env/bin/python}"
ENTRY="${ROOT_DIR}/train_grpo_delta_abs_explicit_kl.py"
ANCHOR="tinker://6adddb5a-0ab5-5a96-9178-b23dcbc0823d:train:0/weights/Layer-2D-11000-RL1e-6"

run_pilot() {
    local beta="$1"
    local suffix="$2"

    "${PYTHON}" "${ENTRY}" \
        --checkpoint "${ANCHOR}" \
        --kl-reference-checkpoint "${ANCHOR}" \
        --num-iterations 100 \
        --save-name "Layer-2D-11100-ExplicitKL-beta${suffix}" \
        --run-tag "pilot11000_explicitkl_beta${suffix}" \
        --learning-rate 1e-6 \
        --refresh-sampling-every 1 \
        --rng-seed 42 \
        --kl-penalty-coef "${beta}" \
        --no-token-prompt \
        --no-proxy-probe
}

run_pilot 0 0
run_pilot 0.01 001
run_pilot 0.03 003
run_pilot 0.05 005

"${PYTHON}" "${ROOT_DIR}/select_explicit_kl_pilot.py"
"${PYTHON}" "${ROOT_DIR}/run_selected_explicit_kl_500.py"
