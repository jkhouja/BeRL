#!/bin/bash
# Smoke test — Qwen2.5 behavior GRPO, short run to verify the pipeline end-to-end.
# RQ=test so it never pollutes real experiment names. Defaults to the combined dialogue+CGA parquet.
set -e
REPO_DIR="${REPO_DIR:-$HOME/repo/BeRL}"
export MODEL_FAMILY=qwen2.5
export TASK=behavior
export EXP_ID="${EXP_ID:-test-qwen2.5_smoke}"
export DATA_NAME="${DATA_NAME:-dcfg_smoke_mix}"
export DATA_TRAIN="${DATA_TRAIN:-$REPO_DIR/data/dcfg_smoke_mix.parquet}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TEST_FREQ="${TEST_FREQ:-5}"
export SAVE_FREQ="${SAVE_FREQ:-999}"
export MAX_RESP="${MAX_RESP:-2048}"
# shellcheck disable=SC1091
source "$REPO_DIR/experiments/lib/common.sh"
berl::run "$@"
