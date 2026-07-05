#!/bin/bash
# Smoke test — Qwen3 behavior GRPO (tag-free data), short run to verify the pipeline end-to-end.
set -e
REPO_DIR="${REPO_DIR:-$HOME/repo/BeRL}"
export MODEL_FAMILY=qwen3
export TASK=behavior
export EXP_ID="${EXP_ID:-test-qwen3_smoke}"
export DATA_NAME="${DATA_NAME:-dcfg_smoke_mix_gemma}"
export DATA_TRAIN="${DATA_TRAIN:-$REPO_DIR/data/dcfg_smoke_mix_gemma.parquet}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TEST_FREQ="${TEST_FREQ:-5}"
export SAVE_FREQ="${SAVE_FREQ:-999}"
export MAX_RESP="${MAX_RESP:-2048}"
# shellcheck disable=SC1091
source "$REPO_DIR/experiments/lib/common.sh"
berl::run "$@"
