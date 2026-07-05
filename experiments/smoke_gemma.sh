#!/bin/bash
# Smoke test — Gemma-2 behavior GRPO (FLASH_ATTN, tag-free), short run to verify the pipeline.
set -e
REPO_DIR="${REPO_DIR:-$HOME/repo/BeRL}"
export MODEL_FAMILY=gemma
export TASK=behavior
export EXP_ID="${EXP_ID:-test-gemma_smoke}"
export DATA_NAME="${DATA_NAME:-dcfg_smoke_mix_notags}"
export DATA_TRAIN="${DATA_TRAIN:-$REPO_DIR/data/merged_all_dialogue_notags.parquet}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TEST_FREQ="${TEST_FREQ:-5}"
export SAVE_FREQ="${SAVE_FREQ:-999}"
# shellcheck disable=SC1091
source "$REPO_DIR/experiments/lib/common.sh"
berl::run "$@"
