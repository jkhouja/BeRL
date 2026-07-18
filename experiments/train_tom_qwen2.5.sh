#!/bin/bash
# Direct rule-based ToM GRPO — Qwen2.5 family (no LM reward model; data_source→rule scoring; KL=0.001).
# Required env: EXP_ID, DATA_NAME. DATA_TRAIN defaults to the native message-format parquet
#   data/cleaned_tom/ToM_train_HiEx_hint_v3.parquet (matches the eval prompts exactly; no
#   <think> prefill). Task-aware defaults: TRAIN_BATCH=8, MAX_RESP=2048 (~400 steps, 1 epoch).
#   Override any of these via env. See experiments/lib/common.sh.
set -e
REPO_DIR="${REPO_DIR:-$HOME/repo/BeRL}"
export MODEL_FAMILY=qwen2.5
export TASK=tom_rulebased
# shellcheck disable=SC1091
source "$REPO_DIR/experiments/lib/common.sh"
berl::run "$@"
