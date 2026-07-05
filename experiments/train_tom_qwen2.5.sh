#!/bin/bash
# Direct rule-based ToM GRPO — Qwen2.5 family (no LM reward model; data_source→rule scoring; KL=0.001).
# Required env: EXP_ID, DATA_NAME, DATA_TRAIN (ToM train parquet). See experiments/lib/common.sh.
set -e
REPO_DIR="${REPO_DIR:-$HOME/repo/BeRL}"
export MODEL_FAMILY=qwen2.5
export TASK=tom_rulebased
# shellcheck disable=SC1091
source "$REPO_DIR/experiments/lib/common.sh"
berl::run "$@"
