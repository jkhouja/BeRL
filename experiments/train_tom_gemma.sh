#!/bin/bash
# Direct rule-based ToM GRPO — Gemma-2 family (no LM reward model; data_source→rule scoring; KL=0.001).
# Required env: EXP_ID, DATA_NAME, DATA_TRAIN (ToM train parquet). Optional: EXP_NUM (tracker short
#   id, e.g. E034/PS2 — prepended to RUN_NAME for WandB search). See experiments/lib/common.sh.
set -e
REPO_DIR="${REPO_DIR:-$HOME/repo/BeRL}"
export MODEL_FAMILY=gemma
export TASK=tom_rulebased
# shellcheck disable=SC1091
source "$REPO_DIR/experiments/lib/common.sh"
berl::run "$@"
