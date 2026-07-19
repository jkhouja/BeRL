#!/bin/bash
# Behavior-reward GRPO — Qwen2.5 family.
# Knobs come from env (see experiments/lib/common.sh); required: EXP_ID, DATA_NAME, DATA_TRAIN.
# Optional: EXP_NUM (tracker short id, e.g. E034/PS2 — prepended to RUN_NAME for WandB search).
# Example:
#   EXP_NUM=E001 EXP_ID=Phase0-p0_power DATA_NAME=dcfg_default \
#   DATA_TRAIN=$HOME/repo/BeRL/data/dcfg_default.parquet \
#   bash experiments/train_behavior_qwen2.5.sh
set -e
REPO_DIR="${REPO_DIR:-$HOME/repo/BeRL}"
export MODEL_FAMILY=qwen2.5
export TASK=behavior
# shellcheck disable=SC1091
source "$REPO_DIR/experiments/lib/common.sh"
berl::run "$@"
