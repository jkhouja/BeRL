#!/bin/bash
# Behavior-reward GRPO — Qwen3 family (tag-free data: cot_eval_notags, add_response_tags=false).
# Required env: EXP_ID, DATA_NAME, DATA_TRAIN. See experiments/lib/common.sh.
set -e
REPO_DIR="${REPO_DIR:-$HOME/repo/BeRL}"
export MODEL_FAMILY=qwen3
export TASK=behavior
# shellcheck disable=SC1091
source "$REPO_DIR/experiments/lib/common.sh"
berl::run "$@"
