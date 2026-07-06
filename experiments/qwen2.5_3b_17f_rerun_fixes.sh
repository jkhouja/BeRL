#!/bin/bash
# Round 17f breakthrough rerun — Qwen2.5-3B dialogue GRPO, actor-as-RM — WITH the anti-collapse fixes.
#
# 17f was the best dialogue->ToM transfer run (tomi 63.4->69.4, explore 47.0->66.9, hi_tom
# 18.7->29.8). It never hard-collapsed, but hi_tom declined in epoch 2 (29.8 -> ~22-25) and
# explore kept climbing — i.e. mild epoch-2 drift. This rerun applies the two training-side
# fixes to that healthy run to measure their impact (should not hurt; may steady epoch 2):
#   Fix 1  actor.think_only_pg=True        -> PG + entropy restricted to <think>...</think>.
#   Fix 2  actor.format_penalty=5.0        -> graded penalty on malformed model responses
#                                             (actor-as-RM reads actor.format_penalty).
#
# Config reproduces 17f: filtered 6k eval-prompt dialogue data (cot_eval), power reward
# k=2.0 ll_min=-8.0, clip (-40,40) [MIN/MAX_REWARD in fsdp_workers.py], actor-as-RM,
# no baseline, KL=0.05, LR=5e-7, batch=32, mini_batch=128, rollout_n=16, 2 epochs.
# Eval uses the standing subsample300 suite (metrics suffixed _sub300).
# Compare against the documented 17f table in grpo_tuning_changelog.md.
# WandB online is mandatory (do NOT unset WANDB_API_KEY / set offline).
set -e
REPO_DIR="${REPO_DIR:-$HOME/repo/BeRL}"

export MODEL_FAMILY=qwen2.5
export TASK=behavior

# --- data: the filtered 6,214-sample eval-prompt (cot_eval) dialogue set from 17f ---
# Eval uses VAL_SUITE=subsample300 default — do NOT hardcode VAL_FILES.
export DATA_NAME="${DATA_NAME:-merged_dialogue_datasets_filtered_eval_prompt}"
export DATA_TRAIN="${DATA_TRAIN:-$REPO_DIR/data/merged_dialogue_datasets_filtered_eval_prompt.parquet}"

# --- 17f context lengths (differ from the qwen2.5 family defaults of 2048/4096) ---
export MAX_PROMPT="${MAX_PROMPT:-1024}"
export MAX_RESP="${MAX_RESP:-2048}"

# --- reward: actor-as-RM + power reward (17f recipe; these are also the behavior defaults) ---
export USE_ACTOR_AS_RM="${USE_ACTOR_AS_RM:-True}"
export REWARD_TYPE="${REWARD_TYPE:-power}"
export POWER_K="${POWER_K:-2.0}"
export POWER_LL_MIN="${POWER_LL_MIN:--8.0}"

# --- run bookkeeping ---
export EXP_ID="${EXP_ID:-round21-17f-rerun-fixes}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
export TEST_FREQ="${TEST_FREQ:-10}"   # matches original 17f cadence
export SAVE_FREQ="${SAVE_FREQ:-50}"

# The two anti-collapse fixes (overridable; set FORMAT_PENALTY=0 / THINK_ONLY_PG=False
# to reproduce the *original* fix-free 17f for a clean A/B).
FORMAT_PENALTY="${FORMAT_PENALTY:-5.0}"
THINK_ONLY_PG="${THINK_ONLY_PG:-True}"

# shellcheck disable=SC1091
source "$REPO_DIR/experiments/lib/common.sh"

# Actor-as-RM reads the fix flags off the actor config node:
#   actor.think_only_pg  (Fix 1)  and  actor.format_penalty (Fix 2, actor-as-RM path).
berl::run \
  actor_rollout_ref.actor.think_only_pg="$THINK_ONLY_PG" \
  actor_rollout_ref.actor.format_penalty="$FORMAT_PENALTY" \
  "$@"
