#!/bin/bash
# Gemma-2-2b dialogue GRPO — anti-collapse validation run.
#
# Reproduces the run that peaked@step30 / collapsed@step60 (frozen RM + power reward on the
# tagged cot_eval dialogue data) and adds the two training-side fixes that prevent the
# eval-format regression:
#   Fix 1  actor.think_only_pg=True     -> policy-gradient + entropy restricted to
#                                          <think>...</think>; KL still anchors the full
#                                          response to base so the answer format can't drift.
#   Fix 2  reward_model.format_penalty  -> graded penalty on the model's OWN malformed
#                                          responses (missing/empty/duplicated think/answer).
#
# Frozen RM (USE_ACTOR_AS_RM=False), power reward k=2.0 ll_min=-8.0. Gemma stability flags
# (flash_attention_2, dynamic bsz, RM forward-token cap) match the known-good gemma recipe.
# WandB online is mandatory (do NOT unset WANDB_API_KEY / set offline).
set -e
REPO_DIR="${REPO_DIR:-$HOME/repo/BeRL}"

export MODEL_FAMILY=gemma
export TASK=behavior

# --- data: the tagged cot_eval dialogue set that produced the peak-then-collapse curve ---
# Eval uses the standing default suite (VAL_SUITE=subsample300 via common.sh) — do NOT
# override VAL_FILES here; subsample300 is the agreed eval set for all scripts.
export DATA_NAME="${DATA_NAME:-merged_all_dialogue_eval_prompt}"
export DATA_TRAIN="${DATA_TRAIN:-$REPO_DIR/data/merged_all_dialogue_eval_prompt.parquet}"

# --- reward: frozen RM + power reward (the collapse-repro recipe) ---
export USE_ACTOR_AS_RM="${USE_ACTOR_AS_RM:-False}"
export REWARD_TYPE="${REWARD_TYPE:-power}"
export POWER_K="${POWER_K:-2.0}"
export POWER_LL_MIN="${POWER_LL_MIN:--8.0}"

# --- run bookkeeping: monitor well past the old step-60 collapse ---
export EXP_ID="${EXP_ID:-round21-gemma-anticollapse}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
export TEST_FREQ="${TEST_FREQ:-15}"   # denser than the original 30 to catch the 30-60 window
export SAVE_FREQ="${SAVE_FREQ:-15}"

# The fix strengths (overridable from the environment).
FORMAT_PENALTY="${FORMAT_PENALTY:-5.0}"
THINK_ONLY_PG="${THINK_ONLY_PG:-True}"

# shellcheck disable=SC1091
source "$REPO_DIR/experiments/lib/common.sh"

# Extra hydra overrides passed through to main_ppo (last-wins over common.sh ARGS):
#  * Gemma stability flags absent from common.sh's gemma path.
#  * RM forward-token cap (the RM inherits use_dynamic_bsz -> would pack 32k tokens ->
#    ~47 GiB single alloc on Gemma2's 256k vocab -> OOM without this cap).
#  * The two anti-collapse fix flags.
berl::run \
  +actor_rollout_ref.model.attn_implementation=flash_attention_2 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072 \
  reward_model.forward_max_token_len_per_gpu=3072 \
  reward_model.model.fsdp_config.param_offload=True \
  reward_model.micro_batch_size=4 \
  actor_rollout_ref.actor.think_only_pg="$THINK_ONLY_PG" \
  reward_model.format_penalty="$FORMAT_PENALTY" \
  "$@"
