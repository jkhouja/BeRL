#!/bin/bash
# Gemma-2-2b dialogue GRPO — anti-collapse v2: kill the residual entropy diffusion.
#
# The v1 run (gemma2_2b_dialogue_anticollapse.sh, Fix 1 + Fix 2) PREVENTED the catastrophic
# step-60 collapse but did not fully stabilise eval: entropy still crept 1.7 -> ~3.0 (past the
# 2.5 ceiling) and eval slowly drifted down (tomi 0.627 -> 0.537 by step 90). The remaining
# driver is entropy diffusion under the positive entropy bonus + moderate KL. This variant
# targets that directly, keeping BOTH prior fixes:
#   Fix 1  actor.think_only_pg=True     -> PG + entropy restricted to <think>...</think>.
#   Fix 2  reward_model.format_penalty  -> graded penalty on malformed model responses.
#   NEW    actor.entropy_coeff=0        -> remove the positive entropy bonus feeding the climb.
#   NEW    KL=0.1                        -> stronger anchor to the base policy (was 0.05).
#
# Frozen RM (USE_ACTOR_AS_RM=False), power reward k=2.0 ll_min=-8.0. Gemma stability flags
# match the known-good gemma recipe. WandB online is mandatory.
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

# --- stronger KL anchor (was 0.05) ---
export KL="${KL:-0.1}"

# --- run bookkeeping: monitor well past the old step-60 collapse ---
export EXP_ID="${EXP_ID:-round21-gemma-anticollapse-noentropy}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
export TEST_FREQ="${TEST_FREQ:-15}"
export SAVE_FREQ="${SAVE_FREQ:-15}"

# The fix strengths (overridable from the environment).
FORMAT_PENALTY="${FORMAT_PENALTY:-5.0}"
THINK_ONLY_PG="${THINK_ONLY_PG:-True}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.0}"

# shellcheck disable=SC1091
source "$REPO_DIR/experiments/lib/common.sh"

# Extra hydra overrides passed through to main_ppo (last-wins over common.sh ARGS):
#  * Gemma stability flags absent from common.sh's gemma path.
#  * RM forward-token cap (avoids Gemma2 256k-vocab single-alloc OOM under dynamic bsz).
#  * The two anti-collapse fixes + the entropy-coeff=0 change.
berl::run \
  +actor_rollout_ref.model.attn_implementation=flash_attention_2 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072 \
  reward_model.forward_max_token_len_per_gpu=3072 \
  reward_model.model.fsdp_config.param_offload=True \
  reward_model.micro_batch_size=4 \
  actor_rollout_ref.actor.think_only_pg="$THINK_ONLY_PG" \
  actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF" \
  reward_model.format_penalty="$FORMAT_PENALTY" \
  "$@"
