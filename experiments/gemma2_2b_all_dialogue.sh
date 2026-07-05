#!/bin/bash
# Gemma equivalent of experiments/qwen2.5_3b_all_dialogue.sh
# Gemma-2-2b-it on the merged "all dialogue" dataset (9 sources, 26,334 rows) —
# actor-as-RM, log-prob reward (matches Qwen: reward_type only set on reward_model.*,
# so actor-as-RM falls back to raw mean log-prob).
# Gemma-specific: Gemma2 uses attention & final logit soft-capping. OLD flash-attn ignored the
# attn soft-cap, so training/ref logits diverged from the soft-capped vLLM rollout -> unstable
# GRPO (empty-output collapse), which forced attn_implementation=eager + use_remove_padding=False.
# That combo pads sequences and materializes the full [batch, seq, 256k-vocab] logits+backward
# tensor -> ~8GB peak -> OOM (even on 8 GPUs, died at step 80). RESOLVED: flash_attn 2.8.3 +
# transformers 4.51.3 DO apply Gemma2 soft-capping in the flash path (flash_attn_varlen_func takes
# a `softcap` arg; HF passes attn_logit_softcapping; final_logit_softcapping is applied at the LM
# head regardless of backend). So we now use the same efficient recipe as Qwen: flash_attention_2
# + use_remove_padding=True (verl registers gemma2 as rmpad-capable). This packs unpadded sequences
# and removes the 256k-vocab padded-logits OOM. vLLM rollout keeps VLLM_ATTENTION_BACKEND=FLASH_ATTN.
# use_dynamic_bsz + ppo_max_token_len_per_gpu=3072 kept as a memory safety cap.

set -x

REPO_DIR=$HOME/repo/BeRL
eval "$($HOME/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate tom
TODAY=$(date +%Y%m%d)
mkdir -p $REPO_DIR/logs/${TODAY}

export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # Gemma2 requires FLASH_ATTN (not XFORMERS) for logits soft capping
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# This SLURM job (5412758) is allocated ALL 8 GPUs on the node and no other SLURM job shares it.
# The ~73GB/GPU OOMs were NOT an external squatter -- they were our OWN FSDP workers peaking (the
# OOM message shows "allocated by PyTorch" against our own worker PIDs). Using all 8 GPUs gives
# more dp ranks, so each GPU holds a smaller slice of the batch -> lower per-GPU activation peak
# (the first 8-GPU run reached step 44 vs ~20 on 4 GPUs). Use GEMMA_GPUS to override. TP=2 needs
# an even count; batch 32 / mini 128 divide cleanly by dp=4 (8 GPUs / TP2).
export CUDA_VISIBLE_DEVICES=${GEMMA_GPUS:-0,1,2,3,4,5,6,7}
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)

train_batch_size=32
enable_gradient_checkpointing=True
ROLLOUT_N=16
SUBTRACT_BASELINE=False
USE_ACTOR_AS_RM=False
REWARD_TYPE="power"
POWER_K=2.0
POWER_LL_MIN=-8.0
DATASET_NAME="all_dialogue_merged"
# FROZEN RM (USE_ACTOR_AS_RM=False): the reward is computed by a separate, frozen copy of the
# base model (reward_model.model.path=model_name), NOT the evolving actor. It applies the power
# transform (REWARD_TYPE=power, k=2.0, ll_min=-8) to the base model's length-normalized mean
# log-prob of the response. This breaks the actor-as-RM reward-hacking feedback loop: with an
# actor-as-RM the actor inflated its own reward by sharpening/lengthening its outputs (observed:
# reward runaway to -15, response_length 120->550, tomi eval collapse 0.66->0.27 at ~step 200).
# A frozen RM's opinion is fixed, so the actor cannot game it. Proven 17f / dialogue_7b_frozenRM
# recipe. NOTE: the RewardModelWorker cannot use rmpad (forwards padded seqs) but defaults to
# attn_implementation=flash_attention_2 so Gemma2 soft-cap is applied; RM param_offload + a small
# RM micro-batch keep the 256k-vocab forward within memory.
EXP_DESC="power-reward-flash-fantom"

# Model weights to load. Defaults to base gemma-2-2b-it, but GEMMA_INIT_MODEL lets the
# supervisor warm-start from the latest saved checkpoint after an environmental restart
# (verl has no training-state resume; warm-starting weights makes restarts cumulative).
MODEL_TAG="gemma-2-2b-it"
model_name="${GEMMA_INIT_MODEL:-google/gemma-2-2b-it}"
lr=5e-7

num_epochs=2

data_train_files=$REPO_DIR/data/merged_all_dialogue_eval_prompt.parquet
test_files="[$REPO_DIR/data/cleaned_tom/ToM_test_HiExTi_hint_v3.parquet,$REPO_DIR/data/cleaned_tom/fantom_test_50pct.parquet]"

RM_TYPE=$( [ "$USE_ACTOR_AS_RM" = "True" ] && echo "actorRM" || echo "frozenRM" )
BASELINE_TAG=$( [ "$SUBTRACT_BASELINE" = "True" ] && echo "baseline" || echo "nobaseline" )
# Use the fixed MODEL_TAG (not basename of a possibly-checkpoint model_name) so the experiment
# name, checkpoint dir, and WandB run stay stable across warm-started restarts.
EXP_NAME="${DATASET_NAME}-${MODEL_TAG}-${RM_TYPE}-${BASELINE_TAG}-lr${lr}-n${ROLLOUT_N}-${EXP_DESC}"

cd $REPO_DIR
HYDRA_FULL_ERROR=1 RAY_BACKEND_LOG_LEVEL=debug python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_train_files \
    data.val_files="$test_files" \
    data.train_batch_size=$train_batch_size \
    data.prompt_is_text=False \
    +data.fold_system_prompt=True \
    +data.truncation=left \
    data.val_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    reward_model.type="lm" \
    reward_model.enable=True \
    reward_model.model.path=$model_name \
    reward_model.micro_batch_size=4 \
    reward_model.forward_max_token_len_per_gpu=3072 \
    reward_model.model.fsdp_config.param_offload=True \
    +reward_model.subtract_baseline=$SUBTRACT_BASELINE \
    +reward_model.use_actor_as_rm=$USE_ACTOR_AS_RM \
    +reward_model.reward_type=$REWARD_TYPE \
    +reward_model.power_k=$POWER_K \
    +reward_model.power_ll_min=$POWER_LL_MIN \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.attn_implementation=flash_attention_2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=$enable_gradient_checkpointing \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="EmpathicDialogue_GRPO" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=10 \
    trainer.test_freq=30 \
    trainer.total_epochs=$num_epochs $@ 2>&1 | tee $REPO_DIR/logs/${TODAY}/${EXP_NAME}.log
