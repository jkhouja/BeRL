#!/bin/bash

set -x

TODAY=$(date +%Y%m%d)
mkdir -p logs/${TODAY}

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate tom

source ~/.bashrc
export VLLM_ATTENTION_BACKEND=XFORMERS

#NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NUM_GPUS=8

train_batch_size=32
enable_gradient_checkpointing=True
ROLLOUT_N=16
SUBTRACT_BASELINE=False  # Set to False to use raw log-likelihood as reward (GRPO normalizes implicitly)
USE_ACTOR_AS_RM=False  # Set to True to use actor model (current weights) instead of frozen RM for reward
REWARD_TYPE="log_prob"  # 'log_prob' or 'neg_perplexity' (only used when USE_LM_REWARD=True)
USE_LM_REWARD=True  # False = rule-based reward (paper's approach), True = LM log-likelihood reward
DATASET_NAME="tom_orig_3k"  # Short name for the training dataset (for experiment tracking)
EXP_DESC="ll-frozenRM-lowKL"  # Meaningful description of what we're testing (appended to experiment name)

#model_names=("Qwen/Qwen2.5-7B-Instruct-1M" "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-0.5B-Instruct")
model_names=("Qwen/Qwen2.5-3B-Instruct")
# lrs=(4e-7 5e-7 3e-7 5e-6)
lrs=(5e-7)

num_epochs=2

for model_name in ${model_names[@]}
do
    for lr in ${lrs[@]}
    do
        data_train_files=$HOME/repo/BeRL/data/tom_train.parquet
        test_files=$HOME/repo/BeRL/data/cleaned_tom/ToM_test_HiExTi_hint_v2.parquet

        # Build descriptive experiment name
        RM_TYPE=$( [ "$USE_ACTOR_AS_RM" = "True" ] && echo "actorRM" || echo "frozenRM" )
        BASELINE_TAG=$( [ "$SUBTRACT_BASELINE" = "True" ] && echo "baseline" || echo "nobaseline" )
        EXP_NAME="${DATASET_NAME}-$(basename $model_name)-${RM_TYPE}-${BASELINE_TAG}-lr${lr}-n${ROLLOUT_N}${EXP_DESC:+-$EXP_DESC}"

        HYDRA_FULL_ERROR=1 RAY_BACKEND_LOG_LEVEL=debug python3 -m verl.trainer.main_ppo \
            algorithm.adv_estimator=grpo \
            data.train_files=$data_train_files \
            data.val_files=$test_files \
            data.train_batch_size=$train_batch_size \
	        data.prompt_is_text=False \
            data.val_batch_size=16 \
            data.max_prompt_length=1024 \
            data.max_response_length=2048 \
            reward_model.type="lm" \
            reward_model.enable=$USE_LM_REWARD \
            reward_model.model.path=$model_name \
            reward_model.micro_batch_size=8 \
            +reward_model.subtract_baseline=$SUBTRACT_BASELINE \
            +reward_model.use_actor_as_rm=$USE_ACTOR_AS_RM \
            +reward_model.reward_type=$REWARD_TYPE \
            actor_rollout_ref.model.path=$model_name \
            actor_rollout_ref.actor.optim.lr=$lr \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.ppo_mini_batch_size=128 \
            actor_rollout_ref.actor.ppo_micro_batch_size=8 \
            actor_rollout_ref.actor.use_kl_loss=True \
            actor_rollout_ref.actor.kl_loss_coef=0.001 \
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
            actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
            actor_rollout_ref.rollout.n=$ROLLOUT_N \
            actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            algorithm.kl_ctrl.kl_coef=0.001 \
            trainer.critic_warmup=0 \
            trainer.logger=['console','wandb'] \
            trainer.project_name="EmpathicDialogue_GRPO" \
            trainer.experiment_name="$EXP_NAME" \
            trainer.n_gpus_per_node=$NUM_GPUS \
            trainer.nnodes=1 \
            trainer.default_hdfs_dir=null \
            trainer.save_freq=30 \
            trainer.test_freq=10 \
            trainer.total_epochs=$num_epochs $@ 2>&1 | tee logs/${TODAY}/${EXP_NAME}.log
    done
done
