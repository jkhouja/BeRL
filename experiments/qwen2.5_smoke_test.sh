#!/bin/bash
# Quick smoke test for Qwen2.5-3B — 1 epoch, test_freq=5, save_freq=999
set -x

REPO_DIR=$HOME/repo/BeRL
TODAY=$(date +%Y%m%d)
mkdir -p $REPO_DIR/logs/${TODAY}

# Unset env var so wandb falls back to ~/.netrc credentials
unset WANDB_API_KEY

eval "$($HOME/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate tom

export VLLM_ATTENTION_BACKEND=XFORMERS

NUM_GPUS=8
model_name="Qwen/Qwen2.5-3B-Instruct"
lr=5e-7
train_batch_size=32
ROLLOUT_N=16
RUN_INDEX="${RUN_INDEX:-1}"
EXP_NAME="test-qwen2.5-3B-smoke-parser-refactor-r${RUN_INDEX}"

data_train_files=$REPO_DIR/data/merged_dialogue_cga_eval_prompt.parquet
test_files="[$REPO_DIR/data/cleaned_tom/ToM_test_HiExTi_hint_v3.parquet,$REPO_DIR/data/cleaned_tom/fantom_test_50pct.parquet]"

cd $REPO_DIR
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_train_files \
    data.val_files=$test_files \
    data.train_batch_size=$train_batch_size \
    data.prompt_is_text=False \
    data.val_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    reward_model.type="lm" \
    reward_model.enable=True \
    reward_model.model.path=$model_name \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=power \
    +reward_model.power_k=2.0 \
    +reward_model.power_ll_min=-8.0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
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
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="EmpathicDialogue_GRPO" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=999 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 $@ 2>&1 | tee $REPO_DIR/logs/${TODAY}/${EXP_NAME}.log
