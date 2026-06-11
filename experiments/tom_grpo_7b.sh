#!/bin/bash
# 7B version of tom_grpo.sh — rule-based reward on ToM training data

set -x

REPO_DIR=$HOME/repo/BeRL
TODAY=$(date +%Y%m%d)
mkdir -p $REPO_DIR/logs/${TODAY}

source ~/.bashrc

export VLLM_ATTENTION_BACKEND=XFORMERS

NUM_GPUS=8

train_batch_size=32
enable_gradient_checkpointing=True
ROLLOUT_N=16

model_name="Qwen/Qwen2.5-7B-Instruct"
lr=5e-7

num_epochs=2

data_train_files=$HOME/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint.parquet
test_files=$HOME/repo/BeRL/data/cleaned_tom/ToM_test_HiExTi_hint_v3.parquet

EXP_NAME="round18-tom3k-rulebased-v3eval-$(basename $model_name)-$lr-$ROLLOUT_N"

cd $REPO_DIR
HYDRA_FULL_ERROR=1 RAY_BACKEND_LOG_LEVEL=debug python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_train_files \
    data.val_files=$test_files \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
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
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=$num_epochs $@ 2>&1 | tee $REPO_DIR/logs/${TODAY}/${EXP_NAME}.log
