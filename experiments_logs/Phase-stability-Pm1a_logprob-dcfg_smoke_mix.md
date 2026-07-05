### Attempt r1 — 2026-07-05T20:08:04+00:00

- **RUN_NAME:** `Phase-stability-Pm1a_logprob-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.01-n16-r1`
- **Host:** slurm-login-0   **git:** `f626df8`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_test_HiExTi_hint_v3.parquet,/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/fantom_test_50pct.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=frozen baseline=False kl=0.01 lr=1e-6 rollout_n=16 epochs=2 max_ctx=2048/512 cot_var=COT_FREEFORM require_answer_tags=True
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=EmpathicDialogue_GRPO run=Phase-stability-Pm1a_logprob-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260705/Phase-stability-Pm1a_logprob-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.01-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_test_HiExTi_hint_v3.parquet,/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/fantom_test_50pct.parquet] \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
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
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.01 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=EmpathicDialogue_GRPO \
    trainer.experiment_name=Phase-stability-Pm1a_logprob-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.01-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=2 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-3B-Instruct \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=log_prob \
    +reward_model.power_k=2.0 \
    +reward_model.power_ll_min=-8.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1a_logprob DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-05T20:10:52+00:00

- **RUN_NAME:** `Phase-stability-Pm1a_logprob-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.01-n16-r1`
- **Host:** h100-027-004   **git:** `f626df8`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_test_HiExTi_hint_v3.parquet,/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/fantom_test_50pct.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=frozen baseline=False kl=0.01 lr=1e-6 rollout_n=16 epochs=2 max_ctx=2048/512 cot_var=COT_FREEFORM require_answer_tags=True
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=EmpathicDialogue_GRPO run=Phase-stability-Pm1a_logprob-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260705/Phase-stability-Pm1a_logprob-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.01-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_test_HiExTi_hint_v3.parquet,/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/fantom_test_50pct.parquet] \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
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
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.01 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=EmpathicDialogue_GRPO \
    trainer.experiment_name=Phase-stability-Pm1a_logprob-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.01-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=2 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-3B-Instruct \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=log_prob \
    +reward_model.power_k=2.0 \
    +reward_model.power_ll_min=-8.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1a_logprob DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_

