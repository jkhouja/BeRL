### Attempt r1 — 2026-07-21T05:40:04+00:00

- **RUN_NAME:** `s2-Q1-B2-01-Q1-B2-tomrl_3B_s1-ToM_HiEx_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1`
- **Host:** h100-007-002   **git:** `6be5ff3`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=n/a power_k=n/a ll_min=n/a rm_mode=frozen baseline=False kl=0.001 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/2048 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=0.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-Q1-B2-01-Q1-B2-tomrl_3B_s1-ToM_HiEx_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1 — https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/le3elpla
- **Log path:** `logs/20260721/s2-Q1-B2-01-Q1-B2-tomrl_3B_s1-ToM_HiEx_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=8 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.think_only_pg=False \
    actor_rollout_ref.actor.format_penalty=0.0 \
    actor_rollout_ref.actor.format_penalty_std_coef=0.0 \
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
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-Q1-B2-01-Q1-B2-tomrl_3B_s1-ToM_HiEx_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Q1-B2-tomrl_3B_s1 DATA_NAME=ToM_HiEx_hint_v3 MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3.parquet RUN_INDEX=1 bash experiments/train_tom_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-21T07:52:24+00:00

- **RUN_NAME:** `s2-Q1-B2-01-Q1-B2-tomrl_3B_s1-ToM_HiEx_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1`
- **Host:** h100-156-003   **git:** `bb80d13`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=n/a power_k=n/a ll_min=n/a rm_mode=frozen baseline=False kl=0.001 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/2048 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=0.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-Q1-B2-01-Q1-B2-tomrl_3B_s1-ToM_HiEx_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1 — https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/le3elpla
- **Log path:** `logs/20260721/s2-Q1-B2-01-Q1-B2-tomrl_3B_s1-ToM_HiEx_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=8 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.think_only_pg=False \
    actor_rollout_ref.actor.format_penalty=0.0 \
    actor_rollout_ref.actor.format_penalty_std_coef=0.0 \
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
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-Q1-B2-01-Q1-B2-tomrl_3B_s1-ToM_HiEx_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Q1-B2-tomrl_3B_s1 DATA_NAME=ToM_HiEx_hint_v3 MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3.parquet RUN_INDEX=1 bash experiments/train_tom_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-21T07:52:38+00:00

- **RUN_NAME:** `s2-Q1-B2-01-Q1-B2-tomrl_3B_s1-ToM_HiEx_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1`
- **Host:** h100-156-003   **git:** `bb80d13`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=n/a power_k=n/a ll_min=n/a rm_mode=frozen baseline=False kl=0.001 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/2048 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=0.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-Q1-B2-01-Q1-B2-tomrl_3B_s1-ToM_HiEx_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1 — https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/le3elpla
- **Log path:** `logs/20260721/s2-Q1-B2-01-Q1-B2-tomrl_3B_s1-ToM_HiEx_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=8 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.think_only_pg=False \
    actor_rollout_ref.actor.format_penalty=0.0 \
    actor_rollout_ref.actor.format_penalty_std_coef=0.0 \
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
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-Q1-B2-01-Q1-B2-tomrl_3B_s1-ToM_HiEx_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Q1-B2-tomrl_3B_s1 DATA_NAME=ToM_HiEx_hint_v3 MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3.parquet RUN_INDEX=1 bash experiments/train_tom_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-21T07:52:49+00:00

- **RUN_NAME:** `s2-Q1-B2-01-Q1-B2-tomrl_3B_s1-ToM_HiEx_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1`
- **Host:** h100-156-003   **git:** `bb80d13`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=n/a power_k=n/a ll_min=n/a rm_mode=frozen baseline=False kl=0.001 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/2048 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=0.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-Q1-B2-01-Q1-B2-tomrl_3B_s1-ToM_HiEx_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1 — https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/le3elpla
- **Log path:** `logs/20260721/s2-Q1-B2-01-Q1-B2-tomrl_3B_s1-ToM_HiEx_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=8 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.think_only_pg=False \
    actor_rollout_ref.actor.format_penalty=0.0 \
    actor_rollout_ref.actor.format_penalty_std_coef=0.0 \
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
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-Q1-B2-01-Q1-B2-tomrl_3B_s1-ToM_HiEx_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Q1-B2-tomrl_3B_s1 DATA_NAME=ToM_HiEx_hint_v3 MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3.parquet RUN_INDEX=1 bash experiments/train_tom_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (r1)

```
eval iters: 15 (step 0..400); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.4725  HM(last3)=0.4727  (baseline step0=0.418)
ToM avg(last5)=0.5393  avg(last3)=0.5374  (baseline step0=0.5035)
gsm8k (separate): 0.7346 (step0=0.66, delta vs step0=+0.075)
mmlu (separate): 0.5934 (step0=0.49, delta vs step0=+0.103)
health(final): kl=0.148 entropy=0.82 resp_len=129.516 reward=1.219 parseable=1.0 max_resp=2048
ToM HM trajectory: 0:0.418 30:0.396 60:0.439 90:0.447 120:0.442 150:0.437 180:0.425 210:0.433 240:0.449 270:0.454 300:0.478 330:0.464 360:0.465 390:0.48 400:0.472
```

**Verdict:** Completed, 400/400. Label-supervised (rule-based ToM labels) RL headline baseline, seed 1. d_avg=+0.0358 (HM 0.418→0.4725, +5.45pp). Clean monotone-ish HM climb, no collapse; kl stayed <0.16 throughout, parseable 1.0. gsm8k/mmlu both improve (no capability regression). Compares against label-free BeRL B1 anchor (ST01/10/28, d_avg=+0.0215): **label-supervised ToM-RL is stronger at 3B**, as expected — establishes the labeled ceiling for the Q1 label-free-vs-labeled headline. Awaits seeds 2/3 (Q1-B2-02/03) for N=3.
