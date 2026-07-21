### Attempt r1 — 2026-07-21T05:46:49+00:00

- **RUN_NAME:** `s2-SS-02-SS-0.5B_frozen_kl0.1-dcfg_smoke_mix-Qwen2.5-0.5B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.1-n16-r1`
- **Host:** h100-156-003   **git:** `6be5ff3`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` (Qwen2.5-0.5B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=4 ll_min=-6 rm_mode=frozen baseline=False kl=0.1 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/4096 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=0.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-SS-02-SS-0.5B_frozen_kl0.1-dcfg_smoke_mix-Qwen2.5-0.5B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.1-n16-r1 — https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/wh3jynnk
- **Log path:** `logs/20260721/s2-SS-02-SS-0.5B_frozen_kl0.1-dcfg_smoke_mix-Qwen2.5-0.5B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.1-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.1 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
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
    algorithm.kl_ctrl.kl_coef=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-SS-02-SS-0.5B_frozen_kl0.1-dcfg_smoke_mix-Qwen2.5-0.5B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.1-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=4 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=0.0 \
    reward_model.format_penalty_std_coef=0.0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=4 \
    +actor_rollout_ref.power_ll_min=-6 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=SS-0.5B_frozen_kl0.1 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r2 — 2026-07-21T05:47:05+00:00

- **RUN_NAME:** `s2-SS-02-SS-0.5B_frozen_kl0.1-dcfg_smoke_mix-Qwen2.5-0.5B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.1-n16-r2`
- **Host:** h100-156-003   **git:** `6be5ff3`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` (Qwen2.5-0.5B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=4 ll_min=-6 rm_mode=frozen baseline=False kl=0.1 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/4096 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=0.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-SS-02-SS-0.5B_frozen_kl0.1-dcfg_smoke_mix-Qwen2.5-0.5B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.1-n16-r2 — https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/wh3jynnk
- **Log path:** `logs/20260721/s2-SS-02-SS-0.5B_frozen_kl0.1-dcfg_smoke_mix-Qwen2.5-0.5B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.1-n16-r2.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.1 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
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
    algorithm.kl_ctrl.kl_coef=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-SS-02-SS-0.5B_frozen_kl0.1-dcfg_smoke_mix-Qwen2.5-0.5B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.1-n16-r2 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=4 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=0.0 \
    reward_model.format_penalty_std_coef=0.0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=4 \
    +actor_rollout_ref.power_ll_min=-6 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=SS-0.5B_frozen_kl0.1 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=2 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (r1)

```
eval iters: 8 (step 0..190); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.0146  HM(last3)=0.0121  (baseline step0=0.0032)
ToM avg(last5)=0.1646  avg(last3)=0.2078  (baseline step0=0.0553)
gsm8k (separate): 0.1274 (step0=0.0, delta vs step0=+0.127)
mmlu (separate): 0.2306 (step0=0.03, delta vs step0=+0.201)
health(final): kl=0.141 entropy=2.016 resp_len=96.711 reward=33.78 parseable=1.0 max_resp=4096
ToM HM trajectory: 0:0.003 30:0.003 60:0.004 90:0.004 120:0.016 150:0.014 180:0.009 190:0.009
```

**Verdict:** Completed, 190/190. d_avg=+0.1093 (avg the meaningful signal at 0.5B; HM near-floor since many ToM benches ~0). gsm8k/mmlu both improve (format learning). **KL_max=2.364** observed mid-run (~step 60–70), intermittent self-recovering spikes back to ~0.002–0.07 (final kl=0.141) — NOT a monotonic runaway like Gemma's 2.6, but **exceeds the SS sweep <1.0 target ⇒ frozen RM + kl0.1 does NOT fully stabilize Qwen2.5-0.5B**. Parseable 1.0, no collapse. Note: max_resp=4096 (launcher default, not the 512 Qwen behavior cap).
