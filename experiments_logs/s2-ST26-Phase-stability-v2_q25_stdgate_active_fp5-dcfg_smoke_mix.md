### Attempt r1 — 2026-07-20T00:07:44+00:00

- **RUN_NAME:** `s2-ST26-Phase-stability-v2_q25_stdgate_active_fp5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-021-003   **git:** `e841f0e`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=4 ll_min=-6 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=5 format_penalty_std_coef=1.5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-ST26-Phase-stability-v2_q25_stdgate_active_fp5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260720/s2-ST26-Phase-stability-v2_q25_stdgate_active_fp5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/dcfg_smoke_mix.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.think_only_pg=False \
    actor_rollout_ref.actor.format_penalty=5 \
    actor_rollout_ref.actor.format_penalty_std_coef=1.5 \
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
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-ST26-Phase-stability-v2_q25_stdgate_active_fp5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-3B-Instruct \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=power \
    +reward_model.power_k=4 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=5 \
    reward_model.format_penalty_std_coef=1.5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=4 \
    +actor_rollout_ref.power_ll_min=-6 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-v2_q25_stdgate_active_fp5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-20T00:08:00+00:00

- **RUN_NAME:** `s2-ST26-Phase-stability-v2_q25_stdgate_active_fp5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-021-003   **git:** `e841f0e`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=4 ll_min=-6 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=5 format_penalty_std_coef=1.5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-ST26-Phase-stability-v2_q25_stdgate_active_fp5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260720/s2-ST26-Phase-stability-v2_q25_stdgate_active_fp5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/dcfg_smoke_mix.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.think_only_pg=False \
    actor_rollout_ref.actor.format_penalty=5 \
    actor_rollout_ref.actor.format_penalty_std_coef=1.5 \
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
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-ST26-Phase-stability-v2_q25_stdgate_active_fp5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-3B-Instruct \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=power \
    +reward_model.power_k=4 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=5 \
    reward_model.format_penalty_std_coef=1.5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=4 \
    +actor_rollout_ref.power_ll_min=-6 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-v2_q25_stdgate_active_fp5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (r1 — authoritative) — appended 2026-07-20 01:15 by agent on h100-021-003
Run **Completed** cleanly: reached step 190/190 (1 epoch confirmed), final subsample300 eval logged, main_ppo proc exited normally.

**Config-selection score** (`scripts/score_run.py`, 8 eval iters step 0..190; 24 ToM benchmarks excl gsm8k/mmlu):
- ToM **HM(last5)=0.4609**, HM(last3)=0.463 (baseline step0=0.417) → **+0.044 over baseline**.
- ToM avg(last5)=0.5211, avg(last3)=0.5232 (baseline 0.5034).
- gsm8k (separate): 0.6462 (step0 0.657, Δ=-0.011 — held).
- mmlu (separate): 0.6106 (step0 0.47, Δ=+0.141).
- HM trajectory (very stable): 0:0.417 → 30:0.459 → 60:0.470(peak) → 90:0.454 → 120:0.460 → 150:0.460 → 180:0.462 → 190:0.464.

**Health (final):** KL=0.102 (stable ~0.06–0.12 throughout), entropy=1.383, response_length≈109, reward/mean≈31.1 (power 0–40), **parseable=1.0**, **format_error_ratio=0.0 throughout** (max-ever 0.0).

**Verdict (PD — first genuinely-ACTIVE std-gated format penalty):** With the gate genuinely active (fp=5, std_coef=1.5; prior "std-gated" rows had fp=0 → no-op per fsdp_workers.py:108), training is **stable and format stays perfect (1.0)**. ToM HM gain (+0.044) is essentially the same as the fp=0 no-penalty reference ST10 (+0.051) and the stability profile is very flat — because Qwen2.5-3B format compliance was ALREADY 1.0 on this config, the active gate has almost nothing to correct, so it's a **safe no-harm addition** rather than a booster here. Its real test is configs that DO drift format (low-KL/high-LR or Gemma); compare with ST27 (active gate on Gemma). vs ST01 anchor (inactive gate) + ST10 (fp0): the active gate does not degrade the honest config.
