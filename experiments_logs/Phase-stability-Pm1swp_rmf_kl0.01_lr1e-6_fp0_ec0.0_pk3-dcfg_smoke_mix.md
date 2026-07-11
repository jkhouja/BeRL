### Attempt r1 — 2026-07-11T21:31:15+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp0_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1`
- **Host:** h100-007-002   **git:** `5cd2331`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=3 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.01 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp0_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp0_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1.log`

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
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.think_only_pg=True \
    actor_rollout_ref.actor.format_penalty=0 \
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
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp0_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=999 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-3B-Instruct \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=3 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=3 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp0_ec0.0_pk3 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:**

#### Findings (completed 2026-07-11, exit 0, 191 steps / 1 epoch)

**Verdict: UNSTABLE / negative transfer — power reward does NOT rescue KL=0.01. Same drift signature as log_prob at kl0.01.**

Config-selection score (HM of 26 subsample300 benchmarks):
- **HM(last-3) = 0.367**; HM(last-5) = 0.317 — **below** step-0 baseline HM = 0.422.
- HM peaks 0.469 @step10 then **declines** with a sharp dip to **0.183 @step150**, partial recovery to 0.322 @step190.
- Parseable-answer rate ≈ **100%** (`format_error_ratio`=0) — degradation is drift, not format collapse.

Eval HM trajectory:
`0:0.422 10:0.469(peak) 20:0.460 30:0.459 40:0.465 50:0.430 60:0.433 70:0.409 80:0.410 90:0.404 100:0.398 110:0.350 120:0.384 130:0.378 140:0.282 150:0.183(trough) 160:0.270 170:0.401 180:0.369 190:0.322`

Health: **reward climbs +to +38** (reward maximized) while ToM declines — classic negative transfer; **KL diverges 0.34→0.38**; entropy climbs to ~3.1; resp_len grows to ~185. All the hallmarks of KL=0.01 under-regularization.

**Key comparisons (HM last-3):**
- vs frozen log_prob kl0.01/lr1e-6/fp5 (PS007 = 0.169): power k=3 is **less catastrophic** (0.367 vs 0.169) but still below-baseline and declining. Reward type modulates severity but does not fix KL=0.01.
- vs frozen power @ kl0.05 (pending PS0xx): KL=0.05 expected to stabilize power too, mirroring log_prob (PS010/12/15 ≈ 0.43).

**Implication:** confirms **KL=0.01 is the instability driver across BOTH reward families** (log_prob and power). Power reward is not a stability substitute for proper KL regularization. Expect the power-block stable cells to be the KL=0.05 ones.

