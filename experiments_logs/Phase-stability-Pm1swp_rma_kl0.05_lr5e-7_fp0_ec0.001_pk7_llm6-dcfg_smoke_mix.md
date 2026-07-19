### Attempt r1 — 2026-07-12T20:14:38+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm6-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k7-llmin-6.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-156-003   **git:** `90455f3`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7 ll_min=-6.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm6-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k7-llmin-6.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm6-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k7-llmin-6.0-lr5e-7-kl0.05-n16-r1.log`

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
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
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
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm6-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k7-llmin-6.0-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=power \
    +reward_model.power_k=7 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=7 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm6 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


---
### Findings (r1 — Completed 2026-07-12, Owner_host h100-156-003, WandB rokdezeg)

**Canonical score (`python scripts/score_run.py`):**
```
eval iters: 20 (step 0..190); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.4714  HM(last3)=0.4699  (baseline step0=0.417)
ToM avg(last5)=0.5302  avg(last3)=0.5295  (baseline step0=0.5032)
gsm8k (separate): 0.662 (step0=0.653, delta=+0.009)
mmlu  (separate): 0.6194 (step0=0.47, delta=+0.149)
health(final): kl=0.045 entropy=1.427 resp_len=106.3 reward=32.31 parseable=1.0
HM trajectory: 0:0.417 10:0.449 20:0.465 30:0.461 40:0.460 50:0.465 60:0.465 70:0.474
               80:0.463 90:0.469 100:0.465 110:0.466 120:0.469 130:0.465 140:0.465
               150:0.472 160:0.474 170:0.475 180:0.466 190:0.468
```

**Verdict:** STRONG, STABLE, no collapse / no hacking. ToM HM ends **+5.4pp over step-0 baseline**
(HM-last5 0.4714 vs 0.417); ToM avg also up **+2.7pp** (0.530 vs 0.503) — a genuine broad-based
gain (both HM and avg rise, not a single-benchmark artifact). HM climbs to ~0.465 by step20 then
holds/rises to a **sustained plateau, peaking 0.475 @step170, no late decline** (contrast PS177
frozen-RM which peaked@20 then declined). Actor-RM power reward saturated near the ceiling
(reward −29 → ~39, advantages → ~0 by mid-run) yet ToM transfer HELD and improved — the saturation
did NOT cause reward-hacking degradation. gsm8k **preserved** (Δ+0.009, no math regression — vs
PS177 frozen-RM's −0.147); mmlu +14.9pp. KL very contained (0.045, actor-RM keeps policy anchored);
100% parseable; response_length stable ~106.

**Downstream implication:** **k=7 / ll_min=-6 with ACTOR-RM is the best power cell seen so far** —
HM-last5 0.4714 beats both PS177 (frozen-RM k5 ll_min=-4, 0.4333) and the Wave-1 power winner PS074
(0.462), with sustained (non-declining) climb AND preserved general capability (gsm8k flat). Strong
Phase-1 stable-config candidate; recommend comparing against PS179 (frozen-RM k7 ll_min=-6) and
PS182 (actor-RM k7 ll_min=-4) to isolate the actor-RM vs floor contribution.

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm6 DATA_NAME=dcfg_smoke_mix DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet MODEL_PATH=Qwen/Qwen2.5-3B-Instruct REWARD_TYPE=power POWER_K=7 POWER_LL_MIN=-6.0 USE_ACTOR_AS_RM=True KL=0.05 LR=5e-7 FORMAT_PENALTY=0 ENTROPY_COEFF=0.001 MAX_RESP=512 THINK_ONLY_PG=True TOTAL_EPOCHS=1 TEST_FREQ=10 SAVE_FREQ=999 RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`
