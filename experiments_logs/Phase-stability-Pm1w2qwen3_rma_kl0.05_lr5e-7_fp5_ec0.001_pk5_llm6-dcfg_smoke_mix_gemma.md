### Attempt r1 — 2026-07-13T19:40:00+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-082-004   **git:** `d481190`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/4096 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
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
    actor_rollout_ref.actor.think_only_pg=False \
    actor_rollout_ref.actor.format_penalty=5 \
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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen3-1.7B \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.001_pk5_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-13T19:40:10+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-082-004   **git:** `d481190`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/4096 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
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
    actor_rollout_ref.actor.think_only_pg=False \
    actor_rollout_ref.actor.format_penalty=5 \
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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen3-1.7B \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.001_pk5_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_


#### Findings (r1, 2026-07-14, h100-082-004) — COMPLETED (clean full run)
**Canonical score (`scripts/score_run.py`, 39 eval iters step0..190, 24 ToM benches excl gsm8k/mmlu):**
- ToM **HM(last5)=0.360** HM(last3)=0.361 (baseline step0=0.342) — +1.8pp
- ToM **avg(last5)=0.452** avg(last3)=0.450 (baseline step0=0.442) — **+0.95pp**
- gsm8k (separate): 0.903 (step0=0.890, **Δ=+0.013**, near-ceiling)
- mmlu (separate): 0.597 (step0=0.587, **Δ=+0.010**)
- health(final): kl=0.002 entropy=0.28 resp_len=681 reward=0.41 **parseable=1.0** (no drift, no OOM)

**Per-benchmark step0→final(step190, subsample300):** tomi 0.583→0.627 · simpletom_mental 0.770→0.810 ·
explore_tom 0.517→0.543 · bigtom_fb 0.860→0.853 · hi_tom 0.403→0.390 · opentom_location_fo 0.430→0.447.
All within ±3-4pp (noise).

**Verdict: NEUTRAL / marginal positive.** Power (k=5, ll_min=−6) with **actor-RM** on Qwen3-1.7B gives
only a small lift (avg +0.95pp, HM +1.8pp; HM trajectory oscillates flat ~0.32–0.37 throughout).
No hacking/collapse (parseable=1.0, gsm8k/mmlu flat).

**KEY CONCLUSION — disambiguates PS148.** Paired with the frozen-RM sibling **PS148** (same k/ll_min,
avg +0.2pp), **both** Qwen3 RM modes are essentially flat (actor 0.452 vs frozen 0.444; HM 0.360 vs
0.349 — actor marginally better but both within noise). So for **Qwen3-1.7B the gating factor is
HEADROOM, not RM-mode**: the base is already strong (ToM HM 0.342, avg 0.442, gsm8k 0.89 near-ceiling)
and the power reward has little room to move it regardless of scorer. This is the **opposite** of
Gemma-2-2B, where RM-mode was decisive (actor-RM +10.3pp/PS124 & +6.5pp/PS135 vs frozen-RM
negative/PS107). **Cross-family takeaway:** BeRL power reward transfers strongly on a *weak* base
(Gemma-2, low ToM/math start) and is neutral on a *strong* base (Qwen3-1.7B) — RM-mode only matters
when there is headroom to exploit.

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.001_pk5_llm6 DATA_NAME=dcfg_smoke_mix_gemma DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-6.0 USE_ACTOR_AS_RM=True KL=0.05 LR=5e-7 FORMAT_PENALTY=5 ENTROPY_COEFF=0.001 TEST_FREQ=5 TOTAL_EPOCHS=1 RUN_INDEX=2 bash experiments/train_behavior_qwen3.sh`
