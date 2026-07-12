### Attempt r1 — 2026-07-11T22:10:25+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-021-003   **git:** `92e1d06`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=3 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.05-n16-r1.log`

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
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.05-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk3 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


---

### Findings — r1 (authoritative) — completed 2026-07-11 ~23:50 UTC

- **Exp #/ID:** PS041 / `Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk3` · **WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/5wnxw362 (state=crashed = unclean teardown only; all 191 steps + step-190 final eval completed) · **Owner_host:** h100-021-003
- **Attempts:** r1 only (authoritative). Full horizon steps 0→190, 19 logged evals (TEST_FREQ=10) + step-190 final validation.
- **Hypothesis:** Does power(k=3, ll_min=−6) behavior reward at **KL=0.05**/lr=5e-7 (frozen RM, fp=0, ec=0.0) climb HM-over-last-X above baseline without collapse — i.e. does KL=0.05 stabilize the drift PS033 (same cell, KL=0.01) showed?

**HM(subsample300) trajectory** (HM_tom excludes gsm8k/mmlu):

| step | HM_tom | HM_all |
|---|---|---|
| 0 (baseline) | 0.418 | 0.425 |
| 30 (peak) | 0.463 | 0.473 |
| 90 | 0.447 | 0.456 |
| 120 | 0.435 | 0.442 |
| 180 | 0.444 | 0.449 |

- **Config-selection score:** HM_tom over **last-3 = 0.443**, **last-5 = 0.442** — both **above** step-0 baseline 0.418 (+2.4pp). Climbs to peak 0.463@step30, then holds a healthy 0.44–0.46 plateau to the end (no drift-down below baseline).
- **End-state (step 190) vs baseline:** tomi 0.587→0.577 (≈flat), bigtom_fwd_belief 0.760→0.710 (−5pp), tombench 0.607→0.697 (+9pp), simpletom_mental 0.857→0.893 (+3.6pp), fantom_belief_mc 0.507→0.510 (flat), gsm8k 0.660→0.457 (−20.3pp, general-cap drift). ToM largely preserved/up; only gsm8k regresses.

**Health / hacking:**
- `reward/format_error_ratio = 0.000` for the entire run — 100% parseable, no format collapse.
- Reward cleanly optimized: `critic/rewards/mean` −35.8 → +34.
- **KL contained:** `actor/kl_loss` 0.007 → 0.175 (vs PS033 kl=0.01 → 0.38); **entropy** `actor/entropy_loss` 1.04 → 2.24 (vs PS033 → 3.27); `response_length/mean` 33 → 113 (vs PS033 → 143). KL=0.05 clearly damps the drift.

**Verdict:** **STABLE, above-baseline — a plausible Phase-1 stable-config candidate.** Direct contrast with PS033 (identical cell at KL=0.01: last-5 HM_tom=0.406 below baseline, KL drifts to 0.38) confirms **KL=0.05 fixes the KL=0.01 drift for the power-k3 reward**, mirroring the log_prob finding (PS009/PS010: KL=0.05 stabilizes). Keep for Wave-2.

**How to rerun:** `ONLY_IDX="41" bash experiments/phase_stability_sweep.sh` (preview with `BERL_DRY_RUN=1`).
