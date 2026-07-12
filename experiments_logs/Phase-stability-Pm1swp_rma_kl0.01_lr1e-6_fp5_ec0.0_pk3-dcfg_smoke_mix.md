### Attempt r1 — 2026-07-12T01:52:46+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1`
- **Host:** h100-021-003   **git:** `d6478ce`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=3 ll_min=-6.0 rm_mode=actor baseline=False kl=0.01 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1.log`

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
    algorithm.kl_ctrl.kl_coef=0.01 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1 \
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
    +reward_model.power_k=3 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=3 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp5_ec0.0_pk3 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


---

### Findings — r1 (authoritative) — completed 2026-07-12 ~03:17 UTC

- **Exp #/ID:** PS055 / `Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp5_ec0.0_pk3` · **WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/afzxg4p5 (all 191 steps + step-190 final eval; state=crashed = unclean teardown only) · **Owner_host:** h100-021-003
- **Attempts:** r1 only (authoritative). Full horizon steps 0→190, 19 logged evals + step-190 final validation.
- **Hypothesis:** With actor-as-RM at KL=0.01, does raising LR to **1e-6** and adding **format penalty fp=5** climb HM-over-last-X and stay stable? (vs PS049 = actor/kl0.01/lr5e-7/fp0.)

**HM(subsample300) trajectory** (HM_tom excludes gsm8k/mmlu; HM_all includes them):

| step | HM_tom | HM_all |
|---|---|---|
| 0 (baseline) | 0.415 | 0.423 |
| 20 (peak) | 0.465 | 0.474 |
| 90 (trough) | 0.406 | 0.391 |
| 140 | 0.427 | 0.414 |
| 180 | 0.437 | 0.407 |

- **Config-selection score:** HM_tom over last-3 = 0.434, last-5 = 0.433 — marginally above baseline 0.415, **but this is misleading**: HM_all sags to ~0.40 (below baseline 0.423) because general-capability benchmarks collapse.
- **End-state (step 190) vs baseline:** tomi 0.587→0.623 (+3.6pp), tombench 0.607→0.713 (+10.6pp), explore_tom 0.480→0.760 (+28pp), bigtom_fwd_belief 0.760→0.720 — several ToM UP; **BUT gsm8k 0.660→0.137 (−52.3pp, catastrophic)**, **simpletom_mental 0.857→0.397 (−46pp)**, fantom_belief_mc 0.507→0.413.

**Health / hacking — RED FLAG (entropy collapse):**
- `actor/entropy_loss` **collapses 1.07 → 0.003 → 0.138** — the policy became near-deterministic (opposite of PS033's entropy *inflation*).
- `actor/kl_loss` **spikes to ~1.48 (mid), 1.25 (end)** — massive divergence from the reference (KL=0.01 far too weak to hold lr=1e-6).
- `response_length/mean` shrinks 43 → 32 → 46; `reward/format_error_ratio = 0.000` throughout (fp=5 kept format valid but did **not** prevent the collapse).

**Verdict:** **PARTIAL COLLAPSE — not a Phase-1 candidate.** lr=1e-6 over-optimizes at KL=0.01: entropy collapses and KL blows up, the policy reward-hacks a subset of ToM formats (tomi/explore_tom/tombench) while catastrophically sacrificing general capability (gsm8k 0.137) and some ToM (simpletom_mental 0.397). The HM_tom-over-last-X selection metric is *fooled* here — HM_all exposes the collapse. **Lesson: keep lr=5e-7 (lr=1e-6 destabilizes); fp=5 alone does not rescue it.** Contrast: PS049 (actor/kl0.01/**lr5e-7**/fp0) held ≈baseline with gsm8k preserved (0.637). Skip in Wave-2.

**How to rerun:** `ONLY_IDX="55" bash experiments/phase_stability_sweep.sh` (preview with `BERL_DRY_RUN=1`).
