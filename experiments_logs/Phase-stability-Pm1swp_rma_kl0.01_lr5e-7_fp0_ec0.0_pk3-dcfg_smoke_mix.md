### Attempt r1 — 2026-07-12T00:15:25+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp0_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1`
- **Host:** h100-021-003   **git:** `f65c24a`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=3 ll_min=-6.0 rm_mode=actor baseline=False kl=0.01 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp0_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp0_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp0_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1 \
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
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=3 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp0_ec0.0_pk3 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


---

### Findings — r1 (authoritative) — completed 2026-07-12 ~01:50 UTC

- **Exp #/ID:** PS049 / `Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp0_ec0.0_pk3` · **WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/e9d7b48s (all 191 steps + step-190 final eval completed) · **Owner_host:** h100-021-003
- **Attempts:** r1 only (authoritative). Full horizon steps 0→190, 19 logged evals + step-190 final validation.
- **Hypothesis:** Does power(k=3, ll_min=−6) at KL=0.01/lr=5e-7 with **actor-as-RM** (fp=0, ec=0.0) climb HM-over-last-X above baseline without collapse — and is actor-as-RM more robust to the KL=0.01 drift than frozen RM (PS033)?

**HM_tom(subsample300) trajectory** (excludes gsm8k/mmlu):

| step | HM_tom |
|---|---|
| 0 (baseline) | 0.416 |
| 40 (peak) | 0.463 |
| 90 | 0.452 |
| 120 | 0.431 |
| 180 | 0.431 |

- **Config-selection score:** HM_tom over **last-3 = 0.433**, **last-5 = 0.434** — both marginally **above** step-0 baseline 0.416 (+1.8pp). Climbs to a 0.45–0.46 plateau (steps 30–90) then gently declines, settling just above baseline (no collapse below baseline).
- **End-state (step 190) vs baseline:** tomi 0.587→0.590 (flat), gsm8k 0.660→0.637 (−2.3pp, well-preserved), bigtom_fwd_belief 0.760→0.677, simpletom_mental 0.857→0.923 (+6.6pp), tombench 0.607→0.677 (+7pp), fantom_belief_mc 0.507→0.493. ToM preserved/up and general-cap barely dented.

**Health / hacking:**
- `reward/format_error_ratio = 0.000` throughout — 100% parseable, no format collapse.
- Reward optimized: `critic/rewards/mean` −20.4 → +28.5.
- **KL drifts** `actor/kl_loss` 0.004 → 0.404 (comparable to PS033's ~0.38); **entropy** 1.1 → 2.96; `response_length/mean` 50 → 110. Despite similar KL drift to PS033, the policy does **not** degrade its benchmarks — the actor-as-RM reward tracks the moving policy and avoids the reward-model-drift negative transfer that hit frozen-RM.

**Verdict:** **MARGINAL-STABLE — holds ≈baseline; a weak Phase-1 candidate.** Key finding: **actor-as-RM is markedly more robust to the KL=0.01 low-KL drift than frozen-RM.** Direct contrast — same knobs, RM only:
- PS033 (frozen, kl=0.01): last-5 HM_tom **0.406 (below baseline)**, gsm8k crashes to 0.483.
- PS049 (actor, kl=0.01): last-5 HM_tom **0.434 (above baseline)**, gsm8k preserved 0.637.
Still below the best frozen cell PS041 (kl=0.05, last-5=0.442). Suggests the strongest cell will combine actor-as-RM *and* KL=0.05 (see PS057). Keep for Wave-2.

**How to rerun:** `ONLY_IDX="49" bash experiments/phase_stability_sweep.sh` (preview with `BERL_DRY_RUN=1`).
