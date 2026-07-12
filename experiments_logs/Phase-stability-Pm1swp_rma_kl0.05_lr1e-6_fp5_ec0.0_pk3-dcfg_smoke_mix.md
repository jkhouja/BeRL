### Attempt r1 — 2026-07-12T03:28:17+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1`
- **Host:** h100-021-003   **git:** `815582e`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=3 ll_min=-6.0 rm_mode=actor baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1.log`

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
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
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
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp5_ec0.0_pk3 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


---

### Findings — r1 (authoritative) — completed 2026-07-12 ~05:01 UTC

- **Exp #/ID:** PS063 / `Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp5_ec0.0_pk3` · **WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/8rke9svg (all 191 steps + step-190 final eval) · **Owner_host:** h100-021-003
- **Attempts:** r1 only (authoritative). Full horizon steps 0→190, 19 logged evals + step-190 final validation.
- **Hypothesis:** With actor-as-RM/lr=1e-6/fp=5, does **KL=0.05** (vs PS055's KL=0.01, which partially collapsed) restore stability and climb HM-over-last-X above baseline?

**HM(subsample300) trajectory:**

| step | HM_tom | HM_all |
|---|---|---|
| 0 (baseline) | 0.415 | 0.423 |
| 10 (peak) | 0.463 | 0.474 |
| 90 | 0.446 | 0.452 |
| 150 | 0.461 | 0.463 |
| 180 | 0.449 | 0.452 |

- **Config-selection score:** HM_tom last-3 = 0.447, last-5 = **0.450** (best of all my power-k3 cells; +3.5pp over baseline 0.415). HM_all last-5 = 0.456 (> baseline 0.423) — **no general-cap collapse**. Holds a healthy 0.44–0.46 plateau the entire run.
- **End-state (step 190) vs baseline:** tomi 0.587→0.560, gsm8k 0.660→0.443 (reduced but **preserved**, vs PS055's 0.137), simpletom_mental 0.857→0.897 (+4pp), tombench 0.607→0.700 (+9.3pp), bigtom_fwd_belief 0.760→0.687, fantom_belief_mc 0.507→0.423.

**Health:**
- `reward/format_error_ratio = 0.000` throughout; reward `critic/rewards/mean` −21.2 → +26.
- **KL contained:** `actor/kl_loss` 0.002 → 0.19 (vs PS055's spike to ~1.5); **entropy healthy** `actor/entropy_loss` 1.0 → 2.15 (NOT collapsed, vs PS055's 0.003); `response_length/mean` 58 → 129.

**Verdict:** **STRONG STABLE — best power-k3 cell so far and a leading Phase-1 candidate.** Decisive contrast with PS055 (identical actor/lr1e-6/fp5, KL only): **KL=0.05 fully rescues the lr=1e-6 destabilization** that entropy-collapsed PS055 (entropy 0.003, gsm8k 0.137) — here entropy stays 2.15, KL 0.19, gsm8k 0.443, and HM_tom last-5 hits 0.450.

**Cross-cell ranking (power-k3, last-5 HM_tom):** PS063 (actor/kl0.05/lr1e-6/fp5) **0.450** > PS041 (frozen/kl0.05/lr5e-7/fp0) 0.442 > PS049 (actor/kl0.01/lr5e-7/fp0) 0.434 > PS055 (actor/kl0.01/lr1e-6/fp5) 0.433 *(collapse — HM_tom fooled)* > PS033 (frozen/kl0.01/lr5e-7/fp0) 0.406. **Dominant lesson: KL=0.05 is essential for stability on Qwen2.5-3B; lr=1e-6 is safe only at KL=0.05, not KL=0.01.**

**How to rerun:** `ONLY_IDX="63" bash experiments/phase_stability_sweep.sh` (preview with `BERL_DRY_RUN=1`).
