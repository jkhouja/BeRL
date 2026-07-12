### Attempt r1 — 2026-07-11T23:42:39+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1`
- **Host:** h100-077-004   **git:** `c2e63f0`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=3 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1 \
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
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=3 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp5_ec0.0_pk3 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


---

### Findings (attempt r1) — completed 2026-07-12T01:24 (h100-077-004)

**Hypothesis:** Does frozen-RM power-reward (k=3, ll_min=−6) at the stable KL kl=0.05, lr=1e-6, fp=5,
ec=0.0 climb stably (HM over last-X ≥ baseline)? Direct kl=0.05 counterpart of PS039 (kl=0.01) —
isolates the KL effect for the power reward family.

**Verdict: STRONG STABLE — best-tier Phase-1 candidate.** Completed full 191-step epoch. HM_tom
holds flat/high the entire run with no late collapse (contrast PS039 which cratered).

**Config-selection metric (HM over ToM subtypes, excl. mmlu/gsm8k):**
- Baseline (step 0) HM_tom = **0.418**
- HM(last-3 evals) = **0.446**, HM(last-5 evals) = **0.446** → **+2.8pp vs baseline**
- Peak HM_tom = **0.460 @ step 90**
- Trajectory flat 0.43–0.46 throughout (0.453@10 … 0.460@90 … 0.440@190) — no drift-down.

**Per-benchmark (step0 → step190):**
| Benchmark | step0 | step190 | note |
|---|---|---|---|
| tomi | 0.590 | 0.610 | held/up |
| bigtom_forward_belief | 0.767 | 0.697 | mild dip |
| simpletom_mental | 0.853 | 0.880 | held (peak 0.920@120) |
| gsm8k (gen-cap) | 0.653 | 0.577 | mild drift, recovers from 0.480@90 |
| mmlu (gen-cap) | 0.473 | 0.607 | up |

**Health/hacking:** No collapse or hacking — `format_error_ratio=0` throughout, KL contained
~0.18–0.23 (kl=0.05), entropy stable ~2.2 (vs PS039's ~3.0 runaway), response_length steady ~120
(vs PS039 drifting to 168). Clean, well-behaved optimization.

**Downstream implication (KEY comparison):**
- **KL effect (PS047 kl=0.05 vs PS039 kl=0.01, same power-k3/lr1e-6/fp5):** HM(last-5) 0.446 vs
  0.309 — kl=0.05 is *necessary* for stability; kl=0.01 drifts to negative transfer regardless of
  reward family.
- **Reward-family effect at stable KL:** PS047 power-k3 HM(last-5)=0.446 edges out the log_prob
  stable cells (PS009=0.425, PS010=0.433, PS032=0.423) → power reward at kl=0.05 is at least as good,
  slightly better, and importantly *climbs above baseline* while log_prob only holds ~baseline.
- ⇒ **power-k3 @ kl=0.05 / lr=1e-6 / fp=5 / ec=0.0 is a leading `stable`-config candidate** for Phase 0.

**How to rerun:** `ONLY_IDX="47" bash experiments/phase_stability_sweep.sh` (from ~/repo/BeRL, conda env `tom`).
