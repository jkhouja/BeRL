### Attempt r1 — 2026-07-11T21:47:14+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1`
- **Host:** h100-077-004   **git:** `92e1d06`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=3 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.01 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp5_ec0.0_pk3 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


---

### Findings (attempt r1) — completed 2026-07-11T23:39 (h100-077-004)

**Hypothesis:** Does frozen-RM power-reward (k=3, ll_min=−6) at the weak KL kl=0.01, lr=1e-6, fp=5,
ec=0.0 climb stably (HM over last-X ≥ baseline) without drift/negative transfer?

**Verdict: BELOW-BASELINE — negative transfer (kl=0.01 drift pattern).** Completed full 191-step
epoch. No *format* collapse (`format_error_ratio=0` throughout, evals stay non-zero), but ToM +
general-cap quality degrades badly after an early peak.

**Config-selection metric (HM over ToM subtypes, excl. mmlu/gsm8k):**
- Baseline (step 0) HM_tom = **0.418**
- Peak HM_tom = **0.462 @ step 10**
- HM(last-3 evals) = **0.312**, HM(last-5 evals) = **0.309** → **−10.9pp vs baseline**
- Trajectory oscillates down: 0.462@10 → 0.317@70 → 0.410@140 → 0.266@160 → 0.310@190.

**Per-benchmark (step0 → step190):**
| Benchmark | step0 | peak | step190 |
|---|---|---|---|
| tomi | 0.593 | 0.640@10 | 0.450 |
| bigtom_forward_belief | 0.770 | 0.777@10 | 0.593 |
| simpletom_mental | 0.850 | 0.943@140 | 0.743 |
| hi_tom | 0.250 | 0.370@10 | 0.200 |
| gsm8k (gen-cap) | 0.653 | 0.723@20 | 0.320 |
| mmlu (gen-cap) | 0.480 | 0.613@140 | 0.550 |

**Health/hacking:** `format_error_ratio=0` (no degenerate-format hacking), but KL is under-penalized
at kl=0.01 → entropy rises to ~3.0 and response_length drifts 122→168; policy drifts off-distribution
and transfers negatively to both ToM benchmarks and gsm8k. Same failure mode as the log_prob kl=0.01
cells (PS001/PS003/PS007).

**Downstream implication:** power-k3 reward does **not** rescue the weak-KL (kl=0.01) regime — the
early climb (peak@step10) is not retained. Reinforces that **kl≥0.05 is required** for stable
behavior-RL on Qwen2.5-3B regardless of reward family (log_prob or power). NOT a Phase-1 candidate.
Compare against the kl=0.05 power cells (PS047/PS048…) to isolate the reward-family effect at stable KL.

**How to rerun:** `ONLY_IDX="39" bash experiments/phase_stability_sweep.sh` (from ~/repo/BeRL, conda env `tom`).
