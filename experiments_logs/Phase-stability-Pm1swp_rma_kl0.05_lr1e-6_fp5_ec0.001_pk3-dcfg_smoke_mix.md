### Attempt r1 — 2026-07-12T03:39:13+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1`
- **Host:** h100-117-001   **git:** `3c2734c`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=3 ll_min=-6.0 rm_mode=actor baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1.log`

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
    actor_rollout_ref.actor.entropy_coeff=0.001 \
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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp5_ec0.001_pk3 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings (r1, completed 2026-07-12):** MODERATE-STRONG / above baseline, sustained, no collapse.

Config-selection metric = harmonic mean over the sub300 benchmarks per eval block; compare vs step-0 baseline.
- Baseline (step 0): HM = **0.4256**
- HM last-3 (170/180/190): **0.4487**
- HM last-5 (150–190): **0.4496**  → **ABOVE baseline** (+0.024)
- Per-step HM: peaks step 10 (0.4777), dips ~step 90 (0.4199), recovers to a stable ~0.45 plateau (150–190). No terminal decline — sustained, unlike PS056.
- gsm8k 0.643 → 0.217 (−42.6pp, math sacrificed); tomi 0.590 → 0.543 (−4.7pp); mmlu 0.457 → 0.603 (+14.6pp).

Health: clean run to step 190 (no crash/OOM, format_error_ratio 0.000 throughout). Entropy fell steadily to ~0.05–0.08 late (policy peaky/deterministic) but this did NOT translate to an eval collapse — the plateau held.

**Verdict:** actor-RM + power k3 + **kl=0.05** is the key contrast to PS056 (actor-RM kl=0.01 = 0.383, POOR). kl=0.05 lifts actor-RM by +0.067 HM and removes the terminal collapse → **confirms kl=0.05 stabilizes the power reward regardless of RM mode** (frozen: PS042/PS048 ≈ 0.455; actor: PS064 = 0.450). Slightly below the frozen best PS048 (0.456) and sacrifices more math (gsm8k −42.6pp vs PS048's −6.7pp), so frozen-RM remains preferable for math preservation, but actor-RM at kl=0.05 is a viable stable config. WandB: https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/h3v5m1km

