### Attempt r1 — 2026-07-12T08:35:47+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1`
- **Host:** h100-117-001   **git:** `c6c382d`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=actor baseline=False kl=0.01 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1.log`

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
    algorithm.kl_ctrl.kl_coef=0.01 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1 \
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
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.001_pk5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings (r1, completed 2026-07-12):** STRONG — strongest confirmation yet that **power k=5 is the primary stabilizer** (holds even in the most collapse-prone setting).

Config-selection metric = harmonic mean over the sub300 benchmarks per eval block; compare vs step-0 baseline.
- Baseline (step 0): HM = **0.4253**
- HM last-3 (170/180/190): **0.4569**
- HM last-5 (150–190): **0.4551**  → **ABOVE baseline** (+0.030)
- Per-step HM: above baseline from step 10 onward, oscillates ~0.45–0.47, no terminal decline. Sustained.
- gsm8k 0.657 → 0.490 (−16.7pp, moderate — worse than frozen k5 runs but FAR better than actor k3 low-KL which crashed to 0.15–0.22); tomi 0.590 → 0.580 (−1.0pp); mmlu 0.477 → 0.573 (+9.6pp).

Health: clean run to step 190 (no crash/OOM). **format_error_ratio stayed 0.000 throughout despite fp=0** (no format hacking even without the penalty). Entropy rose 2.0 → ~2.4 then plateaued (never runaway). Response length ~100 → ~120 plateau.

**Verdict:** actor-RM + power **k=5** + kl=0.01 + **fp=0** = 0.455, STRONG — and this is the most collapse-prone configuration in the sweep (actor RM, lowest KL, NO format penalty). The direct contrast is PS056 (actor + **k=3** + kl=0.01 + fp=5 = 0.383, POOR): the ONLY change that matters is k=3→k=5. This establishes **power k=5 as the primary stabilizer of the reward**, independent of RM mode, KL, and format penalty — the steeper reward concentrates credit on the true human continuation, preventing the reward-hacking/over-optimization that sinks k=3 at low KL. kl=0.05 (PS079=0.475) and fp=5 add incremental math-preservation, but k=5 alone rescues stability. Combined ranking: PS079 (k5 kl0.05 fp5) = 0.475 > PS048 (k3 kl0.05 fp5) = 0.456 ≈ PS086 (k5 kl0.01 fp0) = 0.455 > PS071 (k5 kl0.01 fp5) = 0.447. WandB: https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/sb17xuwg

