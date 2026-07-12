### Attempt r1 — 2026-07-12T08:28:20+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1`
- **Host:** h100-077-004   **git:** `5a590c9`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=actor baseline=False kl=0.01 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1 \
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
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.0_pk5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (r1) — h100-077-004

**Verdict: STABLE. Low lr + format-penalty rescue a weak-KL (kl=0.01) cell — no collapse, near-top ToM.**

### HM_tom (harmonic mean over ToM subtypes, excl mmlu/gsm8k), sub300
| step | 0 | 20 | 40 | 60 | 90 | 110 | 140 | 170 | 190 |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| HM_tom | 0.418 | 0.460 | 0.463 | **0.473** | 0.466 | 0.466 | 0.460 | 0.456 | 0.456 |

- **Baseline(step0)=0.418 · HM(last3)=0.457 · HM(last5)=0.456 · PEAK=0.473@step60.**
- Δ vs baseline **+3.8pp** (last5). Climbs to a peak by step~60, then holds ~0.455-0.466 with a very mild late settle — no collapse.

### Health
- format_error_ratio = 0.000 throughout.
- KL_loss ~0.08-0.11 (tight despite kl_coef=0.01 — low lr + fp5 do the stabilizing).
- entropy_loss ~1.53-1.66 (low/conservative).
- response_length/mean ~110-114 steady.

### Comparison (power-k5 cells)
| cell | KL | lr | fp | HM(last5) | gsm8k end |
|------|-----|-----|-----|-----------|-----------|
| PS078 (TOP) | 0.05 | 1e-6 | 0 | **0.4625** | 0.49 |
| **PS083 (this)** | 0.01 | 5e-7 | 5 | 0.456 | 0.62 |
| PS070 | 0.01 | 1e-6 | 0 | 0.443 | 0.24 (collapse) |

### General-capability note
- gsm8k 0.667 → 0.62 (well-preserved — dramatically better than PS070's weak-KL collapse to 0.24).
- mmlu 0.45 → 0.607 (improved).

### Downstream implication
- **The weak-KL (kl=0.01) collapse observed in PS070/PS039 is driven primarily by LR, not KL alone.** Dropping lr to 5e-7 and adding format_penalty=5 keeps a kl=0.01 power-k5 run stable AND protects general capability (gsm8k) as well as the tight-KL top cell.
- Still slightly below the top cell PS078 (kl=0.05, lr1e-6, +0.6pp) — kl=0.05 @ lr1e-6 remains the best single config, but this shows a robust weak-KL fallback exists.

### Rerun one-liner
`setsid bash -c 'source ~/.bashrc; conda activate tom; cd ~/repo/BeRL; ONLY_IDX=83 bash experiments/phase_stability_sweep.sh' >/tmp/ps083_bootstrap.log 2>&1 &`
