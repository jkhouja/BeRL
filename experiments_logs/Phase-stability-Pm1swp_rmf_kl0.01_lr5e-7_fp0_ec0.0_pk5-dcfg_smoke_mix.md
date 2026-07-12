### Attempt r1 — 2026-07-12T04:06:34+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1`
- **Host:** h100-007-002   **git:** `9b78f2b`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.01 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1 \
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
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp0_ec0.0_pk5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:**

#### Findings (completed 2026-07-12, exit 0, 191 steps / 1 epoch)

**Verdict: STABLE + HEALTHY — BEST HM in the sweep so far (0.458). power k=5 stabilizes even kl0.01 (at lr5e-7) where log_prob failed. Steeper reward shaping is a stability lever.**

Config-selection score (HM of 26 subsample300 benchmarks):
- **HM(last-3) = 0.458**; HM(last-5) = 0.462 — clearly above step-0 baseline HM = 0.424. Mean-of-means(last-3)=0.517.
- HM rises to a **flat plateau 0.46–0.48** (peak 0.482 @step80), ends 0.456 @step190 — genuine improvement, not just non-degradation.
- Parseable-answer rate ≈ **100%** (`format_error_ratio`=0).

**Health checks PASS (clean):**
- **KL contained ~0.08–0.17** across the whole run (no divergence) — remarkable for kl0.01.
- **Entropy healthy** (rises smoothly 1.06→1.98, no collapse).
- Response length stable ~110–117.

Eval HM trajectory:
`0:0.424 10:0.454 20:0.469 30:0.469 40:0.473 50:0.474 60:0.474 70:0.471 80:0.482(peak) 90:0.467 100:0.470 110:0.475 120:0.477 130:0.468 140:0.459 150:0.466 160:0.466 170:0.465 180:0.452 190:0.456`

Health trace: reward ~+28 to +39 (bounded, no runaway); KL 0.003→0.12→0.08→0.13→0.12→0.17; entropy 1.06→1.45→1.51→1.64→1.84→1.98; resp_len 47→113→111→117→108→117.

**Key comparisons (HM last-3):**
- vs frozen **log_prob** kl0.01/lr5e-7/fp0 (PS001 = 0.254, poor/negative transfer): **power k=5 at the SAME kl/lr is dramatically better (0.458)** and healthy. The reward family, not just KL, governs low-KL stability.
- vs frozen **power k=3** kl0.01/lr1e-6 (PS037 = 0.367, declining/unstable): power k=5 @ lr5e-7 is stable — but note the LR differs (5e-7 vs 1e-6). Steeper power (k=5) + lower LR = clean at kl0.01.
- vs stable kl0.05 cells (power PS045=0.447 / PS057=0.449; log_prob 0.432–0.435): **power k=5 edges out all of them (0.458)** — current sweep leader.

**Implication:** two stability levers now identified: (1) **KL=0.05** (universal), and (2) **steeper power shaping (k=5)** which can stabilize even kl0.01 at lr5e-7. power k=5 is the leading reward/stability candidate. Need the k=5 × kl0.05 cells to see if they push higher still (and whether k=5 stays clean at lr1e-6).

