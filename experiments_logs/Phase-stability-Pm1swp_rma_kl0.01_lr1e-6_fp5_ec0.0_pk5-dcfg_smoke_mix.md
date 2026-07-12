### Attempt r1 — 2026-07-12T09:08:23+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1`
- **Host:** h100-007-002   **git:** `c59404d`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=actor baseline=False kl=0.01 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp5_ec0.0_pk5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:**

#### Findings (completed 2026-07-12, exit 0, 191 steps / 1 epoch)

**Verdict: STABLE + HEALTHY — power k=5 survives the HARSHEST corner (actor-RM × kl0.01 × lr1e-6) where power k=3 collapsed. Steeper power shaping is inherently stabilizing.**

Config-selection score (HM of 26 subsample300 benchmarks):
- **HM(last-3) = 0.462**; HM(last-5) = 0.464 — above step-0 baseline HM = 0.424. Mean-of-means(last-3)=0.518.
- HM holds **flat 0.46–0.47 across the run** (peak 0.473 @step40), ends 0.463 @step190.
- Parseable-answer rate ≈ **100%** (`format_error_ratio`=0).

**Health checks PASS (clean — no collapse despite worst-case HPs):**
- **KL bounded ~0.12–0.19** the whole run (vs runaway 1.75 for power k=3 actor kl0.01 PS051).
- **Entropy healthy ~1.6** (no collapse — vs PS051's crash to 0.22).
- Response length stable ~97–124 (no shrinkage).

Eval HM trajectory:
`0:0.424 10:0.463 20:0.466 30:0.463 40:0.473(peak) 50:0.470 60:0.456 70:0.462 80:0.461 90:0.461 100:0.459 110:0.461 120:0.469 130:0.468 140:0.463 150:0.466 160:0.468 170:0.467 180:0.451 190:0.463`

Health trace: reward −23→+37→+39→+32 (bounded); KL 0.002→0.170→0.181→0.122→0.188→0.177; entropy 1.05→1.68→1.60→1.73→1.58→1.61; resp_len 54→103→104→124→97→109.

**Key comparisons (HM last-3):**
- vs actor **power k=3** kl0.01/lr5e-7 (PS051 = 0.453 but HEALTH FAILS, entropy→0.22, KL→1.75): **power k=5 at even HIGHER LR (1e-6) stays clean** (KL~0.18, entropy~1.6). The reward-shape steepness (k=5) prevents the actor-RM entropy collapse.
- vs power k=5 kl0.05 cells (PS073 0.475 / PS080 0.483): kl0.05 still a bit higher, but **k=5 makes even kl0.01 safe** — the two levers are partly redundant, k=5 doing much of the stabilization.

**Implication:** big robustness win — **power k=5 is inherently stabilizing**, keeping KL/entropy healthy even in the actor-RM × kl0.01 × lr1e-6 worst case. Strengthens the case for **power k=5 + kl0.05** as the winning recipe (best HM + double safety margin), and shows k=5 alone confers most of the stability.

