### Attempt r1 — 2026-07-12T05:47:56+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-007-002   **git:** `c8fba8e`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1.log`

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
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
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
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:**

#### Findings (completed 2026-07-12, exit 0, 191 steps / 1 epoch)

**Verdict: STABLE + HEALTHY — NEW SWEEP LEADER (HM 0.475). Both stability levers combined (power k=5 + KL=0.05) give the best + cleanest run. Top Phase −1 stable-config candidate.**

Config-selection score (HM of 26 subsample300 benchmarks):
- **HM(last-3) = 0.475**; HM(last-5) = 0.477 — **+0.05 above** step-0 baseline HM = 0.421. Mean-of-means(last-3)=0.533.
- HM rises fast to a **tight high plateau 0.47–0.48** (peak 0.484 @step140), ends 0.474 @step190 — sustained genuine improvement.
- Parseable-answer rate ≈ **100%** (`format_error_ratio`=0).

**Health checks PASS (cleanest in sweep):**
- **KL tightly contained ~0.08** the entire run (0.075–0.096) — best regularization seen.
- **Entropy stable ~1.4** (no collapse, no runaway).
- Response length stable ~103–118.

Eval HM trajectory:
`0:0.421 10:0.451 20:0.477 30:0.475 40:0.463 50:0.475 60:0.483 70:0.477 80:0.476 90:0.474 100:0.471 110:0.482 120:0.473 130:0.473 140:0.484(peak) 150:0.481 160:0.479 170:0.473 180:0.476 190:0.474`

Health trace: reward −23→+39→+34 (bounded); KL 0.002→0.088→0.082→0.075→0.096→0.076; entropy 1.11→1.42→1.46→1.39→1.38→1.44; resp_len 53→113→119→104→105→109.

**Key comparisons (HM last-3), all healthy unless noted:**
- vs power **k=5** kl0.01 (PS065 = 0.458): KL=0.05 tightens KL (0.08 vs 0.12) and lifts HM → **k=5 × kl0.05 > k=5 × kl0.01**.
- vs power **k=3** kl0.05 (PS045 = 0.447 frozen, PS057 = 0.449 actor): **k=5 > k=3** at kl0.05 (+~0.03).
- vs log_prob kl0.05 (PS010/12/15 = 0.432–0.435): **power k=5 × kl0.05 best of all (+~0.04)**.

**Ranking so far (HM last-3, healthy configs):** PS073 power-k5-kl0.05 **0.475** > PS065 power-k5-kl0.01 0.458 > PS057 power-k3-kl0.05-actor 0.449 ≈ PS045 power-k3-kl0.05-frozen 0.447 > log_prob-kl0.05 0.432–0.435.

**Implication:** the Phase −1 stable/leading recipe trends toward **power reward k=5 + KL=0.05 (lr5e-7)**. Both levers are additive: steeper power shaping + adequate KL give the highest ToM HM with the tightest, healthiest training dynamics. Strong candidate to promote as the `stable` config for Phase 0. Remaining k=5 cells (actor-RM, lr1e-6, fp5) will test robustness of this leader.

