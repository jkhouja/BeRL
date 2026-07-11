### Attempt r1 — 2026-07-11T13:13:30+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp5_ec0.0_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1`
- **Host:** h100-007-002   **git:** `061c3a8`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=frozen baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp5_ec0.0_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp5_ec0.0_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp5_ec0.0_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1 \
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
    +reward_model.reward_type=log_prob \
    +reward_model.power_k=2.0 \
    +reward_model.power_ll_min=-8.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=log_prob \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp5_ec0.0_lp DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_

#### Findings (r1 — completed 2026-07-11, exit 0, 191 steps / 1 epoch)

**Verdict: STABLE even at LR=1e-6 — KL=0.05 rescues the higher LR that cratered at KL=0.01 (PS007). Strongest confirmation that KL is the stability lever.**

Config-selection score (HM of 26 subsample300 benchmarks):
- **HM(last-3) = 0.435**; HM(last-5) = 0.438. Mean-of-means(last-3)=0.497.
- **Step-0 baseline HM = 0.427.** Brief early dip (steps 40–90 to ~0.42) then recovers; ends **0.438 @ step190**. Flat-to-slightly-positive; no collapse.
- Parseable-answer rate ≈ **100%** (`format_error_ratio`=0).

Eval HM trajectory:
`0:0.427 10:0.465 20:0.470(peak) 30:0.457 40:0.446 50:0.429 60:0.421 70:0.429 80:0.416 90:0.422 100:0.424 110:0.427 120:0.435 130:0.420 140:0.429 150:0.440 160:0.443 170:0.433 180:0.433 190:0.438`

Health: reward −70 → ~−2.4 to −7.5; KL contained ~0.25–0.41; entropy ~2.5–2.6; resp_len ~110; 100% parseable.

**Key comparison (LR effect at each KL, HM last-3):**
- KL=0.01: lr5e-7 (PS004)=0.351 → lr1e-6 (PS007)=**0.169** (collapse worsens with LR).
- KL=0.05: lr5e-7 (PS012)=0.433 → lr1e-6 (PS015)=**0.435** (stable, LR-insensitive).

**Implication:** KL=0.05 makes the log_prob/frozen behavior reward **robust to LR** on Qwen2.5-3B — no drift/negative-transfer at either 5e-7 or 1e-6. Stable-config candidate; the KL=0.05 frozen/log_prob cells (PS010/PS012/PS015) are all tied ~0.432–0.435.

