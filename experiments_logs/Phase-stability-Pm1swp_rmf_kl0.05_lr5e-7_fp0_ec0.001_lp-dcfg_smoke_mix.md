### Attempt r1 — 2026-07-11T09:45:36+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp0_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-007-002   **git:** `03533cf`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp0_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp0_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1.log`

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
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp0_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1 \
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
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=log_prob \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp0_ec0.001_lp DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_

#### Findings (r1 — completed 2026-07-11, exit 0, 191 steps / 1 epoch)

**Verdict: FIRST STABLE cell — KL=0.05 contains the drift; behavior reward optimizes while held-out ToM is PRESERVED. Best of the log_prob/frozen cells run so far.**

Config-selection score (HM of 26 subsample300 benchmarks):
- **HM(last-3, steps 170/180/190) = 0.432**; HM(last-5) = 0.433 (avg-then-HM). Mean-of-means(last-3)=0.493.
- **Step-0 baseline HM = 0.423, mean = ~0.49.** HM stays **flat 0.43–0.47 across the whole run** (peak 0.469 @ step40), ends at **0.431 @ step190** — essentially holds/slightly beats baseline. No decline, no collapse.
- Parseable-answer rate ≈ **100%** (`format_error_ratio`=0).

Eval HM trajectory:
`0:0.423 10:0.462 20:0.464 30:0.466 40:0.469(peak) 50:0.465 60:0.463 70:0.457 80:0.438 90:0.437 100:0.451 110:0.446 120:0.434 130:0.449 140:0.432 150:0.433 160:0.433 170:0.429 180:0.434 190:0.431`

Health:
- Reward optimized: `reward/mean` −69 → ~−4.5; `response_length/mean` 53 → ~120.
- **KL contained at ~0.3** (vs 0.5–0.6 for the KL=0.01 cells) — the stronger KL penalty holds the policy near base, so the behavior-prediction gains do NOT come at the cost of ToM. Entropy rises modestly to ~2.7. 100% parseable.

**Comparison (log_prob/frozen, HM last-3):** **PS010 (kl0.05,lr5e-7,fp0,ec0.001)=0.432 (STABLE)** ≫ PS004 (kl0.01,fp5,ec0.001)=0.351 > PS001 (kl0.01,fp0,ec0.0)=0.254 > PS007 (kl0.01,lr1e-6)=0.169. KL=0.05 is decisively better than KL=0.01 at fixed LR=5e-7.

**Implication:** KL=0.05 is the stability lever for the log_prob behavior reward on Qwen2.5-3B — it prevents the policy drift / negative-transfer seen at KL=0.01 while still optimizing the reward. Strong stable-config candidate for the Phase −1 winner (subject to comparison vs the remaining KL=0.05 / power-reward / actor-RM cells).

