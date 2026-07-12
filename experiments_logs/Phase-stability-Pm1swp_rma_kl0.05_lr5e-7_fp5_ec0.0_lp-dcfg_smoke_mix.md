### Attempt r1 — 2026-07-11T18:56:03+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp5_ec0.0_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-076-003   **git:** `967e489`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp5_ec0.0_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp5_ec0.0_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp5_ec0.0_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp5_ec0.0_lp DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


- **WandB link (resolved r1):** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/qevyy40h
- **Hypothesis:** Phase −1 sweep cell PS027 (actor-as-RM, KL=0.05, LR=5e-7, format_penalty=5, entropy_coeff=0.0, log_prob reward). Tests whether a format penalty (fp=5) with actor-RM/log_prob climbs (HM over last-X) without collapse. Contrast vs PS026 (fp=0, ec=0.001).
- **How to rerun:** `ONLY_IDX="27" bash experiments/phase_stability_sweep.sh` (on an 8-GPU node; env `tom`).

- **[2026-07-11T19:32Z] r1 PREEMPTED:** run (WandB qevyy40h) on h100-076-003 received SIGTERM ~19:09 at ~step 16 because another agent's run (`Pm1swp_rma_kl0.05_lr1e-6_fp0_ec0.001`) claimed the same shared interactive allocation (5420875). Not a config failure. Row reverted to Not-started for a clean re-run (`-r2`) on a genuinely-free node. Lesson: only launch on a node whose allocation isn't shared/contended.
### Attempt r2 — 2026-07-11T19:48:27+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp5_ec0.0_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r2`
- **Host:** h100-189-003   **git:** `8e60a98`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp5_ec0.0_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r2 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp5_ec0.0_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r2.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp5_ec0.0_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r2 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp5_ec0.0_lp DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=2 bash experiments/train_behavior_qwen2.5.sh`

**Findings (r2, 2026-07-11, h100-189-003, WandB 612hto4l):** STABLE cell — a plausible Phase-1
candidate (comparable to frozen-RM kl=0.05 stable cells PS009/PS010).

- **Config-selection metric:** HM(last-3)=**0.430**, HM(last-5)=**0.431** vs step-0 baseline
  HM_tom=**0.415** → both slightly above baseline. Peak HM=**0.458 @step40** (+4.3pp), then plateaus
  ~0.43–0.45 and mildly declines to 0.419 @step190. Climbs early, holds, small late fade.
- **Health:** NO collapse, NO reward-hacking. `format_error_ratio=0.000` for the entire run; general
  capability preserved (gsm8k 0.66→0.60; mmlu 0.487→0.617). resp_len stable ~60→111.
- **Reward optimized cleanly:** reward/mean −56.6(step5) → −7.9(step40) → −2.2(step189).
- **Watch item (drift, non-fatal):** entropy 1.16→2.32 and kl_loss 0.012→0.24 rise over the run
  (kl_coef=0.05 partially contains but does not fully pin the actor-RM drift). Late reward wobble
  (−2 to −7) tracks this. Drift is milder than the kl=0.01 cells and never causes collapse.

| step | HM_tom | gsm8k | mmlu | tomi | fantom_belief_mc | bigtom_fwd_belief | simpletom_mental |
|---|---|---|---|---|---|---|---|
| 0   | 0.415 | 0.660 | 0.487 | 0.590 | 0.500 | 0.770 | 0.853 |
| 40  | 0.458 | 0.707 | 0.593 | 0.617 | —     | —     | —     |
| 100 | 0.448 | 0.603 | 0.603 | 0.590 | —     | —     | —     |
| 190 | 0.419 | 0.603 | 0.617 | 0.603 | 0.463 | 0.697 | 0.960 |

Per-benchmark step0→190: tomi +1.3pp, simpletom_mental +10.7pp, mmlu +13.0pp, dyntom_type_a −7.0pp,
bigtom_fwd_belief −7.3pp, fantom_belief_mc −3.7pp, gsm8k −5.7pp. Net roughly neutral on ToM, positive
on general.

**Verdict:** Stable (HM-last-X > baseline, no collapse, general capability intact), but the climb is
modest and front-loaded (peaks @step40). Keep as a stable-config candidate; for Wave-2 gating it is
"not very poor" → not terrible, HM ≥ baseline. Note r1 (WandB qevyy40h) was preempted ~step16 by
shared-node contention on h100-076-003; r2 is the authoritative attempt (full 191 steps).

