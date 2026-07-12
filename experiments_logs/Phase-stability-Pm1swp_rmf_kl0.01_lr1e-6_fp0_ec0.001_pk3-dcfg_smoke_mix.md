### Attempt r1 — 2026-07-11T21:39:46+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp0_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1`
- **Host:** h100-189-003   **git:** `92e1d06`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=3 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.01 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp0_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp0_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp0_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1 \
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
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=3 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp0_ec0.001_pk3 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings (r1, 2026-07-11, h100-189-003, WandB obe4lyr7):** **COLLAPSED / reward-hacked** — the
predicted kl=0.01 + lr=1e-6 instability (cf. PS005/PS006), now confirmed for the power-k3 reward.

- **Config-selection metric:** HM(last-3)=**0.180**, HM(last-5)=**0.165** vs step-0 baseline
  HM_tom=**0.418** → catastrophically below baseline. Peak HM=**0.467 @step10** (early, +4.9pp), then
  monotonic collapse to HM=**0.142 @step190**.
- **Reward hacking:** power reward saturates near the +40 max by ~step40 (reward/mean −8.8@step5 →
  +26@step40 → +31@step189) while every eval crashes: tomi 0.587→0.197, gsm8k 0.657→0.287,
  bigtom_fwd_belief 0.767→0.583. simpletom_mental stays high (0.85→0.66, memorized/degenerate).
- **Health:** entropy explodes 1.39→3.23, kl_loss 0.067→0.44 (kl_coef=0.01 far too weak to contain
  lr=1e-6), resp_len drifts 65→153. format_error=0 throughout (parseable but degenerate/wrong).

| step | HM_tom | gsm8k | mmlu | tomi | bigtom_fwd_belief | reward/mean | entropy | kl_loss |
|---|---|---|---|---|---|---|---|---|
| 0   | 0.418 | 0.657 | 0.473 | 0.587 | 0.767 | −8.8(s5) | 1.39 | 0.067 |
| 10  | 0.467 | 0.710 | 0.583 | 0.630 | 0.757 | —        | —    | —     |
| 40  | 0.379 | 0.370 | 0.553 | 0.390 | 0.680 | +26.0    | 2.81 | 0.505 |
| 100 | 0.265 | 0.377 | 0.573 | 0.413 | 0.713 | +34.9    | 3.12 | 0.363 |
| 190 | 0.142 | 0.287 | 0.537 | 0.197 | 0.583 | +31.4(s189)| 3.23 | 0.440 |

**Verdict:** COLLAPSED — exclude from best-config; not a Phase-1 winner. Confirms that
kl=0.01 + lr=1e-6 is unstable **regardless of reward family** (log_prob PS005/PS006 → power-k3 here):
weak KL cannot contain the fast learning rate, the actor reward-hacks the frozen-RM likelihood and
transfers negatively to every ToM/general benchmark. For Wave-2 gating this cell is **very poor**
(collapsed & HM ≪ baseline) → **skip in Wave 2**.

