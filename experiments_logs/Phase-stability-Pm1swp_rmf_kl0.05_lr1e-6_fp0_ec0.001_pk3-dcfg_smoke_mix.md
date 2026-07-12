### Attempt r1 — 2026-07-11T23:38:33+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp0_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1`
- **Host:** h100-189-003   **git:** `c2e63f0`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=3 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp0_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp0_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp0_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp0_ec0.001_pk3 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings (r1, 2026-07-12, h100-189-003, WandB s3a3upe1):** **STABLE on ToM-HM but severe gsm8k
forgetting** — kl=0.05 rescues the collapse that killed PS038 (kl=0.01, same power-k3/lr=1e-6), yet
lr=1e-6 still destroys general math ability.

- **Config-selection metric (ToM subtypes, excl gsm8k/mmlu):** HM(last-3)=**0.441**, HM(last-5)=**0.441**
  vs step-0 baseline HM_tom=**0.417** → above baseline. Peak HM=**0.459 @step20**; ToM-HM stays flat
  0.42–0.46 the whole run (one transient dip 0.379@step140, recovers). tomi 0.587→0.657,
  bigtom_fwd_belief 0.770→0.697, simpletom_mental 0.85→0.94, mmlu 0.470→0.607 — all held/up.
- **Health RED FLAG:** **gsm8k catastrophically forgotten 0.657→0.197** (dips to 0.047@step140).
  The power reward is optimized cleanly (reward/mean −3.7@s5 → +30@s40 → +34@s189) *without* the
  degenerate length explosion of PS038 (resp_len stable ~116, not 155+); entropy contained
  1.36→2.42, kl_loss 0.07→0.26. So no reward-hacking collapse, but real collateral math forgetting.

| step | HM_tom | gsm8k | mmlu | tomi | bigtom_fwd_belief | reward/mean | entropy | kl_loss |
|---|---|---|---|---|---|---|---|---|
| 0   | 0.417 | 0.657 | 0.470 | 0.587 | 0.770 | −3.7(s5) | 1.36 | 0.071 |
| 20  | 0.459 | 0.657 | 0.580 | 0.613 | 0.743 | —        | —    | —     |
| 100 | 0.440 | 0.320 | 0.607 | 0.587 | 0.710 | +35.2    | 2.36 | 0.224 |
| 190 | 0.439 | 0.197 | 0.607 | 0.657 | 0.697 | +34.3(s189)| 2.42 | 0.257 |

**Verdict:** Qualified STABLE — passes the ToM-HM config-selection metric (HM-last-X > baseline, no
ToM collapse, entropy/KL/len all controlled), but inflicts severe gsm8k catastrophic forgetting
(−46pp). Much better than the kl=0.01 sibling PS038 (which collapsed everything) ⇒ kl=0.05 is
necessary to contain lr=1e-6, but lr=1e-6 still costs general math capability. For the final stable
config, **prefer lr=5e-7 over lr=1e-6** if ToM-HM is comparable, since 5e-7 avoids this gsm8k damage.
Not terrible / HM ≥ baseline ⇒ **keep for Wave-2** but with the gsm8k caveat noted.

