### Attempt r1 — 2026-07-13T00:06:00+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1`
- **Host:** h100-077-004   **git:** `d5f3d58`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=frozen baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=2 max_ctx=2048/4096 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
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
    actor_rollout_ref.actor.think_only_pg=False \
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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=2 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen3-1.7B \
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
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr1e-6_fp0_ec0.0_lp DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-13T00:06:14+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1`
- **Host:** h100-077-004   **git:** `d5f3d58`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=frozen baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=2 max_ctx=2048/4096 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
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
    actor_rollout_ref.actor.think_only_pg=False \
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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=2 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen3-1.7B \
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
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr1e-6_fp0_ec0.0_lp DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (COMPLETED) — 2026-07-13

**Verdict: STABLE / success. Qwen3-1.7B log_prob BeRL improves ToM with a clean epoch-2 recovery, and transfers the Qwen2.5 log_prob recipe across model families.**

Run: 380 steps (2 epochs), WandB ncl690nq. Frozen-RM, log_prob (k2.0, ll_min-8.0), kl=0.05, lr=1e-6, fp=0, ec=0.0, tag-free `dcfg_smoke_mix_gemma`.

### HM_tom trajectory (24 ToM subtypes, harmonic mean, excl mmlu/gsm8k)
| step | HM_tom | gsm8k | mmlu |
|---|---|---|---|
| 0 (base) | 0.3403 | 0.890 | 0.590 |
| 30 | 0.3534 | 0.873 | 0.590 |
| 60 | 0.3701 | 0.907 | 0.603 |
| 90 | 0.3253 | 0.893 | 0.567 |
| 120 | 0.3514 | 0.890 | 0.547 |
| 150 | 0.2983 | 0.893 | 0.537 |
| 180 | 0.3240 | 0.883 | 0.563 |
| 210 | 0.3384 | 0.900 | 0.617 |
| 240 | 0.3889 | 0.897 | 0.650 |
| 270 | 0.3957 | 0.910 | 0.670 |
| 300 | 0.3851 | 0.873 | 0.617 |
| 330 | 0.3967 | 0.883 | 0.630 |
| 360 | 0.3975 | 0.893 | 0.640 |
| 380 | 0.4008 | 0.907 | 0.647 |

**base=0.3403, last3=0.3983, last5=0.3952, peak=0.4008@step380 (+5.5pp last5 vs base).**

### Observations
- **Volatile epoch 1, monotone-strong epoch 2.** ToM dipped to 0.2983@step150 (below base) mid-epoch-1, then recovered decisively in epoch 2 to a rising 0.389→0.401 plateau. Final step is the peak — no overfitting/collapse by 380.
- **General-cap preserved and lifted:** gsm8k stayed ~0.89–0.91 (final 0.907), mmlu rose 0.590→0.647. No capability tax.
- **Health:** reward/mean stayed real and varying (~-7 to -10), within-batch score spread healthy (min!=max), KL~0.008–0.015, pg_loss nonzero, response_length ~800–890 (long native CoT). Never hit the -80 invalid sentinel that blocks Gemma.
- **Model-family transfer:** the Qwen2.5 log_prob winner recipe (PS013, kl0.05/lr1e-6) applied to Qwen3-1.7B (frozen-RM, fp0, ec0) yields a clean +5.5pp ToM gain from a lower Qwen3 base (0.3403 vs Qwen2.5 ~0.415) while preserving gsm8k (0.907 >> Qwen2.5's ~0.65). BeRL is not Qwen2.5-specific.

### How to rerun
```
EXP_ID=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr1e-6_fp0_ec0.0_lp DATA_NAME=dcfg_smoke_mix_gemma \
MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet \
REWARD_TYPE=log_prob POWER_K=2.0 POWER_LL_MIN=-8.0 USE_ACTOR_AS_RM=False SUBTRACT_BASELINE=False \
KL=0.05 LR=1e-6 FORMAT_PENALTY=0 ENTROPY_COEFF=0.0 THINK_ONLY_PG=False bash experiments/train_behavior_qwen3.sh
```
