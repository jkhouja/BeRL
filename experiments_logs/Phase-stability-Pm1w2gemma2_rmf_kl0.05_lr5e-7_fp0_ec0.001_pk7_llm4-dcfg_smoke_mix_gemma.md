### Attempt r1 — 2026-07-13T03:51:29+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-021-003   **git:** `d812f2a`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7 ll_min=-4.0 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1.log`

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
    data.max_response_length=512 \
    actor_rollout_ref.model.path=google/gemma-2-2b-it \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=999 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=google/gemma-2-2b-it \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=7 \
    +reward_model.power_ll_min=-4.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=7 \
    +actor_rollout_ref.power_ll_min=-4.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (r1, authoritative) — completed 2026-07-13 ~06:20 UTC

- **WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/3ejkaih6 · **Log:** `logs/20260713/…ec0.001_pk7_llm4…-r1.log` · full 191 steps (20/20 evals), clean exit.
- **Hypothesis:** does raising the power sharpness to **k=7** (from k=5) plus a small **entropy bonus ec=0.001**, with **ll_min=-4 / fp=0**, help or hurt vs the PS111 Gemma winner (k=5/fp=5/ll_min=-4, +19.4%)? Failure comparator = PS105 (k=5/ll_min=-6/fp=0, NULL -3.2% + entropy collapse).

**AM_tom (arith mean of ToM benches, excl gsm8k/mmlu — reliable for Gemma; parsed from LOG):**

| step | AM_tom | AM_all |
|---|---|---|
| 0 (baseline) | 0.3150 | 0.3163 |
| 90 | 0.3935 | 0.3912 |
| 130 | 0.4055 | 0.4044 |
| 150 (peak) | 0.4282 | 0.4275 |
| 170 | 0.4136 | 0.4119 |
| 190 (final) | 0.4061 | 0.4055 |

- **Selection score:** AM_tom last-3 = 0.401, **last-5 = 0.403** (vs baseline 0.315 → **+0.089, +28.1%**). AM_all last-5 = 0.403. Peak AM_tom = 0.428 @step150. Strong, sustained climb; robust at the end (no step-190 collapse, unlike PS105/PS111 which dipped).
- **End-state (step190) vs baseline — broad, several large gains:** tomi 0.597→0.630, gsm8k 0.277→0.370 (+9.3pp), mmlu 0.387→0.427, bigtom_forward_belief 0.753→0.850 (+9.7pp), **simpletom_mental 0.457→0.587 (+13pp)**, **tombench 0.383→0.517 (+13.4pp)**, bigtom_backward_belief 0.530→0.563. No benchmark regressed.

**Health (excellent, stable):** `critic/rewards/mean` varied around 0 (k=7 sharpens the power reward; NO -40 floor), `reward/format_error_ratio = 0`; `actor/kl_loss` 0.001→0.059 (max 0.095, well controlled); `actor/entropy_loss` **STABLE/rising 1.35→1.70 (min 1.35) — no collapse** (the ec=0.001 entropy bonus keeps exploration healthy); `response_length/mean` 165→129.

**Verdict:** **BEST GEMMA-2-2b CELL TO DATE — +28.1% AM_tom, stable, robust.** Beats PS111 (k=5/fp=5, +19.4%) and dominates PS105 (k=5/ll_min=-6/fp=0, null + collapse). Consolidated Gemma finding: **ll_min=-4 is the essential knob** (the -6 setting is the failure mode); on top of that, **k=7 + a small entropy bonus (ec=0.001)** outperforms **k=5 + format penalty (fp=5)**. Both PS111 and PS118 keep entropy healthy (the collapse in PS105 was driven by the loose clamp, not by fp/ec). **Recommended Gemma-2 power recipe:** frozen-RM / kl0.05 / lr5e-7 / **power-k7 / ll_min=-4 / fp=0 / ec=0.001**.

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4 DATA_NAME=dcfg_smoke_mix_gemma DATA_TRAIN=$HOME/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet REWARD_TYPE=power POWER_K=7 POWER_LL_MIN=-4.0 USE_ACTOR_AS_RM=False SUBTRACT_BASELINE=False KL=0.05 LR=5e-7 FORMAT_PENALTY=0 ENTROPY_COEFF=0.001 THINK_ONLY_PG=True ROLLOUT_N=16 MAX_PROMPT=2048 MAX_RESP=512 TOTAL_EPOCHS=1 TEST_FREQ=10 COT_VAR=cot_eval bash experiments/train_behavior_gemma.sh`
