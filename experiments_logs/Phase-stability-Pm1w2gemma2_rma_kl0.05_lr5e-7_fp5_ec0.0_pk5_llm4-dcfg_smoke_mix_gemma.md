### Attempt r1 — 2026-07-13T06:28:05+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-4.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-021-003   **git:** `6018931`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-4.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-4.0-lr5e-7-kl0.05-n16-r1.log`

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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-4.0-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-4.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-4.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (r1, authoritative) — completed 2026-07-13 ~09:15 UTC

- **WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/cwruv80k · **Log:** `logs/20260713/…rma…fp5_ec0.0_pk5_llm4…-r1.log` · full 191 steps (20/20 evals), clean exit.
- **Hypothesis:** does **actor-as-RM** (the actor scores its own rollouts) match, beat, or destabilize vs **frozen-RM** on Gemma-2, at the PS111 winner knobs (k=5 / ll_min=-4 / fp=5 / ec=0.0)? Comparators: PS111 (frozen, +19.4%), PS118 (frozen k=7/ec0.001, BEST +28.1%).

**AM_tom (arith mean ToM benches, excl gsm8k/mmlu — reliable for Gemma; parsed from LOG):**

| step | AM_tom | AM_all |
|---|---|---|
| 0 (baseline) | 0.3147 | 0.3162 |
| 50 | 0.3547 | 0.3548 |
| 110 (dip) | 0.2695 | 0.2740 |
| 170 (peak) | 0.3720 | 0.3720 |
| 190 (final) | 0.3685 | 0.3674 |

- **Selection score:** AM_tom last-3 = 0.366, **last-5 = 0.355** (vs baseline 0.315 → **+0.041, +12.9%**). AM_all last-5 = 0.356. Peak AM_tom = 0.372 @step170.
- **End-state (step190) vs baseline:** tomi 0.597→0.603, gsm8k 0.28→0.29 (flat), mmlu 0.387→0.420, bigtom_forward_belief 0.753→0.837, **simpletom_mental 0.463→0.587 (+12.4pp)**, tombench 0.377→0.437, bigtom_backward_belief 0.527→0.530 (flat).

**Health (mostly OK, one blemish):** `critic/rewards/mean` stable ≈ -2.8 (varied, NO -40 floor), `reward/format_error_ratio = 0`; `actor/entropy_loss` STABLE 1.48→1.55 (min 1.305, no collapse); `response_length/mean` 176→115. **BUT** `actor/kl_loss` had **one transient spike to 2.762 at the final step** (only 1 of 189 samples exceeds 0.4; KL was otherwise ≤ 0.24) — coincident with the mid-run step-110 AM_tom dip (0.270). Mild actor-RM instability, not sustained.

**Verdict:** **POSITIVE TRANSFER but frozen-RM is clearly better on Gemma-2.** Actor-as-RM gives +12.9% AM_tom — real, but ~6.5pp below the identical-knobs frozen-RM run (PS111, +19.4%) and far below the best frozen cell (PS118 k=7/ec0.001, +28.1%). The trajectory is noisier (a step-110 dip + a single end-of-run KL spike), consistent with the actor scoring its own rollouts introducing mild instability. **Recommendation: prefer frozen-RM for Gemma-2 power runs.** Actor-RM remains viable (no collapse, decent gain) if a separate scorer must be avoided.

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4 DATA_NAME=dcfg_smoke_mix_gemma DATA_TRAIN=$HOME/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-4.0 USE_ACTOR_AS_RM=True SUBTRACT_BASELINE=False KL=0.05 LR=5e-7 FORMAT_PENALTY=5 ENTROPY_COEFF=0.0 THINK_ONLY_PG=True ROLLOUT_N=16 MAX_PROMPT=2048 MAX_RESP=512 TOTAL_EPOCHS=1 TEST_FREQ=10 COT_VAR=cot_eval bash experiments/train_behavior_gemma.sh`
