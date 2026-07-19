### Attempt r1 — 2026-07-13T09:12:57+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-021-003   **git:** `0e18054`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7 ll_min=-4.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm4 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (r1, authoritative) — completed 2026-07-13 ~11:55 UTC

- **WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/b239awed · **Log:** `logs/20260713/…rma…fp0_ec0.0_pk7_llm4…-r1.log` · full 191 steps (20/20 evals), clean exit.
- **Hypothesis:** does **actor-as-RM at k=7** match the frozen-RM k=7 winner (PS118, +28.1%), or stay noisy/weak like actor-RM at k=5 (PS127, +12.9% with a KL spike)?

**AM_tom (arith mean ToM benches, excl gsm8k/mmlu — reliable for Gemma; parsed from LOG):**

| step | AM_tom | AM_all |
|---|---|---|
| 0 (baseline) | 0.3148 | 0.3162 |
| 90 | 0.3625 | 0.3611 |
| 120 (dip) | 0.2875 | 0.2897 |
| 150 | 0.3984 | 0.3973 |
| 170 | 0.4050 | 0.4030 |
| 190 (final/peak) | 0.4249 | 0.4227 |

- **Selection score:** AM_tom last-3 = 0.406, **last-5 = 0.392** (vs baseline 0.315 → **+0.077, +24.5%**). AM_all last-5 = 0.391. Peak AM_tom = 0.425 @step190 — the run ends on an upswing (still climbing at the final eval).
- **End-state (step190) vs baseline — broad, several large:** tomi 0.597→0.607, gsm8k 0.277→0.350 (+7.3pp), mmlu 0.387→0.443 (+5.6pp), bigtom_forward_belief 0.753→0.833, **simpletom_mental 0.463→0.610 (+14.7pp)**, **tombench 0.380→0.547 (+16.7pp)**, bigtom_backward_belief 0.527→0.600 (+7.3pp).

**Health (clean, stable — notably better than the actor-RM k=5 run):** `critic/rewards/mean` varied ≈ 0.1→0.003 (NO -40 floor), `reward/format_error_ratio = 0`; `actor/kl_loss` max 0.193 with **zero samples > 0.4 — NO KL spike** (contrast PS127's transient 2.76); `actor/entropy_loss` STABLE 1.38→1.51 (min 1.311, no collapse); `response_length/mean` 154→103. One mid-run AM_tom dip at step120 (0.288), fully recovered.

**Verdict:** **STRONG POSITIVE + STABLE — 2nd-best Gemma cell, just behind PS118.** The power sharpness k=7 both **strengthens** (+24.5% vs +12.9% at k=5) and **stabilizes** (no KL spike vs the k=5 spike) actor-as-RM on Gemma-2. Actor-RM at k=7 nearly closes the gap to frozen-RM k=7 (PS118 +28.1%). Consolidated: for Gemma-2, **k=7 is the key sharpness knob**; frozen-RM still edges actor-RM, but the RM-mode gap shrinks a lot at k=7. Recommendation ranking (Gemma-2 power): PS118 (frozen k7/ec0.001, +28.1%) ≳ PS133 (actor k7, +24.5%) > PS111 (frozen k5/fp5, +19.4%) > PS127 (actor k5/fp5, +12.9%) ≫ PS105 (frozen k5/ll_min-6/fp0, null).

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm4 DATA_NAME=dcfg_smoke_mix_gemma DATA_TRAIN=$HOME/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet REWARD_TYPE=power POWER_K=7 POWER_LL_MIN=-4.0 USE_ACTOR_AS_RM=True SUBTRACT_BASELINE=False KL=0.05 LR=5e-7 FORMAT_PENALTY=0 ENTROPY_COEFF=0.0 THINK_ONLY_PG=True ROLLOUT_N=16 MAX_PROMPT=2048 MAX_RESP=512 TOTAL_EPOCHS=1 TEST_FREQ=10 COT_VAR=cot_eval bash experiments/train_behavior_gemma.sh`
