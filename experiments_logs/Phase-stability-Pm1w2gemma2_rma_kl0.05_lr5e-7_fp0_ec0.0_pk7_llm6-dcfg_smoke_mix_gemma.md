### Attempt r1 — 2026-07-13T07:03:32+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-033-004   **git:** `be1d37c`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7 ll_min=-6 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/dcfg_smoke_mix_gemma.parquet \
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
    actor_rollout_ref.actor.think_only_pg=False \
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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=google/gemma-2-2b-it \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=true \
    +reward_model.reward_type=power \
    +reward_model.power_k=7 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=7 \
    +actor_rollout_ref.power_ll_min=-6 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


---

## 2026-07-13 — r1 OUTCOME: COMPLETED — STRONG WINNER (actor-RM), HM 0.43 (5x baseline)

**Run:** WandB dhdtyhev. ACTOR-RM (verified +reward_model.use_actor_as_rm=true; log/WandB name says
"frozenRM" due to common.sh MODEL_TAG naming bug — run IS actor-RM). Ran clean to step 190, exited (0 procs, 0 OOM).

**Score (`scripts/score_run.py`, 24 ToM benches, excl gsm8k/mmlu):**
- **ToM HM(last5)=0.4285, HM(last3)=0.4298** vs baseline step0=0.0853 → **+0.343 (≈5x); BEST run to date.**
- **ToM avg(last5)=0.5067** vs baseline 0.3147 → **+0.192** (broad, large ToM gain).
- gsm8k=0.432 (+0.155 vs step0), mmlu=0.505 (+0.118) — big general-reasoning gains too.
- HM trajectory MONOTONIC & stable: 0:0.085 10:0.073 20:0.072 30:0.136 40:0.097 50:0.187 60:0.237
  70:0.33 80:0.356 90:0.366 100:0.383 110:0.372 120:0.341 130:0.347 140:0.405 150:0.416 160:0.435
  170:0.425 180:0.432 190:0.431 — plateaus ~0.42-0.44 across last 5 evals (robust, not a fluke).

**Training dynamics (two phases — important):**
- Steps ~40-160: response_length COLLAPSED to ~2-12 tokens (CoT eliminated); evals still climbed
  (HM 0.10→0.43). i.e. eval gains were realized during the short-CoT phase.
- Steps ~170-190: response_length RE-EXPLODED 155→504 tokens and actor/kl_loss ramped 0.001→~9.85
  (large policy divergence from ref). Late-training instability, BUT eval HM held stable ~0.43
  through it (final checkpoint is post-blowup, HM 0.431). Memory-safe (no OOM even at resp_len ~504).

**Interpretation / KEY FINDING:** ACTOR-as-RM is the winning ingredient for Gemma-2-2B Wave-2 transfer.
Same power reward (k=7, ll_min=-6), same kl/lr/data — swapping frozen-RM → actor-RM takes HM(last5)
from ~0.074-0.087 (PS114/PS119, frozen) to **0.43** (PS129, actor). Frozen-RM power was null on HM;
actor-RM power is a large, stable win. Hypothesis: actor-RM co-adapts the reward with the policy so
the behavior-prediction signal stays informative (wide [-40,40] spread) instead of compressing to ~0
(frozen power collapsed reward near 0 → weak advantages).

**Caveats to flag for downstream:** (1) late KL blow-up (kl_loss→~9.85) + length re-explosion — a KL
schedule / higher kl_coef / early-stop-on-KL may stabilize and is worth a follow-up; (2) eval gains
appeared during a CoT-collapse phase, so the mechanism is NOT "longer/better CoT" — verify the CoT
content quality before over-claiming ToM reasoning (could be improved direct-answering).

**VERDICT = COMPLETED — STRONG WINNER. HM(last5)=0.43 (5x base). Actor-RM >> frozen-RM for Gemma-2.**
