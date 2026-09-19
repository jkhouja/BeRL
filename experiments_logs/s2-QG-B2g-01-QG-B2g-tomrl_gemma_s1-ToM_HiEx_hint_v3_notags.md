### Attempt r1 — 2026-08-09T00:02:15+00:00

- **Entry / Exp #:** `s2-QG-B2g-01-QG-B2g-tomrl_gemma_s1-ToM_HiEx_hint_v3_notags` / `QG-B2g-01`
- **Exp ID:** `QG-B2g-tomrl_gemma_s1`
- **Attempts:** `r1` (invalid, stopped at step 29), `r2` (authoritative)
- **Hypothesis / question:** Test whether direct label-supervised ToM RL reproduces on Gemma-2-2B, providing a second-family baseline against Gemma behavior-prediction RL (D6c/SS-08).
- **RUN_NAME:** `s2-QG-B2g-01-QG-B2g-tomrl_gemma_s1-ToM_HiEx_hint_v3_notags-gemma-2-2b-it-rulebased-lr5e-7-kl0.001-n16-r1`
- **Host:** h100-003-002   **git:** `892a3a3`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3_notags.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=n/a power_k=n/a ll_min=n/a rm_mode=frozen baseline=False kl=0.001 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/2048 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=0.0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/kjog7gsi
- **Log path:** `logs/20260809/s2-QG-B2g-01-QG-B2g-tomrl_gemma_s1-ToM_HiEx_hint_v3_notags-gemma-2-2b-it-rulebased-lr5e-7-kl0.001-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3_notags.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=8 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=google/gemma-2-2b-it \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.think_only_pg=False \
    actor_rollout_ref.actor.format_penalty=0.0 \
    actor_rollout_ref.actor.format_penalty_std_coef=0.0 \
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
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-QG-B2g-01-QG-B2g-tomrl_gemma_s1-ToM_HiEx_hint_v3_notags-gemma-2-2b-it-rulebased-lr5e-7-kl0.001-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False \
    actor_rollout_ref.rollout.temperature=1.0
```

**How to rerun:** `RUN_STAGE=s2 EXP_NUM=QG-B2g-01 EXP_ID=QG-B2g-tomrl_gemma_s1 DATA_NAME=ToM_HiEx_hint_v3_notags DATA_TRAIN=$HOME/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3_notags.parquet MODEL_PATH=google/gemma-2-2b-it KL=0.001 LR=5e-7 TRAIN_BATCH=8 MINI_BATCH=128 MICRO_BATCH=8 ROLLOUT_N=16 MAX_PROMPT=2048 MAX_RESP=2048 ENTROPY_COEFF=0.001 VAL_SUITE=subsample300 SEED=1 GPU_IDS=0,1,2,3,4,5,6,7 NUM_GPUS=8 TP_SIZE=2 RUN_INDEX=1 bash experiments/train_tom_gemma.sh actor_rollout_ref.rollout.temperature=1.0`

**Debugging / issues:**

- **2026-08-09T00:18Z:** Stopped r1 at step 29/400. Every rollout received `reward=-3` with
  `reward/format_error_ratio=1.0`, zero advantages, and `pg_loss=0`, so no learning occurred.
  Gemma correctly emitted tag-free prose, but the direct-ToM rule scorer still called
  `ModelResponseParser.validate_structure()` / `extract_answer()`, which require `<think>` and
  `</think>` even when `REQUIRE_ANSWER_TAGS=False`. This conflicts with the tag-free Gemma parser
  contract implemented by `is_invalid_response()` / `has_format_violation()`. A shared scorer/parser
  fix or a deliberate tagged-Gemma recipe is required before retrying.
- **2026-08-09T00:22Z:** User approved the shared fix. Updated tag-free `extract_answer()` and
  `validate_structure()` semantics while preserving tagged Qwen2.5 behavior; added end-to-end
  Gemma rule-score regression coverage. Targeted parser and ToM scorer tests pass. Relaunching r2.

**Findings:** Invalid attempt; superseded by authoritative r2 below.
### Attempt r2 — 2026-08-09T01:00:10+00:00

- **Entry / Exp #:** `s2-QG-B2g-01-QG-B2g-tomrl_gemma_s1-ToM_HiEx_hint_v3_notags` / `QG-B2g-01`
- **Exp ID:** `QG-B2g-tomrl_gemma_s1`
- **Hypothesis / question:** Test whether direct label-supervised ToM RL reproduces on Gemma-2-2B, providing a second-family baseline against Gemma behavior-prediction RL (D6c/SS-08).
- **RUN_NAME:** `s2-QG-B2g-01-QG-B2g-tomrl_gemma_s1-ToM_HiEx_hint_v3_notags-gemma-2-2b-it-rulebased-lr5e-7-kl0.001-n16-r2`
- **Host:** h100-003-002   **git:** `892a3a3`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3_notags.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=n/a power_k=n/a ll_min=n/a rm_mode=frozen baseline=False kl=0.001 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/2048 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=0.0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/i0905x47
- **Log path:** `logs/20260809/s2-QG-B2g-01-QG-B2g-tomrl_gemma_s1-ToM_HiEx_hint_v3_notags-gemma-2-2b-it-rulebased-lr5e-7-kl0.001-n16-r2.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3_notags.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=8 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=google/gemma-2-2b-it \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.think_only_pg=False \
    actor_rollout_ref.actor.format_penalty=0.0 \
    actor_rollout_ref.actor.format_penalty_std_coef=0.0 \
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
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-QG-B2g-01-QG-B2g-tomrl_gemma_s1-ToM_HiEx_hint_v3_notags-gemma-2-2b-it-rulebased-lr5e-7-kl0.001-n16-r2 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False \
    actor_rollout_ref.rollout.temperature=1.0
```

**How to rerun:** `RUN_STAGE=s2 EXP_NUM=QG-B2g-01 EXP_ID=QG-B2g-tomrl_gemma_s1 DATA_NAME=ToM_HiEx_hint_v3_notags DATA_TRAIN=$HOME/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3_notags.parquet MODEL_PATH=google/gemma-2-2b-it KL=0.001 LR=5e-7 TRAIN_BATCH=8 MINI_BATCH=128 MICRO_BATCH=8 ROLLOUT_N=16 MAX_PROMPT=2048 MAX_RESP=2048 ENTROPY_COEFF=0.001 VAL_SUITE=subsample300 SEED=1 GPU_IDS=0,1,2,3,4,5,6,7 NUM_GPUS=8 TP_SIZE=2 RUN_INDEX=2 bash experiments/train_tom_gemma.sh actor_rollout_ref.rollout.temperature=1.0`

**Debugging / issues:** r2 includes the approved tag-free parser/scorer fix. At step 7,
`format_error_ratio=0.0`, rewards span `-1..3`, advantages span `-3.75..3.75`, and
`pg_loss` is nonzero, confirming learning is restored.

**Findings (final, canonical scorer):**

```text
log: logs/20260809/s2-QG-B2g-01-QG-B2g-tomrl_gemma_s1-ToM_HiEx_hint_v3_notags-gemma-2-2b-it-rulebased-lr5e-7-kl0.001-n16-r2.log
eval iters: 15 (step 0..400); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.5155  HM(last3)=0.5164  (baseline step0=0.4634)
ToM avg(last5)=0.5775  avg(last3)=0.5781  (baseline step0=0.5143)
gsm8k (separate): 0.4932 (step0=0.543, delta vs step0=-0.050)
mmlu (separate): 0.532 (step0=0.53, delta vs step0=+0.002)
health(final): kl=0.439 entropy=0.004 resp_len=6.5 reward=3.0 parseable=1.0 max_resp=2048
ToM HM trajectory: 0:0.463 30:0.502 60:0.498 90:0.514 120:0.493 150:0.510 180:0.505 210:0.512 240:0.511 270:0.516 300:0.507 330:0.515 360:0.511 390:0.520 400:0.518
```

**Verdict:** r2 completed all 400 steps and improved ToM HM by 5.2pp and arithmetic
mean by 6.3pp over the Gemma step-0 baseline. The gain persisted through the final evaluations,
with 100% final parseability and no format collapse or reward/eval divergence. Responses converged
to concise 5-8-token answers and entropy fell to 0.004; this is expected for direct rule-supervised
answer optimization but should be monitored across seeds. The separate gsm8k score regressed 5.0pp,
while mmlu was effectively flat (+0.2pp).
