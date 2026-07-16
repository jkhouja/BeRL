### Attempt r1 — 2026-07-16T02:55:52+00:00

- **RUN_NAME:** `data-recipe-P0g_filter_randlen-dcfg_mix_best_randlen_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-189-003   **git:** `cd16a70`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_randlen_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=data-recipe-P0g_filter_randlen-dcfg_mix_best_randlen_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260716/data-recipe-P0g_filter_randlen-dcfg_mix_best_randlen_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_randlen_gemma.parquet \
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
    actor_rollout_ref.actor.think_only_pg=False \
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
    trainer.experiment_name=data-recipe-P0g_filter_randlen-dcfg_mix_best_randlen_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=google/gemma-2-2b-it \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-4 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-4 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=data-recipe-P0g_filter_randlen DATA_NAME=dcfg_mix_best_randlen_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_randlen_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** BLOCKED / HALTED — shared-code bug, NOT a valid result (r1, WandB rb4pp8rr, killed at ~step 53/171).
- **Symptom:** ALL rewards = -45.000 (invalid sentinel = valid_floor 0 - format_penalty 5 - INVALID_MARGIN 40), constant max=min=mean from step 1; critic/advantages == 0 (std 0) => ZERO gradient, no learning. format_error_ratio=0.000, resp_len ~165, entropy ~1.36 (responses generate fine; the RM universally rejects them).
- **Root cause (same as E103 surprise + E104 filter_off, diagnosed by other agents):** the **frozen-RM Gemma path floors at the invalid sentinel for EVERY rollout on the NEW S3-pipeline parquets (populated answer_pp)**. The frozen-RM path reads `data.batch['answer_pp']` raw (`fsdp_workers.py:1264,1468`) / scores an empty ground_truth region (response_token_count==0 -> NaN -> invalid, `fsdp_workers.py:837`), whereas the actor-RM path None-guards it (`fsdp_workers.py:774`). Old parquets (mix_best3) + frozen RM work; new parquet + actor-RM (Qwen E026) works; **new parquet + frozen RM (Gemma filter arm) is broken.**
- **Scope:** blocks the whole filtered/new-pipeline **frozen-RM Gemma** arm: E103 (surprise), E104 (filter_off), **E105 (randlen)**, and by extension E107 (predictable). E102 unfiltered (old parquet) passed +0.162. **Not a data-quality issue with randlen specifically** — the RM rejects all new-pipeline frozen-RM rollouts regardless of filter mode.
- **Action:** killed the run, GPUs freed (1 MiB). Data build itself is fine (5500 rows, answer_pp non-null). Set tracker E105 -> **Awaiting-input** (mirrors E103/E104): needs a **shared-code fix decision from the user** (add the same None-guard / empty-region guard to the frozen-RM path). Rerun one-liner unchanged once the code is fixed.
- **Verdict:** INVALID/NO-OP under current shared code; do not score or interpret as a control result.

### Attempt r1 — 2026-07-16T02:56:06+00:00

- **RUN_NAME:** `data-recipe-P0g_filter_randlen-dcfg_mix_best_randlen_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-189-003   **git:** `cd16a70`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_randlen_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=data-recipe-P0g_filter_randlen-dcfg_mix_best_randlen_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260716/data-recipe-P0g_filter_randlen-dcfg_mix_best_randlen_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_randlen_gemma.parquet \
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
    actor_rollout_ref.actor.think_only_pg=False \
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
    trainer.experiment_name=data-recipe-P0g_filter_randlen-dcfg_mix_best_randlen_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=google/gemma-2-2b-it \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-4 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-4 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=data-recipe-P0g_filter_randlen DATA_NAME=dcfg_mix_best_randlen_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_randlen_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** BLOCKED / HALTED — shared-code bug, NOT a valid result (r1, WandB rb4pp8rr, killed at ~step 53/171).
- **Symptom:** ALL rewards = -45.000 (invalid sentinel = valid_floor 0 - format_penalty 5 - INVALID_MARGIN 40), constant max=min=mean from step 1; critic/advantages == 0 (std 0) => ZERO gradient, no learning. format_error_ratio=0.000, resp_len ~165, entropy ~1.36 (responses generate fine; the RM universally rejects them).
- **Root cause (same as E103 surprise + E104 filter_off, diagnosed by other agents):** the **frozen-RM Gemma path floors at the invalid sentinel for EVERY rollout on the NEW S3-pipeline parquets (populated answer_pp)**. The frozen-RM path reads `data.batch['answer_pp']` raw (`fsdp_workers.py:1264,1468`) / scores an empty ground_truth region (response_token_count==0 -> NaN -> invalid, `fsdp_workers.py:837`), whereas the actor-RM path None-guards it (`fsdp_workers.py:774`). Old parquets (mix_best3) + frozen RM work; new parquet + actor-RM (Qwen E026) works; **new parquet + frozen RM (Gemma filter arm) is broken.**
- **Scope:** blocks the whole filtered/new-pipeline **frozen-RM Gemma** arm: E103 (surprise), E104 (filter_off), **E105 (randlen)**, and by extension E107 (predictable). E102 unfiltered (old parquet) passed +0.162. **Not a data-quality issue with randlen specifically** — the RM rejects all new-pipeline frozen-RM rollouts regardless of filter mode.
- **Action:** killed the run, GPUs freed (1 MiB). Data build itself is fine (5500 rows, answer_pp non-null). Set tracker E105 -> **Awaiting-input** (mirrors E103/E104): needs a **shared-code fix decision from the user** (add the same None-guard / empty-region guard to the frozen-RM path). Rerun one-liner unchanged once the code is fixed.
- **Verdict:** INVALID/NO-OP under current shared code; do not score or interpret as a control result.

