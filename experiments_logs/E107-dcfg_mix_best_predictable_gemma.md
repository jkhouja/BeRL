### Attempt r1 — 2026-07-16T03:00:49+00:00

- **RUN_NAME:** `E107-dcfg_mix_best_predictable_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-156-003   **git:** `b4184ed`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_predictable_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=E107-dcfg_mix_best_predictable_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260716/E107-dcfg_mix_best_predictable_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_predictable_gemma.parquet \
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
    trainer.experiment_name=E107-dcfg_mix_best_predictable_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=999 \
    trainer.test_freq=5 \
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

**How to rerun:** `EXP_ID=E107 DATA_NAME=dcfg_mix_best_predictable_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_predictable_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-16T03:01:03+00:00

- **RUN_NAME:** `E107-dcfg_mix_best_predictable_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-156-003   **git:** `b4184ed`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_predictable_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=E107-dcfg_mix_best_predictable_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260716/E107-dcfg_mix_best_predictable_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_predictable_gemma.parquet \
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
    trainer.experiment_name=E107-dcfg_mix_best_predictable_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=999 \
    trainer.test_freq=5 \
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

**How to rerun:** `EXP_ID=E107 DATA_NAME=dcfg_mix_best_predictable_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_predictable_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


---
## Findings (E107 — BLOCKED by shared frozen-RM + new-parquet bug, killed 2026-07-16)

**Status: Awaiting-input (Failed to learn — killed @ step 60, all 8 GPUs freed).**

E107 reproduced the exact shared-code bug already documented for its Gemma-mix siblings
E103 (filter_surprise), E104 (filter_off) and E105 (filter_randlen) — all of which were killed and
set to `Awaiting-input`. The E103/E105 tracker notes explicitly predicted E107 would be blocked too.

### Signature
- `critic/rewards/mean = critic/score/mean = -45.000` at **every** logged step (step 1 → 60).
- For power reward with `format_penalty=5`: invalid sentinel = `valid_floor(0) - fp(5) - INVALID_MARGIN(40) = -45`.
  A constant batch mean of exactly -45 ⇒ **100% of rollouts flagged invalid** ⇒ std=0 ⇒ advantages≡0 ⇒ **zero gradient / zero learning**.
- Rollouts themselves generate fine: `format_error_ratio=0.0`, `resp_len≈180`, `kl≈0.001` (not a collapse, not a format-tag problem).
- ToM HM stayed at noise around the base (0.09→0.03–0.08) with no upward signal, consistent with no learning.

### Root cause (shared code, NOT data-recipe-specific)
Frozen-RM path in `verl/workers/fsdp_workers.py` rejects every rollout on the **new S3-pipeline
parquets** (which carry a populated `answer_pp` column):
- `answer_pp = data.batch['answer_pp']` is read **raw** at **:1468** (and **:1264**) with no
  None/empty guard, whereas the **actor-RM** path None-guards it at **:774**.
- The empty ground-truth answer region yields `response_token_count == 0 → NaN → invalid` (**:837**).

Evidence matrix (from siblings + this run):
- old-style parquet + frozen-RM  → **works** (E098 `dcfg_p4g_gemma`, rewards ≈0.05, +15pp HM).
- new S3 parquet + **actor**-RM   → **works** (Qwen E026/E027, +5.0pp HM).
- new S3 parquet + **frozen**-RM  → **BROKEN** (E103/E104/E105/E107 all floor at -45).

Verified the fix is **not yet in the code** (`:1468` still reads `answer_pp` raw), so the bug is
unresolved on branch `jude/paper` at kill time.

### Action taken
- Killed the driver (`main_ppo` pid 1022327) + wrappers; confirmed `main_ppo=0`, ray workers gone,
  all 8 GPUs back to 1 MiB.
- Tracker row E107 → `Awaiting-input` with full diagnosis; WandB + log path recorded.
- **Blocked on a shared-code fix decision from the user** (extend the actor-RM `answer_pp` None/empty
  guard to the frozen-RM path at :1264/:1468/:837, or change the surprise/predictable/randlen build
  so the frozen-RM scorer receives a scoreable ground-truth region). Until then the entire Gemma-mix
  frozen-RM S3 arm (E103/E104/E105/E106/E107) cannot run.

WandB: https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/jxyjo0qr
### Attempt r2 — 2026-07-16T23:53:52+00:00

- **RUN_NAME:** `E107-dcfg_mix_best_predictable_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r2`
- **Host:** h100-156-003   **git:** `f030421`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_predictable_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=2.0 ll_min=-8.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/1024 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0.0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=E107-dcfg_mix_best_predictable_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r2 _(paste link after launch)_
- **Log path:** `logs/20260716/E107-dcfg_mix_best_predictable_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r2.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_predictable_gemma.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
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
    actor_rollout_ref.actor.format_penalty=0.0 \
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
    trainer.experiment_name=E107-dcfg_mix_best_predictable_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r2 \
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
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=power \
    +reward_model.power_k=2.0 \
    +reward_model.power_ll_min=-8.0 \
    reward_model.format_penalty=0.0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=E107 DATA_NAME=dcfg_mix_best_predictable_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_predictable_gemma.parquet RUN_INDEX=2 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r2 — 2026-07-16T23:54:50+00:00

- **RUN_NAME:** `E107-dcfg_mix_best_predictable_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r2`
- **Host:** h100-156-003   **git:** `f030421`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_predictable_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=E107-dcfg_mix_best_predictable_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r2 _(paste link after launch)_
- **Log path:** `logs/20260716/E107-dcfg_mix_best_predictable_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r2.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_predictable_gemma.parquet \
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
    trainer.experiment_name=E107-dcfg_mix_best_predictable_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r2 \
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

**How to rerun:** `EXP_ID=E107 DATA_NAME=dcfg_mix_best_predictable_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_predictable_gemma.parquet RUN_INDEX=2 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


---
## Findings (E107 r2 — RESOLVED by shared fix 8769467, 2026-07-16)

**CORRECTED ROOT CAUSE (supersedes the earlier "frozen-RM answer_pp" diagnosis above):**
The −45 floor was NOT caused by `answer_pp`/empty-region. Runtime diagnostic (`BERL_RM_DEBUG`)
showed `invalid_response=1, empty_region=0, nan_logprob=0`, error "No </think> tag found." The
real cause: **gemma-2-2b-it rarely emits `</think>`** on predictable/low-surprisal turns, and the
reward path hard-invalidated any response missing `</think>` **unconditionally** at
`fsdp_workers.py` actor `:708` / frozen `:1374`. E107's predictable filter (short, casual turns)
→ ~100% of rollouts invalid per GRPO group → `critic/rewards min=max=mean=−45` → advantages≡0 →
zero gradient. (E098 p4g_gemma survived on the same freeform prompt only because its turn mix gave
some `</think>`-emitting rollouts per group → group variance.) `answer_pp` was populated and
non-causal (`subtract_baseline=False`).

**Fix (commit `8769467`, see docs/CHANGE_HISTORY Bug Fixes):** added
`ModelResponseParser.is_invalid_response()` — tagged recipes (Qwen2.5, `REQUIRE_ANSWER_TAGS`) still
require a literal `</think>`; tag-free / native-thinking recipes (Gemma-2, Qwen3) accept any
non-empty response (`split_thinking` supplies the missing tag). Both invalid gates now call it, and
`has_format_violation` no longer treats a missing `</think>` as a violation for tag-free parsers.
Frozen `RewardModelWorker` + actor build the parser via `get_parser(require_answer_tags=…)`.
Qwen2.5 behavior byte-for-byte unchanged. Unblocks E103/E104/E105 (whole Gemma-mix frozen arm) too.

**r2 verification (WandB 9x03dues, on fixed code):** reward/mean healthy and VARIABLE from step 1
(no more −45 floor):

| step | reward/mean | rewards/max | rewards/min | adv max/min | resp_len/mean | kl_loss | format_err |
|------|-------------|-------------|-------------|-------------|---------------|---------|------------|
| 1 | 0.005 | 0.166 | 0.000 | 3.28 / −1.38 | 166 | 0.001 | 0.000 |
| 3 | 0.012 | 0.337 | 0.000 | 3.47 / −1.42 | 156 | 0.001 | 0.000 |
| 4 | 0.064 | 2.769 | 0.000 | 3.38 / −2.04 | 144 | 0.003 | 0.000 |
| 5 | 0.023 | 0.769 | 0.000 | 3.42 / −1.90 | 115 | 0.010 | 0.000 |

Non-zero `pg_loss`/`grad_norm` (0.07–0.26), gradient flowing, no collapse. Run continuing;
canonical ToM HM/avg to be scored on completion via `scripts/score_run.py` (log-results skill).

### r2 canonical score (scripts/score_run.py) — Completed 2026-07-17

```
eval iters: 7 (step 0..171); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.0719  HM(last3)=0.0906  (baseline step0=0.093)
ToM avg(last5)=0.3457  avg(last3)=0.3673  (baseline step0=0.3152)
gsm8k (separate): 0.3574 (step0=0.28, delta vs step0=+0.077)
mmlu  (separate): 0.4006 (step0=0.387, delta vs step0=+0.014)
health(final): kl=0.148 entropy=0.76 resp_len=218.965 reward=0.11 parseable=1.0 max_resp=512
ToM HM trajectory: 0:0.093 30:0.02 60:0.046 90:0.02 120:0.02 150:0.106 171:0.118
```

**Verdict:** Fix VERIFIED end-to-end — reward flowed (0.005→1.27 with variance, never -45), model
learned, parseable=1.0, healthy (kl 0.148, resp_len 219, no collapse/hacking). **ToM avg improved
+2.9pp (last5) / +5.2pp (last3)** and gsm8k +7.7pp; **ToM HM is flat (0.072/0.091 vs base 0.093)**
because two near-zero fantom-list benchmarks (fantom_info_list 0.013, fantom_answerability_list
0.017) dominate the harmonic mean. The predictable Gemma pole underperforms the Qwen predictable
E106 (HM +5.4pp) on HM but shows a genuine broad avg lift. Primary purpose (verify shared fix
`8769467`) achieved; the fix unblocks E103/E104/E105 (whole Gemma-mix frozen arm) and the peer's
PS097–PS176 Gemma/Qwen3 Wave-2 family.
