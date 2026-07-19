### Attempt r1 — 2026-07-12T17:29:42+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-021-003   **git:** `2537109`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-12T17:29:58+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-021-003   **git:** `2537109`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


---

### Findings — r1 — 2026-07-12 (~20:05 UTC) — ⚠️ DEGENERATE / FAILED (systemic Gemma reward bug)

- **Exp #/ID:** PS105 / `Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6` · **WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/7b9r9k6j · **Owner_host:** h100-021-003
- **Launch:** `train_behavior_gemma.sh` (Wave-2 rows are NOT in the ONLY_IDX 96-cell grid) with explicit knobs: `REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-6.0 USE_ACTOR_AS_RM=False KL=0.05 LR=5e-7 FORMAT_PENALTY=0 ENTROPY_COEFF=0.0 THINK_ONLY_PG=True ROLLOUT_N=16 MAX_PROMPT=2048 MAX_RESP=512 TOTAL_EPOCHS=1 TEST_FREQ=10`. Data `dcfg_smoke_mix_gemma` (tag-free, fold_system_prompt=True, require_answer_tags=False).
- **Hypothesis:** Does the Qwen2.5 power winner recipe (PS074: kl0.05/lr5e-7/power-k5) transfer to Gemma-2-2b? **Unanswerable — the run never trained.**

**Critical failure — reward pinned at the invalid sentinel for the entire run:**
- `critic/rewards/mean = -40.000` (EXACT) at **every** logged step 1→189; per-group reward arrays print `[[-40. … -40.]]` — **all 16 rollouts in every GRPO group = -40**. Consequently `critic/advantages = 0`, `actor/pg_loss = 0`, `actor/grad_norm ≈ 0.005` → **zero policy-gradient learning signal** for all 190 steps. The eval scores (tomi ≈ 0.60, gsm8k ≈ 0.28) are just the **frozen base Gemma-2-2b**, not a trained model.
- **Root cause:** `-40` = `invalid_reward_value(power, fp=0)` = `valid_reward_floor(power)=0 − 0 − INVALID_MARGIN(40) = −40` (`fsdp_workers.py:71-80`). So **100% of rollouts are flagged `invalid_response`**. This happens even though generations DO contain a valid `<think>…</think>` (verified in the "Switch template" debug — e.g. `<think>How much does it weigh?</think>`). For **tag-free Gemma** the RM-side scoring stitches the true utterance and builds a `response_mask` over the ground-truth region; that mask comes out **empty** (`response_token_count == 0`), so line 1318/1337 forces the invalid sentinel. The `"</think>" not in response` check (line 1373) plus tag-based answer extraction (`build_stitched_response`, `split_thinking`) don't align with the tag-free Gemma chat template / masking.

**SYSTEMIC — affects the whole Wave-2 Gemma block (verified across other agents' logs today):**
| run | reward_type | distinct `critic/rewards/mean` |
|---|---|---|
| PS105 (this) | power | **-40.000 only** |
| `…rma_kl0.05_lr1e-6_fp0_ec0.0_lp` | log_prob | **-80.000 only** |
| `…rmf_kl0.05_lr1e-6_fp0_ec0.0_lp` | log_prob | **-80.000 only** |
| `…rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5` | power | **-40.000 only** |

(-80 = `invalid_reward_value(log_prob)= −40 − 0 − 40`.) **Every Wave-2 Gemma run is producing zero learning signal.** This is a shared-code (`verl/workers/fsdp_workers.py`) issue in the tag-free reward-scoring/response-masking path, not a per-row misconfig.

**Verdict:** **FAILED (degenerate reward).** Deterministic — a `-r2` resubmit would reproduce identically. Per join-experiments Golden Rule #4 I did **not** edit shared `verl/` code mid-flight (other agents have live Gemma runs). Row set to `Awaiting-input`; raised to the user for a decision on fixing the tag-free Gemma/Qwen3 reward-masking in `verl` (and whether to pause the Gemma Wave-2 rows until fixed).

**How to rerun (once fixed):** `EXP_ID=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6 DATA_NAME=dcfg_smoke_mix_gemma DATA_TRAIN=$HOME/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-6.0 USE_ACTOR_AS_RM=False KL=0.05 LR=5e-7 FORMAT_PENALTY=0 ENTROPY_COEFF=0.0 THINK_ONLY_PG=True ROLLOUT_N=16 TOTAL_EPOCHS=1 TEST_FREQ=10 bash experiments/train_behavior_gemma.sh`
### Attempt r2 — 2026-07-12T21:31:46+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r2`
- **Host:** h100-021-003   **git:** `719757a`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r2 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r2.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r2 \
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
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=2 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


---

### r2 — rerun after fix — launched 2026-07-12 ~20:30 UTC (authoritative)

- **Why:** r1 was degenerate (systemic tag-free Gemma reward bug, all rollouts flagged invalid → reward pinned at -40). User FIXED the shared verl scoring/masking (uncommitted changes to `verl/workers/fsdp_workers.py` + `verl/utils/reward_score/response_parser.py`). Relaunched identical knobs.
- **WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/97fsdj46 · **Log:** `logs/20260712/…-r2.log`
- **Fix verified at launch:** reward is now healthy and varied — `critic/rewards/mean` step1→3 = **1.20 → 5.73 → 6.07** (r1 was -40.000 flat), per-group reward arrays show real spread (~2.0–6.6) instead of uniform -40. Learning signal restored. Monitoring to completion; findings to follow.

**r2 FINDINGS (authoritative) — completed 2026-07-13 ~01:05 UTC**

- **WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/97fsdj46 · **Log:** `logs/20260712/…-r2.log` · full 191 steps, step-190 final validation printed, clean exit.
- **Fix confirmed:** `critic/rewards/mean` 1.20 → 6.29 (min -0.12, max 16.2), varied per-group spread, **zero -40 floor hits** across the whole run; `reward/format_error_ratio = 0` throughout. The user's tag-free Gemma reward-scoring fix works.

**AM_tom (arithmetic mean of ToM benchmarks, excl gsm8k/mmlu) — the reliable metric for Gemma; parsed from the LOG (WandB only synced evals ≤ step 80):**

| step | AM_tom | AM_all | HM_tom |
|---|---|---|---|
| 0 (baseline) | 0.3150 | 0.3165 | 0.093 |
| 10 (peak) | 0.3450 | 0.3450 | 0.047 |
| 30–80 (dip) | 0.24–0.27 | 0.24–0.28 | — |
| 120–150 (recover) | 0.322–0.329 | 0.326–0.335 | — |
| 190 (final, drop) | 0.2588 | 0.2698 | 0.078 |

- **Selection score:** AM_tom last-3 = 0.293, **last-5 = 0.305** (vs baseline 0.315 → **-3.2%, essentially flat/slightly negative**). AM_all last-5 = 0.313 (vs 0.317). Peak AM_tom = 0.345 @step10 (transient, +9.5% but not sustained).
- **HM_tom is uninformative here** (0.03–0.16 all run): the harmonic mean is dominated by near-zero Gemma benchmarks (`opentom_multihop_so`≈0.03, `hi_tom`≈0.10), so it swings on noise. Confirms the PS104 note "HM unreliable for Gemma → use arithmetic mean."
- **End-state (step190) vs baseline:** tomi 0.597→0.540 (-5.7pp), simpletom_mental 0.463→0.313 (-15pp), tombench 0.380→0.273 (-10.7pp), bigtom_forward_belief 0.753→0.620 (-13.3pp), bigtom_backward_belief 0.527→0.460, **gsm8k 0.280→0.423 (+14.3pp, UP)**, mmlu 0.390→0.380. So the late-run step reduced ToM while lifting gsm8k.

**Health:** `actor/kl_loss` 0.0008 → 0.173 (spike max 0.515 mid-run, ended contained); `actor/entropy_loss` **declines** 1.53 → 0.52 (min 0.28) — over-sharpening (not a full →0 collapse, but a clear downward drift, opposite of the Qwen2.5 entropy-inflation pattern); `response_length/mean` 167 → 240 (grows).

**Verdict:** **STABLE (fix works, no reward collapse) but NEGATIVE/NULL TRANSFER.** The Qwen2.5 power-winner recipe (PS074: frozen-RM/kl0.05/lr5e-7/power-k5) does **not** induce ToM gains on Gemma-2-2b — AM_tom is flat-to-slightly-down with a transient early peak, a mid-run dip, a partial recovery, and a late-run degradation that trades ToM for gsm8k. Gemma-2-2b behaves very differently from Qwen2.5 (entropy shrinks rather than inflates). **The Qwen recipe is not directly transferable; Gemma needs its own recipe search.** Report AM_tom (not HM) for all Gemma rows.

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6 DATA_NAME=dcfg_smoke_mix_gemma DATA_TRAIN=$HOME/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-6.0 USE_ACTOR_AS_RM=False KL=0.05 LR=5e-7 FORMAT_PENALTY=0 ENTROPY_COEFF=0.0 THINK_ONLY_PG=True ROLLOUT_N=16 TOTAL_EPOCHS=1 TEST_FREQ=10 bash experiments/train_behavior_gemma.sh`
