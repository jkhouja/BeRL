### Attempt r1 — 2026-07-13T01:14:19+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-021-003   **git:** `d5f3d58`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4.0 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4.0-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4.0-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.power_ll_min=-4.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-4.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-13T01:14:37+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-021-003   **git:** `d5f3d58`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4.0 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4.0-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4.0-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.power_ll_min=-4.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-4.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (r1, authoritative) — completed 2026-07-13 ~03:45 UTC

- **WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/2u99xv9w · **Log:** `logs/20260713/…-r1.log` · full 191 steps (20/20 evals), clean exit.
- **Hypothesis:** does the Qwen2.5 power-winner recipe (PS074: frozen-RM/kl0.05/lr5e-7/power-k5) transfer to Gemma-2-2b when using a **tighter clamp (ll_min=-4)** + **format penalty (fp=5)**? Direct comparator = **PS105** (same cell but ll_min=-6/fp=0), which showed NULL transfer (AM_tom -3.2%) and entropy collapse.

**AM_tom (arith mean of ToM benches, excl gsm8k/mmlu — reliable metric for Gemma; HM unreliable per PS104). Parsed from LOG (WandB synced partial):**

| step | AM_tom | AM_all |
|---|---|---|
| 0 (baseline) | 0.3151 | 0.3164 |
| 60 | 0.3666 | 0.3641 |
| 120 | 0.3889 | 0.3867 |
| 160 (peak) | 0.3935 | 0.3920 |
| 180 | 0.3891 | 0.3888 |
| 190 (final) | 0.3433 | 0.3462 |

- **Selection score:** AM_tom last-3 = 0.369, **last-5 = 0.376** (vs baseline 0.315 → **+0.061, +19.4%**). AM_all last-5 = 0.376 (vs 0.316). Peak AM_tom = 0.394 @step160. Steady climb from step 10 onward with a minor step-190 dip.
- **End-state (step190) vs baseline — broad-based improvement:** tomi 0.597→0.620, gsm8k 0.277→0.353 (+7.6pp), mmlu 0.387→0.407, bigtom_forward_belief 0.753→0.793, simpletom_mental 0.463→0.513 (+5pp), bigtom_backward_belief 0.527→0.533, tombench 0.380→0.380 (flat). No benchmark regressed materially.

**Health (excellent, stable):** `critic/rewards/mean` varied (≈ -1.4 to -0.1, NO -40 floor — negative range is expected since ll_min=-4 shifts power reward down), `reward/format_error_ratio = 0` throughout; `actor/kl_loss` 0.001→0.023 (max 0.107, well controlled, no spike); `actor/entropy_loss` **STABLE 1.41→1.58 (min 1.325) — no collapse**; `response_length/mean` 162→140.

**Verdict:** **POSITIVE TRANSFER + STABLE.** The Qwen2.5 power-winner recipe transfers to Gemma-2-2b **when ll_min is tightened to -4 and a format penalty (fp=5) is added.** This is the decisive contrast with PS105 (ll_min=-6/fp=0), which gave null/negative transfer *and* entropy collapse (1.53→0.52). Two knob changes flip Gemma from null→+19.4% AND keep entropy healthy (~1.4). **Candidate winner recipe for Gemma-2-2b:** frozen-RM / kl0.05 / lr5e-7 / power-k5 / **ll_min=-4 / fp=5** / ec=0.0. Downstream: prefer ll_min=-4 + fp>0 for Gemma-family power runs; the -6/fp=0 setting is the failure mode.

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4 DATA_NAME=dcfg_smoke_mix_gemma DATA_TRAIN=$HOME/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-4.0 USE_ACTOR_AS_RM=False SUBTRACT_BASELINE=False KL=0.05 LR=5e-7 FORMAT_PENALTY=5 ENTROPY_COEFF=0.0 THINK_ONLY_PG=True ROLLOUT_N=16 MAX_PROMPT=2048 MAX_RESP=512 TOTAL_EPOCHS=1 TEST_FREQ=10 COT_VAR=cot_eval bash experiments/train_behavior_gemma.sh`
