### Attempt r1 — 2026-07-13T09:36:50+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-082-004   **git:** `32d0351`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7 ll_min=-4.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/1024 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1.log`

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
    actor_rollout_ref.actor.entropy_coeff=0.0 \
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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
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
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=7 \
    +actor_rollout_ref.power_ll_min=-4.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm4 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-13T09:37:00+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-082-004   **git:** `32d0351`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7 ll_min=-4.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/1024 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1.log`

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
    actor_rollout_ref.actor.entropy_coeff=0.0 \
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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
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
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=7 \
    +actor_rollout_ref.power_ll_min=-4.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm4 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


#### Findings (r1, 2026-07-13, h100-082-004) — COMPLETED (clean full run)
**Canonical score (`scripts/score_run.py`, 39 eval iters step0..190, 24 ToM benches excl gsm8k/mmlu):**
- ToM **HM(last5)=0.152** HM(last3)=0.1725 (baseline step0=0.093)
- ToM **avg(last5)=0.380** avg(last3)=0.391 (baseline step0=0.315) — **+6.5pp**
- gsm8k (separate): 0.336 (step0=0.277, **Δ=+0.059**)
- mmlu (separate): 0.438 (step0=0.387, **Δ=+0.051**)
- health(final): kl=0.042 entropy=1.49 resp_len=113.1 reward=-1.39 **parseable=1.0** (no resp_len drift, no OOM)

**Per-benchmark step0→final(step190, subsample300):** tomi 0.597→0.613 · bigtom_fb 0.753→0.833 ·
bigtom_bb 0.527→0.557 · hi_tom 0.130→0.207 · explore_tom 0.363→0.483 ·
simpletom_mental 0.463→0.560 · opentom_location_fo 0.510→0.613.

**Verdict: POSITIVE TRANSFER (moderate).** Actor-RM power (k=7, ll_min=−4) on Gemma-2 improves ToM
broadly (avg +6.5pp) with no reward hacking (parseable=1.0, evals rose alongside reward) and modest
capability gains (gsm8k +5.9pp, mmlu +5.1pp). Ran the full 191 steps cleanly — **no OOM** (resp_len
stayed ~110, never drifted toward the 1024 cap, unlike the k5/ll_min−6 sibling PS124 which OOM'd near
the end). **Comparison:** weaker than **PS124** (k=5, ll_min=−6, actor-RM: avg +10.3pp, HM ~2×)
⇒ for Gemma-2 actor-RM power the **k=5 / ll_min=−6** reward shaping transfers better than **k=7 /
ll_min=−4**. Both actor-RM configs are positive, reinforcing the actor-RM ≫ frozen-RM (PS107 negative)
rule for Gemma-2 power. HM trajectory is noisy mid-run (Gemma HM is degenerate — floored by near-zero
fantom-list benches; use ToM avg as the Gemma selector) but recovers to ~0.19 in the last evals.

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm4 DATA_NAME=dcfg_smoke_mix_gemma DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet REWARD_TYPE=power POWER_K=7 POWER_LL_MIN=-4.0 USE_ACTOR_AS_RM=True KL=0.05 LR=5e-7 FORMAT_PENALTY=5 ENTROPY_COEFF=0.0 TEST_FREQ=5 TOTAL_EPOCHS=1 RUN_INDEX=2 bash experiments/train_behavior_gemma.sh`
