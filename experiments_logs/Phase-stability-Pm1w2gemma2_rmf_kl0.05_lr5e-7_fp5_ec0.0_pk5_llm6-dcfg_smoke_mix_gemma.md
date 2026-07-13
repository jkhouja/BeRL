### Attempt r1 — 2026-07-12T21:27:22+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-082-004   **git:** `719757a`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/1024 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-12T21:28:07+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-082-004   **git:** `719757a`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/1024 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


#### Debugging note (r1, 2026-07-12, h100-082-004)
- **Blocker context:** This tag-free Gemma-2 power/frozen-RM config previously produced 100% INVALID
  rewards (-40 sentinel, min=max=mean, zero advantage/gradient) — see PS097/PS105. Root cause:
  the old invalid gate flagged tag-free rollouts (no literal `</think>`) as invalid.
- **Fix applied (local tree):** model-aware `ModelResponseParser.is_invalid_response` in
  `response_parser.py` + both gates in `fsdp_workers.py` (~L709, ~L1374). Tag-free (Gemma2/Qwen3)
  rollouts now valid.
- **Verified live (r1):** first PPO steps show `reward/mean≈+5.3→+5.8`, `critic/rewards` range
  −45..+40 (real variance), `advantages/max≈3.7`, `pg_loss≈−0.018`, `grad_norm≈0.33`. Learning
  signal restored — blocker resolved. WandB run v16twu8b.

#### Findings (r1, 2026-07-13, h100-082-004) — COMPLETED
**Canonical score (`scripts/score_run.py`, 39 eval iters, 24 ToM benches excl gsm8k/mmlu):**
- ToM **HM(last5)=0.0267** HM(last3)=0.0248 (baseline step0=0.0931) — regressed
- ToM **avg(last5)=0.2422** avg(last3)=0.2409 (baseline step0=0.3151) — regressed −7.3pp
- gsm8k (separate): 0.4188 (step0=0.277, **Δ=+0.142**) — capability improved
- mmlu (separate): 0.3474 (step0=0.39, Δ=−0.043)
- health(final): kl=0.081 entropy=0.155 resp_len=523.8 reward=10.58 **parseable=1.0**

**Per-benchmark step0→final (subsample300):** tomi 0.597→0.530 · bigtom_fb 0.753→0.640 ·
bigtom_bb 0.530→0.467 · hi_tom 0.130→0.070 · explore_tom 0.360→0.193 ·
fantom_answerability_binary 0.257→0.193 · opentom_location_fo 0.510→0.470 ·
simpletom_mental 0.463→0.280.

**Verdict: NEGATIVE TRANSFER (Completed).** Training was healthy — the invalid-detection fix
restored a real learning signal (reward 0→~10.6, 100% parseable, no reward-hacking collapse: eval
dipped hard then partially recovered but ended below step0). Yet the behavior-prediction **power**
reward (k=5, ll_min=−6, frozen-RM) — the Qwen2.5 sweep winner (PS074) — **degrades ToM on Gemma-2**
(avg −7.3pp; HM unreliable here, floored by fantom-list benches as noted for PS098). gsm8k rose
(+14pp) so the model learned format/reasoning but not ToM. Contrast: log_prob Gemma-2 runs
(PS098/PS104) gave small **positive** avg transfer. **Downstream implication:** confirms the
per-family rule — do not port the Qwen2.5 power recipe to Gemma-2; Gemma-2 favors log_prob. HM is
not a reliable selector for Gemma-2 (use ToM avg).

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm6 DATA_NAME=dcfg_smoke_mix_gemma DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-6.0 USE_ACTOR_AS_RM=False KL=0.05 LR=5e-7 FORMAT_PENALTY=5 ENTROPY_COEFF=0.0 TEST_FREQ=5 TOTAL_EPOCHS=1 RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`
