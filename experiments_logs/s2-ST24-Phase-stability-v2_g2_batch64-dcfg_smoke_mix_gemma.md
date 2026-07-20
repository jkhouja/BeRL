### Attempt r1 — 2026-07-20T00:03:37+00:00

- **RUN_NAME:** `s2-ST24-Phase-stability-v2_g2_batch64-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-117-001   **git:** `e841f0e`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=4 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=1.0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-ST24-Phase-stability-v2_g2_batch64-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260720/s2-ST24-Phase-stability-v2_g2_batch64-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=64 \
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
    actor_rollout_ref.actor.format_penalty=0.0 \
    actor_rollout_ref.actor.format_penalty_std_coef=1.0 \
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
    trainer.experiment_name=s2-ST24-Phase-stability-v2_g2_batch64-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.power_k=4 \
    +reward_model.power_ll_min=-4 \
    reward_model.format_penalty=0.0 \
    reward_model.format_penalty_std_coef=1.0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=4 \
    +actor_rollout_ref.power_ll_min=-4 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-v2_g2_batch64 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-20T00:03:48+00:00

- **RUN_NAME:** `s2-ST24-Phase-stability-v2_g2_batch64-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-117-001   **git:** `e841f0e`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=4 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=1.0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-ST24-Phase-stability-v2_g2_batch64-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260720/s2-ST24-Phase-stability-v2_g2_batch64-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=64 \
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
    actor_rollout_ref.actor.format_penalty=0.0 \
    actor_rollout_ref.actor.format_penalty_std_coef=1.0 \
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
    trainer.experiment_name=s2-ST24-Phase-stability-v2_g2_batch64-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.power_k=4 \
    +reward_model.power_ll_min=-4 \
    reward_model.format_penalty=0.0 \
    reward_model.format_penalty_std_coef=1.0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=4 \
    +actor_rollout_ref.power_ll_min=-4 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-v2_g2_batch64 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


- **WandB link (resolved):** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/xd49hw84

---

## Findings — r1 (2026-07-20, Completed)

**Canonical score (`scripts/score_run.py`, 5 eval iters, steps 0..95; 24 ToM benchmarks excl gsm8k/mmlu):**
```
ToM HM(last5)=0.0708  HM(last3)=0.0733  (baseline step0=0.093)
ToM avg(last5)=0.3222 avg(last3)=0.3358 (baseline step0=0.3152)
gsm8k (separate): 0.288   (step0=0.28,  Δ=+0.008)
mmlu  (separate): 0.3914  (step0=0.383, Δ=+0.008)
health(final): kl=0.027 entropy=1.565 resp_len=118.2 reward=-0.696 parseable=1.0 max_resp=512
ToM HM trajectory: 0:0.093 30:0.019 60:0.064 90:0.064 95:0.084
```

**Verdict:** batch64 is **training-stable** on gemma frozen-RM with the power reward — KL stayed
clean (~0.013–0.027 throughout, no spikes), entropy ~1.5–1.6, format_error 0.0, no clipping, no
collapse; the larger batch behaved smoothly. **However, the power reward produced NO ToM improvement
on gemma**: ToM avg stayed flat (0.315→~0.33) and the HM actually *fell below* the step-0 baseline
(0.093→0.019 at step 30, recovering only to 0.084 by step 95). gsm8k/mmlu essentially unchanged
(+0.008 each).

**Cross-run contrast (key finding):** compare with ST14 (gemma, same family, **log_prob** reward),
which reached ToM HM=0.294 / avg=0.465 — a large lift — albeit with KL volatility. So on the gemma
family the reward choice dominates the batch-size axis: **log_prob drives real ToM/format gains but
is KL-volatile; power (k4) is KL-clean but gives ~zero ToM lift.** The batch-size increase (32→64)
did not itself destabilise or help. This points to a reward-vs-stability trade-off for gemma that the
power anchor (ST13) and neg_perplexity (ST15) rows should further triangulate; a stable *and*
effective gemma recipe may need log_prob with a tightened KL/entropy schedule rather than power.

**How to rerun:**
`RUN_STAGE=s2 EXP_NUM=ST24 EXP_ID=Phase-stability-v2_g2_batch64 DATA_NAME=dcfg_smoke_mix_gemma DATA_TRAIN=$HOME/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet REWARD_TYPE=power POWER_K=4 POWER_LL_MIN=-4 USE_ACTOR_AS_RM=False SUBTRACT_BASELINE=False KL=0.05 LR=5e-7 TRAIN_BATCH=64 MINI_BATCH=128 MICRO_BATCH=8 ROLLOUT_N=16 MAX_PROMPT=2048 MAX_RESP=512 ENTROPY_COEFF=0.001 FORMAT_PENALTY=0.0 FORMAT_PENALTY_STD_COEF=1.0 NUM_GPUS=8 TP_SIZE=2 bash experiments/train_behavior_gemma.sh`
