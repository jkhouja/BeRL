### Attempt r1 — 2026-07-19T19:42:23+00:00

- **RUN_NAME:** `s2-ST03-Phase-stability-v2_q25_reward_negppl-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-neg_perplexity-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-076-003   **git:** `e841f0e`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=neg_perplexity power_k=2.0 ll_min=-8.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=0.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-ST03-Phase-stability-v2_q25_reward_negppl-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-neg_perplexity-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260719/s2-ST03-Phase-stability-v2_q25_reward_negppl-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-neg_perplexity-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
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
    actor_rollout_ref.actor.format_penalty=0.0 \
    actor_rollout_ref.actor.format_penalty_std_coef=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-ST03-Phase-stability-v2_q25_reward_negppl-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-neg_perplexity-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-3B-Instruct \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=neg_perplexity \
    +reward_model.power_k=2.0 \
    +reward_model.power_ll_min=-8.0 \
    reward_model.format_penalty=0.0 \
    reward_model.format_penalty_std_coef=0.0 \
    +actor_rollout_ref.reward_type=neg_perplexity \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-v2_q25_reward_negppl DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_

> **NOTE:** The three "Attempt r1" blocks below are launcher-emitted duplicates (two `BERL_DRY_RUN=1`
> previews + the real launch, all within the same second). There was a **single** actual run, the
> authoritative attempt is **r1** (WandB `axc03qdr`). Consolidated findings are at the bottom of this file.

### Attempt r1 — 2026-07-19T19:42:38+00:00

- **RUN_NAME:** `s2-ST03-Phase-stability-v2_q25_reward_negppl-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-neg_perplexity-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-076-003   **git:** `e841f0e`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=neg_perplexity power_k=2.0 ll_min=-8.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=0.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-ST03-Phase-stability-v2_q25_reward_negppl-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-neg_perplexity-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260719/s2-ST03-Phase-stability-v2_q25_reward_negppl-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-neg_perplexity-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
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
    actor_rollout_ref.actor.format_penalty=0.0 \
    actor_rollout_ref.actor.format_penalty_std_coef=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-ST03-Phase-stability-v2_q25_reward_negppl-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-neg_perplexity-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-3B-Instruct \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=neg_perplexity \
    +reward_model.power_k=2.0 \
    +reward_model.power_ll_min=-8.0 \
    reward_model.format_penalty=0.0 \
    reward_model.format_penalty_std_coef=0.0 \
    +actor_rollout_ref.reward_type=neg_perplexity \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-v2_q25_reward_negppl DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-19T19:42:47+00:00

- **RUN_NAME:** `s2-ST03-Phase-stability-v2_q25_reward_negppl-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-neg_perplexity-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-076-003   **git:** `e841f0e`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=neg_perplexity power_k=2.0 ll_min=-8.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=0.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-ST03-Phase-stability-v2_q25_reward_negppl-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-neg_perplexity-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260719/s2-ST03-Phase-stability-v2_q25_reward_negppl-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-neg_perplexity-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
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
    actor_rollout_ref.actor.format_penalty=0.0 \
    actor_rollout_ref.actor.format_penalty_std_coef=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-ST03-Phase-stability-v2_q25_reward_negppl-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-neg_perplexity-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-3B-Instruct \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=neg_perplexity \
    +reward_model.power_k=2.0 \
    +reward_model.power_ll_min=-8.0 \
    reward_model.format_penalty=0.0 \
    reward_model.format_penalty_std_coef=0.0 \
    +actor_rollout_ref.reward_type=neg_perplexity \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-v2_q25_reward_negppl DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


---

## Consolidated Findings (authoritative — attempt r1, WandB `axc03qdr`)

- **WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/axc03qdr
- **Log:** `logs/20260719/s2-ST03-Phase-stability-v2_q25_reward_negppl-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-neg_perplexity-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1.log`
- **Run:** 190 steps (1 epoch, 6100 rows / batch 32), 8 eval iters (step 0..190), ~18s/step.

**Hypothesis.** neg_perplexity reward (length-normalized) was never in the old sweep; test whether it
trains stably under the merged v2 fixes and how ToM transfer compares to the power/log_prob families.

### Canonical scores (`scripts/score_run.py`)
```
eval iters: 8 (step 0..190); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.4376  HM(last3)=0.4336  (baseline step0=0.4192)
ToM avg(last5)=0.5004  avg(last3)=0.4978  (baseline step0=0.5032)
gsm8k (separate): 0.5392 (step0=0.667, delta vs step0=-0.128)
mmlu  (separate): 0.594  (step0=0.47,  delta vs step0=+0.124)
health(final): kl=0.257 entropy=2.162 resp_len=104.488 reward=-18.36 parseable=1.0 max_resp=512
ToM HM trajectory: 0:0.419 30:0.46 60:0.454 90:0.449 120:0.436 150:0.437 180:0.431 190:0.431
```

### Verdict
- **Stable, no collapse, no reward-hacking.** parseable=1.0 throughout; format_pass ~99-100% on every
  subtype at all eval iters; KL drifts modestly (kl_loss 0.03→~0.25, coef held 0.05); entropy rises
  gently 1.1→2.16; resp_len steady ~100. reward/mean improves −54→−18 (RM = actor, moving target).
- **ToM transfer: marginal.** HM(last5)=0.4376 is only +0.018 over the step-0 baseline (0.4192), and
  ToM **avg** is essentially flat (0.5004 vs 0.5032). HM peaks early (step 30 = 0.46) then declines to
  ~0.43 — i.e. best transfer is in the first ~30 steps, gently eroding thereafter.
- **Capability regression:** gsm8k −0.128 (math reasoning drops), mmlu +0.124 (knowledge up) —
  mixed, net a real gsm8k hit typical of behavior-reward drift.
- **Implication for Phase −1 v2 (reward-family axis):** neg_perplexity is *stable* but a *weak* ToM
  driver on Qwen2.5-3B (small HM gain, early-peaking, gsm8k cost). Compare directly against ST01
  (power k4 anchor), ST02 (log_prob), ST04 (power k6) once those complete to rank the reward family.
