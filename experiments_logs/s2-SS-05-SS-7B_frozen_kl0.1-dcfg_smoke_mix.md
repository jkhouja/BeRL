### Attempt r1 — 2026-07-21T06:04:13+00:00

- **RUN_NAME:** `s2-SS-05-SS-7B_frozen_kl0.1-dcfg_smoke_mix-Qwen2.5-7B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.1-n16-r1`
- **Host:** h100-077-004   **git:** `7d1de6b`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-7B-Instruct` (Qwen2.5-7B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=4 ll_min=-6 rm_mode=frozen baseline=False kl=0.1 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=1.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-SS-05-SS-7B_frozen_kl0.1-dcfg_smoke_mix-Qwen2.5-7B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.1-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260721/s2-SS-05-SS-7B_frozen_kl0.1-dcfg_smoke_mix-Qwen2.5-7B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.1-n16-r1.log`

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
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.1 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-SS-05-SS-7B_frozen_kl0.1-dcfg_smoke_mix-Qwen2.5-7B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.1-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-7B-Instruct \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=4 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=0.0 \
    reward_model.format_penalty_std_coef=1.0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=4 \
    +actor_rollout_ref.power_ll_min=-6 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=SS-7B_frozen_kl0.1 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-7B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-21T06:04:23+00:00

- **RUN_NAME:** `s2-SS-05-SS-7B_frozen_kl0.1-dcfg_smoke_mix-Qwen2.5-7B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.1-n16-r1`
- **Host:** h100-077-004   **git:** `7d1de6b`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-7B-Instruct` (Qwen2.5-7B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=4 ll_min=-6 rm_mode=frozen baseline=False kl=0.1 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=1.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-SS-05-SS-7B_frozen_kl0.1-dcfg_smoke_mix-Qwen2.5-7B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.1-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260721/s2-SS-05-SS-7B_frozen_kl0.1-dcfg_smoke_mix-Qwen2.5-7B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.1-n16-r1.log`

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
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.1 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-SS-05-SS-7B_frozen_kl0.1-dcfg_smoke_mix-Qwen2.5-7B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.1-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-7B-Instruct \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=4 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=0.0 \
    reward_model.format_penalty_std_coef=1.0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=4 \
    +actor_rollout_ref.power_ll_min=-6 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=SS-7B_frozen_kl0.1 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-7B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


---
## Findings (r1, h100-077-004, 2026-07-21)

`python scripts/score_run.py <log>`:
```
eval iters: 8 (step 0..190); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.3952  HM(last3)=0.4023  (baseline step0=0.3919)
ToM avg(last5)=0.5352  avg(last3)=0.5365  (baseline step0=0.5328)
gsm8k (separate): 0.6804 (step0=0.837, delta vs step0=-0.157)
mmlu  (separate): 0.7112 (step0=0.743, delta vs step0=-0.032)
health(final): kl=0.08 entropy=0.709 resp_len=84.561 reward=22.044 parseable=1.0 max_resp=512
ToM HM trajectory: 0:0.392 30:0.335 60:0.377 90:0.361 120:0.399 150:0.411 180:0.403 190:0.39
```

**Verdict: STABLE ✅ (primary objective met).** This row is a scale-stability screen — the
Q3-D6 3B-locked recipe caused off-3B KL blowups (7B/0.5B under actor-RM; Gemma frozen but KL~2.6).
Swapping the 7B to a **frozen 7B LM RM + KL=0.1** holds KL bounded: **actor/kl_loss max = 0.115
across the full 190-step run** (mean ~0.08), never approaching the 1.0 blowup threshold. Rank metric
KL_max<1.0 = PASS. Reward healthy (range −2.7→+30.6, final ~22), entropy stable ~0.71, resp_len ~85,
parseable=1.0, no collapse, no crash (final val at step 190, then normal Ray shutdown / GPUs→1MiB).

ToM signal near-flat: **d_avg = avg(last5) − step0 = 0.5352 − 0.5328 = +0.0024** (vs 3B smoke_mix
anchor +0.0215); HM(last5) +0.0033 over baseline. Capability regressions: **gsm8k −0.157** (notable
math drop), mmlu −0.032. So the frozen-RM/KL=0.1 config **unblocks a clean 7B point for the D6
scaling curve** (stability achieved) but yields little ToM gain and a math hit at this LR — the
value here is the stability fix, not a ToM win. max_resp=512.
