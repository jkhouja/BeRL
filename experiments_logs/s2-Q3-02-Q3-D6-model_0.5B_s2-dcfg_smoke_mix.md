### Attempt r1 — 2026-07-20T21:16:12+00:00

- **RUN_NAME:** `s2-Q3-02-Q3-D6-model_0.5B_s2-dcfg_smoke_mix-Qwen2.5-0.5B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-156-003   **git:** `9e9bed3`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` (Qwen2.5-0.5B-Instruct)
- **Data (train):** `data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=4 ll_min=-6 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=1.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-Q3-02-Q3-D6-model_0.5B_s2-dcfg_smoke_mix-Qwen2.5-0.5B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1 — https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/fe7zg0x7
- **Log path:** `logs/20260720/s2-Q3-02-Q3-D6-model_0.5B_s2-dcfg_smoke_mix-Qwen2.5-0.5B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/dcfg_smoke_mix.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
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
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-Q3-02-Q3-D6-model_0.5B_s2-dcfg_smoke_mix-Qwen2.5-0.5B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
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

**How to rerun:** `EXP_ID=Q3-D6-model_0.5B_s2 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct DATA_TRAIN=data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-20T21:16:29+00:00

- **RUN_NAME:** `s2-Q3-02-Q3-D6-model_0.5B_s2-dcfg_smoke_mix-Qwen2.5-0.5B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-156-003   **git:** `9e9bed3`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` (Qwen2.5-0.5B-Instruct)
- **Data (train):** `data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=4 ll_min=-6 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=1.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-Q3-02-Q3-D6-model_0.5B_s2-dcfg_smoke_mix-Qwen2.5-0.5B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1 — https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/fe7zg0x7
- **Log path:** `logs/20260720/s2-Q3-02-Q3-D6-model_0.5B_s2-dcfg_smoke_mix-Qwen2.5-0.5B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/dcfg_smoke_mix.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
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
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-Q3-02-Q3-D6-model_0.5B_s2-dcfg_smoke_mix-Qwen2.5-0.5B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
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

**How to rerun:** `EXP_ID=Q3-D6-model_0.5B_s2 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct DATA_TRAIN=data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


---

## Findings (r1) — Completed 2026-07-20

**Run:** Qwen2.5-0.5B-Instruct · smoke_mix (dcfg_smoke_mix, 190 steps = 1 epoch) · locked Qwen2.5 recipe (power k4, ll_min-6, actor-RM, no-baseline, kl0.05, lr5e-7, n16, fp0, ec0). Q3-D6 model-scaling replicate 2 (s2).

```
eval iters: 8 (step 0..190); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.012  HM(last3)=0.0041  (baseline step0=0.0032)
ToM avg(last5)=0.1174  avg(last3)=0.0902  (baseline step0=0.055)
gsm8k (separate): 0.0508 (step0=0.0, delta vs step0=+0.051)
mmlu (separate): 0.2318 (step0=0.033, delta vs step0=+0.199)
health(final): kl=0.344 entropy=1.364 resp_len=45.426 reward=35.077 parseable=1.0 max_resp=512
ToM HM trajectory: 0:0.003 30:0.004 60:0.004 90:0.012 120:0.01 150:0.005 180:0.003 190:0.004
```

**d_avg (ranking metric) = avg(last5) − avg(step0) = 0.1174 − 0.055 = +0.0624** vs 3B anchor +0.0215 (ST01/10/28, SD0.001).

**Verdict:** 0.5B model-scale point (replicate 2). Absolute ToM is near-floor — many of the 24 ToM benchmarks read ~0 at 0.5B, which tanks the avg-then-HM harmonic mean (HM≈0.012, statistically noise). The arithmetic **avg** is the meaningful scale signal here; the BeRL delta d_avg=+0.062 is present but on a very low base (avg 0.117 vs ~0.52 at 3B). HM peaks step90 (0.012) then erodes back to baseline by step190. Health fully clean: kl 0.344 (elevated but no runaway, distinct from Gemma blowups), parseable=1.0, no collapse, resp_len 45 (short answers). gsm8k/mmlu both improve off a near-zero 0.5B base. Pairs with Q3-01 (s1) for the 0.5B mean±SD in the D6 model-scaling law.

**Rerun one-liner:**
```
setsid bash -c "source ~/.bashrc; conda activate tom; RUN_STAGE=s2 EXP_NUM=Q3-02 EXP_ID=Q3-D6-model_0.5B_s2 DATA_NAME=dcfg_smoke_mix DATA_TRAIN=data/dcfg_smoke_mix.parquet MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct REWARD_TYPE=power POWER_K=4 POWER_LL_MIN=-6 USE_ACTOR_AS_RM=True SUBTRACT_BASELINE=False KL=0.05 LR=5e-7 TRAIN_BATCH=32 MINI_BATCH=128 MICRO_BATCH=8 ROLLOUT_N=16 MAX_PROMPT=2048 MAX_RESP=512 ENTROPY_COEFF=0.0 FORMAT_PENALTY=0.0 FORMAT_PENALTY_STD_COEF=1.0 SEED=1 RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh"
```
