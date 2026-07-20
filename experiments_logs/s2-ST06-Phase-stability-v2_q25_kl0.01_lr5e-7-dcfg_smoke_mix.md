### Attempt r1 — 2026-07-19T19:45:06+00:00

- **RUN_NAME:** `s2-ST06-Phase-stability-v2_q25_kl0.01_lr5e-7-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.01-n16-r1`
- **Host:** h100-189-003   **git:** `e841f0e`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=4 ll_min=-6 rm_mode=actor baseline=False kl=0.01 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=1.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-ST06-Phase-stability-v2_q25_kl0.01_lr5e-7-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260719/s2-ST06-Phase-stability-v2_q25_kl0.01_lr5e-7-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.01-n16-r1.log`

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
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
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
    algorithm.kl_ctrl.kl_coef=0.01 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-ST06-Phase-stability-v2_q25_kl0.01_lr5e-7-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.01-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-v2_q25_kl0.01_lr5e-7 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-19T19:45:27+00:00

- **RUN_NAME:** `s2-ST06-Phase-stability-v2_q25_kl0.01_lr5e-7-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.01-n16-r1`
- **Host:** h100-189-003   **git:** `e841f0e`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=4 ll_min=-6 rm_mode=actor baseline=False kl=0.01 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=1.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-ST06-Phase-stability-v2_q25_kl0.01_lr5e-7-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260719/s2-ST06-Phase-stability-v2_q25_kl0.01_lr5e-7-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.01-n16-r1.log`

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
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
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
    algorithm.kl_ctrl.kl_coef=0.01 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-ST06-Phase-stability-v2_q25_kl0.01_lr5e-7-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.01-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-v2_q25_kl0.01_lr5e-7 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


---

### Findings (r1 authoritative) — completed 2026-07-19, owner h100-189-003

Run completed a full 1 epoch (189 update steps + end-of-training final validation at step 190),
process exited cleanly, no traceback/OOM, GPUs freed.

`scripts/score_run.py` output:
```
eval iters: 8 (step 0..190); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.461  HM(last3)=0.4589  (baseline step0=0.4179)
ToM avg(last5)=0.5187  avg(last3)=0.5175  (baseline step0=0.5027)
gsm8k (separate): 0.5928 (step0=0.657, delta vs step0=-0.064)
mmlu (separate): 0.5988 (step0=0.46, delta vs step0=+0.139)
health(final): kl=0.258 entropy=2.018 resp_len=104.412 reward=39.469 parseable=1.0 max_resp=512
ToM HM trajectory: 0:0.418 30:0.461 60:0.464 90:0.462 120:0.464 150:0.459 180:0.455 190:0.459
```

**Verdict:** kl=0.01 (low-KL stress corner) is **stable** under the merged v2 fixes for Qwen2.5-3B.
ToM HM rose +4.3pp over the step-0 baseline (0.418 -> 0.461), peaked ~0.464 at steps 60-120, then a
mild settle to 0.459 (stable plateau, **no collapse**). Format perfect throughout (parseable=1.0,
format_error_ratio=0.0). Reward/mean climbed -23 -> ~+39 (approaching the +40 clip, expected for
actor-as-RM). KL drifted up to ~0.26 and entropy rose to ~2.0 (low-KL run permits more drift) but
neither exploded; response_length stable ~100-120 tok. Capability regression modest: gsm8k -6.4pp,
mmlu +13.9pp. Overall: KL=0.01 remains usable under the fixes; comparable ToM gain to the anchor
family, no instability observed.
