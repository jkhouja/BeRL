### Attempt r1 — 2026-07-19T21:08:35+00:00

- **RUN_NAME:** `s2-ST12-Phase-stability-v2_q25_batch64-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-077-004   **git:** `e841f0e`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=4 ll_min=-6 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=1.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-ST12-Phase-stability-v2_q25_batch64-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260719/s2-ST12-Phase-stability-v2_q25_batch64-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/dcfg_smoke_mix.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=64 \
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
    trainer.experiment_name=s2-ST12-Phase-stability-v2_q25_batch64-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-v2_q25_batch64 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


---

**Findings (r1, completed 2026-07-19 ~21:57 UTC, 95 steps / 1 epoch @ train_batch=64, exited clean).**

`scripts/score_run.py` output:
```
eval iters: 5 (step 0..95); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.4563  HM(last3)=0.4602  (baseline step0=0.4209)
ToM avg(last5)=0.5167  avg(last3)=0.5199  (baseline step0=0.5038)
gsm8k (separate): 0.6678 (step0=0.657, delta vs step0=+0.011)
mmlu  (separate): 0.5832 (step0=0.463, delta vs step0=+0.120)
health(final): kl=0.13 entropy=1.414 resp_len=110.476 reward=35.197 parseable=1.0 max_resp=512
ToM HM trajectory: 0:0.421 30:0.465 60:0.463 90:0.46 95:0.457
```

- **Verdict:** Stable, healthy, no collapse or reward-hacking. train_batch=64 (vs 32 anchor ST01),
  else Qwen2.5-3B PA anchor: power k4 ll_min=-6, actor-RM, kl=0.05, lr=5e-7, fp=0, max_resp=512.
- **Caveat on HM(last5):** only 5 eval iters exist (0/30/60/90/95), so "last5" includes the step-0
  baseline (0.421) and understates the plateau. **HM(last3)=0.460** (steps 60/90/95) is the fair
  number and matches ST09's plateau (~0.46) and the anchor trajectory.
- **ToM:** HM(last3) +3.9pp over step-0 (0.421->0.460); avg(last3) +1.6pp (0.504->0.520). HM jumps by
  step 30 then plateaus ~0.46 — same shape as batch=32.
- **Capability:** no gsm8k regression (delta +0.011); mmlu up (delta +0.120), mostly format_pass gain.
- **Health:** reward/mean -18->~35 (near cap 40); kl_loss rose gently to ~0.13 (still controlled),
  entropy healthy ~1.41, response_length 56->~110, parseable=1.0 throughout, format_error_ratio=0.
- **Conclusion:** Larger batch (64) is as stable and effective as 32 on Qwen2.5-3B, with ~2x fewer
  optimizer steps per epoch (95 vs 190) at similar wall-clock per step (~higher throughput). No
  stability cost from doubling the batch.
