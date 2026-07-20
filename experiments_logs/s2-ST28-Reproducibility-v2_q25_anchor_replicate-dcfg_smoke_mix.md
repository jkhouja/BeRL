### Attempt r1 — 2026-07-20T00:49:02+00:00

- **RUN_NAME:** `s2-ST28-Reproducibility-v2_q25_anchor_replicate-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-077-004   **git:** `e841f0e`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=4 ll_min=-6 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=1.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-ST28-Reproducibility-v2_q25_anchor_replicate-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260720/s2-ST28-Reproducibility-v2_q25_anchor_replicate-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=s2-ST28-Reproducibility-v2_q25_anchor_replicate-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1 \
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

**How to rerun:** `EXP_ID=Reproducibility-v2_q25_anchor_replicate DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


---

**Findings (r1 = the real launch; a BERL_DRY_RUN preview block may precede it). Completed 2026-07-20 ~01:57 UTC, 190 steps / 1 epoch, exited clean.**

`scripts/score_run.py` output:
```
eval iters: 8 (step 0..190); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.4670  HM(last3)=0.4676  (baseline step0=0.4172)
ToM avg(last5)=0.5255  avg(last3)=0.5267  (baseline step0=0.504)
gsm8k (separate): 0.6426 (step0=0.66,  delta vs step0=-0.017)
mmlu  (separate): 0.6246 (step0=0.473, delta vs step0=+0.152)
health(final): kl=0.069 entropy=1.452 resp_len=109.066 reward=35.463 parseable=1.0 max_resp=512
ToM HM trajectory: 0:0.417 30:0.463 60:0.461 90:0.471 120:0.459 150:0.471 180:0.467 190:0.463
```

- **Verdict: reproducible anchor.** Verbatim ST01-config replicate (Qwen2.5-3B, power k4 ll_min=-6,
  actor-RM, kl=0.05, lr=5e-7, ec=0.0, fp=0, max_resp=512). Stable, healthy, no collapse/hacking.
- **Reproducibility:** HM(last5)=0.467 — essentially identical to the sibling anchor runs
  (ST09 flat-fp5 HM(last5)=0.461; ST12 batch64 HM(last3)=0.460). ToM HM held ~0.46-0.47 across all
  eval iters (0.463->0.471 band), i.e. flat/stable, tight run-to-run agreement at seed=1.
- **ToM:** HM +5.0pp over step-0 (0.417->0.467); avg +2.3pp (0.504->0.527).
- **Capability:** gsm8k Δ-0.017 (within noise, no meaningful regression); mmlu Δ+0.152 (format-pass
  driven, mmlu format_pass rose over training).
- **Health:** reward/mean climbed -18->~35 (near cap 40), kl_loss steady ~0.07-0.12, entropy healthy
  ~1.45, response_length 56->~109, parseable=1.0 throughout, format_error_ratio=0.
- **Paper note:** for the N=3 anchor error bar the tracker asks for `d_cavg` (format-controlled
  conditional accuracy delta, via `scripts/reassess_runs.py`), not the HM reported here — run
  reassess_runs.py on ST01/ST10/ST28 to firm up SD(d_cavg). This run's HM confirms the anchor is
  stable and reproducible.
