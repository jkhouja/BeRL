### Attempt r1 — 2026-07-11T05:22:16+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp5_ec0.0_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.01-n16-r1`
- **Host:** h100-013-002   **git:** `8c953da`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=frozen baseline=False kl=0.01 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp5_ec0.0_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp5_ec0.0_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.01-n16-r1.log`

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
    actor_rollout_ref.actor.think_only_pg=True \
    actor_rollout_ref.actor.format_penalty=5 \
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
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp5_ec0.0_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.01-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=999 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-3B-Instruct \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=log_prob \
    +reward_model.power_k=2.0 \
    +reward_model.power_ll_min=-8.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=log_prob \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp5_ec0.0_lp DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings (attempt -r1, completed 2026-07-11, Owner h100-013-002):**

Full 191-step (1-epoch) run completed cleanly; 20 evals (every 10 steps) on `subsample300`.
WandB: https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/3ozyf3ly

Config-selection metric = harmonic mean over the ToM benchmarks (excludes gsm8k/mmlu controls),
tracked per eval; parseable-answer rate tracked separately.

| step | HM_tom | tomi | hi_tom | simpletom_mental | gsm8k | mmlu | fmt_err |
|-----:|-------:|-----:|-------:|-----------------:|------:|-----:|--------:|
|   0  | 0.413  | 0.590| 0.243  | 0.853            | 0.657 | 0.477| —       |
|  20  | 0.457  | 0.650| 0.390  | 0.867            | 0.693 | 0.593| 0.000   |
|  50  | 0.466  | 0.600| 0.377  | 0.903            | 0.660 | 0.607| 0.000   |
|  90  | 0.371  | 0.467| 0.273  | 0.687            | 0.487 | 0.497| 0.000   |
| 130  | 0.303  | 0.420| 0.187  | 0.633            | 0.453 | 0.510| 0.000   |
| 190  | 0.370  | 0.517| 0.213  | 0.793            | 0.553 | 0.590| —       |

- **Baseline (step 0) HM_tom = 0.413; peak = 0.466 @ step 50 (+5.3pp); final = 0.370 (−4.4pp).**
- **Config-selection score = HM over last-5 evals (steps 150–190) = 0.319 — BELOW baseline (0.413).**
- **Health:** `reward/format_error_ratio = 0.000` at every step → ~100% parseable answers, no format
  collapse or reward hacking. `reward/mean` stays negative (expected for raw log-prob reward), no floor collapse.
- **Verdict:** climbs early (peak ~step 50) then steadily degrades over the horizon → net-negative by
  end; not a stable-climbing cell. Health is clean but ToM accuracy regresses. **Not a Phase −1 winner.**
- **How to rerun:** see command above (`ONLY_IDX="3" bash experiments/phase_stability_sweep.sh`).

