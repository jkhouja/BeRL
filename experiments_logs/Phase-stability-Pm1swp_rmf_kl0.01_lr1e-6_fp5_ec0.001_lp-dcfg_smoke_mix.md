### Attempt r1 — 2026-07-11T09:02:28+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp5_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.01-n16-r1`
- **Host:** h100-076-003   **git:** `3c14990`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=frozen baseline=False kl=0.01 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp5_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp5_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.01-n16-r1.log`

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
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
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
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp5_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.01-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp5_ec0.001_lp DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**WandB link:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/vyklqeep

**Findings (r1, completed 2026-07-11 ~11:00 UTC, 190 steps / 1 epoch):**

- **Verdict: PARTIAL COLLAPSE — fp=5 format penalty markedly mitigates the reward-hacking** seen in PS002/PS005 (both fp=0). Reward still climbed (`reward/mean` −70.9 @step1 → ~−2 to −7), `format_error_ratio` stayed 0.0 (well-formed), but ToM degradation is far milder and several benchmarks are *preserved or improved*.
- **Health:** `response_length/mean` still grew 48 → ~141 tokens, but outputs stayed well-formed (no degenerate tag breakage). Format penalty curbs the worst hacking without fully stopping length growth. gsm8k still collapses hard (math reasoning most fragile).
- **Eval (subsample300), step 0 → step 190, pp deltas:**

  | Benchmark | step0 | step190 | Δpp |
  |---|---|---|---|
  | tomi | 0.583 | 0.413 | −17.0 |
  | hi_tom | 0.250 | 0.103 | −14.7 |
  | simpletom_mental | 0.853 | 0.857 | +0.4 |
  | simpletom_judgment | 0.287 | 0.360 | +7.3 |
  | simpletom_behavior | 0.550 | 0.533 | −1.7 |
  | bigtom_forward_belief | 0.767 | 0.653 | −11.4 |
  | bigtom_forward_action | 0.733 | 0.477 | −25.6 |
  | bigtom_backward_belief | 0.633 | 0.617 | −1.6 |
  | fantom_belief_mc | 0.500 | 0.400 | −10.0 |
  | opentom_multihop_so | 0.483 | 0.333 | −15.0 |
  | dyntom_type_a | 0.507 | 0.343 | −16.4 |
  | explore_tom | 0.480 | 0.547 | +6.7 |
  | tombench | 0.607 | 0.610 | +0.3 |
  | gsm8k | 0.660 | 0.093 | −56.7 |
  | mmlu | 0.467 | 0.590 | +12.3 |

  Downstream: `fp=5` is the key stabilizer vs fp=0 — preserves several ToM benchmarks (simpletom_mental, bigtom_backward_belief, tombench, explore_tom) and lifts mmlu, but net ToM still negative (tomi/bigtom_forward_action/dyntom drop 15–26pp) and gsm8k unusable. Still below a "climbs cleanly" bar, but the fp=5 vs fp=0 contrast is the clearest signal so far: **format penalty is necessary but not sufficient** for plain log_prob@kl0.01.

