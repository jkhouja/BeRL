### Attempt r1 — 2026-07-11T04:49:38+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp0_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.01-n16-r1`
- **Host:** h100-076-003   **git:** `8c953da`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=frozen baseline=False kl=0.01 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp0_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp0_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.01-n16-r1.log`

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
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.think_only_pg=True \
    actor_rollout_ref.actor.format_penalty=0 \
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
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp0_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.01-n16-r1 \
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
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=log_prob \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp0_ec0.001_lp DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**WandB link:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/b20a3o01

**Findings (r1, completed 2026-07-11 ~06:39 UTC, 190 steps / 1 epoch):**

- **Verdict: COLLAPSED — reward-hacked.** Behavior-LL reward climbed steadily (`reward/mean` −53.9 @step1 → ~−2 to −7 @step119) while **every** eval benchmark crashed. Config is *not viable* for the sweep; HM-over-last-X is far below baseline. Wave-2 should skip this cell.
- **Health/hacking:** `response_length/mean` drifted up 62 → ~140 tokens; generations show degenerate repetition ("She would be aware that the pot was not yet fired…" repeated; rambling math CoT). Classic behavior-LL reward-hacking: generic/repetitive CoT raises the held-out utterance likelihood but destroys actual ToM accuracy. `grad_norm` stayed finite (~0.5–4), no NaN/OOM — collapse is behavioral, not numerical.
- **Eval (subsample300), step 0 (baseline) → step 190 (final), pp deltas:**

  | Benchmark | step0 | step190 | Δpp |
  |---|---|---|---|
  | tomi | 0.590 | 0.290 | −30.0 |
  | hi_tom | 0.243 | 0.067 | −17.6 |
  | simpletom_mental | 0.853 | 0.647 | −20.6 |
  | simpletom_judgment | 0.290 | 0.150 | −14.0 |
  | bigtom_forward_belief | 0.767 | 0.573 | −19.4 |
  | bigtom_forward_action | 0.740 | 0.560 | −18.0 |
  | bigtom_backward_belief | 0.633 | 0.530 | −10.3 |
  | fantom_belief_mc | 0.507 | 0.123 | −38.4 |
  | opentom_multihop_so | 0.480 | 0.223 | −25.7 |
  | dyntom_type_a | 0.510 | 0.143 | −36.7 |
  | explore_tom | 0.483 | 0.217 | −26.6 |
  | tombench | 0.607 | 0.420 | −18.7 |
  | gsm8k | 0.663 | 0.300 | −36.3 |
  | mmlu | 0.477 | 0.470 | −0.7 |

  Universal, severe degradation across ToM + reasoning (only mmlu ~flat). Downstream implication: `rm=frozen, kl=0.01, lr=5e-7, fp=0, ec=0.001, reward=log_prob` collapses on smoke_mix → exclude from best-config selection.

