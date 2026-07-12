### Attempt r1 — 2026-07-11T17:22:04+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.01-n16-r1`
- **Host:** h100-076-003   **git:** `a329b31`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=actor baseline=False kl=0.01 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.01-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.01-n16-r1 \
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
    +reward_model.use_actor_as_rm=True \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.001_lp DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** WandB `iiwenshg`. Ran to completion (step 190, final validation logged). **Verdict: MIXED / brevity-drift — NOT healthy, but a distinct failure mode from length-inflation collapse.**

Config: actor-as-RM, kl=0.01, lr=1e-6, fp=0, ec=0.001, log_prob reward.

Training dynamics:
- `critic/rewards/mean`: −67.9 → −5.3 (climbs normally, expected).
- `response_length/mean`: **42.8 → 57.0** — stays SHORT, no inflation. Contrast: PS018 (actor kl0.01 lr5e-7) inflated 44→90; frozen collapses (PS002/PS005) ballooned to 140+. Higher LR under actor-RM drives **brevity**, not length-explosion (short outputs = fewer tokens to score = easier high per-token likelihood under actor-RM).
- `format_error_ratio`: 0.000 throughout.

Eval (subsample300, 26 benchmarks), step0 → step190:
| Benchmark | step0 | step190 | Δpp |
|---|---|---|---|
| **aggregate mean (26)** | 0.5075 | 0.4556 | **−5.2** |
| explore_tom | 0.480 | 0.747 | +26.7 |
| simpletom_judgment | 0.287 | 0.507 | +22.0 |
| opentom_attitude | 0.440 | 0.467 | +2.7 |
| simpletom_behavior | 0.553 | 0.550 | −0.3 |
| tomi | 0.590 | 0.570 | −2.0 |
| bigtom_forward_action | 0.737 | 0.713 | −2.4 |
| fantom_answerability_binary | 0.207 | 0.177 | −3.0 |
| bigtom_forward_belief | 0.767 | 0.690 | −7.7 |
| dyntom_type_a | 0.510 | 0.430 | −8.0 |

Interpretation & hypothesis test: The monitor asked "does higher LR under actor-RM collapse harder?" Answer: **no — it collapses *differently*.** Higher LR (1e-6) under actor-RM avoids the length-inflation seen in PS018 (lr5e-7) and instead drifts to very short (57-tok) responses. Net **aggregate eval regresses −5.2pp** despite large idiosyncratic gains on explore_tom/simpletom_judgment (likely artifacts of the brevity regime favoring terse answer formats on those benchmarks). Conclusion: **kl=0.01 is unstable regardless of RM mode or LR** — it produces either length-inflation or brevity-drift, both net-negative. PS011 (kl0.05 + fp5) remains the leading `Pm1swp_best` candidate. Note: process hung in ray/wandb teardown post-final-val (~10min) but all training+eval results were fully captured before the hang.

