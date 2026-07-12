### Attempt r1 — 2026-07-11T20:53:31+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1`
- **Host:** h100-076-003   **git:** `5b2c335`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=3 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.01 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1 \
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
    +reward_model.reward_type=power \
    +reward_model.power_k=3 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=3 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp5_ec0.001_pk3 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** WandB `e7v0b9i1`. Ran to completion (step 190, final validation logged); clean exit, GPUs freed to 1 MiB (no teardown hang). **Verdict: COLLAPSED via length-inflation — the power-law reward did NOT stabilize low-KL training; it made it WORSE.**

Config: frozen RM, kl=0.01, lr=5e-7, fp=5, ec=0.001, **reward=power (k=3, ll_min=−6)** — first power-law reward cell.

Training dynamics (collapse signature):
- `critic/rewards/mean`: −36.6 → **+23.9** (power reward climbs strongly into positive territory — a much steeper reward gradient than log_prob's −70→−5, driving harder optimization pressure).
- `response_length/mean`: **50.6 → 146–155** — **LENGTH-INFLATION COLLAPSE** (>140 band). Even with fp=5 format penalty, the steep power reward blows response length past the healthy 110–126 band.
- `format_error_ratio`: 0.000.

Eval (subsample300), step0 → final:
| Benchmark | step0 | final | Δpp |
|---|---|---|---|
| **aggregate mean (~26)** | 0.508 | ~0.435 | **~−7.3** |
| simpletom_behavior | 0.560 | 0.560 | 0.0 |
| simpletom_judgment | 0.283 | 0.270 | −1.3 |
| opentom_attitude | 0.443 | 0.433 | −1.0 |
| fantom_answerability_binary | 0.203 | 0.173 | −3.0 |
| bigtom_forward_action | 0.730 | 0.657 | −7.3 |
| tomi | 0.590 | 0.510 | −8.0 |
| explore_tom | 0.480 | 0.357 | −12.3 |
| bigtom_forward_belief | 0.767 | 0.640 | −12.7 |
| dyntom_type_a | 0.503 | 0.360 | −14.3 |

Interpretation & hypothesis test: The monitor asked "does the power-law reward (k=3, ll_min=−6) stabilize training at low KL=0.01 where plain log_prob struggled?" **Answer: NO — it collapses harder.** The power reward's much steeper gradient (reward −36→+24 vs log_prob's shallow −70→−5) amplifies optimization pressure, and at kl=0.01 the policy escapes into length-inflation (146–155 tokens) with a −7.3pp aggregate eval regression across nearly all benchmarks. This is **worse** than the log_prob equivalent PS008 (frozen kl0.01 lr1e-6 fp5 log_prob → PARTIAL rescue), and fp=5 alone cannot contain it. **Reinforces the central Phase −1 conclusion: high KL (0.05) is the essential stabilizer — reward shape (power vs log_prob) cannot fix low-KL collapse, and the steeper power reward actively worsens it. Power-law reward should be paired with kl=0.05, not kl=0.01.** PS011 (frozen kl0.05+fp5, log_prob) remains the leading `Pm1swp_best` candidate.

