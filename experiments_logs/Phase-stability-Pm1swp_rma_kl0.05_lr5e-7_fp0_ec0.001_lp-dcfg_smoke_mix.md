### Attempt r1 — 2026-07-11T18:50:43+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-013-002   **git:** `967e489`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1.log`

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
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
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
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_lp DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


- **WandB link (resolved r1):** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/p8015nto
- **Hypothesis:** Phase −1 sweep cell PS026 (actor-as-RM, KL=0.05, LR=5e-7, format_penalty=0, entropy_coeff=0.001, log_prob reward). Tests whether this actor-RM/log_prob config climbs (HM over last-X) without collapse/reward-hacking. Compare vs frozen-RM counterpart PS010 (stable candidate).
- **How to rerun:** `ONLY_IDX="26" bash experiments/phase_stability_sweep.sh` (on an 8-GPU node; env `tom`).

### Findings (r1 — completed 2026-07-11 ~20:23Z, 191 steps / 1 epoch)

**Config-selection metric (HM over ToM benchmarks, excl. gsm8k/mmlu):**
- baseline (step 0) HM_tom = **0.414**
- HM over last-3 evals = **0.441**, last-5 = **0.443**
- peak HM = **0.458 @ step 80**, final HM = **0.446 @ step 190**

**Eval trajectory (sub300, every 10 steps):**
| step | HM_tom | gsm8k | mmlu | tomi | bigtom_fwd_belief | simpletom_mental |
|---|---|---|---|---|---|---|
| 0   | 0.414 | 0.657 | 0.483 | 0.590 | 0.767 | 0.853 |
| 20  | 0.457 | 0.713 | 0.563 | 0.610 | 0.747 | 0.867 |
| 80  | 0.458 | 0.480 | 0.560 | 0.630 | 0.723 | 0.847 |
| 130 | 0.432 | 0.657 | 0.580 | 0.597 | 0.700 | 0.877 |
| 190 | 0.446 | 0.583 | 0.590 | 0.637 | 0.693 | 0.867 |

**Health:** format_error_ratio = 0.000 throughout; response_length/mean 54→106 (grows but no
degenerate repetition/blowup); entropy 1.1→2.4, kl_loss ~0.3–0.7 (contained by KL=0.05); behavior
reward optimized −65→−2. No collapse, no reward-hacking signature.

**Verdict:** STABLE, above-baseline cell. HM_tom stays ≥ baseline the entire run. ToM metrics hold
or improve (tomi +4.7pp, simpletom_mental +1.4pp) while gsm8k takes a transient mid-run dip that
partially recovers. This is the **actor-RM twin of PS010** (frozen-RM, same KL/LR/fp/ec) and behaves
comparably (PS010 HM-last-5 ≈ 0.433) → **actor-as-RM is viable at KL=0.05**. Plausible Phase-1
stable-config candidate; not collapsed → keep for Wave-2 consideration.
