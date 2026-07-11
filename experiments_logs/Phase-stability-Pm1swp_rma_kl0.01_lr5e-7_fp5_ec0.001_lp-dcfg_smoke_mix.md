### Attempt r1 — 2026-07-11T16:32:33+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.01-n16-r1`
- **Host:** h100-007-002   **git:** `935424d`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=actor baseline=False kl=0.01 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.01-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.01-n16-r1 \
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
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=log_prob \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.001_lp DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_

#### Findings (r1 — completed 2026-07-11, exit 0, 191 steps / 1 epoch)

**Verdict: Below-baseline, moderate late decline. Adding fp5+ec0.001 to actor-RM slightly HURT vs PS017 (fp0/ec0.0) — opposite of the frozen-RM case.**

Config-selection score (HM of 26 subsample300 benchmarks):
- **HM(last-3) = 0.337**; HM(last-5) = 0.347. Mean-of-means(last-3)=0.417.
- **Step-0 baseline HM = 0.426.** Holds ~0.43–0.48 through step 100 (peak 0.475 @ step50), then declines to **0.323 @ step190**.
- Parseable-answer rate ≈ **100%** (`format_error_ratio`=0).

Eval HM trajectory:
`0:0.426 10:0.455 20:0.462 30:0.465 40:0.460 50:0.475(peak) 60:0.449 70:0.435 80:0.437 90:0.431 100:0.428 110:0.378 120:0.363 130:0.362 140:0.366 150:0.354 160:0.360 170:0.356 180:0.328 190:0.323`

Health: reward −75 → ~−2 to −7; KL drifts to ~0.64–0.84; entropy rises to 3.2; resp_len ~99–113; 100% parseable.

**Comparisons (HM last-3):**
- actor-RM @ kl0.01: PS017 (fp0,ec0.0)=**0.400** > PS020 (fp5,ec0.001)=**0.337** — here fp5+ec0.001 *hurts* (opposite of frozen, where PS004 fp5>PS001 fp0). Suggests fp/ec interaction is RM-mode-dependent and small.
- vs frozen @ same knobs: PS020 (actor,fp5,ec0.001)=0.337 ≈ PS004 (frozen,fp5,ec0.001)=0.351 — comparable at kl0.01 with fp5.
- All KL=0.01 cells (0.169–0.400) remain below the KL=0.05 stable cells (0.432–0.435).

**Implication:** confirms KL=0.05 dominance; actor-RM helps most in the fp0/ec0.0 corner (PS017), and format_penalty is not a reliable win. Not a stable-config candidate on its own.

