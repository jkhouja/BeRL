### Attempt r1 — 2026-07-11T17:21:46+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.01-n16-r1`
- **Host:** h100-013-002   **git:** `a329b31`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=actor baseline=False kl=0.01 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.01-n16-r1.log`

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
    actor_rollout_ref.actor.entropy_coeff=0.0 \
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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.01-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.0_lp DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (PS021 — actor-RM, kl=0.01, lr=1e-6, fp0, ec0.0, log_prob)

Owner_host: h100-013-002 · WandB run u5n54j7q · 191 steps (1 epoch), TEST_FREQ=10, subsample300 eval.

**Config-selection metric** = harmonic mean over ToM benchmark subtypes EXCLUDING gsm8k & mmlu, per eval.

| step | HM_tom | tomi | gsm8k | mmlu |
|-----:|-------:|-----:|------:|-----:|
|   0  | 0.417  | 0.590| 0.660 | 0.477|
|  10  | 0.455  | 0.657| 0.733 | 0.560|
|  20  | 0.454  | 0.630| 0.617 | 0.557|
|  50  | 0.290  | 0.357| 0.187 | 0.500|
|  90  | 0.406  | 0.593| 0.123 | 0.603|
| 130  | 0.348  | 0.407| 0.087 | 0.563|
| 190  | 0.348  | 0.387| 0.087 | 0.513|

- **baseline HM(step0) = 0.417**; peak 0.455 @step10; **HM-last5 = 0.341** (< baseline).
- gsm8k crashes 0.660 → 0.087 (severe general-capability loss); mmlu roughly held (0.48→0.51).
- reward/mean rises −61.2 → −1.97 (near ceiling −1.86) while ToM eval **falls** ⇒ reward↑/eval↓ over-optimization at too-low KL.
- response_length bounded 47.7 → 42.0 (max 123.8): **no length-hacking**.

**Verdict: climbs-then-degrades (below baseline).** Actor-RM at kl0.01+lr1e-6 does NOT fully collapse ToM
(tomi holds ~0.39–0.45, unlike frozen-RM PS006 which drove tomi→0.000), but it still degrades ToM below
baseline and destroys gsm8k. Confirms kl=0.01 is too weak a leash even for actor-RM; kl=0.05 is the
stability lever (cf PS013 best). Actor-RM is modestly more robust than frozen-RM at kl0.01 but not stable.

Rerun: `ONLY_IDX=21 bash experiments/phase_stability_sweep.sh`
