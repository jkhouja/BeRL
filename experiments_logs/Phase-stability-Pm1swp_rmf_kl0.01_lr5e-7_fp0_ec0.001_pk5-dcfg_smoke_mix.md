### Attempt r1 — 2026-07-12T04:13:31+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp0_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1`
- **Host:** h100-013-002   **git:** `6ed1e1b`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.01 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp0_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp0_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp0_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1 \
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
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp0_ec0.001_pk5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (PS066 — frozen-RM, POWER reward k=5/ll_min=-6, kl=0.01, lr=5e-7, fp0, ec0.001)

Owner_host: h100-013-002 · WandB run 4kk4m5ay · 191 steps (1 epoch), TEST_FREQ=10, subsample300 eval.

**Config-selection metric** = harmonic mean over ToM benchmark subtypes EXCLUDING gsm8k & mmlu, per eval.

| step | HM_tom | tomi | gsm8k | mmlu |
|-----:|-------:|-----:|------:|-----:|
|   0  | 0.415  | 0.587| 0.653 | 0.470|
|  40  | 0.465  | 0.647| 0.757 | 0.613|
|  70  | 0.470  | 0.637| 0.723 | 0.560|
| 120  | 0.448  | 0.620| 0.727 | 0.593|
| 160  | 0.444  | 0.600| 0.627 | 0.573|
| 190  | 0.440  | 0.593| 0.613 | 0.617|

- **baseline HM(step0) = 0.415**; peak 0.470 @step70; holds 0.44–0.47 through step120, eases to ~0.44; **HM-last5 = 0.439 (> baseline)**.
- tomi stable 0.57–0.66 (ends 0.593); mmlu improves 0.47 → 0.62; **gsm8k barely declines 0.653 → 0.613 (even peaks 0.757 mid-run) — near-perfect general-capability preservation.**
- resp_len grows 54.0 → 131.2 (max 138.5); reward/mean rises −16.2 → +36.4 while eval stays high ⇒ healthy optimization, NOT hacking.

**Verdict: STABLE, above baseline.** KEY RESULT: **sharper power exponent k=5 is markedly more stable
than k=3 at the shaky kl0.01** — HM-last5 0.439 (k=5) vs 0.390 (k=3, PS034, climbs-then-degrades), and
gsm8k preservation is dramatically better (0.61 end vs k=3 ~0.42–0.48). Mechanism: a sharper power curve
(higher k with ll_min=-6) flattens the reward gradient for already-good tokens, reducing over-optimization
pressure ⇒ gentler updates ⇒ stability even at low KL. k=5@kl0.01 nearly matches the k=3@kl0.05 winner-tier
(0.451/0.446). Implication: **higher power-k is an alternative stability lever to higher KL.**

Rerun: `ONLY_IDX=66 bash experiments/phase_stability_sweep.sh`
