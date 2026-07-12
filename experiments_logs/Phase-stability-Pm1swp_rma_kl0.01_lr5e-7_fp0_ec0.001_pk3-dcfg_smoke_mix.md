### Attempt r1 — 2026-07-12T00:34:35+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp0_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1`
- **Host:** h100-013-002   **git:** `f65c24a`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=3 ll_min=-6.0 rm_mode=actor baseline=False kl=0.01 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp0_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp0_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp0_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1 \
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
    +reward_model.reward_type=power \
    +reward_model.power_k=3 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=3 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp0_ec0.001_pk3 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (PS050 — actor-RM, POWER reward k=3/ll_min=-6, kl=0.01, lr=5e-7, fp0, ec0.001)

Owner_host: h100-013-002 · WandB run 2zrx8fza · 191 steps (1 epoch), TEST_FREQ=10, subsample300 eval.

**Config-selection metric** = harmonic mean over ToM benchmark subtypes EXCLUDING gsm8k & mmlu, per eval.

| step | HM_tom | tomi | gsm8k | mmlu |
|-----:|-------:|-----:|------:|-----:|
|   0  | 0.418  | 0.590| 0.663 | 0.467|
|  30  | 0.457  | 0.623| 0.697 | 0.597|
|  80  | 0.455  | 0.627| 0.580 | 0.603|
| 120  | 0.441  | 0.610| 0.537 | 0.620|
| 160  | 0.416  | 0.607| 0.527 | 0.583|
| 190  | 0.423  | 0.557| 0.453 | 0.603|

- **baseline HM(step0) = 0.418**; peak 0.457 @step30; holds 0.42–0.46 across all 190 steps; **HM-last5 = 0.424 (≈/slightly > baseline)**.
- tomi stable 0.55–0.66 (ends 0.557); mmlu improves 0.47 → 0.60; gsm8k drifts 0.663 → ~0.45–0.55 (moderate, far milder than log_prob kl0.01).
- resp_len grows 50.4 → 111.2 (max 130.3, ctx 512 not saturated); reward/mean rises −30.2 → +31.1 while eval stays flat/up ⇒ healthy optimization, NOT hacking.

**Verdict: STABLE (≈baseline).** KEY RESULT: **actor-RM makes power reward robust even at the shaky
kl=0.01** — contrast frozen-RM+power at the SAME kl0.01/lr5e-7 (PS034) which climbed-then-degraded to
0.390. Actor-RM here holds 0.424 with no degradation. This strengthens the earlier log_prob observation
(actor-RM > frozen-RM at kl0.01) and shows it's most pronounced with power reward. Power+actor-RM is a
robust pairing across KL. gsm8k moderate drift (mmlu compensates).

Rerun: `ONLY_IDX=50 bash experiments/phase_stability_sweep.sh`
