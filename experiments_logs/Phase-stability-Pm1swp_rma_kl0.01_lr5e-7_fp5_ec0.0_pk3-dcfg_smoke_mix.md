### Attempt r1 — 2026-07-12T01:01:49+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1`
- **Host:** h100-007-002   **git:** `bc2bb8b`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=3 ll_min=-6.0 rm_mode=actor baseline=False kl=0.01 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1 \
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
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=3 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.0_pk3 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:**

#### Findings (completed 2026-07-12, exit 0, 191 steps / 1 epoch)

**Verdict: HEALTH FAILS despite good eval HM — actor-RM @ kl0.01 collapses to a degenerate low-entropy policy. Do NOT adopt.**

Config-selection score (HM of 26 subsample300 benchmarks):
- HM(last-3) = **0.453**; HM(last-5) = 0.458 — above step-0 baseline HM = 0.426. Eval score looks stable/flat 0.45–0.47.
- Parseable-answer rate ≈ **100%** (`format_error_ratio`=0).

**BUT health checks FAIL (collapse/hacking):**
- **KL diverges to ~1.75** (step160), ~1.36 @step189 — far past the 0.05 target; policy drifting hard.
- **Entropy collapses**: 1.12 → 2.57 (step80) → **0.45 (step120) → 0.22 (step189)** — model becoming deterministic.
- **Response length shrinks**: ~130 (step40) → **46–62** (step120+) — short, confident outputs.
- Signature of reward-model-is-actor at low KL: the moving reward target + weak KL lets the policy converge to a degenerate short-answer mode that happens to score OK on benchmarks. **High HM masks an unhealthy trajectory.**

Eval HM trajectory (deceptively flat):
`0:0.426 10:0.452 20:0.466 30:0.473 40:0.472 50:0.468 60:0.456 70:0.450 80:0.474 90:0.450 100:0.445 110:0.451 120:0.445 130:0.473 140:0.456 150:0.457 160:0.467 170:0.457 180:0.446 190:0.455`

Health trace: reward ~+27 (bounded); KL 0.002→0.13→0.49→**1.57→1.75→1.36**; entropy 1.12→1.69→2.57→**0.48→0.45→0.23**; resp_len 51→130→79→**47→46→62**.

**Key comparisons (HM last-3):**
- vs actor log_prob kl0.01/lr5e-7/fp5 (PS020 = 0.337): power gives higher HM here (0.453) but at the cost of entropy collapse — the score is not trustworthy.
- vs frozen power kl0.01/lr1e-6 (PS037 = 0.367, declining): different failure mode (frozen = eval decline; actor = eval-stable-but-policy-collapse).
- vs kl0.05 stable cells (PS045 power = 0.447, healthy KL~0.2): **KL=0.05 avoids this collapse**.

**Implication:** reinforces that **KL=0.01 is unsafe** — for actor-RM it produces entropy collapse (hidden by benchmark HM), for frozen-RM it produces eval decline. **HM alone is insufficient; must gate on KL/entropy health.** KL=0.05 remains the safe lever.

