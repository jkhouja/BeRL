### Attempt r1 — 2026-07-12T08:33:50+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1`
- **Host:** h100-038-001   **git:** `5a590c9`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=actor baseline=False kl=0.01 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1 \
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
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.0_pk5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


### Findings (r1 — ran to step 189/191, 2026-07-12 ~10:12Z)

**Config:** actor-RM, power-k5/ll_min=−6, kl=0.01, lr=1e-6, format_penalty=0, entropy_coeff=0.0.
WandB iwkh6740; job 5421638 on h100-038-001. This is the AGGRESSIVE regime (low kl + high lr=1e-6)
that catastrophically collapsed frozen-RM PS040.

**Config-selection metric (HM over ToM benchmarks, excl. gsm8k/mmlu):**
- baseline (step 0) HM_tom = **0.420**
- HM last-3 = **0.426**, last-5 = **0.429**
- peak HM = **0.460 @ step 50**, HM @ step 180 = **0.432**

**Eval trajectory (sub300, selected):**
| step | HM_tom | tomi | simpletom_mental | gsm8k |
|---|---|---|---|---|
| 0   | 0.420 | 0.590 | 0.853 | 0.653 |
| 50  | 0.460 | 0.570 | 0.890 | 0.650 |
| 100 | 0.442 | 0.597 | 0.903 | 0.577 |
| 150 | 0.430 | 0.600 | 0.933 | 0.383 |
| 180 | 0.432 | 0.597 | 0.910 | 0.440 |

**Health:** reward ~+35; format_error_ratio=0.000 throughout; response_length/mean controlled ~108
(NO blowup — unlike PS040's 46→181 blowup). BUT: HM_tom rises to 0.46@50 then **drifts back down to
~0.43 by step 180** (ToM gains fade), and **gsm8k erodes badly 0.65→0.44** (dipped to 0.383@150) —
general-capability tax from the aggressive lr=1e-6. simpletom_mental actually climbs 0.853→0.91.

**Verdict:** MARGINAL / DRIFTING (weakly stable, degrading). Actor-RM **avoids the catastrophic
collapse** that frozen-RM suffered in this same aggressive regime (PS040: length blowup + all-eval
collapse) — no length blowup, format intact, HM stays ≥ baseline. This confirms **actor-RM is
substantially more robust than frozen-RM to low-kl/high-lr**. HOWEVER lr=1e-6 still costs: late ToM
fade + heavy gsm8k erosion, leaving last-5 HM=0.429 barely above baseline 0.420 and well below the
gentle-lr cells (PS076 last-5=0.468). Not a preferred config; the gentle lr=5e-7 regime dominates.
Dedicated exclusive srun job — no preemption; benign Ray teardown crash at step 189, evals 0–180 captured.
