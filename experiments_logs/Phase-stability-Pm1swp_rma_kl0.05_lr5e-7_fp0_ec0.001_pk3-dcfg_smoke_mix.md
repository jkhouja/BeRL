### Attempt r1 — 2026-07-12T02:36:51+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-079-001   **git:** `815582e`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=3 ll_min=-6.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.05-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_pk3 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


### Findings (r1 — completed 2026-07-12 ~04:11Z, 191 steps / 1 epoch)

**Config-selection metric (HM over ToM benchmarks, excl. gsm8k/mmlu):**
- baseline (step 0) HM_tom = **0.417**
- HM over last-3 = **0.444**, last-5 = **0.442**
- peak HM = **0.463 @ step 30**, final HM = **0.443 @ step 180**

**Eval trajectory (sub300, selected):**
| step | HM_tom | gsm8k | tomi | bigtom_fwd_belief | simpletom_mental |
|---|---|---|---|---|---|
| 0   | 0.417 | 0.663 | 0.590 | 0.767 | 0.847 |
| 30  | 0.463 | 0.673 | 0.650 | 0.743 | 0.870 |
| 100 | 0.443 | 0.477 | 0.620 | 0.700 | 0.867 |
| 180 | 0.443 | 0.540 | 0.623 | 0.697 | 0.883 |

**Health:** power reward optimized −33→+34; format_error_ratio=0.000 throughout; response_length/mean
42→116 (moderate growth, no degenerate blowup); entropy 1.0→2.27. No collapse, no reward-hacking
signature (contrast PS040 which blew up to 181 and collapsed all evals).

**Verdict:** STABLE, above-baseline cell — HM_tom stays ≥ baseline across all 19 evals. The
**power-reward (k=3, ll_min=−6) analog of PS026** (log_prob), same actor-RM/KL=0.05/LR=5e-7/fp0/ec0.001.
Both reward families are stable in this regime → **actor-RM + KL=0.05 + LR=5e-7 is a robust operating
point independent of reward family**. Strong Phase-1 stable-config candidate; keep for Wave-2.
Launched as a dedicated exclusive srun job (h100-079-001) — no preemption.
