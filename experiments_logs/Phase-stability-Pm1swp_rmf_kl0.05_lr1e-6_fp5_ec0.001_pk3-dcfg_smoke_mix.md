### Attempt r1 — 2026-07-12T00:04:13+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1`
- **Host:** h100-117-001   **git:** `2913354`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=3 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1.log`

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
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
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
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp5_ec0.001_pk3 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings (r1, authoritative — completed 2026-07-12, Owner_host h100-117-001):**

Full horizon (191 steps, 1 epoch), evals every 10 steps on subsample300 (26 subtypes). Config-selection metric = harmonic mean over all 26 sub300 benchmark scores per eval:

| step | 0(base) | 20(peak) | 60 | 120 | 160 | 180 | 190 |
|------|---------|----------|-----|-----|-----|-----|-----|
| HM_all | 0.4286 | 0.4734 | 0.4480 | 0.4600 | 0.4600 | 0.4510 | 0.4568 |

- **HM(last-3)=0.455, HM(last-5)=0.456, HM@190=0.457** vs step-0 baseline **HM=0.4286** → **sustained above baseline (+2.8pp)**. Peak HM=0.473 @step20; **holds a stable plateau (~0.45–0.46) the entire horizon**, no regression — same stability profile as PS042.
- **Health: fully stable.** `format_error_ratio=0.000` throughout; entropy steady ~2.4 (no runaway); kl_loss controlled ~0.2; response_length stable ~122.
- Per-benchmark step0→step190: **hi_tom 0.250→0.357 (+10.7)**, mmlu 0.457→0.597 (+14.0), simpletom_mental 0.853→0.910 (+5.7), tomi 0.590→0.643 (+5.3); bigtom_fwd_belief −4.4, fantom_belief_mc flat. **gsm8k 0.657→0.590 (only −6.7)** — far better math retention than PS042 (−23pp).

**Verdict:** STRONG cell — ties PS042 on HM (0.456 vs 0.455) but with **much less math degradation** and broader ToM gains (hi_tom/tomi/simpletom_mental all up). Frozen-RM power(k3,ll_min=−6), **kl=0.05**, lr=1e-6, **fp=5**, ec=0.001. The fp=5 format penalty (+ lr1e-6) appears to keep the policy on-distribution and protect gsm8k while still climbing ToM. **Best Phase-1 candidate so far; strong stable-config / Wave-2 pick.** Reinforces kl=0.05 as the stabilizer (cf. PS035 kl=0.01 collapse).

