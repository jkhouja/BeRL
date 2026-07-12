### Attempt r1 — 2026-07-12T04:53:44+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1`
- **Host:** h100-189-003   **git:** `6ed1e1b`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.01 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp0_ec0.0_pk5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** STABLE — SURPRISE result (r1, WandB 738xy56j, 191 steps, h100-189-003).

PS069 = frozen-RM, kl=0.01, lr=1e-6, power-**k5**, ll_min=-6. Expectation going in: collapse, because kl=0.01+lr=1e-6 collapsed in every prior cell (PS038 frozen power-**k3** → high-entropy blowup + length explosion; PS053 actor power-**k3** → entropy-collapse to terse outputs). **It did not collapse.** The only difference from the collapsing PS038 is power_k (5 vs 3).

Health envelope (full run):

| Signal | Envelope | Read |
|---|---|---|
| actor/entropy_loss | 1.08 → 2.32 | rising but contained; no blowup (PS038 blew up), no collapse (PS053 → 0.009) |
| response_length/mean | 44 → 148 (max 148, cap 512) | grew but contained; no explosion |
| critic/rewards/mean | −27 → +39.9 | saturates near +40 clip but oscillates (not pinned) — k=5 steepens the power curve so reward hits the clip more readily |
| reward/format_error_ratio | max 0.000 | perfect formatting |
| actor/kl_loss | 0.002 → 0.262 | controlled |

Evals (standalone consolidated lines; intermediate inline evals noisy):

| step | ToM-HM | gsm8k | mmlu | tomi | simpletom_mental |
|---|---|---|---|---|---|
| 0 (baseline) | 0.416 | 0.663 | 0.470 | 0.590 | 0.853 |
| 190 (final) | 0.427 | 0.577 | 0.620 | 0.590 | 0.870 |

ToM-HM 0.416 → 0.427 (+0.011, marginally above baseline). gsm8k **well-preserved** 0.663 → 0.577 — in stark contrast to the kl=0.05/lr=1e-6 cells (PS046 frozen .66→.20, PS061 actor .66→.33) which catastrophically forgot gsm8k. mmlu rose 0.470 → 0.620.

**Key implication — the kl0.01/lr1e-6 collapse is power-k dependent:**
- power-k3: COLLAPSES (PS038 frozen, PS053 actor).
- power-k5: STABLE (PS069 frozen) — sharper reward exponent apparently stabilizes training at low KL.

This revises the earlier over-generalization "kl0.01+lr1e-6 ALWAYS collapses". It holds for k=3 but not k=5. Caveats: the HM gain is marginal (+0.011), reward saturates near the clip (inherent to k=5's steeper transform, not necessarily hacking since ToM/gsm8k both held), and this is a single r1 run. Not a config winner, but a clean stability data point and a reward-shape × KL interaction worth noting for Phase-0 recipe selection.

**Rerun:** `cd /mnt/home/judekhouja/repo/BeRL && source ~/.bashrc && conda activate tom && setsid bash -c 'ONLY_IDX=69 bash experiments/phase_stability_sweep.sh'`

