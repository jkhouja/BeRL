### Attempt r1 — 2026-07-12T06:39:45+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.05-n16-r1`
- **Host:** h100-189-003   **git:** `7ba14d4`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.05-n16-r1.log`

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
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.05-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp0_ec0.0_pk5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** STABLE — standout cell (r1, WandB 9s71k6gi, 191 steps, h100-189-003).

PS077 = frozen-RM, kl=0.05, lr=1e-6, power-**k5**, ll_min=-6. The power-k5 analogue of PS046 (frozen, kl0.05, lr1e-6, power-**k3**), which was QUALIFIED-STABLE on ToM-HM but catastrophically forgot gsm8k (.66→.20). Key question: does k5 preserve gsm8k at kl0.05? **Yes.**

Health envelope (full run):

| Signal | Envelope | Read |
|---|---|---|
| actor/entropy_loss | 1.09 → 1.69 | contained; no blowup (PS038), no collapse (PS053) — tighter than PS069's 2.32 |
| response_length/mean | 42 → 129 (cap 512) | contained, no explosion |
| critic/rewards/mean | −31 → +39.8 | saturates near +40 clip (k5 steepens power curve) but ToM held ⇒ not hacking |
| reward/format_error_ratio | max 0.000 | perfect formatting |
| actor/kl_loss | 0.003 → 0.161 | very controlled (kl0.05 keeps drift tight) |

Evals (standalone consolidated lines):

| step | ToM-HM | gsm8k | mmlu | tomi | simpletom_mental |
|---|---|---|---|---|---|
| 0 (baseline) | 0.417 | 0.660 | 0.470 | 0.587 | 0.850 |
| 190 (final) | 0.453 | 0.577 | 0.640 | 0.633 | 0.887 |

ToM-HM 0.417 → 0.453 (**+0.036**, clearly above baseline). gsm8k **preserved** 0.660 → 0.577 (mild −0.083 drift) — in stark contrast to k3 at identical kl/lr (PS046 .66→.20). mmlu 0.470 → 0.640; tomi and simpletom_mental both up.

**Key implication — power-k5 solves the gsm8k catastrophic forgetting seen with power-k3:**
- k3 @ kl0.05/lr1e-6: ToM-HM survives but gsm8k collapses (.66→.20 frozen PS046; .66→.33 actor PS061).
- k5 @ kl0.05/lr1e-6: ToM-HM improves (+0.036) AND gsm8k preserved (.66→.58) — this cell.
- k5 @ kl0.01/lr1e-6 (PS069): also stable, gsm8k preserved (.66→.58), where k3 collapsed entirely (PS038/PS053).

Taken together, **power-k5 is markedly more robust than power-k3** across the kl grid: it prevents both the low-KL training collapse (PS038/PS053) and the catastrophic gsm8k forgetting (PS046/PS061). This is the best combined ToM-improvement + general-capability-preservation result in my sweep so far, and a strong Phase-0 recipe candidate. Caveats: single r1 run; reward saturates near the clip (inherent to k5's steep transform); should be confirmed at lr=5e-7 (the cleaner LR per PS027) before adoption.

**Rerun:** `cd /mnt/home/judekhouja/repo/BeRL && source ~/.bashrc && conda activate tom && setsid bash -c 'ONLY_IDX=77 bash experiments/phase_stability_sweep.sh'`

