### Attempt r1 — 2026-07-12T03:08:00+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp0_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1`
- **Host:** h100-189-003   **git:** `815582e`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=3 ll_min=-6.0 rm_mode=actor baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp0_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp0_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp0_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.05-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp0_ec0.0_pk3 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** QUALIFIED-STABLE (r1, WandB 7b276p8y, 191 steps, h100-189-003).

PS061 = actor-RM, kl=0.05, lr=1e-6, power-k3, ll_min=-6 — the direct actor-RM analogue of PS046 (frozen). Question: does kl=0.05 save the actor-RM (as it did the frozen RM in PS046), or does the actor-RM still entropy-collapse (as PS053 did with kl=0.01)?

Verdict: **kl=0.05 rescues the actor-RM.** No entropy-collapse (contrast PS053 → entropy 0.009, terse 32-tok outputs). Health envelope over the full run:

| Signal | Envelope | Read |
|---|---|---|
| actor/entropy_loss | 0.94 → 2.40 (body ~2.0–2.4) | stable, no collapse |
| response_length/mean | 35 → 132 (body ~95–130) | no terse collapse, no explosion |
| critic/rewards/mean | −36 → +37 (end ~30–35) | not saturated at +40 clip |
| reward/format_error_ratio | max 0.000 | perfect formatting throughout |
| actor/kl_loss | 0.002 → 0.396 (body ~0.2) | controlled |

ToM-HM (harmonic mean over ToM subtypes, excl. gsm8k/mmlu):

| step | ToM-HM | gsm8k | mmlu |
|---|---|---|---|
| 0 (baseline) | 0.420 | 0.657 | 0.477 |
| 190 (final) | 0.458 | 0.327 | 0.597 |

Only step-0 and step-190 are standalone consolidated eval lines; intermediate evals appear inline and are noisy (band ~0.32–0.46), so the step-190 final is the reliable headline. Final ToM-HM 0.458 > baseline 0.420 (+0.038).

**Catastrophic gsm8k forgetting:** gsm8k 0.657 → 0.327 (halved). Same lr=1e-6 forgetting signature as PS046 (frozen, .66→.20) — milder here but still severe. mmlu actually rose 0.477 → 0.597.

**Implications for Phase-1 sweep:**
- kl=0.05 is required for actor-RM stability: PS053 (actor, kl0.01) COLLAPSED via entropy-collapse; PS061 (actor, kl0.05) is stable. Mirrors frozen-RM: PS038 (kl0.01) collapsed, PS046 (kl0.05) stable.
- lr=1e-6 causes gsm8k catastrophic forgetting regardless of RM mode (frozen PS046 and actor PS061 both forget) even when ToM-HM survives. Prefer lr=5e-7 (PS027 log_prob was cleanly STABLE with no forgetting).

**Rerun:** `cd /mnt/home/judekhouja/repo/BeRL && source ~/.bashrc && conda activate tom && setsid bash -c 'ONLY_IDX=61 bash experiments/phase_stability_sweep.sh'`

