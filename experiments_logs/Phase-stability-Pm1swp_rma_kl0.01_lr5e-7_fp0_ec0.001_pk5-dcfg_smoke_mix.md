### Attempt r1 — 2026-07-12T08:25:22+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp0_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1`
- **Host:** h100-189-003   **git:** `5a590c9`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=actor baseline=False kl=0.01 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp0_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp0_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp0_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp0_ec0.001_pk5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** STABLE (r1, WandB kzzr3ymf, 191 steps, h100-189-003).

PS082 = actor-RM, kl=0.01, lr=5e-7, **ec=0.001** (first nonzero entropy coeff in my sweep), power-k5, ll_min=-6. Tests whether the cleaner LR (5e-7) + a small entropy bonus keeps the actor-RM stable at low kl0.01 — recall PS053 (actor, kl0.01, lr1e-6, power-k3) entropy-collapsed to near-deterministic terse outputs. **It stayed stable.**

Health envelope (full run):

| Signal | Envelope | Read |
|---|---|---|
| actor/entropy_loss | 1.04 → 2.42 | healthy; NO collapse (PS053 → 0.009). ec=0.001 kept entropy up |
| response_length/mean | 39 → 138 (cap 512) | contained, no explosion, no terse collapse |
| critic/rewards/mean | −31 → +39.96 | k5 saturates near +40 clip; ToM held ⇒ not hacking |
| reward/format_error_ratio | max 0.000 | perfect formatting |
| actor/kl_loss | 0.002 → 0.279 | controlled |

Evals (standalone consolidated lines):

| step | ToM-HM | gsm8k | mmlu | tomi | simpletom_mental |
|---|---|---|---|---|---|
| 0 (baseline) | 0.419 | 0.657 | 0.463 | 0.590 | 0.853 |
| 190 (final) | 0.453 | 0.463 | 0.627 | 0.583 | 0.877 |

ToM-HM 0.419 → 0.453 (**+0.035**, ties the standout PS077). gsm8k 0.657 → 0.463 final — moderate forgetting (−0.19); intermediate inline evals were higher (band ~0.46–0.67), so gsm8k drift is noisy/moderate, not the catastrophic collapse of k3 (PS046 → .20) but worse than the frozen-RM k5 cells (PS069/PS077 both ~.58). mmlu 0.463 → 0.627.

**Key implications:**
- **k5 + entropy-bonus (ec=0.001) + lr5e-7 stabilizes the actor-RM at low KL (kl0.01)** where k3/lr1e-6 entropy-collapsed (PS053). The entropy bonus is a viable guard against actor-RM entropy-collapse.
- **Frozen-RM preserves gsm8k better than actor-RM** at the same power-k5 reward shape: frozen PS069/PS077 kept gsm8k ~.58, actor PS082 dropped to .46. Suggests the actor-RM (bootstrapping reward from the policy itself) induces more general-capability drift.
- Confirms the broader pattern: power-k5 avoids training collapse across RM modes and KL values; ToM-HM gains are consistent (~+0.035).

Caveats: single r1 run; gsm8k final-eval variance is notable (intermediate was higher); reward saturates near clip (inherent to k5). Frozen-RM k5 (PS077) remains the better all-round recipe; actor-RM viable if entropy bonus is used.

**Rerun:** `cd /mnt/home/judekhouja/repo/BeRL && source ~/.bashrc && conda activate tom && setsid bash -c 'ONLY_IDX=82 bash experiments/phase_stability_sweep.sh'`

