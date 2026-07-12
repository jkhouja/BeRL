### Attempt r1 — 2026-07-12T10:11:10+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-189-003   **git:** `afc71ac`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_pk5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** STABLE — STANDOUT cell, best combined result of my sweep (r1, WandB 6ts7zdpn, 191 steps, h100-189-003).

PS090 = actor-RM, kl=0.05, lr=5e-7, ec=0.001, power-k5, ll_min=-6 — the kl0.05 analogue of PS082 (kl0.01) and the best-expected actor-RM config. Key question: does tighter kl0.05 preserve gsm8k better than PS082's kl0.01 (.46), approaching frozen-RM (~.58)? **Yes — and it exceeds frozen-RM (.637).**

Health envelope (full run):

| Signal | Envelope | Read |
|---|---|---|
| actor/entropy_loss | 0.98 → 1.67 | contained; no collapse (ec=0.001 + tight kl) |
| response_length/mean | 42 → 126 (cap 512) | contained |
| critic/rewards/mean | −30 → +39.95 | k5 saturates near +40 clip; ToM held ⇒ not hacking |
| reward/format_error_ratio | max 0.000 | perfect formatting |
| actor/kl_loss | 0.002 → 0.156 | very tight (kl0.05 keeps drift minimal) |

Evals (standalone consolidated lines):

| step | ToM-HM | gsm8k | mmlu | tomi | simpletom_mental |
|---|---|---|---|---|---|
| 0 (baseline) | 0.417 | 0.657 | 0.470 | 0.590 | 0.853 |
| 190 (final) | 0.459 | 0.637 | 0.610 | 0.640 | 0.877 |

ToM-HM 0.417 → 0.459 (**+0.043** — the highest ToM-HM gain among all my cells, beating frozen PS077's +0.036). gsm8k 0.657 → 0.637 (**−0.02**, the best general-capability preservation of any cell in my sweep, even beating the frozen-RM k5 cells at ~.58). mmlu 0.47 → 0.61; tomi and simpletom_mental both up.

**Key implications (this cell resolves an earlier confound):**
- **The gsm8k-forgetting driver is the KL coefficient, not the RM mode.** PS082 (actor, kl0.01) forgot gsm8k to .46; PS090 (actor, kl0.05) preserved it at .637. Same RM, same lr/ec/k — only KL differs. Tight KL (0.05) is what protects general capability.
- With power-k5 + kl0.05 + lr5e-7 + ec0.001, the **actor-RM matches/beats the frozen-RM on BOTH ToM-HM and gsm8k** — so actor-as-RM is not inherently worse; the earlier "frozen preserves gsm8k better" pattern was an artifact of comparing frozen@kl0.05 vs actor@kl0.01.
- **Winning stable recipe candidate: power-k5, kl=0.05, lr=5e-7, ec=0.001** — best ToM improvement + near-zero general-capability loss, no collapse, no hacking.

Caveats: single r1 run; reward saturates near the clip (inherent to k5); ToM-HM gains across the k5 grid are modest (~+0.04) and should be confirmed with reruns / longer training before Phase-0 adoption.

**Rerun:** `cd /mnt/home/judekhouja/repo/BeRL && source ~/.bashrc && conda activate tom && setsid bash -c 'ONLY_IDX=90 bash experiments/phase_stability_sweep.sh'`

