### Attempt r1 — 2026-07-19T21:16:02+00:00

- **RUN_NAME:** `s2-ST15-Phase-stability-v2_g2_reward_negppl-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-neg_perplexity-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-021-001   **git:** `e841f0e`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=neg_perplexity power_k=2.0 ll_min=-8.0 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=1.0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-ST15-Phase-stability-v2_g2_reward_negppl-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-neg_perplexity-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260719/s2-ST15-Phase-stability-v2_g2_reward_negppl-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-neg_perplexity-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=google/gemma-2-2b-it \
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
    actor_rollout_ref.actor.think_only_pg=False \
    actor_rollout_ref.actor.format_penalty=0.0 \
    actor_rollout_ref.actor.format_penalty_std_coef=1.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-ST15-Phase-stability-v2_g2_reward_negppl-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-neg_perplexity-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=google/gemma-2-2b-it \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=neg_perplexity \
    +reward_model.power_k=2.0 \
    +reward_model.power_ll_min=-8.0 \
    reward_model.format_penalty=0.0 \
    reward_model.format_penalty_std_coef=1.0 \
    +actor_rollout_ref.reward_type=neg_perplexity \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-v2_g2_reward_negppl DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-19T21:16:16+00:00

- **RUN_NAME:** `s2-ST15-Phase-stability-v2_g2_reward_negppl-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-neg_perplexity-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-021-001   **git:** `e841f0e`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=neg_perplexity power_k=2.0 ll_min=-8.0 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=1.0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-ST15-Phase-stability-v2_g2_reward_negppl-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-neg_perplexity-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260719/s2-ST15-Phase-stability-v2_g2_reward_negppl-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-neg_perplexity-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=google/gemma-2-2b-it \
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
    actor_rollout_ref.actor.think_only_pg=False \
    actor_rollout_ref.actor.format_penalty=0.0 \
    actor_rollout_ref.actor.format_penalty_std_coef=1.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-ST15-Phase-stability-v2_g2_reward_negppl-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-neg_perplexity-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=google/gemma-2-2b-it \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=neg_perplexity \
    +reward_model.power_k=2.0 \
    +reward_model.power_ll_min=-8.0 \
    reward_model.format_penalty=0.0 \
    reward_model.format_penalty_std_coef=1.0 \
    +actor_rollout_ref.reward_type=neg_perplexity \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-v2_g2_reward_negppl DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


**WandB link (r1):** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/og7l8pf1

**Hypothesis / question (ST15, RQ Phase-stability, PA reward family):** Tests the **neg_perplexity**
behavior reward on the Gemma-2-2B family (frozen RM, tag-free cot_eval_notags data) — a reward that
was **never** swept in the old stage. Compares against the Gemma-2 power anchor (ST13) and log_prob
(ST14) reward families. Expected: neg_perplexity (length-normalised) should give a smoother, less
saturating signal than the 0–40 power reward; watch for KL/entropy stability and whether it lifts
ToM-HM without format collapse on Gemma (frozen RM, entropy_coeff=0.001).

**Authoritative attempt:** r1.

---

## Findings (r1 — completed 2026-07-19)

**Canonical score (`scripts/score_run.py`, 8 eval iters, steps 0..190; 24 ToM benchmarks excl gsm8k/mmlu):**
```
ToM HM(last5)=0.090   HM(last3)=0.0995  (baseline step0=0.0929)  => flat
ToM avg(last5)=0.3404 avg(last3)=0.3477 (baseline step0=0.3145)  => +2.6pp
gsm8k (separate): 0.320  (step0=0.28,  delta=+0.040)
mmlu  (separate): 0.3874 (step0=0.383, delta=+0.004)
health(final): kl=0.020 entropy=1.568 resp_len=163.916 reward=-40.0 parseable=1.0 max_resp=512
ToM HM trajectory: 0:0.093 30:0.086 60:0.116 90:0.083 120:0.063 150:0.102 180:0.113 190:0.073
```

**Verdict:** neg_perplexity behavior reward on **Gemma-2-2B (frozen RM)** is **stable and healthy**
(no collapse, parseable=1.0 throughout, kl≈0.02, entropy≈1.57 flat, resp_len≈164) but delivers a
**weak/noisy ToM signal**: HM stays essentially at the step-0 baseline (~0.09) and bounces
non-monotonically across evals; only the arithmetic ToM avg shows a small lift (+2.6pp). The very low
absolute HM is characteristic of the Gemma-2 base model — its per-benchmark ToM scores are far below
Qwen2.5's (base HM 0.093 vs 0.415), so a single near-zero benchmark tanks the harmonic mean. Small
capability gains on gsm8k (+0.04) / mmlu (+0.004). No reward-hacking observed.

**Comparison note:** vs the Qwen2.5 neg_perplexity sibling (ST03) and the Gemma-2 power anchor (ST13,
pending), neg_perplexity does not appear to strongly drive ToM on Gemma; power/log_prob family
comparison on Gemma (ST13/ST14) needed to judge the best Gemma reward.

**Authoritative attempt:** r1 (WandB https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/og7l8pf1).
