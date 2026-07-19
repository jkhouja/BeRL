### Attempt r1 — 2026-07-13T19:07:10+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-4-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-013-002   **git:** `40d792b`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-4-lr5e-7-kl0.05-n16-r1 https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/2f4w2w42
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-4-lr5e-7-kl0.05-n16-r1.log`

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
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.think_only_pg=False \
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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-4-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=999 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen3-1.7B \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=7 \
    +reward_model.power_ll_min=-4 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=7 \
    +actor_rollout_ref.power_ll_min=-4 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm4 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_



## Hypothesis (PS159, Qwen3-1.7B power k7 ll_min-4 fp5 ec0.0 frozen-RM)
Power + format-penalty variant on Qwen3 (k=7, ll_min=-4, fp=5, ec=0.0). Frozen-RM, kl0.05, lr5e-7. Tag-free parquet, require_answer_tags=False, Gen ctx 2048/512.

Key question: does adding fp=5 (format penalty) to power on Qwen3 destabilize? fp=5 was a Gemma COLLAPSE DRIVER (PS123/PS125 isolated it). But Qwen3 is tag-free (require_answer_tags=False) so the format penalty may be inert / apply differently. Compare: PS146 (power k5 fp0) & PS153 (power k7 fp0) both FLAT/STABLE/neutral on Qwen3; PS139 (log_prob fp5) NET NEGATIVE. Does fp5+power stay neutral like the fp0 power runs, or does fp5 hurt? Also ll_min=-4 (higher floor) vs -6 in PS146/153. Watch resp_len (Qwen3 thinking ~450-490), kl, evals.

- WandB: https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/2f4w2w42 ; main_ppo clean, no crash, max_response=512, require_answer_tags=False, micro_batch=8.

## FINAL findings (PS159, Qwen3-1.7B power k7 ll_min-4 fp5 ec0.0 frozen-RM) — MILDLY NEGATIVE (fp5 slightly hurts power on Qwen3)
Completed cleanly (step 189/191, GPUs→1MiB, no crash). Canonical scorer (`scripts/score_run.py --last 5`):
- **ToM avg(last5)=0.2577 vs base 0.2719 → −1.4pp (mildly negative)**. HM(last5)=0.142 vs base 0.161 (−1.9pp).
- **mmlu=0.208 vs base 0.243 → −3.5pp (mild regression)**; gsm8k=0.480 vs base 0.487 → −0.7pp.
- health: resp_len=453 (Qwen3 thinking, stable from start, no hack), format_error_ratio=0 throughout (fp5 inert on format since tag-free), parseable=1.0, kl=0.002 (rock-stable the ENTIRE run), entropy=0.276.
- HM trajectory: ~0.16–0.18 early, gently DECLINES to ~0.14 in the last third — mild erosion, no collapse.

**Verdict:** power k7/fp5 on Qwen3 is **MILDLY NEGATIVE** — no crash/collapse (KL rock-stable 0.002, 100% parseable, no length-hack) but a small consistent erosion vs the fp0 power siblings.

**fp=0 vs fp=5 A/B on Qwen3 power (all frozen-RM, kl0.05, lr5e-7):**
- PS146 (k5 ll_min-6 fp0 ec0.001): −0.1pp (neutral).
- PS153 (k7 ll_min-6 fp0 ec0.0): +0.3pp (neutral/slightly+).
- **PS159 (k7 ll_min-4 fp5 ec0.0): −1.4pp, mmlu −3.5pp (mildly negative).**
=> **fp=5 shifts power from NEUTRAL to MILDLY NEGATIVE on Qwen3.** This is the gentle cross-family echo of fp=5's role on Gemma (a collapse driver, PS123/PS125) and with log_prob on Qwen3 (PS139 fp5 = −4.0pp). Even though the format penalty is nominally inert on the tag-free Qwen3 parser (format_error_ratio=0), its reward shaping still degrades outcomes. (ll_min=-4 higher floor here vs -6 in PS146/153 may add a minor contribution.)

**Consolidated takeaway (fp across families):** fp=0 is the robust choice everywhere. fp>0 = collapse on Gemma, net-negative with log_prob on Qwen3, and mildly-negative with power on Qwen3. Prefer fp=0.

Rerun: `EXP_ID=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm4 REWARD_TYPE=power POWER_K=7 POWER_LL_MIN=-4 USE_ACTOR_AS_RM=False KL=0.05 LR=5e-7 FORMAT_PENALTY=5 ENTROPY_COEFF=0.0 MAX_PROMPT=2048 MAX_RESP=512 bash experiments/smoke_qwen3.sh`
