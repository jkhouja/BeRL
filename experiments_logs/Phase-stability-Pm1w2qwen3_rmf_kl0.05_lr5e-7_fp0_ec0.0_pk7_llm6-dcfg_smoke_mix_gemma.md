### Attempt r1 — 2026-07-13T16:02:02+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-013-002   **git:** `b36bdee`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7 ll_min=-6 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1 https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/o6ztbgmt
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=7 \
    +actor_rollout_ref.power_ll_min=-6 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_



## Hypothesis (PS153, Qwen3-1.7B power k7 ll_min-6 frozen-RM, ec=0.0)
Higher-k / no-entropy power variant on Qwen3 (k=7 vs PS146's k=5; ec=0.0 vs 0.001). Frozen-RM, fp=0, kl0.05, lr5e-7. Tag-free parquet, require_answer_tags=False, Gen ctx 2048/512.

Key question: does raising the power exponent (k=7 sharpens the reward toward high-LL tokens) and removing the entropy floor (ec=0.0) shift Qwen3 off PS146's FLAT/neutral result? On Gemma, k=7 (PS116) DESTABILIZED vs k=5 (PS110) — KL blew up 3x. Watch: does k=7 similarly destabilize Qwen3, or does the thinking model tolerate it? ec=0.0 removes the entropy floor (was a collapse driver on Gemma actor-RM). Expect either (a) still flat/neutral like PS146, or (b) k=7 destabilizes (KL drift, eval decline). Compare PS146 (k5 neutral) and PS139 (log_prob negative).

- WandB: https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/o6ztbgmt ; main_ppo clean, no crash, max_response=512, require_answer_tags=False, micro_batch=8.

## FINAL findings (PS153, Qwen3-1.7B power k7 ll_min-6 frozen-RM ec=0.0) — NEUTRAL/STABLE (k=7 does NOT destabilize Qwen3)
Completed cleanly (step 189/191, GPUs→1MiB, no crash). Canonical scorer (`scripts/score_run.py --last 5`):
- **ToM avg(last5)=0.2748 vs base 0.2719 → +0.3pp (FLAT, marginally positive)**. HM(last5)=0.163 vs base 0.161 (+0.3pp).
- **mmlu=0.276 vs base 0.243 → +3.3pp**; gsm8k=0.482 vs base 0.487 → −0.5pp. Capabilities PRESERVED.
- health: resp_len=450 (Qwen3 thinking, stable from start, no hack), parseable=1.0, kl=0.003 (rock-stable the ENTIRE run), entropy=0.279.
- HM trajectory: **perfectly flat ~0.15–0.18 the whole run**, no decline, no collapse.

**Verdict:** power k7/ec0.0 on Qwen3-1.7B is **STABLE and NEUTRAL** (+0.3pp ToM, mmlu +3.3pp, gsm8k preserved). Essentially indistinguishable from the k5 sibling PS146 (−0.1pp).

**KEY CONTRAST — k=7 stability is model-family-dependent:**
- **Gemma PS116 (power k7 fp5 ec0.001): NET NEGATIVE −3.4pp, KL blew up to 0.193 (~3× the k5 winner)** — k=7 DESTABILIZED Gemma.
- **Qwen3 PS153 (power k7 fp0 ec0.0): NEUTRAL +0.3pp, KL 0.003 rock-stable** — k=7 does NOT destabilize Qwen3.
=> Qwen3 (thinking model, XFORMERS, no logit-softcap quirks) tolerates the sharper k=7 reward fine; Gemma-2 does not. (Note PS116 also carried fp=5 which is itself a Gemma collapse-driver per PS123/PS125 — so the Gemma instability may be fp-driven more than k-driven; regardless, on Qwen3 neither k7 nor ec0 caused trouble.)

**Qwen3 power summary (all frozen-RM, kl0.05, lr5e-7):** PS146 (k5 ll_min-6 ec0.001) = −0.1pp; **PS153 (k7 ll_min-6 ec0.0) = +0.3pp**. Both FLAT/STABLE, both preserve capabilities. Power on Qwen3 = SAFE but NO ToM gain (vs Gemma power +3–5pp and vs Qwen3 log_prob PS139 −4.0pp). Likely the 512 response cap on a thinking model and/or a strong base limit upside; future: try actor-RM / larger resp budget.

Rerun: `EXP_ID=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm6 REWARD_TYPE=power POWER_K=7 POWER_LL_MIN=-6 USE_ACTOR_AS_RM=False KL=0.05 LR=5e-7 FORMAT_PENALTY=0 ENTROPY_COEFF=0.0 MAX_PROMPT=2048 MAX_RESP=512 bash experiments/smoke_qwen3.sh`
