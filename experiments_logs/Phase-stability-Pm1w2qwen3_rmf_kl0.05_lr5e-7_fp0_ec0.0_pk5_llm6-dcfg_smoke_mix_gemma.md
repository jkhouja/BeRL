### Attempt r1 — 2026-07-13T12:27:20+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-033-004   **git:** `855d28f`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/dcfg_smoke_mix_gemma.parquet \
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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen3-1.7B \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=false \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_


---
## OUTCOME (r1, scored 2026-07-13T14:35Z) — NULL (frozen-RM weak, as expected)

**Run:** WandB d33rrp4k, Qwen3-1.7B, FROZEN-RM (use_actor_as_rm=false; 'frozenRM' name is CORRECT here), power k=5, ll_min=-6, fp=0, ec=0.0, kl0.05, lr5e-7, MAX_RESP=512, 190 steps. Full horizon, clean exit, 0 OOM/0 tracebacks.

**Score (24 ToM benches, subsample300):**
- HM(last5)=0.154, HM(last3)=0.154 vs step0 baseline 0.161 → **-0.007 (NULL / flat, marginally below baseline)**
- ToM avg(last5)=0.268 vs base 0.272 → -0.004 (flat)
- gsm8k 0.473 (step0 0.487, -0.014); mmlu 0.257 (step0 0.243, +0.014)
- Health(final): kl=0.003 (very stable), entropy=0.278, resp_len=459, reward=-2.017, parseable=1.0

**HM trajectory (flat, no learning):** 0:0.16 → 50:0.15 → 100:0.18 → 150:0.16 → 190:0.16. Oscillates 0.14-0.18 the whole run; no upward trend.

**Dynamics:** Memory-stable throughout (Qwen3-1.7B small + bounded power reward). resp_len steady ~435-459, clip_ratio 0.47-0.60 (native-thinking runs long but never pinned at 1.0; no length-hacking). KL flat at 0.002-0.003 — the frozen power reward gives weak/compressed advantages, so the policy barely moves.

**VERDICT:** NULL. Frozen-RM power on Qwen3-1.7B produces no ToM transfer — mirrors the Gemma-2 frozen-RM result (PS114 HM 0.074, PS119 HM 0.087, both ~baseline). CONSISTENT with the key cross-model finding: **frozen-RM is weak; actor-RM is the winning ingredient** (Gemma actor-RM PS129 hit HM 0.43, ~5x baseline). A Qwen3 actor-RM row should be the real test of Wave-2 Qwen3 transfer. Note Qwen3's baseline HM (0.16) starts higher than Gemma's (0.085), so absolute numbers aren't directly comparable across model families — compare each to its own step0.

**Rerun:** `setsid env EXP_ID=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6 DATA_NAME=dcfg_smoke_mix_gemma DATA_TRAIN=data/dcfg_smoke_mix_gemma.parquet REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-6 USE_ACTOR_AS_RM=false KL=0.05 LR=5e-7 FORMAT_PENALTY=0 ENTROPY_COEFF=0.0 MAX_RESP=512 TOTAL_EPOCHS=1 TEST_FREQ=10 bash experiments/train_behavior_qwen3.sh </dev/null >/dev/null 2>&1 &`
