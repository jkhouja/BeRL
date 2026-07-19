### Attempt r1 — 2026-07-13T18:52:10+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-033-004   **git:** `40d792b`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1.log`

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
    actor_rollout_ref.actor.entropy_coeff=0.001 \
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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.use_actor_as_rm=true \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_


---
## OUTCOME (r1, scored 2026-07-13T20:58Z) — MARGINAL/NULL (entropy bonus did NOT unlock Qwen3)

**Run:** WandB 4pgjilxs, Qwen3-1.7B, ACTOR-RM (use_actor_as_rm=true VERIFIED; 'frozenRM' name = common.sh bug), power k=5, ll_min=-6, fp=0, ec=0.001, kl0.05, lr5e-7, MAX_RESP=512, 190 steps. Full horizon, clean, 0 OOM.

**Score (24 ToM benches, sub300):**
- HM(last5)=0.170, HM(last3)=0.167 vs step0 baseline 0.161 → **+0.009 (MARGINAL, within noise)**
- ToM avg(last5)=0.273 vs base 0.272 → +0.001 (flat)
- gsm8k 0.473 (step0 0.487, -0.014); mmlu 0.278 (step0 0.247, +0.031)
- Health(final): kl=0.003 (flat), entropy=0.271, resp_len=460, reward=3.383, parseable=1.0

**HM trajectory:** 0:0.16 → 30:0.13 → 100:0.17 → 150:0.17 → 190:0.16. Oscillates 0.13-0.18, tiny upward drift in 2nd half but within noise.

**Dynamics:** ec=0.001 did NOT unlock divergence — KL stayed FLAT 0.002-0.004 the whole run (same as PS161), resp_len stayed HIGH ~460 (no CoT collapse). The entropy bonus was too small to move Qwen3's stuck policy. Slightly higher final reward (3.38) than PS161 (-0.37) but no eval payoff.

**VERDICT: MARGINAL/NULL — confirms Qwen3 sweep is low-yield.** HM 0.170 is nominally the best Qwen3 run so far (vs PS145 0.154, PS151 0.154, PS161 0.159) but the +0.009 gain is within run-to-run noise and far from a real effect (Gemma actor PS129 was +0.34). The ec=0.001 entropy bonus did not induce the divergence dynamics that drove the Gemma win. **4th consecutive Qwen3 null (2 frozen + 2 actor) — strongly confirms the actor-RM ToM win is Gemma-2-specific, not a general BeRL mechanism.** Qwen3-1.7B's native-thinking policy resists the collapse-then-generalize dynamic under all knobs tried. Remaining rma variants (fp5, higher k, shallower ll_min) may be worth 1-2 more probes but the sweep is diminishing-returns; a bigger lever (much higher LR, lower/scheduled KL, or larger ec) would be needed to test if Qwen3 can move at all.

**Rerun:** `setsid env EXP_ID=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm6 DATA_NAME=dcfg_smoke_mix_gemma DATA_TRAIN=data/dcfg_smoke_mix_gemma.parquet REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-6 USE_ACTOR_AS_RM=true KL=0.05 LR=5e-7 FORMAT_PENALTY=0 ENTROPY_COEFF=0.001 MAX_RESP=512 TOTAL_EPOCHS=1 TEST_FREQ=10 bash experiments/train_behavior_qwen3.sh </dev/null >/dev/null 2>&1 &`
