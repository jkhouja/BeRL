### Attempt r1 — 2026-07-13T16:43:58+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-033-004   **git:** `155d1a2`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_


---
## OUTCOME (r1, scored 2026-07-13T18:50Z) — NULL — KEY NEGATIVE: actor-RM win does NOT transfer to Qwen3

**Run:** WandB qv20om3p, Qwen3-1.7B, ACTOR-RM (use_actor_as_rm=true VERIFIED via dry-run; 'frozenRM' in name = common.sh naming bug), power k=5, ll_min=-6, fp=0, ec=0.0, kl0.05, lr5e-7, MAX_RESP=512, 190 steps. Full horizon, clean, 0 OOM/tracebacks.

**Score (24 ToM benches, sub300):**
- HM(last5)=0.159, HM(last3)=0.152 vs step0 baseline 0.161 → **-0.002 (NULL/FLAT)**
- ToM avg(last5)=0.269 vs base 0.272 → -0.003 (flat)
- gsm8k 0.485 (step0 0.487, -0.002); mmlu 0.246 (step0 0.243, +0.003)
- Health(final): kl=0.002 (very stable), entropy=0.285, resp_len=468, reward=-0.368, parseable=1.0

**HM trajectory (flat):** 0:0.16 → 70:0.18 → 100:0.17 → 140:0.14 → 190:0.15. Oscillates 0.14-0.18, no trend.

**Dynamics — CRITICAL difference vs Gemma PS129:** NONE of the PS129 actor-RM dynamics appeared on Qwen3. KL stayed FLAT at 0.002-0.003 the entire run (PS129 blew up to ~9.85). resp_len stayed HIGH ~455-468 (PS129's CoT COLLAPSED to ~2 tokens mid-run). Reward wide spread [0,40] as expected for actor-RM, but the policy simply did NOT diverge/exploit. Memory-stable throughout.

**VERDICT: KEY NEGATIVE RESULT — actor-RM is NOT a universal ingredient.** The dramatic Gemma-2 actor-RM win (PS129 HM 0.43, ~5x) does NOT replicate on Qwen3-1.7B: HM 0.159 = flat, identical to Qwen3 frozen-RM (PS145 0.154, PS151 0.154). So on Qwen3, BOTH frozen-RM AND actor-RM power give NULL ToM transfer. The Gemma win was coupled to aggressive policy divergence (CoT collapse + KL blow-up) that Qwen3's native-thinking policy did NOT undergo under identical knobs — Qwen3 kept long stable CoT and low KL, so no learning signal drove ToM up. **Implication:** the actor-RM finding is model-family-specific (Gemma-2), not a general BeRL mechanism. Qwen3 may need stronger divergence pressure (higher LR, lower/scheduled KL, or entropy bonus) to move at all, OR Qwen3-1.7B simply doesn't exhibit the collapse-then-generalize dynamic. Remaining Qwen3 rma variants (fp/ec/k/ll_min sweeps PS162-176) test whether ANY knob combination unlocks movement.

**Cross-model summary (power, kl0.05, lr5e-7):**
| Model | RM | k,ll_min,fp,ec | HM(last5) | base | dynamics |
|---|---|---|---|---|---|
| Gemma-2-2B | frozen | k7,-6,0,.001 (PS114) | 0.074 | 0.093 | stable, weak |
| Gemma-2-2B | frozen | k7,-4,5,0 (PS119) | 0.087 | 0.093 | stable, null |
| Gemma-2-2B | ACTOR | k7,-6,0,0 (PS129) | **0.43** | 0.085 | WIN + CoT collapse + KL->9.85 |
| Gemma-2-2B | ACTOR | k7,-4,5,.001 (PS136) | 0.127 | 0.085 | stable, modest (knobs killed win) |
| Qwen3-1.7B | frozen | k5,-6,0,0 (PS145) | 0.154 | 0.161 | flat null |
| Qwen3-1.7B | frozen | k5,-4,5,0 (PS151) | 0.154 | 0.160 | flat null |
| Qwen3-1.7B | ACTOR | k5,-6,0,0 (PS161) | 0.159 | 0.161 | flat null — NO collapse/blow-up |

**Rerun:** `setsid env EXP_ID=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm6 DATA_NAME=dcfg_smoke_mix_gemma DATA_TRAIN=data/dcfg_smoke_mix_gemma.parquet REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-6 USE_ACTOR_AS_RM=true KL=0.05 LR=5e-7 FORMAT_PENALTY=0 ENTROPY_COEFF=0.0 MAX_RESP=512 TOTAL_EPOCHS=1 TEST_FREQ=10 bash experiments/train_behavior_qwen3.sh </dev/null >/dev/null 2>&1 &`
