### Attempt r1 — 2026-07-13T21:00:25+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-033-004   **git:** `d6b0553`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7 ll_min=-6 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.power_k=7 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=7 \
    +actor_rollout_ref.power_ll_min=-6 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_


---
## OUTCOME (r1, scored 2026-07-13T23:06Z) — NULL — k7 does NOT unlock Qwen3; CLINCHES Gemma-specificity

**Run:** WandB 06xwgnd7, Qwen3-1.7B, ACTOR-RM (use_actor_as_rm=true VERIFIED; 'frozenRM' name = common.sh bug), power k=7, ll_min=-6, fp=0, ec=0.0, kl0.05, lr5e-7, MAX_RESP=512, 190 steps. Full horizon, clean, 0 OOM.

**Score (24 ToM benches, sub300):**
- HM(last5)=0.154, HM(last3)=0.147 vs step0 baseline 0.157 → **-0.003 (NULL/FLAT)**
- ToM avg(last5)=0.268 vs base 0.272 → -0.004 (flat)
- gsm8k 0.483 (step0 0.483, -0.000); mmlu 0.263 (step0 0.243, +0.020)
- Health(final): kl=0.003 (flat), entropy=0.297, resp_len=454, reward=0.604, parseable=1.0

**HM trajectory (flat):** 0:0.16 → 20:0.18 → 100:0.16 → 150:0.16 → 190:0.15. Oscillates 0.14-0.18, no trend.

**Dynamics:** k7's sharper reward gradient did NOT induce divergence — KL stayed FLAT 0.002-0.003 the whole run, resp_len stayed HIGH ~453-468 (no CoT collapse), same stuck pattern as every prior Qwen3 run. The higher k made no difference.

**VERDICT: NULL — this CLINCHES the Gemma-specificity conclusion.** k=7 (the exact k of the Gemma winner PS129) still gives flat-null on Qwen3 with flat KL. That is now **5 consecutive Qwen3-1.7B nulls** spanning both RM modes, both k (5,7), both ll_min (-6,-4), fp{0,5}, ec{0,0.001}:
| Exp | RM | k,ll_min,fp,ec | HM(last5) | base | KL dynamics |
|---|---|---|---|---|---|
| PS145 | frozen | 5,-6,0,0 | 0.154 | 0.161 | flat null |
| PS151 | frozen | 5,-4,5,0 | 0.154 | 0.160 | flat null |
| PS161 | ACTOR | 5,-6,0,0 | 0.159 | 0.161 | flat null |
| PS162 | ACTOR | 5,-6,0,.001 | 0.170 | 0.161 | flat null (marginal) |
| PS169 | ACTOR | 7,-6,0,0 | 0.154 | 0.157 | flat null |

**CONCLUSION: The BeRL power-reward actor-RM ToM win (Gemma-2-2B PS129 HM 0.43, ~5x) does NOT transfer to Qwen3-1.7B under ANY power knob combination in this sweep.** The Gemma win was mechanistically coupled to aggressive policy divergence (CoT collapse + KL blow-up to ~9.85); Qwen3-1.7B's native-thinking policy simply does not diverge at kl=0.05/lr=5e-7 — KL stays pinned at ~0.002-0.003 regardless of reward sharpness, so no learning pressure reshapes the CoT. To move Qwen3 at all would require a fundamentally stronger divergence lever (much higher LR, substantially lower or scheduled KL coefficient, or a large entropy bonus), which is outside this fixed-knob sweep. **Recommendation: deprioritize remaining Qwen3 power rma variants (PS170-176) — they will almost certainly replicate this null. Redirect compute to Gemma rows / PS129 follow-ups (milder-knob or early-stop-by-eval-HM variants to capture the Gemma win cleanly) or a Qwen3 LR/KL-lever probe.**

**Rerun:** `setsid env EXP_ID=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.0_pk7_llm6 DATA_NAME=dcfg_smoke_mix_gemma DATA_TRAIN=data/dcfg_smoke_mix_gemma.parquet REWARD_TYPE=power POWER_K=7 POWER_LL_MIN=-6 USE_ACTOR_AS_RM=true KL=0.05 LR=5e-7 FORMAT_PENALTY=0 ENTROPY_COEFF=0.0 MAX_RESP=512 TOTAL_EPOCHS=1 TEST_FREQ=10 bash experiments/train_behavior_qwen3.sh </dev/null >/dev/null 2>&1 &`
