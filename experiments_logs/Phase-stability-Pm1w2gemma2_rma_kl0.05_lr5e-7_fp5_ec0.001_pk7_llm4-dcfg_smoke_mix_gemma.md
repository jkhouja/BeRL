### Attempt r1 — 2026-07-13T09:42:53+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k7-llmin-4-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-033-004   **git:** `32d0351`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k7-llmin-4-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k7-llmin-4-lr5e-7-kl0.05-n16-r1.log`

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
    actor_rollout_ref.actor.format_penalty=5 \
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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k7-llmin-4-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=google/gemma-2-2b-it \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=true \
    +reward_model.reward_type=power \
    +reward_model.power_k=7 \
    +reward_model.power_ll_min=-4 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=7 \
    +actor_rollout_ref.power_ll_min=-4 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm4 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


---

## 2026-07-13 — r1 OUTCOME: COMPLETED (STABLE) — MODEST ToM gain; stability cost the PS129 win

**Run:** WandB wzgzjwgm. ACTOR-RM (log name 'frozenRM' = common.sh naming bug; run IS actor-RM).
Ran clean to step 190, exited (0 procs, 0 OOM/tracebacks).

**Score (`scripts/score_run.py`, 24 ToM benches, excl gsm8k/mmlu):**
- **ToM HM(last5)=0.1271, HM(last3)=0.1124** vs baseline step0=0.0854 → **+0.042 (~1.5x)**: modest.
- ToM avg(last5)=0.3779 vs baseline 0.315 → +0.063 (mild broad gain).
- gsm8k=0.299 (+0.022), mmlu=0.409 (+0.022) — small.
- health(final): **kl=0.026 (LOW/stable), entropy=1.6, resp_len=139.2 (stable), reward=-4.21, parseable=1.0.**
- HM trajectory (noisy, no strong trend): 0:0.085 10:0.042 ... 140:0.129 150:0.102 160:0.171 170:0.081
  180:0.12 190:0.121 — ends ~0.12, far below PS129's 0.43.

**Cross-run comparison (all Gemma-2-2B power k=7, kl0.05, lr5e-7, dcfg_smoke_mix_gemma):**
| Row  | RM     | ll_min | fp | ec    | HM(last5) | final kl | dynamics |
|------|--------|--------|----|-------|-----------|----------|----------|
| PS114| frozen | -6     | 0  | 0.001 | 0.074     | low      | null on HM, stable |
| PS119| frozen | -4     | 5  | 0.0   | 0.087     | 0.03     | null on HM, stable |
| PS129| ACTOR  | -6     | 0  | 0.0   | **0.43**  | **9.85** | BIG win, late KL blow-up |
| PS136| ACTOR  | -4     | 5  | 0.001 | 0.127     | 0.026    | modest, STABLE (no blow-up) |

**Interpretation / KEY FINDING:** The hypothesis (fp=5/ec=0.001/ll_min=-4 would stabilize PS129's
KL blow-up) HELD — PS136 kept kl≈0.026 and steady resp_len with NO late explosion. BUT stability came
at the cost of the win: HM dropped 0.43 → 0.127. PS129's large ToM gain was COUPLED to its aggressive
policy divergence (KL→~10); the format penalty + entropy bonus + shallower ll_min that tame KL also
suppress the very policy change that produced the gain. So among actor-RM cells, the "unstable but big"
PS129 >> "stable but modest" PS136. Actionable: to chase PS129's win with stability, try milder single
knobs (e.g. ll_min=-6 + small fp only, or a KL schedule / kl_coef bump) rather than stacking fp+ec+shallow-llmin,
and add an early-stop/best-checkpoint-by-eval-HM selection so the win is captured before any blow-up.

**VERDICT = COMPLETED (stable); MODEST ToM (HM-last5 0.127 vs base 0.085). Beats frozen-RM (PS114/119)
but far below PS129 (0.43) — the stabilizing knobs traded away the win.**
