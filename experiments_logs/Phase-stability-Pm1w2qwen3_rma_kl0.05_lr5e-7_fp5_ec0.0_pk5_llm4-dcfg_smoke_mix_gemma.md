### Attempt r1 — 2026-07-13T21:22:05+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-076-003   **git:** `d6b0553`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/2048 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1.log`

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
    data.max_response_length=2048 \
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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-4 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-4 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (r1 — authoritative, completed 2026-07-14 02:14)

**Verdict: PASSED — power reward with ACTOR-RM (+k5+fp5) yields real downstream gains on Qwen3.**
This recovers the positive transfer that the frozen-RM variant (PS154) missed, and it flips exactly
the benchmarks PS154 hurt into the largest gains here.

**Training health:** power reward hovered ~0 with fp=5 penalty active (band ~−1.4…+1.4), format
error stayed ~0, advantages non-zero, response_length stable ~585–750 (no collapse/inflation). Full
1 epoch (step 189 + final val step:190), clean exit, GPU → 1 MiB.

**Downstream eval (26 × *_sub300, step:0 → step:190):**
- **MEAN 0.455 → 0.480 = +2.5pp**
- **13 improved (>2pp), only 1 worsened (>2pp, simpletom_behavior −2.3), 0 collapsed-to-zero.**
- Biggest gains: opentom_multihop_fo +9.4, opentom_multihop_so +8.0, simpletom_judgment +5.4,
  simpletom_mental +5.3, fantom_belief_mc +5.0, dyntom_type_c +4.4, dyntom_type_d +4.0, tomi +3.7.
- **Direct contrast with PS154 (Qwen3 power, frozen-RM, k7, fp0):** the three benchmarks PS154 hurt
  most — opentom_multihop_fo (−9.0 → **+9.4**), opentom_multihop_so (−6.4 → **+8.0**),
  simpletom_mental (−5.4 → **+5.3**) — are now this run's top gains. The sign flip is systematic,
  not benchmark noise.

**Cross-family × reward × RM × fp summary (all tag-free gemma data, kl0.05):**

| Run   | Model      | Reward   | RM     | k | fp | LR    | Mean Δ      | Verdict |
|-------|------------|----------|--------|---|----|-------|-------------|---------|
| PS101 | Gemma-2-2B | log_prob | actor  | – | 0  | 1e-6  | **−23.5pp** | FAILED (collapse) |
| PS141 | Qwen3-1.7B | log_prob | actor  | – | 0  | 1e-6  | **−11.0pp** | FAILED (collapse) |
| PS154 | Qwen3-1.7B | power    | frozen | 7 | 0  | 5e-7  | **−1.2pp**  | SAFE (flat) |
| PS167 | Qwen3-1.7B | power    | actor  | 5 | 5  | 5e-7  | **+2.5pp**  | PASSED (gains) |
| PS122 | Gemma-2-2B | power    | actor  | 5 | 0  | 5e-7  | **+9.7pp**  | PASSED (gains) |

**Interpretation:** two consistent, orthogonal signals now emerge:
1. **Reward family (dominant):** log_prob collapses downstream on both non-Qwen2.5 families (2/2);
   power never collapses (0/3). This is the headline safety result.
2. **RM mode (magnitude):** among power runs, **actor-RM produces downstream gains on BOTH families**
   (Gemma +9.7pp, Qwen3 +2.5pp), whereas the single frozen-RM power run was flat (−1.2pp). PS167
   changes three knobs vs PS154 (RM actor-vs-frozen, k 5-vs-7, fp 5-vs-0), but the systematic sign
   flip on the very benchmarks frozen-RM degraded points at **actor-RM (self-scoring)** as the most
   likely driver of the recovered gain, with k=5 (softer than k=7) and fp=5 (format-parseability
   pressure) plausibly contributing. Qwen3's smaller gain vs Gemma is consistent with its higher
   starting accuracy (0.455 vs 0.316 → less headroom).

**Paper takeaway:** power + actor-RM is the safe, transferable, *and* beneficial BeRL recipe across
Qwen2.5, Gemma-2, and Qwen3; bare log_prob is unsafe cross-family. A clean isolation run
(Qwen3 power, actor-RM, k5, **fp0**) would separate the RM effect from the fp effect.

**Rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm4 DATA_NAME=dcfg_smoke_mix_gemma REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-4 USE_ACTOR_AS_RM=True KL=0.05 LR=5e-7 FORMAT_PENALTY=5 ENTROPY_COEFF=0.0 bash experiments/smoke_qwen3.sh`
