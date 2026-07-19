### Attempt r1 — 2026-07-13T11:16:13+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rma_kl0.05_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1`
- **Host:** h100-076-003   **git:** `fccf2f9`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=actor baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/2048 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rma_kl0.05_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1.log`

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
    actor_rollout_ref.actor.optim.lr=1e-6 \
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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1 \
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
    +reward_model.reward_type=log_prob \
    +reward_model.power_k=2.0 \
    +reward_model.power_ll_min=-8.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=log_prob \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr1e-6_fp0_ec0.0_lp DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (r1 — authoritative, completed 2026-07-13 15:56)

**Verdict: FAILED transfer — log_prob is UNSAFE on Qwen3, just as on Gemma-2.** The Qwen2.5
log_prob winner (PS013) recipe does NOT transfer to Qwen3-1.7B. Training looked healthy but
downstream eval collapsed.

**Training health (fine, but not predictive):** reward negative & bounded well above the −40 floor
(~−7 to −10, gently improving, never pinned at the −80 invalid or −40 valid sentinel → tag-free
`is_invalid_response` fix works for Qwen3), format_error_ratio 0.000, advantages non-zero,
response_length verbose but stable ~700–870 (no brevity collapse, low clip_ratio). Ran full 1 epoch
(step 189 + final val step:190), clean exit, GPU → 1 MiB (no teardown hang).

**Downstream eval (26 × *_sub300, step:0 → step:190):**
- **MEAN 0.455 → 0.345 = −11.0pp**
- **20/26 worsened (>2pp), only 1 improved (hi_tom +4.3pp), 0 exact-zero.**
- Worst hits: simpletom_mental −45.0pp, opentom_multihop_fo −38.6pp, opentom_multihop_so −32.0pp,
  mmlu −25.0pp, opentom_attitude −24.0pp, tombench −22.4pp.
- gsm8k held (−0.4pp, still 0.853); the collapse is concentrated in ToM/MC reasoning.

**Cross-family × cross-reward summary (all actor-RM, tag-free gemma data, kl0.05):**

| Run   | Model      | Reward   | Base→Final   | Mean Δ      | Worsened | Verdict |
|-------|------------|----------|--------------|-------------|----------|---------|
| PS101 | Gemma-2-2B | log_prob | 0.316→0.081  | **−23.5pp** | 20/26    | FAILED (parseable-answer collapse, many exact-0) |
| PS141 | Qwen3-1.7B | log_prob | 0.455→0.345  | **−11.0pp** | 20/26    | FAILED (gradual degradation, no exact-0) |
| PS122 | Gemma-2-2B | power k5 | 0.316→0.413  | **+9.7pp**  | 0/26     | PASSED |

**Mechanism / lesson:** log_prob reward degrades downstream ToM on BOTH non-Qwen2.5 families,
whereas power reward improves it. On Gemma-2 the log_prob failure mode was a discrete
parseable-answer collapse (MC scores → exactly 0); on Qwen3 it is a softer, broad capability
degradation from a stronger base (no exact-zeros), but the sign and magnitude class are the same.
This strongly generalizes the Wave-2 conclusion: **bare log_prob reward (fp=0) does not transfer
across model families and is unsafe; power reward is the safe, transferable family.** The Qwen2.5
log_prob "winner" was family-specific and does not hold for cross-family transfer.

**Rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr1e-6_fp0_ec0.0_lp DATA_NAME=dcfg_smoke_mix_gemma REWARD_TYPE=log_prob USE_ACTOR_AS_RM=True KL=0.05 LR=1e-6 FORMAT_PENALTY=0 ENTROPY_COEFF=0.0 bash experiments/smoke_qwen3.sh`
