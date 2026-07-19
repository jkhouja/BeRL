### Attempt r1 — 2026-07-13T04:38:48+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-076-003   **git:** `ac8969f`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/1024 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1.log`

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
    data.max_response_length=1024 \
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
    actor_rollout_ref.actor.format_penalty=0 \
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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=999 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=google/gemma-2-2b-it \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (r1 — authoritative, completed 2026-07-13 10:44)

**Verdict: SUCCESSFUL transfer — power reward is safe on Gemma-2.** The Qwen2.5 power winner
(PS074) recipe transferred cleanly to Gemma-2-2b-it. Both training AND downstream eval are healthy.

**Training health:** reward positive & climbing throughout (5.7 → stable ~10–16, ended ~11–15),
format_error_ratio 0.000, 0 invalid, response_length bounded/steady ~135–180 (no brevity collapse,
no inflation toward 1024), advantages non-zero. Ran full 1 epoch (step 189 + final val step:190),
clean exit, GPU → 1 MiB (no teardown hang).

**Downstream eval (26 × *_sub300, step:0 → step:190):**
- **MEAN 0.316 → 0.413 = +9.7pp**
- **22/26 improved (>2pp), 0 worsened (>2pp), 0 collapsed-to-zero.**
- Biggest gains: opentom_multihop_fo +30.0pp, gsm8k +22.0pp, dyntom_type_a +19.0pp,
  tombench +17.7pp, opentom_location_so +17.0pp, explore_tom +16.4pp, simpletom_mental +13.4pp.
- Only tomi essentially flat (−0.4pp); no benchmark regressed materially.

**Cross-family / cross-reward comparison (all Gemma-2-2b-it, actor-RM, gemma tag-free data):**

| Run   | Reward   | Mean eval Δ | Worsened | Zero-collapse | Verdict |
|-------|----------|-------------|----------|---------------|---------|
| PS101 | log_prob | **−23.5pp** | 20/26    | many (tomi, simpletom_mental, hi_tom → exactly 0) | FAILED |
| PS122 | power k5 | **+9.7pp**  | 0/26     | none          | PASSED |

**Mechanism / lesson:** log_prob's invalid/floor sentinel is −40 and its valid floor is also −40,
so degenerate (unparseable) rollouts are not distinguished from bad-but-valid ones — the policy
drifts into a non-answer basin (parseable-answer collapse, many MC → exactly 0.000). Power reward's
floor is **0** (bounded, non-negative), which keeps the reward landscape from rewarding the collapse
basin; the model instead climbs on genuine LL improvement and downstream ToM/QA accuracy rises.
**On Gemma-2, reward family choice is decisive: power >> log_prob.** This mirrors the Qwen2.5 result
where power (PS074) was the winner, and extends it: power is not just best-performing but the only
*safe* family for cross-family transfer to Gemma-2. Bare log_prob on Gemma-2 needs anti-collapse
scaffolding; power does not.

**Rerun:** `EXP_ID=Pm1w2gemma2 DATA_NAME=dcfg_smoke_mix_gemma REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-6 USE_ACTOR_AS_RM=True KL=0.05 LR=5e-7 FORMAT_PENALTY=0 ENTROPY_COEFF=0.001 bash experiments/smoke_gemma.sh`
