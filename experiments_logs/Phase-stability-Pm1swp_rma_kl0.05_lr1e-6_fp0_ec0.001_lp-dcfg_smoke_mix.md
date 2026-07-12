### Attempt r1 — 2026-07-11T19:10:03+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp0_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1`
- **Host:** h100-076-003   **git:** `967e489`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=actor baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp0_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp0_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.think_only_pg=True \
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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp0_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=999 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-3B-Instruct \
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
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp0_ec0.001_lp DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** WandB `55suvvvr`. Ran to completion (step 190, final validation logged); process **exited cleanly this time — no ray/vLLM teardown hang**, GPUs freed to 1 MiB. **Verdict: HEALTHY / STABLE — kl=0.05 stabilizes actor-RM, confirming high-KL as the primary stabilizer regardless of RM mode.**

Config: actor-as-RM, kl=0.05, lr=1e-6, fp=0, ec=0.001, log_prob reward.

Training dynamics (all healthy):
- `critic/rewards/mean`: −66.1 → −4.8 (normal climb).
- `response_length/mean`: **48.0 → 121.4** — squarely in the HEALTHY band (~110–126). No length-inflation, no brevity collapse. Direct contrast to same-LR PS022 (actor kl0.01 → brevity-drift to 57) and the frozen kl0.01 collapses (140+). **kl=0.05 holds response length healthy under actor-RM.**
- `format_error_ratio`: 0.000 throughout.

Eval (subsample300), step0 → final validation:
| Benchmark | step0 | final | Δpp |
|---|---|---|---|
| **aggregate mean (~26)** | 0.509 | ~0.505 | **~−0.4 (flat)** |
| simpletom_judgment | 0.293 | 0.407 | +11.4 |
| simpletom_behavior | 0.557 | 0.630 | +7.3 |
| explore_tom | 0.480 | 0.500 | +2.0 |
| bigtom_forward_action | 0.733 | 0.710 | −2.3 |
| fantom_answerability_binary | 0.207 | 0.170 | −3.7 |
| dyntom_type_a | 0.510 | 0.443 | −6.7 |
| tomi | 0.590 | 0.520 | −7.0 |
| opentom_attitude | 0.440 | 0.363 | −7.7 |
| bigtom_forward_belief | 0.767 | 0.677 | −9.0 |

Interpretation & hypothesis test: The monitor asked "does kl=0.05 stabilize actor-RM like it does frozen-RM?" **Answer: yes.** On the health axis (response length, reward trajectory, format), PS030 is clean/stable — the healthy 121-token responses are the signature difference from its kl=0.01 actor twin PS022 (which brevity-collapsed to 57 at the same LR). Aggregate eval is roughly FLAT (mixed per-benchmark; no net gain from plain log_prob reward on the small smoke_mix data — expected for a stability probe, not an accuracy-optimized run). This mirrors the frozen twin PS014 (kl0.05 lr1e-6 fp0 → MIXED/mostly-stable). **Reinforces the Phase −1 conclusion: kl=0.05 is the primary stabilizer across BOTH frozen and actor RM modes; fp=5 (as in PS011) adds further gains.** PS011 (frozen kl0.05+fp5) remains the leading `Pm1swp_best` candidate.

