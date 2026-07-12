### Attempt r1 — 2026-07-12T01:23:21+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1`
- **Host:** h100-189-003   **git:** `c8ba900`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=3 ll_min=-6.0 rm_mode=actor baseline=False kl=0.01 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1.log`

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
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
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
    algorithm.kl_ctrl.kl_coef=0.01 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1 \
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
    +reward_model.reward_type=power \
    +reward_model.power_k=3 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=3 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp0_ec0.0_pk3 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings (r1, 2026-07-12, h100-189-003, WandB nkuj96xz):** **COLLAPSED — entropy-collapse
instability** (a distinct failure mode from PS038's high-entropy blow-up, but same conclusion:
kl=0.01 + lr=1e-6 is unstable, now confirmed for actor-RM + power-k3).

- **Config-selection metric:** HM(last-3)=**0.272**, HM(last-5)=**0.174** vs step-0 baseline
  HM_tom=**0.419** → far below baseline. Peak HM=**0.460 @step30**, then catastrophic mid-run crater to
  **near-zero** (HM 0.017–0.054 across steps 120–170) with a partial oscillatory "recovery" to
  0.412 @step190 (unstable, not a real recovery — last-5 still 0.174).
- **Entropy collapse:** entropy 1.29→2.07(s40)→0.80(s100)→0.11(s160)→**0.009**(s189) — the actor-RM
  drives the policy to a near-deterministic, terse mode (resp_len 80→32). kl_loss explodes
  0.06→0.82→1.74→1.29. Reward saturates ~+24 (hacked) but the policy degenerates.
- **What crashed:** mmlu 0.470→0.06–0.11, simpletom_mental 0.86→0.14–0.20, bigtom_fwd_belief
  0.767→0.07–0.20 during the crater. Oddly **tomi held ~0.60 throughout** (single-benchmark
  robustness, not overall). gsm8k volatile 0.66→0.23–0.57. format_error=0 (parseable but degenerate).

| step | HM_tom | mmlu | tomi | bigtom_fwd_belief | simpletom_mental | reward | entropy | kl_loss | resp_len |
|---|---|---|---|---|---|---|---|---|---|
| 0   | 0.419 | 0.470 | 0.593 | 0.767 | 0.857 | −2.0(s5) | 1.29 | 0.06 | 80 |
| 30  | 0.460 | 0.590 | 0.613 | 0.700 | 0.850 | —        | —    | —    | —  |
| 100 | 0.176 | 0.150 | 0.610 | 0.430 | 0.150 | +17.9    | 0.80 | 1.74 | 40 |
| 160 | 0.017 | 0.063 | 0.607 | 0.070 | 0.187 | +25.2    | 0.11 | 1.03 | 46 |
| 190 | 0.412 | 0.523 | 0.557 | 0.693 | 0.577 | +24.0(s189)| 0.009| 1.29 | 32 |

**Verdict:** COLLAPSED — exclude; not a Phase-1 winner. Third confirmation that **kl=0.01 + lr=1e-6 is
unstable regardless of reward family or RM mode** (frozen log_prob PS005/PS006, frozen power PS038,
actor power PS053 all collapse). Actor-RM here fails via entropy-collapse/determinism rather than
entropy blow-up, but the config-selection HM-last-5 (0.174 ≪ 0.419) is a clear reject. Very poor ⇒
**skip in Wave-2**.

