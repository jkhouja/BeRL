### Attempt r3 — 2026-07-17T00:38:24+00:00

- **RUN_NAME:** `data-recipe-P0_single_thoughttrace-dcfg_thoughttrace_notags-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r3`
- **Host:** h100-082-004   **git:** `15ecb88`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `data/dcfg_thoughttrace_notags.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=data-recipe-P0_single_thoughttrace-dcfg_thoughttrace_notags-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r3 _(paste link after launch)_
- **Log path:** `logs/20260717/data-recipe-P0_single_thoughttrace-dcfg_thoughttrace_notags-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r3.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/dcfg_thoughttrace_notags.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
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
    trainer.experiment_name=data-recipe-P0_single_thoughttrace-dcfg_thoughttrace_notags-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r3 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-3B-Instruct \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=data-recipe-P0_single_thoughttrace DATA_NAME=dcfg_thoughttrace_notags MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=data/dcfg_thoughttrace_notags.parquet RUN_INDEX=3 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


---

## Findings (r3 — tag-free relaunch, COMPLETED)

**Verdict: NEGATIVE / NULL transfer.** The tag-free path fixed the mechanics (run trained
to completion, parseable=1.0, gradients flowed all 177 steps — no −45 floor), but the
ThoughtTrace single domain does **not** induce ToM transfer.

Canonical scorer (`scripts/score_run.py`):
```
eval iters: 37 (step 0..177); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.4048  HM(last3)=0.4132  (baseline step0=0.4167)
ToM avg(last5)=0.4925  avg(last3)=0.4962  (baseline step0=0.503)
gsm8k (separate): 0.5218 (step0=0.67, delta=-0.148)
mmlu  (separate): 0.474  (step0=0.47, delta=+0.004)
health(final): kl=1.284 entropy=0.465 resp_len=15.834 reward=-0.021 parseable=1.0
```
- **ToM:** HM flat-to-slightly-down (−1.19pp), avg −1.05pp. HM trajectory hovers at/below
  the step-0 baseline (0.417) for the entire run — never meaningfully exceeds it. No transfer.
- **Capability:** gsm8k regresses hard (−14.8pp); mmlu flat. Training on short human↔AI
  next-turn prediction erodes math reasoning without buying ToM.
- **Health:** clean — parseable=1.0, moderate KL drift (1.28), short responses (resp_len≈16,
  as expected for next-user-turn prediction). No collapse/hacking.

**Interpretation (single-domain transfer picture):** ThoughtTrace is a POOR single-domain
transfer source for ToM, in contrast to cga (E019, POSITIVE) and the 10-domain mix (E025,
POSITIVE). Human↔AI dialogue predicting the human's short next turn does not carry the
mental-state signal that human↔human negotiation/social dialogue does. Answers the
"which single domain transfers best?" question in the negative for thoughttrace.

**Mechanics note:** unblocked by the tag-free `is_invalid_response` fix (commit 8769467);
ran on `dcfg_thoughttrace_notags` (cot_eval_notags, REQUIRE_ANSWER_TAGS=False, actor-RM,
power k=5 ll_min=−6). Rerun: see `How to rerun` above with RUN_INDEX bumped.
