### Attempt r1 — 2026-07-13T19:19:44+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-156-003   **git:** `40d792b`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7 ll_min=-4.0 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/4096 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1.log`

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
    data.max_response_length=4096 \
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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=7 \
    +reward_model.power_ll_min=-4.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=7 \
    +actor_rollout_ref.power_ll_min=-4.0 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm4 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (r1) — COMPLETED (best Qwen3 cell so far, +4.1pp HM)

Ran to completion (step 190), fully healthy — NO collapse (frozen-RM fp=5, as expected). Trajectory smoothly rising.

Score (`scripts/score_run.py`):
```
ToM HM(last5)=0.3833  HM(last3)=0.3852  (baseline step0=0.3425)
ToM avg(last5)=0.468   avg(last3)=0.4708  (baseline step0=0.4423)
gsm8k (separate): 0.8922 (step0=0.893, delta=-0.001)
mmlu  (separate): 0.6312 (step0=0.593, delta=+0.038)
health(final): kl=0.002 entropy=0.303 resp_len=577.074 reward=0.0 parseable=1.0
peak HM ~0.394 (steps 95/150/190); trajectory rising 0.34->0.39
```
- HM(last5) 0.383 vs baseline 0.343 = **+4.1pp** (beats PS147 fp5 ec0.0 k5 llm-6 which was +2.2pp).
- avg(last5) 0.468 vs baseline 0.442 = +2.6pp.
- Capabilities: gsm8k flat (-0.1pp), mmlu +3.8pp — no regression.
- resp_len stable ~560-800 (Qwen3 native-thinking), KL ~0.002 throughout, parseable=1.0 — textbook-clean.

**Verdict:** Best Qwen3-1.7B cell so far. Differences vs PS147: k=7 (vs 5), ll_min=-4 (vs -6), ec=0.001 (vs 0.0). The sharper power (k=7) + tighter floor (ll_min=-4) + small entropy floor gives a stronger, monotone HM gain (+4.1pp) with clean stability. Confirms frozen-RM fp=5 is safe on Qwen3 and BeRL transfers to Qwen3-1.7B with a solid ToM gain. Status=Completed.
