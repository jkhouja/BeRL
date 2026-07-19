### Attempt r1 — 2026-07-13T08:53:47+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-6.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-156-003   **git:** `0e18054`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7 ll_min=-6.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/1024 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-6.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-6.0-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-6.0-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.power_k=7 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=7 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (r1) — FAILED (response-length collapse; ec=0.001 did NOT rescue fp=5)

**Killed at step ~34** (driver PID 921402) — actor-as-RM co-adaptation reward-hacking collapse.

Rescue test result: **fp=5 collapses Gemma-2-2B actor-RM even WITH entropy floor ec=0.001.**
- resp_len crashed monotonically 156 → 18 tokens by step 13, stayed ~17–26 through step 31 (steps 32–34 minor uptick 36→49→72 = thrashing, not recovery; KL still ~0.5–0.6, far from healthy ~120–160).
- KL exploded past 1.0 from step 17 onward (peaks 1.6 @ step 30), same signature as PS123 (fp5 ec0.0).
- reward stayed high/positive (+4 to +13) throughout the collapse — actor-as-RM reward is UNTRUSTWORTHY under co-adaptation.

Score (partial, `scripts/score_run.py`):
```
ToM HM(last5)=0.089  HM(last3)=0.0826  (baseline step0=0.093)
ToM avg(last5)=0.307  avg(last3)=0.2871  (baseline step0=0.3151)
gsm8k (separate): 0.3626 (step0=0.277, delta=+0.086)
mmlu  (separate): 0.3712 (step0=0.387, delta=-0.016)
health(final): kl=0.219 entropy=1.578 resp_len=87.584 reward=13.538 parseable=1.0
ToM HM trajectory: 0:0.093 5:0.083 10:0.074 15:0.092 20:0.097 25:0.072 30:0.086 35:0.079
```
Both HM (0.089) and avg (0.307) BELOW baseline (0.093 / 0.315) → collapse degraded eval, no ToM gain.

**Conclusion:** `format_penalty=5` is a hard collapse driver for Gemma-2-2B actor-RM; entropy_coeff=0.001 does NOT rescue it (KL still blows past 1.0, resp_len still crashes to ~18). Combined with PS123 (fp5 ec0.0 collapsed) and PS125 (fp0 ec0.0 stable, best Gemma cell HM 0.123), the isolated driver is **fp>0**, not entropy. Recommendation for Gemma actor-RM: **keep fp=0**. Status=Failed.
