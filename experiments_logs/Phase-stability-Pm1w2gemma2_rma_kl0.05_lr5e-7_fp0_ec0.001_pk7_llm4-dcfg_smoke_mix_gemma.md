### Attempt r1 — 2026-07-13T09:35:53+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-156-003   **git:** `24b1a82`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7 ll_min=-4.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/1024 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.power_ll_min=-4.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=7 \
    +actor_rollout_ref.power_ll_min=-4.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (r1) — COMPLETED (stable, marginal HM gain, caps↑)

Ran to completion (step 190), fully healthy — NO collapse (fp=0 confirms fp>0 is the sole collapse driver; cf. PS132 fp5 ec0.001 which collapsed).

Score (`scripts/score_run.py`):
```
ToM HM(last5)=0.095  HM(last3)=0.0896  (baseline step0=0.093)
ToM avg(last5)=0.3621  avg(last3)=0.3624  (baseline step0=0.3152)
gsm8k (separate): 0.3134 (step0=0.273, delta=+0.040)
mmlu  (separate): 0.3968 (step0=0.387, delta=+0.010)
health(final): kl=0.038 entropy=1.558 resp_len=121.021 reward=-1.25 parseable=1.0
peak HM ~0.111 (steps 70/170); trajectory volatile ~0.02-0.11
```
- HM(last5) 0.095 vs baseline 0.093 = +0.2pp (within noise; Gemma HM is volatile).
- avg(last5) 0.362 vs baseline 0.315 = **+4.7pp** (meaningful broad shift).
- Capabilities: gsm8k +4.0pp, mmlu +1.0pp — no regression, slight gain.
- resp_len stable ~90-200 throughout, KL <0.05, parseable=1.0 — textbook-clean run.

**Verdict:** Stable, marginal ToM HM gain but solid avg + capability gains. Comparable to PS125 (fp0 ec0.0, HM 0.123, best Gemma) — slightly lower HM but similar avg/caps profile. Confirms fp=0 Gemma actor-RM cells are safe & mildly beneficial; ll_min=-4 vs -6 (PS125=-4 too) not clearly separable at Gemma's noise floor. Status=Completed.
