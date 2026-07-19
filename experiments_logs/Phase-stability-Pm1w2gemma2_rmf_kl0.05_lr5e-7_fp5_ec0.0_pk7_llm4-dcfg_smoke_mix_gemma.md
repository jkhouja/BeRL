### Attempt r1 — 2026-07-13T04:25:12+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k7-llmin-4-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-033-004   **git:** `d812f2a`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k7-llmin-4-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k7-llmin-4-lr5e-7-kl0.05-n16-r1.log`

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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k7-llmin-4-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.use_actor_as_rm=false \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm4 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


---

## 2026-07-13 — r1 OUTCOME: COMPLETED (full horizon, memory-stable) — MARGINAL/NULL on HM, mild avg gain

**Run:** WandB l21tkaut. Ran cleanly to step 190 final eval, process exited (0 procs, 0 OOM/tracebacks).

**Score (`scripts/score_run.py`, 24 ToM benchmarks, excl gsm8k/mmlu):**
- **ToM HM(last5)=0.0872, HM(last3)=0.0893** vs **baseline step0=0.093 → ~AT baseline** (marginally below; null on HM).
- **ToM avg(last5)=0.3639, avg(last3)=0.3632** vs baseline 0.315 → **arithmetic mean +0.049** (broad ToM up).
- gsm8k=0.323 (+0.046 vs step0), mmlu=0.416 (+0.033).
- health(final): kl=0.029, entropy=1.356, resp_len=127.4, reward=-2.812, parseable=1.0.
- HM trajectory (early dip 0.01-0.05 @steps10-80, then RECOVERS to ~0.08-0.10 back-half — full-horizon
  recovery, would've been mis-killed if stopped early):
  0:0.093 10:0.045 20:0.019 30:0.02 40:0.011 50:0.042 60:0.046 70:0.019 80:0.031 90:0.093 100:0.073
  110:0.088 120:0.095 130:0.081 140:0.086 150:0.076 160:0.087 170:0.081 180:0.099 190:0.084.

**Interpretation:** power k=7, ll_min=-4, fp=5, ec=0.0 (frozen RM, kl0.05, lr5e-7) on Gemma-2-2B is
NULL on the HM metric (HM(last5) 0.087 ~ base 0.093, dominated by near-zero benches
fantom_info_list/answerability_list ~0.01, opentom_multihop_so ~0.09) but shows a mild broad ToM gain
on the arithmetic mean (+0.049) plus modest gsm8k/mmlu gains. Slightly BETTER than sibling PS114
(k=7 ll_min=-6 fp=0 ec=0.001: HM 0.074, avg flat) — i.e. ll_min=-4 + fp=5 edges out ll_min=-6 + fp=0
here, though neither clears baseline on HM.

**Memory stability:** bounded power reward held — resp_len ~127/512, clip 0.0 throughout, no OOM, full
190 steps. Again confirms power >> log_prob (PS100) for Gemma-2 Wave-2 stability.

**VERDICT = COMPLETED (healthy, stable); NULL on HM (0.087 ~ base 0.093) with mild avg-ToM +0.049.**
