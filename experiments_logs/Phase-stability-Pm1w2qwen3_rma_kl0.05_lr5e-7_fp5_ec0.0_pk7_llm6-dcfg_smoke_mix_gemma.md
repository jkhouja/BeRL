### Attempt r1 — 2026-07-13T22:22:31+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-021-003   **git:** `83ffea6`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7 ll_min=-6 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1.log`

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
    data.max_response_length=512 \
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
    actor_rollout_ref.actor.think_only_pg=True \
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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=999 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen3-1.7B \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=power \
    +reward_model.power_k=7 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=7 \
    +actor_rollout_ref.power_ll_min=-6 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-13T22:22:44+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-021-003   **git:** `83ffea6`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7 ll_min=-6 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1.log`

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
    data.max_response_length=512 \
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
    actor_rollout_ref.actor.think_only_pg=True \
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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=999 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen3-1.7B \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=power \
    +reward_model.power_k=7 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=7 \
    +actor_rollout_ref.power_ll_min=-6 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_


---

## Findings (r1, authoritative; completed 2026-07-14)

**NEAR-NEUTRAL — no ToM transfer gain, but (unlike PS144/PS166) NO capability collapse.** This is the
best-behaved Qwen3-1.7B actor-RM cell so far. Full 20-eval trajectory (steps 0→190) parsed from the log.

- **HM_tom** (primary for Qwen-family): base 0.161 → last-5 0.151 (**−6.1%**), last-3 0.151 (−5.9%).
- **AM_tom** (cross-check): base 0.272 → last-5 0.265 (**−2.5%**), last-3 0.264 (−3.0%).
- **HM_all/AM_all** (incl. gsm8k+mmlu): −5.8% / −2.4% (last-5) — much milder than PS144/PS166.
- **End-state benches vs step-0 (all near baseline, no collapse):** tomi 0.493→0.527, mmlu 0.243→**0.26**
  (up!), gsm8k 0.487→**0.47** (preserved), bigtom_forward_belief 0.32→0.337, hi_tom 0.367→0.387 slight ↑;
  simpletom_mental 0.573→0.45, tombench 0.39→0.363, opentom_multihop_so 0.183→0.123 mild ↓.
- **Health = clean:** reward −5.6 to positive (power k=7 ll_min=−6; never at −40 floor); format_error_ratio
  0.000; entropy stable 0.243–0.283 (no collapse); kl_loss max 0.004, zero steps >0.4; response_length ~458.

**Verdict + REVISED cross-cell synthesis:** PS171 (actor-RM, k=7, **ll_min=−6**) does NOT collapse
gsm8k/mmlu, whereas PS166 (actor-RM, k=5, **ll_min=−4**) cratered them (gsm8k 0.487→0.097) and PS144
(actor-RM, log_prob) likewise. Since all three are actor-RM, **actor-RM is NOT the sole culprit** — the
capability-destroying failure mode is driven by the **aggressive ll_min=−4** (tight LL floor → high-variance
reward that over-optimizes). The looser **ll_min=−6** keeps the effective reward gentle and the policy near
baseline (near-neutral, capability-preserving). NB this is the OPPOSITE of Gemma-2-2b, where ll_min=−4 was
essential and ll_min=−6 was the null/failure mode (PS105) — so the optimal LL floor is **model-family-specific**
(Gemma wants tight −4, Qwen3 wants loose −6). Net: no Qwen3-1.7B actor-RM cell yet yields POSITIVE transfer;
best case (PS171) is a capability-preserving wash. Recommend testing Qwen3 with ll_min=−6 at tighter/other
knobs and, ideally, frozen-RM ll_min=−6 to see if a positive Qwen3 transfer exists at all.

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet REWARD_TYPE=power POWER_K=7 POWER_LL_MIN=-6 USE_ACTOR_AS_RM=True KL=0.05 LR=5e-7 FORMAT_PENALTY=5 ENTROPY_COEFF=0.0 RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`
