### Attempt r1 — 2026-07-14T00:26:48+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k7.0-llmin-4.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-077-004   **git:** `d5feebd`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7.0 ll_min=-4.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=2 max_ctx=2048/4096 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k7.0-llmin-4.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260714/Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k7.0-llmin-4.0-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k7.0-llmin-4.0-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=2 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen3-1.7B \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=power \
    +reward_model.power_k=7.0 \
    +reward_model.power_ll_min=-4.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=7.0 \
    +actor_rollout_ref.power_ll_min=-4.0 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-14T00:27:02+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k7.0-llmin-4.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-077-004   **git:** `d5feebd`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7.0 ll_min=-4.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=2 max_ctx=2048/4096 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k7.0-llmin-4.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260714/Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k7.0-llmin-4.0-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k7.0-llmin-4.0-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=2 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen3-1.7B \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=power \
    +reward_model.power_k=7.0 \
    +reward_model.power_ll_min=-4.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=7.0 \
    +actor_rollout_ref.power_ll_min=-4.0 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_


---

## Findings — PS174 COMPLETED (2026-07-14, h100-077-004)

**Verdict: NEUTRAL/FLAT (clean run, no collapse).** ACTOR-as-RM softens the POWER k7/llm4 regression seen under frozen-RM, landing ~baseline; but still below log_prob.

### HM_tom per step (harmonic mean over 24 ToM subtypes, excl mmlu/gsm8k)
| step | HM_tom | gsm8k | mmlu |
|------|--------|-------|------|
| 0 (base) | 0.3416 | 0.887 | 0.593 |
| 30 | 0.3136 | 0.907 | 0.560 |
| 60 | 0.3090 | 0.877 | 0.497 |
| 90 | **0.3619 (peak)** | 0.903 | 0.617 |
| 120 | 0.3185 | 0.887 | 0.543 |
| 150 | 0.3348 | 0.867 | 0.570 |
| 180 | 0.3321 | 0.900 | 0.573 |
| 210 | 0.3183 | 0.890 | 0.537 |
| 240 | 0.3158 | 0.887 | 0.550 |
| 270 | 0.3164 | 0.890 | 0.560 |
| 300 | 0.3064 | 0.907 | 0.550 |
| 330 | 0.3389 | 0.900 | 0.613 |
| 360 | 0.3559 | 0.897 | 0.620 |
| 380 | 0.3407 | 0.903 | 0.603 |

- **base(step0)=0.3416 → last5=0.3317 (−1.0pp), last3=0.3452 (+0.4pp, ~baseline); peak=0.3619@90 (+2.0pp).**
- **Late epoch-2 recovery**: after a mid-run trough (0.306@300) it climbs 0.339@330 → 0.356@360 → 0.341@380, i.e. it returns to ~baseline by the end rather than continuing to fall.
- General caps stable: gsm8k ~0.87–0.91, mmlu volatile 0.50–0.62 (final 0.603).

### Health
- Reward sparse/spiky (mostly 0 with spikes ±1.25 and occasional MAX clamp +40), `critic/advantages/max` nonzero on most steps (up to 3.75), KL ~0.002 throughout. No collapse — expected POWER signature.

### Comparison — actor-RM vs frozen-RM for high-floor POWER on Qwen3
| run | RM | ec | last5 | last3 | peak | verdict |
|-----|-----|-----|-------|-------|------|---------|
| PS157 | frozen | 0.0 | 0.3019 (−4.1pp) | 0.3087 | 0.3628@60 | REGRESSION (drifted down, no recovery) |
| **PS174** | **actor** | **0.001** | **0.3317 (−1.0pp)** | **0.3452 (+0.4pp)** | **0.3619@90** | **NEUTRAL/FLAT (late recovery to baseline)** |
| PS137 (log_prob) | frozen | 0.0 | 0.3952 (+5.5pp) | — | 0.4008@380 | STABLE ✅ |
| PS138 (log_prob) | frozen | 0.001 | 0.4015 (+6.0pp) | — | 0.4113@380 | STABLE ✅ |

- **Actor-as-RM is clearly better than frozen-RM for POWER k7/llm4 on Qwen3** (−1.0pp vs −4.1pp; recovers late vs keeps dropping). The actor scoring the held-out utterance evidently provides a less-mis-calibrated (self-consistent) signal than a frozen scorer under the high ll_min=−4 floor.
- **But neither POWER config reaches log_prob** (+5.5–6.0pp). For Qwen3, **log_prob remains the recommended reward**; the Qwen2.5 POWER winner recipe (high floor) does not transfer well because Qwen3's lower utterance LLs (~−7) clamp most rewards under ll_min=−4. A lower floor (ll_min≈−6/−8) would likely be needed to make POWER competitive on Qwen3 (cf Qwen2.5 power-ext PS179 k7/llm6 last5 0.469).

### Rerun
```
EXP_ID=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4 DATA_NAME=dcfg_smoke_mix_gemma \
MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet \
REWARD_TYPE=power POWER_K=7.0 POWER_LL_MIN=-4.0 USE_ACTOR_AS_RM=True SUBTRACT_BASELINE=False \
KL=0.05 LR=5e-7 FORMAT_PENALTY=0 ENTROPY_COEFF=0.001 THINK_ONLY_PG=False bash experiments/train_behavior_qwen3.sh
```
WandB u83sx80i · log logs/20260714/Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k7.0-llmin-4.0-lr5e-7-kl0.05-n16-r1.log
