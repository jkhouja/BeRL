### Attempt r1 — 2026-07-12T10:39:46+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.05-n16-r1`
- **Host:** h100-007-002   **git:** `8008172`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=actor baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.05-n16-r1.log`

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
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.05-n16-r1 \
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
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp0_ec0.0_pk5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:**

#### Findings (completed 2026-07-12, exit 0, 191 steps / 1 epoch)

**Verdict: STABLE + HEALTHY — actor-RM version of the winning recipe (power k=5 × kl0.05 × lr1e-6) is top-band, ties the frozen versions. The recipe is RM-mode-agnostic.**

Config-selection score (HM of 26 subsample300 benchmarks):
- **HM(last-3) = 0.478**; HM(last-5) = 0.477 — +0.05 above step-0 baseline HM = 0.426. Mean-of-means(last-3)=0.535.
- HM holds a **high plateau 0.47–0.485** (peak 0.485 @step120), ends 0.476 @step190.
- Parseable-answer rate ≈ **100%** (`format_error_ratio`=0).

**Health checks PASS (clean):**
- **KL contained ~0.07–0.12** the whole run.
- **Entropy stable ~1.4** (no collapse).
- Response length stable ~104–113.

Eval HM trajectory:
`0:0.426 10:0.465 20:0.470 30:0.475 40:0.481 50:0.478 60:0.467 70:0.471 80:0.475 90:0.473 100:0.478 110:0.481 120:0.485(peak) 130:0.479 140:0.466 150:0.479 160:0.471 170:0.478 180:0.479 190:0.476`

Health trace: reward −28→+32→+40→+36→+35 (bounded); KL 0.017→0.079→0.098→0.110→0.073→0.117; entropy 1.11→1.40→1.45→1.48→1.49→1.23; resp_len 48→113→113→111→112→104.

**Key comparisons (HM last-3), all healthy:**
- vs frozen power k=5 kl0.05 (PS073 lr5e-7 = 0.475, PS080 lr1e-6 = 0.483): **actor (0.478) ≈ frozen** — RM mode is second-order for the winning recipe.
- vs actor power k=5 kl0.01 lr1e-6 (PS087 = 0.462): kl0.05 lifts it ~+0.016.

**Ranking (HM last-3, healthy, top band):** PS080 power-k5-kl0.05-frozen-lr1e-6 **0.483** ≈ PS093 power-k5-kl0.05-actor-lr1e-6 **0.478** ≈ PS073 power-k5-kl0.05-frozen-lr5e-7 **0.475** > PS087 power-k5-kl0.01-actor 0.462 ≈ PS065 power-k5-kl0.01-frozen 0.458.

**Implication:** the **power k=5 × kl0.05 winning recipe is robust across RM mode (frozen≈actor) and LR (5e-7≈1e-6)**, all landing HM ~0.475–0.483 with healthy KL/entropy. This is a strong, reproducible Phase −1 `stable` config: {reward=power, k=5, ll_min=−6, kl=0.05}, RM mode and lr∈[5e-7,1e-6] both fine.

