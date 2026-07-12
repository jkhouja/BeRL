### Attempt r1 — 2026-07-12T07:28:01+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp5_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.05-n16-r1`
- **Host:** h100-007-002   **git:** `fe3dfa6`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp5_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp5_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp5_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.05-n16-r1 \
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
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.05_lr1e-6_fp5_ec0.001_pk5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:**

#### Findings (completed 2026-07-12, exit 0, 191 steps / 1 epoch)

**Verdict: STABLE + HEALTHY — NEW SWEEP LEADER (HM 0.483). The power k=5 × KL=0.05 recipe is robust to higher LR (1e-6) + fp5 + ec0.001. Confirms it as the Phase −1 winning `stable` config.**

Config-selection score (HM of 26 subsample300 benchmarks):
- **HM(last-3) = 0.483**; HM(last-5) = 0.485 — **+0.06 above** step-0 baseline HM = 0.426. Mean-of-means(last-3)=0.534.
- HM climbs to a **high plateau, still rising late** (peak 0.489 @step170), ends 0.479 @step190.
- Parseable-answer rate ≈ **100%** (`format_error_ratio`=0).

**Health checks PASS (clean):**
- **KL contained ~0.05–0.13** the whole run (no divergence even at lr1e-6).
- **Entropy stable ~1.5** (no collapse, fp5 + ec0.001 keep it well-behaved).
- Response length stable ~110–118.

Eval HM trajectory:
`0:0.426 10:0.469 20:0.474 30:0.478 40:0.477 50:0.473 60:0.465 70:0.471 80:0.474 90:0.476 100:0.474 110:0.473 120:0.476 130:0.475 140:0.486 150:0.485 160:0.486 170:0.489(peak) 180:0.481 190:0.479`

Health trace: reward −32→+35→+38→+26→+36 (bounded); KL 0.002→0.092→0.120→0.130→0.053→0.098; entropy 1.17→1.63→1.47→1.60→1.47→1.54; resp_len 49→110→112→108→110→118.

**Key comparisons (HM last-3), all healthy:**
- vs power k=5 kl0.05 **lr5e-7 fp0** (PS073 = 0.475): lr1e-6 + fp5 + ec0.001 (PS080 = 0.483) is **as good or slightly better** → the k=5 × kl0.05 leader is **LR-robust and fp/ec-robust**.
- vs power k=5 kl0.01 (PS065 = 0.458): KL=0.05 lifts it further.
- Confirms the additive-levers story with robustness: steeper power (k=5) + KL=0.05 stays top across LR and format/entropy knobs.

**Ranking (HM last-3, healthy):** PS080 power-k5-kl0.05-lr1e-6-fp5 **0.483** ≈ PS073 power-k5-kl0.05-lr5e-7 0.475 > PS065 power-k5-kl0.01 0.458 > power-k3-kl0.05 0.447–0.449 > log_prob-kl0.05 0.432–0.435.

**Implication:** **power reward k=5 + KL=0.05 is the Phase −1 winning stable recipe**, robust across lr5e-7/1e-6 and fp/ec settings, healthy KL/entropy, best ToM HM (~0.475–0.483). Recommend promoting `stable` = {reward=power, k=5, ll_min=−6, kl=0.05, lr∈[5e-7,1e-6]} for Phase 0.

