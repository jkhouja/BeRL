### Attempt r1 — 2026-07-19T23:25:25+00:00

- **RUN_NAME:** `s2-ST20-Phase-stability-v2_g2_kl0.01_lr1e-6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr1e-6-kl0.01-n16-r1`
- **Host:** h100-156-003   **git:** `e841f0e`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=4 ll_min=-4 rm_mode=frozen baseline=False kl=0.01 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=1.0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-ST20-Phase-stability-v2_g2_kl0.01_lr1e-6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr1e-6-kl0.01-n16-r1 — https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/sexh8bay
- **Log path:** `logs/20260719/s2-ST20-Phase-stability-v2_g2_kl0.01_lr1e-6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr1e-6-kl0.01-n16-r1.log`

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
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.think_only_pg=False \
    actor_rollout_ref.actor.format_penalty=0.0 \
    actor_rollout_ref.actor.format_penalty_std_coef=1.0 \
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
    algorithm.kl_ctrl.kl_coef=0.01 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-ST20-Phase-stability-v2_g2_kl0.01_lr1e-6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr1e-6-kl0.01-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=google/gemma-2-2b-it \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=4 \
    +reward_model.power_ll_min=-4 \
    reward_model.format_penalty=0.0 \
    reward_model.format_penalty_std_coef=1.0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=4 \
    +actor_rollout_ref.power_ll_min=-4 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-v2_g2_kl0.01_lr1e-6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-19T23:26:00+00:00

- **RUN_NAME:** `s2-ST20-Phase-stability-v2_g2_kl0.01_lr1e-6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr1e-6-kl0.01-n16-r1`
- **Host:** h100-156-003   **git:** `e841f0e`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=4 ll_min=-4 rm_mode=frozen baseline=False kl=0.01 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=1.0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-ST20-Phase-stability-v2_g2_kl0.01_lr1e-6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr1e-6-kl0.01-n16-r1 — https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/sexh8bay
- **Log path:** `logs/20260719/s2-ST20-Phase-stability-v2_g2_kl0.01_lr1e-6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr1e-6-kl0.01-n16-r1.log`

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
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.think_only_pg=False \
    actor_rollout_ref.actor.format_penalty=0.0 \
    actor_rollout_ref.actor.format_penalty_std_coef=1.0 \
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
    algorithm.kl_ctrl.kl_coef=0.01 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-ST20-Phase-stability-v2_g2_kl0.01_lr1e-6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr1e-6-kl0.01-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=google/gemma-2-2b-it \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=4 \
    +reward_model.power_ll_min=-4 \
    reward_model.format_penalty=0.0 \
    reward_model.format_penalty_std_coef=1.0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=4 \
    +actor_rollout_ref.power_ll_min=-4 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-v2_g2_kl0.01_lr1e-6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


---

### Findings — r1 (2026-07-20, completed 190 steps)

**Hypothesis.** PC stability corner: **kl=0.01 + lr=1e-6** on Gemma-2 (vs the ST13 anchor's
kl=0.05/lr=5e-7). Flagged "most drift-prone in v1" — does the low-KL/high-LR combination blow up KL
under the merged v2 fixes? Differs from ST13 only in KL and LR.

**Canonical score (`scripts/score_run.py`):**
```
eval iters: 8 (step 0..190); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.3071  HM(last3)=0.3696  (baseline step0=0.093)
ToM avg(last5)=0.4435  avg(last3)=0.4877  (baseline step0=0.315)
gsm8k (separate): 0.4626 (step0=0.277, delta vs step0=+0.186)
mmlu (separate): 0.4702 (step0=0.383, delta vs step0=+0.087)
health(final): kl=2.652 entropy=2.403 resp_len=448.129 reward=-2.4 parseable=1.0 max_resp=512
ToM HM trajectory: 0:0.093 30:0.008 60:0.007 90:0.045 120:0.267 150:0.352 180:0.389 190:0.364
final training-step KL: 184:0.597 185:0.839 186:0.278 187:1.317 188:0.285 189:2.652
```

**Verdict.** The kl=0.01/lr=1e-6 corner is **KL-UNSTABLE** — training KL is erratic and blows up by
the end (final steps 0.6→0.84→1.32→2.65; final-eval kl=2.65 vs anchor ST13's 0.043). This
**confirms the "most drift-prone" label** even under the merged v2 fixes. The run also shows a
**mid-run collapse** (HM near-zero at steps 30–60 with repetition-loop CoT degeneration and response
length inflating toward the 512 cap ~450) followed by a **strong recovery** to the best Gemma ToM
scores seen (HM 0.37, avg 0.49; gsm8k +18.6pp, mmlu +8.7pp) with parseable=1.0. Net: attractive
endpoint metrics but the trajectory is **high-variance and non-reproducible** — the KL blowup makes
this corner **unsafe as a recipe**; prefer the stable anchor (ST13, kl=0.05/lr=5e-7). Use ToM **avg**
(not HM) for Gemma comparison per the family caveat; even so this corner tops the anchor on avg but
only by riding an unstable trajectory.

**How to rerun:** `RUN_STAGE=s2 EXP_NUM=ST20 EXP_ID=Phase-stability-v2_g2_kl0.01_lr1e-6 DATA_NAME=dcfg_smoke_mix_gemma DATA_TRAIN=data/dcfg_smoke_mix_gemma.parquet MODEL_PATH=google/gemma-2-2b-it REWARD_TYPE=power POWER_K=4 POWER_LL_MIN=-4 USE_ACTOR_AS_RM=False SUBTRACT_BASELINE=False KL=0.01 LR=1e-6 TRAIN_BATCH=32 MINI_BATCH=128 MICRO_BATCH=8 ROLLOUT_N=16 MAX_PROMPT=2048 MAX_RESP=512 ENTROPY_COEFF=0.001 FORMAT_PENALTY=0.0 FORMAT_PENALTY_STD_COEF=1.0 SEED=1 RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`
