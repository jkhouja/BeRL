### Attempt r1 — 2026-07-19T21:10:13+00:00

- **RUN_NAME:** `s2-ST13-Phase-stability-v2_g2_anchor_pk4_stdgate-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-156-003   **git:** `e841f0e`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=4 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=1.0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-ST13-Phase-stability-v2_g2_anchor_pk4_stdgate-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr5e-7-kl0.05-n16-r1 — https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/nix3o6rs
- **Log path:** `logs/20260719/s2-ST13-Phase-stability-v2_g2_anchor_pk4_stdgate-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr5e-7-kl0.05-n16-r1.log`

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
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-ST13-Phase-stability-v2_g2_anchor_pk4_stdgate-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr5e-7-kl0.05-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-v2_g2_anchor_pk4_stdgate DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-19T21:10:28+00:00

- **RUN_NAME:** `s2-ST13-Phase-stability-v2_g2_anchor_pk4_stdgate-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-156-003   **git:** `e841f0e`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=4 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=1.0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-ST13-Phase-stability-v2_g2_anchor_pk4_stdgate-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr5e-7-kl0.05-n16-r1 — https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/nix3o6rs
- **Log path:** `logs/20260719/s2-ST13-Phase-stability-v2_g2_anchor_pk4_stdgate-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr5e-7-kl0.05-n16-r1.log`

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
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-ST13-Phase-stability-v2_g2_anchor_pk4_stdgate-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k4-llmin-4-lr5e-7-kl0.05-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-v2_g2_anchor_pk4_stdgate DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


---

### Findings — r1 (2026-07-19, completed 190 steps)

**Hypothesis.** PA v2 **Gemma-2 family anchor**: power k4 + std-gated fp (fp_std_coef=1.0), frozen
base-LM RM, tag-free CoT (cot_eval_notags, require_answer_tags=False). Establishes the stable Gemma-2
reference all other Gemma cells (ST14–ST24) vary one axis from.

**Canonical score (`scripts/score_run.py`):**
```
eval iters: 8 (step 0..190); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.069  HM(last3)=0.062  (baseline step0=0.0931)
ToM avg(last5)=0.3298 avg(last3)=0.3343 (baseline step0=0.3155)
gsm8k (separate): 0.306 (step0=0.277, delta vs step0=+0.029)
mmlu (separate): 0.3846 (step0=0.39, delta vs step0=-0.005)
health(final): kl=0.043 entropy=1.733 resp_len=137.529 reward=-1.25 parseable=1.0 max_resp=512
ToM HM trajectory: 0:0.093 30:0.044 60:0.08 90:0.071 120:0.082 150:0.066 180:0.046 190:0.063
```

**Verdict.** Gemma-2 2B anchor is **stable** — no collapse, no KL blowup (KL ~0.01–0.11, final
0.043), parseable=1.0 throughout, response length healthy (~138). Capability regressions negligible
(gsm8k +2.9pp, mmlu −0.5pp). **Metric caveat:** ToM **HM is unreliable for this family** — two
benchmarks are floored at baseline (`fantom_answerability_list`=0.007, `fantom_info_list`=0.007,
both list-type FANToM tasks Gemma-2 2B effectively can't do), and the harmonic mean is dominated by
these near-zero terms. The **arithmetic avg is the reliable selection signal here** and is slightly
up (0.316→0.330, +1.4pp). Net: BeRL on Gemma-2 does not harm ToM and is training-stable; use **avg**
(not HM) for Gemma cross-cell comparison.

**How to rerun:** `RUN_STAGE=s2 EXP_NUM=ST13 EXP_ID=Phase-stability-v2_g2_anchor_pk4_stdgate DATA_NAME=dcfg_smoke_mix_gemma DATA_TRAIN=data/dcfg_smoke_mix_gemma.parquet MODEL_PATH=google/gemma-2-2b-it REWARD_TYPE=power POWER_K=4 POWER_LL_MIN=-4 USE_ACTOR_AS_RM=False SUBTRACT_BASELINE=False KL=0.05 LR=5e-7 TRAIN_BATCH=32 MINI_BATCH=128 MICRO_BATCH=8 ROLLOUT_N=16 MAX_PROMPT=2048 MAX_RESP=512 ENTROPY_COEFF=0.001 FORMAT_PENALTY=0.0 FORMAT_PENALTY_STD_COEF=1.0 SEED=1 RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`
