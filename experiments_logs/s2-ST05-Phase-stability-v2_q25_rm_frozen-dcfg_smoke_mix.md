### Attempt r1 — 2026-07-19T19:44:57+00:00

- **RUN_NAME:** `s2-ST05-Phase-stability-v2_q25_rm_frozen-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-156-003   **git:** `e841f0e`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=4 ll_min=-6 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=1.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-ST05-Phase-stability-v2_q25_rm_frozen-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1 — https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/o0d04nwj
- **Log path:** `logs/20260719/s2-ST05-Phase-stability-v2_q25_rm_frozen-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/dcfg_smoke_mix.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
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
    actor_rollout_ref.actor.format_penalty=0.0 \
    actor_rollout_ref.actor.format_penalty_std_coef=1.0 \
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
    trainer.experiment_name=s2-ST05-Phase-stability-v2_q25_rm_frozen-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-3B-Instruct \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=4 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=0.0 \
    reward_model.format_penalty_std_coef=1.0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=4 \
    +actor_rollout_ref.power_ll_min=-6 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-v2_q25_rm_frozen DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-19T19:45:10+00:00

- **RUN_NAME:** `s2-ST05-Phase-stability-v2_q25_rm_frozen-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-156-003   **git:** `e841f0e`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=4 ll_min=-6 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=1.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-ST05-Phase-stability-v2_q25_rm_frozen-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1 — https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/o0d04nwj
- **Log path:** `logs/20260719/s2-ST05-Phase-stability-v2_q25_rm_frozen-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/dcfg_smoke_mix.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
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
    actor_rollout_ref.actor.format_penalty=0.0 \
    actor_rollout_ref.actor.format_penalty_std_coef=1.0 \
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
    trainer.experiment_name=s2-ST05-Phase-stability-v2_q25_rm_frozen-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-3B-Instruct \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=4 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=0.0 \
    reward_model.format_penalty_std_coef=1.0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=4 \
    +actor_rollout_ref.power_ll_min=-6 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-v2_q25_rm_frozen DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


---

### Findings — r1 (2026-07-19, completed 190 steps)

**Hypothesis.** PB RM-mode axis: does the **frozen** base-LM reward model blow up KL under the
merged v2 fixes (tag-free invalid gate, relative invalid sentinel, std-gated format penalty,
single-epoch guard) at kl=0.05? Old stage: actor blew KL in 12/72 runs vs 0/80 frozen. Everything
else = Qwen2.5 PA anchor (ST01) except RM mode.

**Canonical score (`scripts/score_run.py`):**
```
eval iters: 8 (step 0..190); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.4622  HM(last3)=0.4653  (baseline step0=0.4163)
ToM avg(last5)=0.5205  avg(last3)=0.5229  (baseline step0=0.5034)
gsm8k (separate): 0.628 (step0=0.663, delta vs step0=-0.035)
mmlu (separate): 0.6098 (step0=0.483, delta vs step0=+0.127)
health(final): kl=0.086 entropy=1.419 resp_len=112.352 reward=34.852 parseable=1.0 max_resp=512
ToM HM trajectory: 0:0.416 30:0.468 60:0.464 90:0.461 120:0.453 150:0.463 180:0.462 190:0.47
```

**Verdict.** Frozen RM is **stable** at kl=0.05 under the merged fixes — no KL blowup (KL stayed
~0.07–0.12 throughout; final 0.086), parseable=1.0, response length healthy (~112, no collapse),
reward climbed to ceiling (~35/40). ToM HM improved +4.6pp over step0 (0.416→0.462), avg +1.7pp.
Capability regressions minimal: gsm8k −3.5pp, mmlu **+12.7pp**. Confirms the old-stage observation
that frozen RM does not exhibit the actor-RM KL instability; the merged fixes did not reintroduce it.

**How to rerun:** `RUN_STAGE=s2 EXP_NUM=ST05 EXP_ID=Phase-stability-v2_q25_rm_frozen DATA_NAME=dcfg_smoke_mix DATA_TRAIN=data/dcfg_smoke_mix.parquet MODEL_PATH=Qwen/Qwen2.5-3B-Instruct REWARD_TYPE=power POWER_K=4 POWER_LL_MIN=-6 USE_ACTOR_AS_RM=False SUBTRACT_BASELINE=False KL=0.05 LR=5e-7 TRAIN_BATCH=32 MINI_BATCH=128 MICRO_BATCH=8 ROLLOUT_N=16 MAX_PROMPT=2048 MAX_RESP=512 ENTROPY_COEFF=0.0 FORMAT_PENALTY=0.0 FORMAT_PENALTY_STD_COEF=1.0 SEED=1 RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`
