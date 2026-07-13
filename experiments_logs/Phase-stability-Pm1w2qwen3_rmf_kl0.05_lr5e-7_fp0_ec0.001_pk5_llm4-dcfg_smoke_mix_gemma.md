### Attempt r1 — 2026-07-13T13:57:29+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-189-003   **git:** `455f255`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=999 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen3-1.7B \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-4 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-4 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm4 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-13T13:57:41+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-189-003   **git:** `455f255`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=999 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen3-1.7B \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-4 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-4 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm4 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (r1 — completed 2026-07-13, host h100-189-003)

Run: WandB 9lmrhbh2; log `logs/20260713/Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1.log`

Canonical score (`scripts/score_run.py`):
```
eval iters: 39 (step 0..190); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.0754  HM(last3)=0.0695  (baseline step0=0.1605)
ToM avg(last5)=0.2572  avg(last3)=0.2583  (baseline step0=0.2719)
gsm8k (separate): 0.1706 (step0=0.487, delta vs step0=-0.316)
mmlu  (separate): 0.0846 (step0=0.243, delta vs step0=-0.158)
health(final): kl=0.002 entropy=0.275 resp_len=455.361 reward=0.033 parseable=1.0
```

Verdict: **NEGATIVE transfer + severe capability regression.** ToM HM −8.5pp (0.075 vs 0.160);
AM only mildly down (−1.5pp: 0.257 vs 0.272). Critically, gsm8k crashes −31.6pp (0.171 vs 0.487) and
mmlu −15.8pp (0.085 vs 0.243) — a large hit to math/general capability. HM trajectory holds ~0.15
through step ~100 then declines to 0.04–0.09 in the final third.

Health: NO entropy-collapse self-hacking signature (contrast PS142 actor-RM). entropy stable ~0.27
throughout (that is Qwen3's native low-entropy reasoning baseline, not a collapse), resp_len ~455
(Qwen3 native-thinking near the 512 cap), kl tiny (0.002), parseable=1.0. So this is not reward
hacking — the frozen-RM power reward simply pushes Qwen3 off-distribution, eroding both ToM and
(especially) capability.

MONITORING LESSON: during training the single-benchmark `tomi` watch stayed healthy (~0.54, even
rising), yet the aggregate HM declined and gsm8k crashed −31.6pp. Do NOT judge Qwen3 power runs by
tomi alone — always score the full suite; tomi is not representative of HM/capability here.

Wave-2 Qwen3 pattern so far: Qwen2.5 winner recipes do NOT transfer to Qwen3 —
- PS142 actor-RM log_prob: HM −7.8pp (entropy-collapse self-hack)
- PS150 frozen-RM power k5 ll_min−4: HM −8.5pp, gsm8k −31.6pp (capability crash, no hacking sig)
Both negative but via different mechanisms.
