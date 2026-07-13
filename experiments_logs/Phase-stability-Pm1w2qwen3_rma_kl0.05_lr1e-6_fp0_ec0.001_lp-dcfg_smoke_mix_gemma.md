### Attempt r1 — 2026-07-13T11:17:47+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rma_kl0.05_lr1e-6_fp0_ec0.001_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1`
- **Host:** h100-189-003   **git:** `8f3b0df`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=actor baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr1e-6_fp0_ec0.001_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rma_kl0.05_lr1e-6_fp0_ec0.001_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr1e-6_fp0_ec0.001_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1 \
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
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=log_prob \
    +reward_model.power_k=2.0 \
    +reward_model.power_ll_min=-8.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=log_prob \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr1e-6_fp0_ec0.001_lp DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-13T11:17:58+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rma_kl0.05_lr1e-6_fp0_ec0.001_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1`
- **Host:** h100-189-003   **git:** `8f3b0df`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=actor baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr1e-6_fp0_ec0.001_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rma_kl0.05_lr1e-6_fp0_ec0.001_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr1e-6_fp0_ec0.001_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1 \
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
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=log_prob \
    +reward_model.power_k=2.0 \
    +reward_model.power_ll_min=-8.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=log_prob \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr1e-6_fp0_ec0.001_lp DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (r1 — completed 2026-07-13, host h100-189-003)

Run: WandB yc4r5z1c; log `logs/20260713/Phase-stability-Pm1w2qwen3_rma_kl0.05_lr1e-6_fp0_ec0.001_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1.log`

Canonical score (`scripts/score_run.py`):
```
eval iters: 39 (step 0..190); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.0824  HM(last3)=0.0706  (baseline step0=0.1605)
ToM avg(last5)=0.2322  avg(last3)=0.2312  (baseline step0=0.2719)
gsm8k (separate): 0.4186 (step0=0.487, delta vs step0=-0.068)
mmlu  (separate): 0.154  (step0=0.243, delta vs step0=-0.089)
health(final): kl=0.01 entropy=0.299 resp_len=475.191 reward=-6.908 parseable=1.0
```

Verdict: **NEGATIVE transfer / degradation.** Both HM (−7.8pp: 0.082 vs 0.160) and AM (−4.0pp:
0.232 vs 0.272) fall below baseline; gsm8k (−6.8pp) and mmlu (−8.9pp) also regress. HM trajectory
holds ~0.14 through step ~115 then declines steadily to 0.04–0.09 in the final third — a slow
degradation, not a sharp collapse.

Health/hacking signature: entropy collapsed to 0.299 (very low), response length saturated at ~475
(clip_ratio ~0.7 throughout, responses hitting the 512 cap), kl tiny (0.01). parseable stayed 1.0.
This is consistent with **actor-as-RM self-hacking on log_prob**: the actor can raise its own assigned
likelihood of the held-out human utterance by producing long, low-entropy (over-confident/repetitive)
CoT, which inflates reward without improving ToM — and in fact erodes it. The Qwen2.5 log_prob winner
recipe (PS013) does NOT transfer to Qwen3 under actor-RM at lr1e-6/fp0/ec0.001.

Implication for Wave-2 Qwen3: actor-RM + log_prob is a degenerate combination here (entropy collapse
+ length saturation). Contrast with frozen-RM log_prob and power variants (PS143+ / PS145+) — a frozen
scorer should not be hackable this way; watch whether frozen-RM avoids the entropy/length pathology.
