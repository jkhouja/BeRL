### Attempt r1 — 2026-07-14T20:01:23+00:00

- **RUN_NAME:** `data-recipe-P0g_mix_all-dcfg_mix_all_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-082-004   **git:** `743fe84`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `data/dcfg_mix_all_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=data-recipe-P0g_mix_all-dcfg_mix_all_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260714/data-recipe-P0g_mix_all-dcfg_mix_all_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/dcfg_mix_all_gemma.parquet \
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
    trainer.experiment_name=data-recipe-P0g_mix_all-dcfg_mix_all_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=google/gemma-2-2b-it \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-4 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-4 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=data-recipe-P0g_mix_all DATA_NAME=dcfg_mix_all_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=data/dcfg_mix_all_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings (E101, completed 2026-07-15, r1):** POSITIVE.

Canonical scorer output (`python scripts/score_run.py <log>`):
```
eval iters: 165 (step 0..816); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.1734  HM(last3)=0.1773  (baseline step0=0.0935)
ToM avg(last5)=0.4124  avg(last3)=0.4141  (baseline step0=0.3151)
gsm8k (separate): 0.372 (step0=0.28, delta vs step0=+0.092)
mmlu  (separate): 0.404 (step0=0.387, delta vs step0=+0.017)
health(final): kl=6.909 entropy=0.36 resp_len=362.305 reward=0.274 parseable=1.0 max_resp=512
```

**Verdict:** POSITIVE — Gemma-2-2B mix_all (frozen RM, tag-free) roughly **doubles ToM HM**:
HM(last5)=0.1734 vs base 0.0935 (**+7.99pp**); avg(last5)=0.4124 vs 0.3151 (**+9.73pp**).
No capability regression — gsm8k +0.092, mmlu +0.017 (both up). Parseable rate 1.0 throughout.

**Health / caveats:**
- Training was stable & advancing all 816 steps; step-1 healthy (reward -1.4, advantages nonzero
  max +3.45, grad_norm 0.19), no E023-style invalid-sentinel deadlock (mixed data + tag-free parser).
- HM trajectory rose steadily, **peaked ~0.28-0.29 around steps 470-630**, then **declined to
  ~0.15-0.18 by the end** (late-training softening). The last5 window (0.173) sits below the peak;
  a mid-run checkpoint (~step 620) may be the stronger operating point if selecting for peak ToM.
- **KL drift is high (final kl=6.909)** — the policy moved substantially from the frozen reference
  despite KL coef 0.05. Not catastrophic (no collapse, parseable=1.0, capabilities preserved), but
  worth flagging vs the Qwen2.5 arm; a higher KL coef or earlier stop could tighten this.
- Single transient grad_norm spike to 2.71 at step ~524 (clipped by grad_clip=1.0), self-resolved.

**Comparison:** mirrors E025 (Qwen2.5 mix_all, +5.45pp HM) — the mix_all recipe transfers to the
Gemma-2 family and is even larger in relative terms here (base HM was lower, 0.094).

