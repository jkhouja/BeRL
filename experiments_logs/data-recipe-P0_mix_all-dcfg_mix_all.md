### Attempt r1 — 2026-07-14T09:58:08+00:00

- **RUN_NAME:** `data-recipe-P0_mix_all-dcfg_mix_all-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-082-004   **git:** `22b5f49`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `data/dcfg_mix_all.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=data-recipe-P0_mix_all-dcfg_mix_all-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260714/data-recipe-P0_mix_all-dcfg_mix_all-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/dcfg_mix_all.parquet \
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
    trainer.experiment_name=data-recipe-P0_mix_all-dcfg_mix_all-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-3B-Instruct \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=data-recipe-P0_mix_all DATA_NAME=dcfg_mix_all MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=data/dcfg_mix_all.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings (r1, WandB ap14j21d, completed 816/816 steps 2026-07-14):** VERDICT = **POSITIVE, stable.**

`scripts/score_run.py` output:
```
eval iters: 165 (step 0..816); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.4718  HM(last3)=0.4724  (baseline step0=0.4173)
ToM avg(last5)=0.5266  avg(last3)=0.527  (baseline step0=0.5034)
gsm8k (separate): 0.6552 (step0=0.657, delta vs step0=-0.002)
mmlu  (separate): 0.6066 (step0=0.48,  delta vs step0=+0.127)
health(final): kl=0.107 entropy=1.284 resp_len=99.557 reward=32.687 parseable=1.0 max_resp=512
```
- **ToM HM +5.45pp** (0.4173 -> 0.4718 last5), **ToM avg +2.32pp** (0.5034 -> 0.5266). HM rises fast to ~0.47 by step 25 and holds flat/slightly rising to 0.475 through step 816 — no collapse, no reward-hacking drift.
- **No capability regression:** gsm8k flat (Δ-0.002); mmlu +0.127 (step0 mmlu=0.48 is a low warm-up baseline, so treat the mmlu gain cautiously but at minimum no regression).
- **Health clean:** parseable=1.0 (format_err 0 throughout), kl=0.107, entropy=1.284, resp_len ~100 (well under the 512 cap, no runaway). Reward stayed non-saturated with real within-group variance the entire run (score spread +40..-45, advantages nonzero) — the intended learning signal, contrast E023 thoughttrace-only which floored.
- **Implication:** the 10-domain mixed data recipe (`dcfg_mix_all`, 26,128 rows) trains stably and transfers to held-out ToM at Qwen2.5-3B under the best-known config (power k5 ll_min-6, actor-RM, KL0.05, LR5e-7). Strong candidate `dcfg_default` for Phase 0 vs the single-domain arms (E019 cga was +4.8pp; mix_all edges it at +5.5pp and is far more stable/longer-horizon).

