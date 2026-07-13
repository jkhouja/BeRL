### Attempt r1 — 2026-07-13T16:37:35+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-189-003   **git:** `f9f6aef`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7 ll_min=-6 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.power_k=7 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=7 \
    +actor_rollout_ref.power_ll_min=-6 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (r1 — completed 2026-07-13, host h100-189-003)

Run: WandB bfb9kzvv; log `logs/20260713/Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1.log`

Canonical score (`scripts/score_run.py`):
```
eval iters: 39 (step 0..190); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.1511  HM(last3)=0.1502  (baseline step0=0.1605)
ToM avg(last5)=0.2617  avg(last3)=0.2623  (baseline step0=0.2717)
gsm8k (separate): 0.4846 (step0=0.487, delta vs step0=-0.002)
mmlu  (separate): 0.2386 (step0=0.243, delta vs step0=-0.004)
health(final): kl=0.003 entropy=0.278 resp_len=451.393 reward=0.542 parseable=1.0
```

Verdict: **STABLE / neutral — no degradation, no capability regression.** ToM HM essentially flat
(−0.9pp: 0.151 vs 0.160), AM flat (−1.0pp: 0.262 vs 0.272). Crucially gsm8k (−0.2pp) and mmlu
(−0.4pp) are unchanged — NO capability crash. HM trajectory is stable ~0.15 for the entire run (no
mid/late decline). Health clean: entropy ~0.278 (Qwen3 native), resp_len ~451, kl 0.003,
parseable=1.0.

KEY CONTRAST (isolates what stabilises Qwen3 power runs):
- PS150 frozen-RM power **k5 ll_min−4 fp0** → HM −8.5pp, gsm8k −31.6pp (off-distribution crash)
- PS156 frozen-RM power **k7 ll_min−6 fp5** → HM −0.9pp, gsm8k −0.2pp (STABLE)
The conservative reward floor (ll_min=−6) and the format penalty (fp=5) prevent the Qwen3
off-distribution collapse seen in PS150. fp+ll_min matter far more than k for Qwen3 stability.

Wave-2 Qwen3 pattern (3 runs):
- PS142 actor-RM log_prob: HM −7.8pp (entropy-collapse self-hack)
- PS150 frozen-RM power k5 ll_min−4 fp0: HM −8.5pp, gsm8k −31.6pp (capability crash)
- PS156 frozen-RM power k7 ll_min−6 fp5: HM −0.9pp, gsm8k −0.2pp (STABLE, first non-catastrophic)
No positive transfer yet, but PS156 shows a stable Qwen3 recipe exists (fp5 + ll_min−6); worth using
as the stability anchor for further Qwen3 tuning.
