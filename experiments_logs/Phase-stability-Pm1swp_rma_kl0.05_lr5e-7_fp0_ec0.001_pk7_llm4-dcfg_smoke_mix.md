### Attempt r1 — 2026-07-12T21:51:21+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-156-003   **git:** `2ba54e8`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7 ll_min=-4.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.power_k=7 \
    +reward_model.power_ll_min=-4.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=7 \
    +actor_rollout_ref.power_ll_min=-4.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


---

## Findings (r1 — PS182, run by h100-156-003, 2026-07-12)

WandB: https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/ood14jno
Log: logs/20260712/Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm4-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k7-llmin-4.0-lr5e-7-kl0.05-n16-r1.log

Canonical scorer (scripts/score_run.py) output:
```
eval iters: 20 (step 0..190); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.4524  HM(last3)=0.4554  (baseline step0=0.4178)
ToM avg(last5)=0.5074  avg(last3)=0.51  (baseline step0=0.5032)
gsm8k (separate): 0.4806 (step0=0.66, delta vs step0=-0.179)
mmlu (separate): 0.6072 (step0=0.47, delta vs step0=+0.137)
health(final): kl=0.196 entropy=2.351 resp_len=94.77 reward=21.793 parseable=1.0
ToM HM trajectory: 0:0.418 10:0.438 20:0.46 30:0.46 40:0.458 50:0.463 60:0.459 70:0.457 80:0.444 90:0.454 100:0.44 110:0.434 120:0.441 130:0.438 140:0.449 150:0.448 160:0.446 170:0.447 180:0.455 190:0.461
```

**Verdict: MODEST gain, but weaker than sibling PS180 (ll_min=-6).** ToM HM +3.5pp over baseline
(0.4524 vs 0.4178), avg only +0.4pp. HM peaks early (~0.46 @20-60), dips mid-run to ~0.434-0.44,
recovers to 0.461@190 — noisier / less stable plateau than PS180's sustained rise. **gsm8k
regressed notably (Δ=-0.179, 0.66→0.48)** — a real math-reasoning cost, unlike PS180 which
preserved gsm8k (+0.009). KL drift elevated (0.196 vs PS180's 0.045). Parse=1.0, no collapse/hacking.

**Cross-cell conclusion (actor-RM power k=7, floor sweep):** ll_min=-6 (PS180, HM 0.4714, gsm8k
preserved) CLEARLY BEATS ll_min=-4 (PS182, HM 0.4524, gsm8k -0.179). The lower floor (-6) gives
higher ToM transfer AND protects math capability. => Prefer ll_min=-6 for actor-RM power.
Ranking so far: PS180 (0.4714) > PS074 Wave-1 (0.462) > PS182 (0.4524) > PS177 (0.4333).
