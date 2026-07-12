### Attempt r1 — 2026-07-12T05:04:46+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1`
- **Host:** h100-117-001   **git:** `1d91a59`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.01 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1.log`

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
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
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
    algorithm.kl_ctrl.kl_coef=0.01 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp5_ec0.0_pk5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings (r1, completed 2026-07-12):** STRONG / above baseline, sustained, math largely preserved — and a notable counter-example to "kl=0.01 collapses".

Config-selection metric = harmonic mean over the sub300 benchmarks per eval block; compare vs step-0 baseline.
- Baseline (step 0): HM = **0.4218**
- HM last-3 (170/180/190): **0.4461**
- HM last-5 (150–190): **0.4474**  → **ABOVE baseline** (+0.026)
- Per-step HM: rises immediately, peaks step 50 (0.4768), gentle decline to a ~0.444 plateau (170–190). Stays well above baseline throughout — no collapse.
- gsm8k 0.653 → 0.543 (−11.0pp, math LARGELY PRESERVED); tomi 0.590 → 0.603 (+1.3pp); mmlu 0.463 → 0.613 (+15.0pp).

Health: clean run to step 190 (no crash/OOM, format_error_ratio 0.000 throughout). Entropy stayed moderate (~1.5 → ~2.1, never runaway). Response length grew ~100 → ~120 then plateaued (k=5 favors longer CoT), no blowup.

**Verdict:** frozen-RM + power **k=5** + kl=0.01 + lr1e-6 + fp5 is STRONG (0.447) and does NOT exhibit the PS035 collapse (frozen power **k=3** kl=0.01 lr5e-7 fp5 = 0.379, POOR). The stabilizing differences vs PS035 are **k=5 (steeper power reward, concentrates credit on the true continuation → less reward-hacking) and lr=1e-6**. This challenges the "kl=0.01 always collapses unless mitigated" heuristic: at k=5 the reward shape itself is stabilizing even at low KL. Math preservation here (gsm8k −11pp) is far better than the k=3 low-KL runs (PS056/PS064 gsm8k crashed to 0.15–0.22). Candidate worth comparing head-to-head with PS048 (k=3 kl0.05, 0.456). WandB: https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/77z79n2n

