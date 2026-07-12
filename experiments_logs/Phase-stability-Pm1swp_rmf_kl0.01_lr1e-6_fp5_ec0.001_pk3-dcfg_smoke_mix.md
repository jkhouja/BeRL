### Attempt r1 — 2026-07-11T22:11:42+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1`
- **Host:** h100-232-001   **git:** `92e1d06`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=3 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.01 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1.log`

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
    actor_rollout_ref.actor.entropy_coeff=0.001 \
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
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr1e-6-kl0.01-n16-r1 \
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
    +reward_model.power_k=3 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=3 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.01_lr1e-6_fp5_ec0.001_pk3 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


- **WandB link (resolved r1):** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/1ixts8m0
- **Owner_host:** h100-232-001 (dedicated exclusive srun job 5421376 — not a shared/overlap allocation, avoids preemption).
- **Hypothesis:** Phase −1 sweep cell PS040 (frozen-RM, power reward k=3/ll_min=-6, KL=0.01, LR=1e-6, format_penalty=5, entropy_coeff=0.001). Tests whether the power reward family climbs (HM over last-X) without collapse at the aggressive kl0.01/lr1e-6 corner where log_prob cells collapsed. fp=5 expected to help stability.
- **How to rerun:** `ONLY_IDX="40" bash experiments/phase_stability_sweep.sh` (on an 8-GPU node; env `tom`).

### Findings (r1 — completed 2026-07-12 ~00:08Z, 191 steps / 1 epoch)

**Config-selection metric (HM over ToM benchmarks, excl. gsm8k/mmlu):**
- baseline (step 0) HM_tom = **0.417**
- peak HM = **0.456 @ step 10**, then monotonic decline; final HM = **0.242 @ step 180**
- HM over last-3 = **0.244**, last-5 = **0.262** (well below baseline)

**Eval trajectory (sub300):**
| step | HM_tom | gsm8k | tomi | bigtom_fwd_belief | simpletom_mental |
|---|---|---|---|---|---|
| 0   | 0.417 | 0.660 | 0.593 | 0.767 | 0.853 |
| 10  | 0.456 | 0.733 | 0.637 | 0.770 | 0.877 |
| 80  | 0.303 | 0.297 | 0.360 | 0.567 | 0.697 |
| 130 | 0.363 | 0.377 | 0.440 | 0.693 | 0.847 |
| 170 | 0.171 | 0.230 | 0.200 | 0.330 | 0.450 |
| 180 | 0.242 | 0.287 | 0.217 | 0.513 | 0.627 |

**Health:** power reward optimized to +36.8 (hacked); format_error_ratio=0.000 throughout (fp=5 keeps
`<think>`/`<answer>` tags present) but content degenerates; response_length/mean blows up 46→134→181.
Classic reward-hacking with format intact → severe negative transfer to every benchmark.

**Verdict:** DEGRADED / reward-hacked cell. KL=0.01 is too weak and LR=1e-6 too aggressive for the
power-k3 (ll_min=-6) reward — the policy games the LL reward via length inflation while all ToM and
general benchmarks collapse. NOT a stable-config candidate; exclude from best-config and skip in
Wave-2. Mirrors the log_prob@kl0.01/lr1e-6 collapses (PS005-PS007).
