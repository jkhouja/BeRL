### Attempt r1 — 2026-07-11T20:27:52+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1`
- **Host:** h100-117-001   **git:** `6fde828`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=3 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.01 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp5_ec0.0_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp5_ec0.0_pk3 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings (r1, authoritative — completed 2026-07-11, Owner_host h100-117-001):**

Full horizon (191 steps, 1 epoch), evals every 10 steps on subsample300 (26 subtypes). Config-selection metric = harmonic mean over all 26 sub300 benchmark scores per eval:

| step | 0(base) | 30(peak) | 60 | 100 | 150 | 180 | 190 |
|------|---------|----------|-----|-----|-----|-----|-----|
| HM_all | 0.4258 | 0.4694 | 0.4691 | 0.4250 | 0.3843 | 0.3714 | 0.3706 |

- **HM(last-3)=0.379, HM(last-5)=0.379, HM@190=0.371** vs step-0 baseline **HM=0.4258** → **well below baseline (−4.7pp)**. Peak HM=0.469 @step30 (+4.4pp) then **steady monotonic decline** through the second half.
- **Health: no format collapse** (`format_error_ratio=0.000` throughout), but **entropy runaway** (2.38→3.03) and **response_length inflation** (111→136) accompany the late regression — classic over-optimization drift under frozen-RM power + weak KL (kl=0.01) + format_penalty=5.
- Per-benchmark step0→step190 broadly negative: tomi 0.590→0.400 (−19), hi_tom 0.250→0.170 (−8), bigtom_fwd_belief 0.767→0.637 (−13), fantom_belief_mc 0.507→0.367 (−14), simpletom_mental 0.853→0.757 (−10); only mmlu up (0.480→0.560).

**Verdict:** POOR cell. Frozen-RM power(k3,ll_min=−6) + kl=0.01 + fp5 climbs to an early peak (~step30) then over-optimizes and **regresses below baseline** across nearly all ToM benchmarks with entropy/length drift. kl=0.01 is too weak to hold the frozen-RM power reward. **Not a Phase-1 winner; exclude from best-config.** Contrast with actor-RM cells and stronger-KL (kl=0.05) variants which hold up better.

