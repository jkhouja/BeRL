### Attempt r1 — 2026-07-11T22:59:05+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-076-003   **git:** `a62147b`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=3 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.05-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp5_ec0.001_pk3 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** WandB `x0055c1s`. Ran to completion (step 190, final validation logged); clean exit, GPUs freed to 1 MiB (no teardown hang). **Verdict: HEALTHY / STABLE — kl=0.05 tames the steep power reward that collapsed at kl=0.01 (PS036). Confirms the PS036 prediction: pair power with kl=0.05.**

Config: frozen RM, kl=0.05, lr=5e-7, fp=5, ec=0.001, **reward=power (k=3, ll_min=−6)**.

Training dynamics (all healthy):
- `critic/rewards/mean`: −28.2 → **+27.0** (power reward climbs strongly, same steep gradient as PS036 — but here the policy does NOT escape into length-inflation).
- `response_length/mean`: **53.9 → 105.2** — HEALTHY band (~105–110), NO collapse. Direct contrast: same power reward at kl=0.01 (PS036) blew up to 146–155. **kl=0.05 is what contains the steep power reward.**
- `format_error_ratio`: 0.000.

Eval (subsample300, 26 benchmarks), step0 → final:
| Benchmark | step0 | final | Δpp |
|---|---|---|---|
| **aggregate mean (26)** | 0.5072 | 0.5067 | **−0.05 (flat)** |
| explore_tom | 0.477 | 0.587 | +11.0 |
| fantom_answerability_binary | 0.207 | 0.220 | +1.3 |
| tomi | 0.590 | 0.593 | +0.3 |
| opentom_attitude | 0.440 | 0.433 | −0.7 |
| simpletom_judgment | 0.290 | 0.250 | −4.0 |
| simpletom_behavior | 0.557 | 0.513 | −4.3 |
| bigtom_forward_action | 0.733 | 0.687 | −4.7 |
| bigtom_forward_belief | 0.767 | 0.710 | −5.7 |
| dyntom_type_a | 0.510 | 0.453 | −5.7 |

Interpretation & hypothesis test: The monitor asked "does kl=0.05 tame the power reward, and does power+kl0.05 BEAT the log_prob best PS011?" **Answers: (1) YES, kl=0.05 tames it** — response length stays in the healthy 105-band and reward climbs cleanly, a textbook contrast to the PS036 collapse (same power reward, only kl differs: 0.01→collapse vs 0.05→healthy). **(2) NO, it does not beat PS011 on aggregate** — like PS011 (log_prob kl0.05+fp5) and PS030 (actor kl0.05), the aggregate 26-eval mean is essentially FLAT (0.507→0.507) on the small smoke_mix stability probe. Neither reward shape produces net climbing on this tiny dataset (expected — this is a stability probe, not an accuracy-optimized run). **Central Phase −1 conclusion now firmly established: kl=0.05 is the essential, reward-agnostic stabilizer** — it holds response length healthy for log_prob (PS011), actor-RM (PS030), and the steep power reward (PS044), whereas kl=0.01 collapses all of them (PS002/005/036 length-inflation, PS022 brevity). Power reward is stability-viable ONLY when paired with kl=0.05. PS011 (log_prob kl0.05+fp5) and PS044 (power kl0.05+fp5) are co-leading healthy `Pm1swp_best` candidates; the winner will be decided on a real accuracy dataset in Phase 0, not on smoke_mix.

