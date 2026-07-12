### Attempt r1 — 2026-07-11T20:25:34+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp0_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1`
- **Host:** h100-013-002   **git:** `deeb330`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=3 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.01 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp0_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp0_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1.log`

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
    algorithm.kl_ctrl.kl_coef=0.01 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp0_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1 \
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
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=3 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp0_ec0.001_pk3 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (PS034 — frozen-RM, POWER reward k=3/ll_min=-6, kl=0.01, lr=5e-7, fp0, ec0.001)

Owner_host: h100-013-002 · WandB run rkpnfi5o · 191 steps (1 epoch), TEST_FREQ=10, subsample300 eval.

**Config-selection metric** = harmonic mean over ToM benchmark subtypes EXCLUDING gsm8k & mmlu, per eval.

| step | HM_tom | tomi | gsm8k | mmlu |
|-----:|-------:|-----:|------:|-----:|
|   0  | 0.421  | 0.590| 0.657 | 0.483|
|  20  | 0.451  | 0.617| 0.720 | 0.567|
|  70  | 0.448  | 0.587| 0.560 | 0.563|
| 100  | 0.412  | 0.457| 0.520 | 0.547|
| 140  | 0.350  | 0.410| 0.423 | 0.503|
| 180  | 0.400  | 0.557| 0.513 | 0.550|
| 190  | 0.389  | 0.463| 0.533 | 0.567|

- **baseline HM(step0) = 0.421**; peak 0.451 @step20; holds ~0.45 to step70; trough 0.350 @140; recovers ~0.40; **HM-last5 = 0.390** (< baseline).
- gsm8k degrades only 0.657 → ~0.42–0.53 (mmlu ~held): **power reward preserves general capability far better than log_prob** (cf log_prob PS021 gsm8k crash to 0.087).
- response_length grows 56.8 → 144.6 (max 169.2, ctx 512 not saturated); reward/mean rises −22.6 → +32.2 while ToM eval falls after step 70 ⇒ **mild length-driven over-optimization** (flag, not full hacking).

**Verdict: climbs-then-degrades (below baseline), but markedly milder than log_prob at same kl.**
Power reward (k=3, ll_min=-6) at kl0.01/lr5e-7 holds ToM longer and protects gsm8k/mmlu vs log_prob
(PS003 log_prob 0.319, gsm8k drift; PS034 power 0.390). Still below baseline ⇒ kl=0.01 remains too weak;
kl=0.05 expected to stabilize power reward as it did log_prob (cf PS013). Response-length growth +
reward↑/eval↓ after step 70 is the over-optimization signature to watch for power reward.

Rerun: `ONLY_IDX=34 bash experiments/phase_stability_sweep.sh`
