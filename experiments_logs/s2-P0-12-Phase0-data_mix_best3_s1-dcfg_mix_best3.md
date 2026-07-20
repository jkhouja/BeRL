### Attempt r1 — 2026-07-20T08:24:56+00:00

- **RUN_NAME:** `s2-P0-12-Phase0-data_mix_best3_s1-dcfg_mix_best3-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-117-001   **git:** `57903cd`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best3.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=4 ll_min=-6 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=1.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-P0-12-Phase0-data_mix_best3_s1-dcfg_mix_best3-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260720/s2-P0-12-Phase0-data_mix_best3_s1-dcfg_mix_best3-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best3.parquet \
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
    actor_rollout_ref.actor.format_penalty=0.0 \
    actor_rollout_ref.actor.format_penalty_std_coef=1.0 \
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
    trainer.experiment_name=s2-P0-12-Phase0-data_mix_best3_s1-dcfg_mix_best3-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-3B-Instruct \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=power \
    +reward_model.power_k=4 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=0.0 \
    reward_model.format_penalty_std_coef=1.0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=4 \
    +actor_rollout_ref.power_ll_min=-6 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase0-data_mix_best3_s1 DATA_NAME=dcfg_mix_best3 MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best3.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (P0-12, scored 2026-07-20)

Canonical scoring (`scripts/score_run.py`), eval iters 13 (step 0..343), 24 ToM benchmarks (excl gsm8k/mmlu):

```
ToM HM(last5)=0.462  HM(last3)=0.4615  (baseline step0=0.4151)
ToM avg(last5)=0.5202  avg(last3)=0.5193  (baseline step0=0.5033)
gsm8k (separate): 0.724 (step0=0.653, delta=+0.071)
mmlu (separate): 0.6282 (step0=0.47, delta=+0.158)
health(final): kl=0.069 entropy=1.468 resp_len=112.469 reward=38.937 parseable=1.0 max_resp=512
ToM HM trajectory: 0:0.415 30:0.466 60:0.466 90:0.455 120:0.474 150:0.454 180:0.455 210:0.459 240:0.46 270:0.464 300:0.465 330:0.46 343:0.459
```

- **mix_best3 mixture, replicate 1/3** (casino+empathetic+dailydialog, v1 best3): dcfg_mix_best3 = 11,000 rows → 342 steps (~1 epoch, -r1).
- **Health**: KL stayed clean (0.04-0.09, final 0.069), entropy ~1.47, format_error 0, parseable 1.0, no clipping, no collapse.
- **ToM lift**: HM +4.7pp (last5) over step-0 baseline; avg +1.7pp; stable HM plateau 0.45-0.47. Also strong capability retention/gain: gsm8k +7.1pp, mmlu +15.8pp.
- Comparable ToM HM lift to P0-02 cga (+4.0pp) and P0-07 p4g (+4.7pp). Rank on d_avg vs smoke_mix baseline (+0.0215±0.001) once all 3 mix_best3 replicates (P0-12/13/14) complete.
- WandB: https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/7t2pehyb
