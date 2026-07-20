### Attempt r1 — 2026-07-20T07:52:44+00:00

- **RUN_NAME:** `s2-P0-11-Phase0-data_mix_all_s3-dcfg_mix_all-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-076-003   **git:** `57903cd`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_all.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=4 ll_min=-6 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=0.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-P0-11-Phase0-data_mix_all_s3-dcfg_mix_all-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260720/s2-P0-11-Phase0-data_mix_all_s3-dcfg_mix_all-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_all.parquet \
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
    actor_rollout_ref.actor.format_penalty_std_coef=0.0 \
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
    trainer.experiment_name=s2-P0-11-Phase0-data_mix_all_s3-dcfg_mix_all-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k4-llmin-6-lr5e-7-kl0.05-n16-r1 \
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
    reward_model.format_penalty_std_coef=0.0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=4 \
    +actor_rollout_ref.power_ll_min=-6 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase0-data_mix_all_s3 DATA_NAME=dcfg_mix_all MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_all.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


---

## Consolidated Findings (r1 — completed)

**Run.** Phase-0 (v2) S2 mixture, `dcfg_mix_all` (ALL dialogue domains mixed), replicate 3/3 of the
mix_all seed set (P0-09/10/11). Locked Qwen2.5 recipe (power k4 llmin-6 actor-RM fp0 ec0 kl0.05
lr5e-7). Full 1 epoch = **816 steps** (~4.5h, 8×H100). WandB `hvndmsyd`. No OOM/crash; clean exit.

**Scores** (`scripts/score_run.py`, 24 ToM benchmarks excl gsm8k/mmlu, avg-then-HM):
```
ToM HM(last5)=0.4681  HM(last3)=0.4666  (baseline step0=0.4178)
ToM avg(last5)=0.5258  avg(last3)=0.5251 (baseline step0=0.5035)  -> d_avg=+0.0223
gsm8k (separate): 0.6466 (step0=0.657, delta=-0.010)
mmlu  (separate): 0.6226 (step0=0.477, delta=+0.146)
health(final): kl=0.178 entropy=1.361 resp_len=91.5 reward=28.35 parseable=1.0 max_resp=512
HM trajectory: stable ~0.45-0.47 across all 816 steps (0:0.418 → 30:0.451 → ... → 816:0.462)
```

**Verdict.** mix_all (all domains) is **stable and healthy** over a long 816-step run (KL rose to
~0.18 by end but no blowup, entropy 1.36, parseable=1.0, no collapse; HM holds a flat ~0.46 plateau).
ToM lift: HM +0.050, **d_avg = +0.0223** — essentially **identical to the smoke_mix baseline**
(ST01/10/28 d_avg +0.0215±0.001). Mixing all domains gives **no advantage** over the smaller
smoke_mix, consistent with the v1 finding that "mixtures don't compose" — reconfirmed here with the
power reward. mmlu up +0.146, gsm8k flat. This is replicate 3/3; pair with P0-09/P0-10 for the
3-seed mean±SD comparison.
