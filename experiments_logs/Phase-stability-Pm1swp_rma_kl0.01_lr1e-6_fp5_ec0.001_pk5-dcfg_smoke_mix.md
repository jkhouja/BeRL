### Attempt r1 — 2026-07-12T09:57:44+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp5_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1`
- **Host:** h100-013-002   **git:** `afc71ac`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=actor baseline=False kl=0.01 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp5_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp5_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp5_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.01-n16-r1 \
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
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.01_lr1e-6_fp5_ec0.001_pk5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


---

## Findings (PS088 — completed 2026-07-12)

**Verdict: STABLE — winner-tier (key stress-test result).** Actor-RM + power k=5 + fp5 at the
AGGRESSIVE lr=1e-6 / kl=0.01: HM_tom 0.416 (baseline) → 0.459 (last-5), peak 0.465. This is the
exact hyperparameter regime (kl0.01 + lr1e-6) that made log_prob reward COLLAPSE (PS006). Power +
k=5 + fp5 not only tames it — it BEATS the gentler-LR analog PS081 (actor k5 kl01 lr5e-7, 0.444)
via a faster climb, nearly matching the double-lever best PS074 (0.462).

Health: stable — reward↑ (29→~38) with eval climbing/plateauing high (no over-optimization
collapse); response_length bounded ~110–130 (fp5 working, no length hacking). Mild general-capability
erosion: gsm8k 0.66→0.59 (softer than the gentle cells' full preservation, but nowhere near
log_prob's crash to ~0.09 under the same LR). mmlu steady ~0.61.

| step | HM_tom | tomi | gsm8k | mmlu | reward | resp_len |
|-----:|-------:|-----:|------:|-----:|-------:|---------:|
|   0  | 0.416  | 0.593| 0.660 | 0.477|   —    |   —      |
|  40  | 0.465  | 0.660| 0.687 | 0.617| 40.0   | 108.6    |
| 100  | 0.460  | 0.617| 0.587 | 0.587| 32.0   | 100.7    |
| 170  | 0.465  | 0.600| 0.567 | 0.613| 39.4   | 124.7    |
| 190  | 0.460  | 0.603| 0.593 | 0.613|   —    |   —      |

Takeaway: power reward + k=5 + fp5 is robust enough to run at aggressive lr1e-6 WITHOUT collapse —
the strongest evidence yet that power's gentler gradient (vs log_prob) is what preserves stability
and capability under high learning pressure. lr1e-6 gives a faster/higher climb than lr5e-7 here,
at only a mild gsm8k cost — a promising fast-training recipe for Phase 0.
