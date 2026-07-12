### Attempt r1 — 2026-07-12T10:04:59+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-076-003   **git:** `afc71ac`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1.log`

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
    actor_rollout_ref.actor.entropy_coeff=0.0 \
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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 \
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
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp0_ec0.0_pk5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** VERDICT = **HEALTHY / STABLE** (hypothesis CONFIRMED + notable fp=0 result). WandB run `aio8hclu`; 190 steps completed cleanly (final metrics logged, GPU→1 MiB, no teardown hang).

Config: power reward **k=5**, ll_min=−6, **actor** RM, kl=0.05, lr=5e-7, **fp=0 (no format penalty)**, ec=0.0, smoke_mix.

**Health/drift:** response_length/mean 52→112 (final), peak 121 — under the 140+ inflation threshold. reward/mean −21.6→+33.0 (power climbs positive). **format_error_ratio = 0.000 throughout (max 0.000)** — CRITICAL: even with NO format penalty (fp=0), the model kept format perfectly. No collapse, no format-hacking.

**Hypothesis result / key finding:** Predicted actor+kl0.05 stabilizes and worried fp=0 might induce format drift. **Confirmed stable, and fp=0 did NOT cause format-hacking** — actor-RM + kl0.05 alone keeps format at 0.000. Implication: the format penalty (fp=5) may be *unnecessary scaffolding* when actor-RM + kl0.05 is used; the reward/KL structure self-maintains format. Worth verifying on real data in Phase 0 (could simplify the recipe).

**Eval (subsample300, step0→step190, pp):**
| bench | s0 | s190 | Δ | | bench | s0 | s190 | Δ |
|---|---|---|---|---|---|---|---|---|
| tomi | 59.0 | 59.7 | +0.7 | | mmlu | 47.7 | 62.3 | +14.6 |
| hi_tom | 25.0 | 36.3 | +11.3 | | gsm8k | 66.3 | 65.0 | −1.3 |
| bigtom_bwd_belief | 63.0 | 63.0 | 0.0 | | bigtom_fwd_action | 73.3 | 69.7 | −3.6 |
| bigtom_fwd_belief | 76.7 | 75.3 | −1.4 | | explore_tom | 48.0 | 57.3 | +9.3 |
| exploretom_infilled | 58.7 | 58.3 | −0.4 | | dyntom_type_a | 50.7 | 49.3 | −1.4 |
| dyntom_type_c | 51.3 | 50.0 | −1.3 | | dyntom_type_d | 37.3 | 37.0 | −0.3 |
| fantom_ans_binary | 20.3 | 20.7 | +0.4 | | fantom_ans_list | 13.7 | 30.7 | +17.0 |
| fantom_belief_mc | 50.0 | 49.7 | −0.3 | | fantom_info_binary | 43.3 | 43.3 | 0.0 |
| fantom_info_list | 28.7 | 28.7 | 0.0 | | opentom_attitude | 44.3 | 40.7 | −3.6 |
| opentom_loc_fo | 66.0 | 67.0 | +1.0 | | opentom_loc_so | 53.3 | 55.0 | +1.7 |
| opentom_multihop_fo | 66.0 | 64.7 | −1.3 | | opentom_multihop_so | 48.7 | 50.7 | +2.0 |
| simpletom_behavior | 56.0 | 56.3 | +0.3 | | simpletom_judgment | 28.7 | 29.0 | +0.3 |
| simpletom_mental | 85.3 | 88.7 | +3.4 | | tombench | 60.7 | 66.7 | +6.0 |

**Aggregate:** +53.1pp summed over 26 benchmarks (avg **+2.04pp**) — **tied best** with PS075 (also +53.1). Biggest gains: fantom_answerability_list +17.0, mmlu +14.6, hi_tom +11.3, explore_tom +9.3, tombench +6.0. Notably **gsm8k only −1.3** here (vs −5 to −12 in fp=5 power cells) — least math regression of any power cell. Note: smoke_mix is a tiny noisy probe — the HEALTH + fp=0 format-robustness is the finding, not the aggregate. **k=5 power @ kl0.05 actor-RM fp0 = stable AND format-robust; candidate for simplified recipe.**

