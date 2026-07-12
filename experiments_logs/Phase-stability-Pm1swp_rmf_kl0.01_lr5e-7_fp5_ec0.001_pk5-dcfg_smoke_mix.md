### Attempt r1 — 2026-07-12T04:18:55+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp5_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1`
- **Host:** h100-076-003   **git:** `6ed1e1b`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.01 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp5_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp5_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp5_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.01_lr5e-7_fp5_ec0.001_pk5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** VERDICT = **HEALTHY / STABLE** (hypothesis REFUTED). WandB run `tap051po`; 190 steps completed cleanly (final metrics logged, GPU→1 MiB, no teardown hang).

Config: power reward **k=5**, ll_min=−6, frozen RM, kl=0.01, lr=5e-7, fp=5, ec=0.001, smoke_mix.

**Health/drift:** response_length/mean 38→124 (final), peak 138 — stayed under the 140+ inflation-collapse threshold; format_error_ratio 0.000 throughout. reward/mean −38.3→+38.6 (power reward climbs strongly positive, as expected). No brevity or inflation collapse.

**Hypothesis result:** Predicted k=5 (steeper power) at kl=0.01 frozen would collapse *harder* than the k=3 twin PS036 (which COLLAPSED via length-inflation 146–155, aggregate −7.3pp). **Refuted:** k=5 stayed healthy (length 124, peak 138) with a **positive** aggregate. Steeper exponent did NOT worsen stability here — likely because the steeper power gradient rewards *content correctness* faster before length can runaway, or run-to-run variance on the tiny smoke_mix probe. Notable given PS036/PS068 differ only in k (3→5).

**Eval (subsample300, step0→step190, pp):**
| bench | s0 | s190 | Δ | | bench | s0 | s190 | Δ |
|---|---|---|---|---|---|---|---|---|
| tomi | 59.0 | 63.7 | +4.7 | | mmlu | 46.0 | 60.0 | +14.0 |
| hi_tom | 25.0 | 32.0 | +7.0 | | gsm8k | 66.3 | 55.0 | **−11.3** |
| bigtom_bwd_belief | 63.3 | 65.3 | +2.0 | | bigtom_fwd_action | 73.7 | 67.7 | −6.0 |
| bigtom_fwd_belief | 76.7 | 75.7 | −1.0 | | explore_tom | 48.3 | 58.3 | +10.0 |
| exploretom_infilled | 58.7 | 56.0 | −2.7 | | dyntom_type_a | 51.0 | 47.0 | −4.0 |
| dyntom_type_c | 51.3 | 50.3 | −1.0 | | dyntom_type_d | 37.3 | 35.7 | −1.6 |
| fantom_ans_binary | 20.7 | 21.0 | +0.3 | | fantom_ans_list | 13.7 | 30.0 | +16.3 |
| fantom_belief_mc | 50.0 | 46.3 | −3.7 | | fantom_info_binary | 43.3 | 43.0 | −0.3 |
| fantom_info_list | 28.3 | 30.7 | +2.4 | | opentom_attitude | 44.0 | 43.3 | −0.7 |
| opentom_loc_fo | 65.7 | 71.3 | +5.6 | | opentom_loc_so | 53.0 | 57.0 | +4.0 |
| opentom_multihop_fo | 66.0 | 65.3 | −0.7 | | opentom_multihop_so | 48.3 | 48.0 | −0.3 |
| simpletom_behavior | 55.7 | 55.7 | 0.0 | | simpletom_judgment | 29.0 | 23.7 | −5.3 |
| simpletom_mental | 85.3 | 87.0 | +1.7 | | tombench | 61.0 | 62.7 | +1.7 |

**Aggregate:** +31.1pp summed over 26 benchmarks (avg **+1.20pp**) — best aggregate of any Phase −1 cell so far. Biggest gains: fantom_answerability_list +16.3, mmlu +14.0, explore_tom +10.0, hi_tom +7.0. Biggest regressions: gsm8k −11.3 (math), bigtom_forward_action −6.0, simpletom_judgment −5.3. Note: smoke_mix is a tiny stability probe — deltas are noisy and NOT the paper signal; the finding of interest is HEALTH/STABILITY, not the aggregate. **k=5 power @ kl0.01+fp5 frozen = stable candidate**, and unexpectedly more stable than its k=3 twin.

