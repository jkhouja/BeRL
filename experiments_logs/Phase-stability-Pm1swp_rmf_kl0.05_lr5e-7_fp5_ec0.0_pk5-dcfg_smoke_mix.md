### Attempt r1 — 2026-07-12T06:25:21+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-076-003   **git:** `7ba14d4`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-frozenRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rmf_kl0.05_lr5e-7_fp5_ec0.0_pk5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** VERDICT = **HEALTHY / STABLE** (hypothesis CONFIRMED). WandB run `ihgo8knk`; 190 steps completed cleanly (final metrics logged, GPU→1 MiB, no teardown hang).

Config: power reward **k=5**, ll_min=−6, frozen RM, **kl=0.05**, lr=5e-7, fp=5, ec=0.0, smoke_mix.

**Health/drift:** response_length/mean 47→106 (final), peak 124 — well under the 140+ inflation threshold and *lower* than the kl0.01 twin PS068 (final 124, peak 138), as predicted (kl0.05 = primary stabilizer). format_error_ratio 0.000 throughout. reward/mean −29.5→+36.2 (power climbs positive). No collapse.

**Hypothesis result:** Predicted kl0.05 twin of PS068 should be HEALTHY with lower final length. **Confirmed:** length 106 (< PS068's 124), stable throughout, and higher aggregate (+53.1pp vs PS068 +31.1pp). Also beats k=3 kl0.05 twin PS044 (length 105, aggregate ~flat) — steeper power k=5 @ kl0.05 gave a stronger positive aggregate while staying healthy.

**Eval (subsample300, step0→step190, pp):**
| bench | s0 | s190 | Δ | | bench | s0 | s190 | Δ |
|---|---|---|---|---|---|---|---|---|
| tomi | 59.0 | 64.0 | +5.0 | | mmlu | 46.7 | 60.0 | +13.3 |
| hi_tom | 25.0 | 35.7 | +10.7 | | gsm8k | 65.7 | 60.3 | −5.4 |
| bigtom_bwd_belief | 63.7 | 60.0 | −3.7 | | bigtom_fwd_action | 73.7 | 71.7 | −2.0 |
| bigtom_fwd_belief | 76.7 | 75.3 | −1.4 | | explore_tom | 47.7 | 57.3 | +9.6 |
| exploretom_infilled | 58.7 | 58.3 | −0.4 | | dyntom_type_a | 50.7 | 46.0 | −4.7 |
| dyntom_type_c | 51.0 | 47.7 | −3.3 | | dyntom_type_d | 37.3 | 38.7 | +1.4 |
| fantom_ans_binary | 20.3 | 20.3 | 0.0 | | fantom_ans_list | 14.0 | 27.3 | +13.3 |
| fantom_belief_mc | 49.3 | 50.3 | +1.0 | | fantom_info_binary | 43.3 | 39.3 | −4.0 |
| fantom_info_list | 29.3 | 30.3 | +1.0 | | opentom_attitude | 44.0 | 45.3 | +1.3 |
| opentom_loc_fo | 65.7 | 69.3 | +3.6 | | opentom_loc_so | 53.7 | 55.7 | +2.0 |
| opentom_multihop_fo | 65.7 | 64.3 | −1.4 | | opentom_multihop_so | 48.7 | 50.3 | +1.6 |
| simpletom_behavior | 56.0 | 58.3 | +2.3 | | simpletom_judgment | 28.7 | 32.0 | +3.3 |
| simpletom_mental | 85.3 | 89.3 | +4.0 | | tombench | 60.3 | 66.3 | +6.0 |

**Aggregate:** +53.1pp summed over 26 benchmarks (avg **+2.04pp**) — best aggregate of any Phase −1 cell so far (> PS068 +31.1). Biggest gains: mmlu +13.3, fantom_answerability_list +13.3, hi_tom +10.7, explore_tom +9.6, tombench +6.0, tomi +5.0. Biggest regressions: gsm8k −5.4, dyntom_type_a −4.7, fantom_info_binary −4.0. Note: smoke_mix is a tiny noisy probe — HEALTH/STABILITY is the finding of interest, not the aggregate. **k=5 power @ kl0.05+fp5 frozen = strong stable candidate** (healthiest length + best aggregate among power-frozen cells).

