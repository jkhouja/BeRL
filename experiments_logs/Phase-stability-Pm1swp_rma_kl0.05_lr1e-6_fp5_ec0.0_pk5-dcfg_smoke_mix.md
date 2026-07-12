### Attempt r1 — 2026-07-12T11:39:56+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.05-n16-r1`
- **Host:** h100-076-003   **git:** `bd1c2d5`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=actor baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr1e-6-kl0.05-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp5_ec0.0_pk5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** VERDICT = **HEALTHY / STABLE** (hypothesis REFUTED — higher LR helped). WandB run `kfnkkua7`; 190 steps completed cleanly (final metrics logged, GPU→1 MiB, no teardown hang).

Config: power reward **k=5**, ll_min=−6, **actor** RM, kl=0.05, **lr=1e-6 (2× default)**, fp=5, ec=0.0, smoke_mix.

**Health/drift:** response_length/mean 48→111 (final), peak 124 — under the 140+ inflation threshold, essentially same as lr5e-7 twins (PS089 112, PS075 106) despite 2× LR. format_error_ratio 0.000 throughout. reward/mean −21.8→+37.3. No collapse, no faster drift.

**Hypothesis result:** Predicted lr1e-6 (2× default) = more aggressive updates → faster destabilization. **Refuted:** stayed fully healthy (length 111) with NO faster drift, and produced the **highest aggregate of the whole sweep (+70.2pp)**. On smoke_mix, actor-RM + kl=0.05 absorbs the 2× LR — higher LR = faster learning without collapse. gsm8k actually +0.7 (no math regression, unlike lr5e-7 power cells which lost 1–12pp).

**Eval (subsample300, step0→step190, pp):**
| bench | s0 | s190 | Δ | | bench | s0 | s190 | Δ |
|---|---|---|---|---|---|---|---|---|
| tomi | 59.0 | 60.3 | +1.3 | | mmlu | 47.0 | 63.0 | +16.0 |
| hi_tom | 25.0 | 36.7 | +11.7 | | gsm8k | 66.0 | 66.7 | +0.7 |
| bigtom_bwd_belief | 63.7 | 62.3 | −1.4 | | bigtom_fwd_action | 73.7 | 71.7 | −2.0 |
| bigtom_fwd_belief | 76.7 | 75.0 | −1.7 | | explore_tom | 48.0 | 57.7 | +9.7 |
| exploretom_infilled | 58.7 | 54.7 | −4.0 | | dyntom_type_a | 50.7 | 47.3 | −3.4 |
| dyntom_type_c | 51.0 | 50.0 | −1.0 | | dyntom_type_d | 37.3 | 35.7 | −1.6 |
| fantom_ans_binary | 20.7 | 19.0 | −1.7 | | fantom_ans_list | 13.3 | 35.0 | +21.7 |
| fantom_belief_mc | 50.7 | 49.0 | −1.7 | | fantom_info_binary | 43.3 | 45.3 | +2.0 |
| fantom_info_list | 28.3 | 33.3 | +5.0 | | opentom_attitude | 44.0 | 42.0 | −2.0 |
| opentom_loc_fo | 65.3 | 68.0 | +2.7 | | opentom_loc_so | 53.7 | 57.0 | +3.3 |
| opentom_multihop_fo | 66.0 | 64.3 | −1.7 | | opentom_multihop_so | 48.3 | 47.0 | −1.3 |
| simpletom_behavior | 55.7 | 60.0 | +4.3 | | simpletom_judgment | 28.7 | 37.0 | +8.3 |
| simpletom_mental | 85.3 | 86.0 | +0.7 | | tombench | 61.0 | 67.3 | +6.3 |

**Aggregate:** +70.2pp summed over 26 benchmarks (avg **+2.70pp**) — **NEW BEST of the entire Phase −1 sweep** (> PS075/PS089 +53.1). Biggest gains: fantom_answerability_list +21.7, mmlu +16.0, hi_tom +11.7, explore_tom +9.7, simpletom_judgment +8.3, tombench +6.3, fantom_info_list +5.0. Biggest regressions: exploretom_infilled −4.0, dyntom_type_a −3.4. Note: smoke_mix is a tiny noisy probe — HEALTH + LR-robustness is the finding, aggregate ranking is noisy. **k=5 power @ kl0.05 actor-RM lr1e-6 = stable at 2× LR; higher LR safe under actor+kl0.05.**

