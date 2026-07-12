### Attempt r1 — 2026-07-12T08:30:13+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1`
- **Host:** h100-076-003   **git:** `5a590c9`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=actor baseline=False kl=0.01 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.001_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.01-n16-r1 \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.001_pk5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** VERDICT = **HEALTHY / STABLE** (hypothesis CONFIRMED). WandB run `3vlzp3om`; 190 steps completed cleanly (final metrics logged, GPU→1 MiB, no teardown hang).

Config: power reward **k=5**, ll_min=−6, **actor** RM, kl=0.01, lr=5e-7, fp=5, ec=0.001, smoke_mix.

**Health/drift:** response_length/mean 39→115 (final), peak 126 — under the 140+ inflation threshold. format_error_ratio 0.000 throughout. reward/mean −36.4→+39.7 (power climbs positive). No collapse.

**Hypothesis result:** Predicted actor-RM k=5 @ kl0.01 stays HEALTHY (like actor k=3 twin PS052 which self-limited length, and like frozen k=5 twin PS068). **Confirmed:** length 115 stable, no collapse. Actor-RM at low KL continues to self-limit length; consistent with the finding that both actor-RM AND frozen-RM stay healthy for k=5 @ kl0.01 (unlike k=3 frozen PS036 which collapsed — so the k=3→k=5 stability flip holds for both RM types).

**Eval (subsample300, step0→step190, pp):**
| bench | s0 | s190 | Δ | | bench | s0 | s190 | Δ |
|---|---|---|---|---|---|---|---|---|
| tomi | 58.3 | 63.0 | +4.7 | | mmlu | 47.3 | 60.3 | +13.0 |
| hi_tom | 25.0 | 34.0 | +9.0 | | gsm8k | 65.7 | 54.0 | **−11.7** |
| bigtom_bwd_belief | 63.0 | 67.7 | +4.7 | | bigtom_fwd_action | 73.3 | 69.7 | −3.6 |
| bigtom_fwd_belief | 76.7 | 75.3 | −1.4 | | explore_tom | 48.0 | 61.3 | +13.3 |
| exploretom_infilled | 59.0 | 54.3 | −4.7 | | dyntom_type_a | 51.0 | 46.3 | −4.7 |
| dyntom_type_c | 50.7 | 52.3 | +1.6 | | dyntom_type_d | 37.3 | 35.0 | −2.3 |
| fantom_ans_binary | 20.3 | 22.3 | +2.0 | | fantom_ans_list | 14.0 | 31.3 | +17.3 |
| fantom_belief_mc | 50.3 | 46.7 | −3.6 | | fantom_info_binary | 43.3 | 41.3 | −2.0 |
| fantom_info_list | 28.7 | 28.7 | 0.0 | | opentom_attitude | 44.0 | 44.3 | +0.3 |
| opentom_loc_fo | 65.3 | 66.7 | +1.4 | | opentom_loc_so | 53.7 | 55.7 | +2.0 |
| opentom_multihop_fo | 66.3 | 63.0 | −3.3 | | opentom_multihop_so | 48.3 | 44.0 | −4.3 |
| simpletom_behavior | 55.7 | 56.0 | +0.3 | | simpletom_judgment | 29.0 | 26.3 | −2.7 |
| simpletom_mental | 85.3 | 86.3 | +1.0 | | tombench | 61.0 | 71.3 | +10.3 |

**Aggregate:** +36.6pp summed over 26 benchmarks (avg **+1.41pp**) — between PS068 (+31.1) and PS075 (+53.1). Biggest gains: fantom_answerability_list +17.3, explore_tom +13.3, mmlu +13.0, tombench +10.3, hi_tom +9.0. Biggest regressions: gsm8k −11.7 (consistent math regression across power cells), dyntom_type_a/exploretom_infilled −4.7, opentom_multihop_so −4.3. Note: smoke_mix is a tiny noisy probe — HEALTH/STABILITY is the finding, not the aggregate. **k=5 power @ kl0.01+fp5 actor-RM = stable candidate.**

