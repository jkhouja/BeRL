### Attempt r1 — 2026-07-14T05:39:26+00:00

- **RUN_NAME:** `data-recipe-P0_single_diplomacy-dcfg_diplomacy-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-076-003   **git:** `99d72d1`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_diplomacy.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=2 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=data-recipe-P0_single_diplomacy-dcfg_diplomacy-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260714/data-recipe-P0_single_diplomacy-dcfg_diplomacy-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_diplomacy.parquet \
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
    trainer.experiment_name=data-recipe-P0_single_diplomacy-dcfg_diplomacy-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=2 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-3B-Instruct \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=data-recipe-P0_single_diplomacy DATA_NAME=dcfg_diplomacy MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_diplomacy.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

## Hypothesis / question (Phase 0 — data-recipe, single-domain)

**RQ:** Which single training domain transfers best to ToM? This is the `diplomacy` arm (E020) of
the P0 single-domain sweep (E016–E023: empathetic/casino/craigslist/cga/diplomacy/p4g/dailydialog/
thoughttrace), all on Qwen2.5-3B with the fixed Phase −1 winner recipe (power k5 ll_min−6, actor-RM,
kl0.05, lr5e-7, fp5, ec0.0, COT_FREEFORM=cot_eval). Each domain trains on its own `dcfg_<domain>`
parquet; runs are compared by ToM HM (24 benchmarks, ex gsm8k/mmlu) step0→final on the shared
`eval_subsample_300` suite. The 8 single-domain ranks then define the best-mix (E024 best3) that the
gated mixture rows consume.

- **Data:** `dcfg_diplomacy` = 612 rows from 39 diplomacy conversations (203 convs filtered), Qwen2.5
  tagged format (cot_eval + answer tags), answer_pp computed with Qwen2.5-3B-Instruct.
- **Run length:** 38 steps (2 epochs × 19 batches @ bs32), test_freq=30 (evals at step0/30/final),
  save_freq=50.
- **Expected:** power+actor-RM is the never-collapse, gains-on-both-families config from Phase −1
  Wave-2 (PS122 gemma +9.7pp, PS167 qwen3 +2.5pp). Expect healthy training (power reward positive,
  length stable, fmt_err→0) and a non-negative ToM HM delta; magnitude vs the other 7 domains is the
  actual signal of interest.

**Live:** WandB `e1s6m3c7`; launched 2026-07-14 05:39 on h100-076-003. Findings appended on completion.

**Findings (E020 — COMPLETE, 2026-07-14):** ✅ **PASSED — strong single-domain transfer, no collapse.**
- **ToM HM (24, ex gsm8k/mmlu): 0.4186 → 0.4683 (+0.0498)**; ToM mean 0.5043 → 0.5185 (+1.41pp).
- 8 improved (>2pp) / 3 worsened (>2pp) / **0 exact-zeros** (no collapse). Evals at step0 & step38 (final).
- **Capabilities IMPROVED (not regressed):** gsm8k 0.657→0.713 (+5.6pp), mmlu 0.480→0.637 (+15.7pp) —
  unusual & healthy for a power+actor-RM run (Phase −1 power cells typically cost a few pp gsm8k).
- Top gains: fantom_answerability_list +15.7, explore_tom +11.0, hi_tom +9.7, simpletom_judgment +4.7,
  tomi +4.3. Top drops: bigtom_forward_action −6.4, bigtom_forward_belief −3.7, dyntom_type_a −2.4.
- **Training health:** reward positive/stable ~34–35 (floor 0, never pinned at −40/−80), fmt_err 0
  throughout, response_length steady ~120–128 (no inflation/collapse), advantages nonzero, kl~0.09–0.11,
  entropy ~1.6. 38 steps (2 epochs) + step0/final eval, clean exit (GPU→1 MiB).
- **Verdict:** diplomacy = a strong P0 single-domain transfer candidate (HM +0.05, capability-positive).
  Consistent with the Phase −1 Wave-2 result that power+actor-RM never collapses and yields ToM gains.
  Ranks among the 8 single-domain arms (E016–E023) will define the best-mix (E024).
- **How to rerun:** `EXP_ID=data-recipe-P0_single_diplomacy DATA_NAME=dcfg_diplomacy DATA_TRAIN=data/dcfg_diplomacy.parquet MAX_PROMPT=2048 MAX_RESP=512 KL=0.05 REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-6 USE_ACTOR_AS_RM=True LR=5e-7 FORMAT_PENALTY=5 ENTROPY_COEFF=0.0 bash experiments/train_behavior_qwen2.5.sh`

