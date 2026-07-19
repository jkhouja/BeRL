### Attempt r1 — 2026-07-14T07:39:59+00:00

- **RUN_NAME:** `data-recipe-P0_single_dailydialog-dcfg_dailydialog-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-076-003   **git:** `1046011`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_dailydialog.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=2 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=data-recipe-P0_single_dailydialog-dcfg_dailydialog-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260714/data-recipe-P0_single_dailydialog-dcfg_dailydialog-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_dailydialog.parquet \
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
    trainer.experiment_name=data-recipe-P0_single_dailydialog-dcfg_dailydialog-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 \
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

**How to rerun:** `EXP_ID=data-recipe-P0_single_dailydialog DATA_NAME=dcfg_dailydialog MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_dailydialog.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

## Hypothesis / question (Phase 0 — data-recipe, single-domain)

**RQ:** Which single training domain transfers best to ToM? This is the `dailydialog` arm (E022) of
the P0 single-domain sweep (E016–E023), all Qwen2.5-3B with the fixed Phase −1 winner recipe (power
k5 ll_min−6, actor-RM, kl0.05, lr5e-7, fp5, ec0.0, COT_FREEFORM=cot_eval). dailydialog is the
**"smalltalk baseline"** domain in the plan (expected to be a LOW-ToM-dependence control) — the point
of the ranking is that a chit-chat corpus should transfer *worse* to ToM than negotiation/social
domains (casino, diplomacy, cga…).

- **Data:** `dcfg_dailydialog` = **55,157 rows from 6,397 conversations** (4,721 convs filtered),
  Qwen2.5 tagged format, answer_pp fully computed (55157/55157) with Qwen2.5-3B-Instruct (~50 min build).
- **Run length:** **3,446 steps** (2 epochs × ~1,723 batches @ bs32) — by far the largest single-domain
  corpus (vs diplomacy 612 rows/38 steps, craigslist 30k, empathetic 18.7k). ~19h wall.
- **Note on comparability:** single-domain corpora are full-size and vary ~90× in rows (cga 2.5k …
  dailydialog 55k). Ranking is on ToM HM step0→final on the shared `eval_subsample_300` suite; the
  data-quantity confound is inherent to the full-corpus design (other hosts also run full-size).
- **Expected:** healthy training (power+actor-RM never collapses); ToM HM delta likely small/flat or
  weakest of the 8 domains if the smalltalk-baseline hypothesis holds.
- **First rebuild attempt wrote a null `answer_pp` column (build interrupted silently, exit 0); the
  parquet was deleted and rebuilt cleanly (55157/55157 non-null) before launch.**

**Live:** WandB `32a0jste`; launched 2026-07-14 ~07:20 on h100-076-003. Findings appended on completion.

**Findings:** _(fill on completion via log-results skill)_


## Findings (E022, r1 authoritative) — COMPLETED 2026-07-15

**Verdict: PASSED (positive transfer).** Training healthy throughout (power reward warmed up
negative then held steady ~+38–40, format_error 0, advantages nonzero, response_length stable
~90–100, no collapse/OOM over all 3446 steps / 2 epochs).

**ToM transfer (24 ToM benchmarks, excl. gsm8k+mmlu):**
- ToM HM step0→final: **0.4197 → 0.4674 (+0.0477)**
- mean delta: **+2.37pp**; improved(>2pp): 10; worsened(>2pp): 4; exact-zeros(collapse): 0
- Capabilities (reported separately): gsm8k 0.663→0.757 (**+9.4pp**), mmlu 0.463→0.643 (**+18.0pp**) — POSITIVE, no capability regression.

**Framing vs E020 diplomacy (HM +0.050) & smalltalk-baseline hypothesis:**
- Hypothesis predicted dailydialog (open-domain smalltalk) would transfer WEAKER than
  social/negotiation domains. **NOT supported** — dailydialog (+0.048 HM) is essentially on par
  with diplomacy (+0.050 HM). Behavior-prediction RL on plain chit-chat induces ToM gains
  comparable to strategic-dialogue domains at this recipe. Suggests the transfer signal is not
  strongly domain-gated among conversational corpora (at least for Qwen2.5-3B / power-k5).
- Biggest per-benchmark winners: fantom_answerability_list +18.7pp, explore_tom +11.0pp,
  tombench +9.7pp, hi_tom +9.3pp, opentom_location_fo +5.6pp. Losers: bigtom_forward_action
  -4.7pp, bigtom_backward_belief -3.6pp, opentom_multihop_fo -3.0pp.

**Rerun one-liner:**
`EXP_ID=data-recipe-P0_single_dailydialog DATA_NAME=dcfg_dailydialog DATA_TRAIN=$HOME/repo/BeRL/data/dcfg_dailydialog.parquet MAX_PROMPT=2048 MAX_RESP=512 KL=0.05 REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-6 USE_ACTOR_AS_RM=True LR=5e-7 FORMAT_PENALTY=5 ENTROPY_COEFF=0.0 setsid bash experiments/train_behavior_qwen2.5.sh`

WandB: 32a0jste | Log: logs/20260714/data-recipe-P0_single_dailydialog-dcfg_dailydialog-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1.log | Owner: h100-076-003
