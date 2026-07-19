### Attempt r1 — 2026-07-12T17:27:53+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rma_kl0.05_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1`
- **Host:** h100-076-003   **git:** `16570d4`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=actor baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/1024 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1w2gemma2_rma_kl0.05_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=google/gemma-2-2b-it \
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
    actor_rollout_ref.actor.think_only_pg=False \
    actor_rollout_ref.actor.format_penalty=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=999 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=google/gemma-2-2b-it \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=log_prob \
    +reward_model.power_k=2.0 \
    +reward_model.power_ll_min=-8.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=log_prob \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr1e-6_fp0_ec0.0_lp DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


### Debugging note (r1) — 2026-07-12T18:xx — BLOCKER: 100% invalid rollouts, no learning signal
- **Symptom:** `reward/mean` pinned at exactly **-80.000** for all 27 steps; `critic/advantages/mean:0.000` → GRPO has zero within-group reward variance → **no learning**. Length ~140-160 (not collapsed), format_error_ratio 0.000.
- **Root cause:** -80 = `invalid_reward_value(log_prob)` = `valid_reward_floor(-40) - format_penalty(0) - INVALID_MARGIN(40)` (fsdp_workers.py:55,68,80). Every rollout is flagged **invalid** at `fsdp_workers.py:707`: a response is invalid iff it does NOT contain `THINK_CLOSE` (`"</think>"`, response_parser.py:19). `GemmaResponseParser` inherits this tag (response_parser.py:224).
- **Why:** `dcfg_smoke_mix_gemma` is a **tag-free "native-thinking" recipe** — system prompt: *"The assistant first thinks about the reasoning process in the mind and then provides the answer"* with **no `<think>`/`</think>` instruction**. gemma-2-2b-it therefore emits plain text (e.g. `[actor-RM] Model response: "Listener will probably say something like:"`) with no think tags → always invalid.
- **Scope:** systemic to the entire gemma (and likely qwen3) tag-free Wave-2 block — every such run will pin at -80 with zero signal. Other hosts running gemma rows (PS097/PS098) will hit the same wall.
- **Action:** run left alive; PS101 set `Awaiting-input`; raised with user for a decision (fix parser/recipe vs skip gemma tag-free block vs document as negative). Not editing shared `verl/` code unilaterally (join-experiments Golden Rule #4).
### Attempt r2 — 2026-07-12T21:28:10+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rma_kl0.05_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r2`
- **Host:** h100-076-003   **git:** `719757a`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=actor baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/1024 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r2 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1w2gemma2_rma_kl0.05_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r2.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=google/gemma-2-2b-it \
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
    actor_rollout_ref.actor.think_only_pg=False \
    actor_rollout_ref.actor.format_penalty=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr1e-6_fp0_ec0.0_lp-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r2 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=999 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=google/gemma-2-2b-it \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=log_prob \
    +reward_model.power_k=2.0 \
    +reward_model.power_ll_min=-8.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=log_prob \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr1e-6_fp0_ec0.0_lp DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=2 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


### Findings (r2 — authoritative; r1 was the pre-fix DEAD run) — 2026-07-13T04:35
**Verdict: COMPLETED but FAILED cross-family transfer — catastrophic downstream eval collapse despite healthy training signals.**

- **Run health (training):** healthy throughout. Reward climbed off the invalid floor and stabilized ~-5 (final -4.92, never near the -80 invalid sentinel); advantages non-zero; format_error_ratio 0.000; 0 invalid rollouts. Response length was noisy (dipped ~12, spiked ~600) but bounded, ending ~609 (never hit the 1024 cap → no inflation-collapse; no brevity-collapse). ~190 steps, 1 epoch, clean exit (GPU→1MiB, no teardown hang).
- **Downstream eval (the real result):** mean over 26 sub300 benchmarks **collapsed 0.316 → 0.081 (-23.5pp)**; **20/26 worsened**, only 2 improved (gsm8k 0.277→0.383 +0.106; opentom_multihop_so 0.047→0.270 +0.223). Severe drops: tomi 0.597→0.000, bigtom_forward_belief 0.753→0.017, bigtom_backward_belief 0.527→0.007, simpletom_behavior 0.517→0.037, simpletom_judgment 0.410→0.000, simpletom_mental 0.463→0.000, tombench 0.380→0.063, mmlu 0.387→0.063, opentom_attitude 0.283→0.000, fantom_info_binary 0.423→0.043, explore_tom 0.363→0.000, opentom_location_fo 0.510→0.147, hi_tom 0.130→0.000.
- **Mechanism (inferred):** many multiple-choice tasks fell to **exactly 0.000** (tomi, explore_tom, simpletom_mental/judgment, opentom_attitude, hi_tom, dyntom_c/d, fantom_info_list) and mmlu (0.063) sits far below chance — the signature of **parseable-answer collapse**, not gradual capability loss. The BeRL log_prob (utterance-LL) objective on gemma-2-2b-it optimized long free-form native-thinking text that maximizes the held-out human-utterance likelihood, but the resulting policy no longer emits an eval-parseable final answer, so accuracy craters. Training reward/length/format all look fine because they never measure eval answer extraction. (No explicit parseable-answer-rate metric is logged; conclusion rests on the exactly-0 MC scores.)
- **Cross-family comparison:** the Qwen2.5 log_prob winner recipe (PS013, HM-last5 0.435) does NOT transfer to Gemma-2 as-is. Qwen2.5 log_prob stayed healthy; gemma-2-2b log_prob (actor-RM, kl0.05, lr1e-6, fp0, ec0.0) collapses downstream. Suggests gemma needs stronger anti-collapse scaffolding (format penalty fp>0, entropy, and/or an answer-parseability guard) and/or a smaller step — bare log_prob is unsafe on this family.
- **Attempts:** r1 = DEAD (100% invalid, reward pinned -80) due to the tag-free </think> validity bug — superseded. r2 = authoritative (post user parser fix `is_invalid_response`). WandB: y2ymu6oj. Log: logs/20260712/...-r2.log.
