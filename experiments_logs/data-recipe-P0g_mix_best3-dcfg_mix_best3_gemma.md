### Attempt r1 — 2026-07-15T15:03:32+00:00

- **RUN_NAME:** `data-recipe-P0g_mix_best3-dcfg_mix_best3_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-076-003   **git:** `c887e83`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best3_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=2 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=data-recipe-P0g_mix_best3-dcfg_mix_best3_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260715/data-recipe-P0g_mix_best3-dcfg_mix_best3_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best3_gemma.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=google/gemma-2-2b-it \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.think_only_pg=False \
    actor_rollout_ref.actor.format_penalty=5 \
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
    trainer.experiment_name=data-recipe-P0g_mix_best3-dcfg_mix_best3_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=2 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=google/gemma-2-2b-it \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-4 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-4 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=data-recipe-P0g_mix_best3 DATA_NAME=dcfg_mix_best3_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best3_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


---

## Hypothesis / question (E102, Gemma-2 2B best-3 mix)

**Exp #:** E102 | **Run-name stem:** data-recipe-P0g_mix_best3-dcfg_mix_best3_gemma | authoritative attempt: r1
**Owner:** h100-076-003 | **Start:** 2026-07-15 15:03

**Question:** Which domain mixture best induces ToM transfer for the Gemma-2 arm? This is the
S2 "best-3" mixture = the 3 top single domains from the Gemma S1 arm ranked on d_avg (primary)
+ d_cavg (format-controlled cross-check), clean runs only (casino excluded — KL exploded to 9.65
= reward-hacking): **craigslist + dailydialog + empathetic**. Mirror of the Qwen2.5 E024 arm.

**Expected outcome:** the best-3 mixture should match or beat the best single Gemma domain and the
full mix, with healthy (non-exploding) KL and no collapse, demonstrating that a curated multi-domain
behavior-prediction mixture transfers to ToM at least as well as any single domain.

## Implementation details (resolved knobs)

- Model: google/gemma-2-2b-it (Gemma-2 2B)
- Data: dcfg_mix_best3_gemma (tag-free base; craigslist 3000 + dailydialog 4000 + empathetic 4000 = 11000 rows; answer_pp all non-null)
- Reward: power, k=5, ll_min=-4, **frozen RM** (USE_ACTOR_AS_RM=False, RM=gemma-2-2b-it), no baseline
- KL=0.05, LR=5e-7, format_penalty=5, entropy_coeff=0.001, gen ctx 2048/512, COT_FREEFORM
- batch=32, mini_batch=128, rollout_n=16, 2 epochs → ~688 steps
- fold_system_prompt=True, require_answer_tags=False (tag-free Gemma)

**Exact launch command:**
```
EXP_ID=data-recipe-P0g_mix_best3 DATA_NAME=dcfg_mix_best3_gemma \
DATA_TRAIN=$HOME/repo/BeRL/data/dcfg_mix_best3_gemma.parquet \
MAX_PROMPT=2048 MAX_RESP=512 KL=0.05 REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-4 \
USE_ACTOR_AS_RM=False LR=5e-7 FORMAT_PENALTY=5 ENTROPY_COEFF=0.001 \
setsid bash experiments/train_behavior_gemma.sh
```

**WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP (run: data-recipe-P0g_mix_best3-dcfg_mix_best3_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1)
**Log:** logs/20260715/data-recipe-P0g_mix_best3-dcfg_mix_best3_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1.log

## Findings (E102 — COMPLETE, PASSED) — 2026-07-15 21:40

**Verdict: ✅ PASSED (strong).** Gemma-2 2B best-3 mix (craigslist+dailydialog+empathetic),
frozen-RM power k5 ll_min−4, ran clean to step 686/~688, no collapse, KL healthy throughout
(~0.03–0.15, never near the 9.65 casino-hacking regime), format_error 0 throughout.

**ToM harmonic mean (24 benches, excl gsm8k+mmlu): 0.0931 → 0.2550 (+0.1619).**
- Mean per-bench delta: **+11.85pp**; improved(>2pp)=**24/24**; worsened(>2pp)=**0**; exact-zeros(final)=**0** (no collapse).
- Held-out general: gsm8k 0.280→0.407 (+12.7pp); mmlu 0.387→0.453 (+6.6pp).

**Largest gains:** opentom_multihop_fo +34.3pp, opentom_multihop_so +30.3pp, opentom_location_so
+28.3pp, tombench +24.0pp, explore_tom +23.0pp, simpletom_judgment +15.3pp, bigtom_forward_action
+14.3pp, dyntom_type_a +14.0pp. Even the smallest gain (dyntom_type_c) was +2.0pp — uniformly positive.

**Interpretation:** For the Gemma-2 arm, the curated best-3 behavior-prediction mixture induces broad,
uniform ToM transfer with no reward-hacking and no benchmark regressions — every single ToM task
improves, with the multi-hop/second-order OpenTom tasks (hardest) gaining most. The low base HM
(0.093) reflects several near-zero base benchmarks on Gemma-2-2b; the mixture lifts all of them.
This is the Gemma mirror of the Qwen2.5 E024 best-3 arm and confirms behavior-prediction RL transfers
across model families.

**Rerun one-liner:**
```
EXP_ID=data-recipe-P0g_mix_best3 DATA_NAME=dcfg_mix_best3_gemma DATA_TRAIN=$HOME/repo/BeRL/data/dcfg_mix_best3_gemma.parquet MAX_PROMPT=2048 MAX_RESP=512 KL=0.05 REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-4 USE_ACTOR_AS_RM=False LR=5e-7 FORMAT_PENALTY=5 ENTROPY_COEFF=0.001 setsid bash experiments/train_behavior_gemma.sh
```
**WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP (run …-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1)
**Log:** logs/20260715/data-recipe-P0g_mix_best3-dcfg_mix_best3_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1.log | **Owner:** h100-076-003 | **End:** 2026-07-15 21:39
