### Attempt r1 — 2026-07-15T23:21:31+00:00

- **RUN_NAME:** `data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-076-003   **git:** `00a37c2`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_surprise_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260715/data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_surprise_gemma.parquet \
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
    trainer.experiment_name=data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
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

**How to rerun:** `EXP_ID=data-recipe-P0g_filter_surprise DATA_NAME=dcfg_mix_best_surprise_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_surprise_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


---

## Hypothesis / question (E103, Gemma-2 2B surprise-filter ON)

**Exp #:** E103 | **Run-name stem:** data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma | authoritative attempt: r1
**Owner:** h100-076-003 | **Start:** 2026-07-15 23:22

**Question (S3 turn-filtering ablation, Gemma-2 arm — mirror of Qwen2.5 E026):** Does training only
on the **most surprising / ToM-dependent** human turns improve ToM transfer per-token? Surprise filter
ON = keep the 50% of turns with the **lowest answer_pp** (highest surprisal), scored by the shared
Qwen2.5-3B perplexity model, over the S2-winner best-3 Gemma mix (craigslist+dailydialog+empathetic).

**Expected outcome:** if ToM-dependent turns carry the transfer signal, surprise-ON should match or
beat filter_off (E104) at half the turns, and clearly beat the length-matched random control (E105).
Compared against filter_off (E104) and randlen control (E105).

## Implementation details (resolved knobs)

- Model: google/gemma-2-2b-it (Gemma-2 2B)
- Data: dcfg_mix_best_surprise_gemma (best3 Gemma mix, surprise filter keep_fraction=0.5, seed=42; **5500 rows**, answer_pp all non-null, range −18.99…−5.20 = low tail kept)
- Reward: power, k=5, ll_min=-4, **frozen RM** (USE_ACTOR_AS_RM=False, RM=gemma-2-2b-it), no baseline
- KL=0.05, LR=5e-7, format_penalty=5, entropy_coeff=0.001, gen ctx 2048/512, COT_FREEFORM
- batch=32, mini_batch=128, rollout_n=16, 2 epochs → ~344 steps
- fold_system_prompt=True, require_answer_tags=False (tag-free Gemma)

**Exact launch command:**
```
EXP_ID=data-recipe-P0g_filter_surprise DATA_NAME=dcfg_mix_best_surprise_gemma \
DATA_TRAIN=$HOME/repo/BeRL/data/dcfg_mix_best_surprise_gemma.parquet \
MAX_PROMPT=2048 MAX_RESP=512 KL=0.05 REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-4 \
USE_ACTOR_AS_RM=False LR=5e-7 FORMAT_PENALTY=5 ENTROPY_COEFF=0.001 \
setsid bash experiments/train_behavior_gemma.sh
```

**WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP (run: data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1)
**Log:** logs/20260715/data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1.log

## Debugging / issue (E103 — HALTED at step 125, awaiting user decision) — 2026-07-16 00:15

**Symptom:** From step 1 onward, **every** rollout's reward = **−45.000** (the power invalid sentinel
= valid_floor 0 − format_penalty 5 − INVALID_MARGIN 40), mean=max=min=−45, **advantages ≡ 0** → zero
learning across all 125 steps. format_error_ratio=0, response_length healthy (7–318 tok), KL ~0.001,
no OOM/Traceback. GPUs busy the whole time but the run was doing nothing useful.

**Diagnosis:** The frozen RM flags **all** surprise-filtered rollouts as *invalid* even though the
model responses are structurally valid (they contain `</think>` + an answer, e.g.
`</think>Our house special is our Cuervo Gold margarita.<end_of_turn>`). The invalid path that fires
is the **empty scored ground-truth region** (`response_token_count==0` → NaN → invalid, verl/workers/
fsdp_workers.py:837-840), i.e. after stitching [prompt + long CoT + ground_truth] and right-truncating
to the RM context, the ground_truth answer region is left with 0 scored tokens.

**Why it is surprise-filter-specific (not a generic reward bug):** the sibling run **E102**
(dcfg_mix_best3_gemma, identical knobs: Gemma-2-2b frozen RM, power k5 ll_min−4, fp5, ec0.001) had
`critic/rewards/min == 0.000` for the *entire* run — **zero** −45/invalid groups — and passed strongly
(+0.162 HM). The surprise build is a *fresh* rebuild (not a subset of E102's parquet) and diverges:
E102's parquet has `answer_pp` all-NaN while E103's `dcfg_mix_best_surprise_gemma` has valid answer_pp
(−18.99…−5.20). `answer_pp` itself does NOT feed rm_score (subtract_baseline=False), so it is not the
cause — but the divergent builds indicate the surprise/filter build path produces data the frozen-RM
scoring universally rejects (plausibly the low-answer_pp turns elicit longer CoT that truncates the
stitched ground_truth away, or a boundary/stitching interaction unique to the filtered set).

**Scope / blast radius:** likely affects the whole S3 turn-filtering *filtered-data* family that uses
freshly built filter parquets — **E103 (surprise-gemma), E105 (randlen-gemma), E106 (predictable-qwen),
E107 (predictable-gemma)**, and the Qwen mirrors **E026/E028**. `filter_off` arms (E104 =
dcfg_mix_best_gemma, no filter) should be unaffected (that path == E102's working data).

**Action taken:** killed the run (no learning); GPUs freed. Set tracker E103 → **Awaiting-input**
(needs a shared decision on the surprise-filter build / RM invalid-region handling, which I must not
change unilaterally mid-flight while other runs may load shared verl/ + dcfg_* files). NOT re-run
as-is (deterministic failure from step 1).

---

## Deep-dive update (user-approved) — truncation/stitching is NOT the root cause

I faithfully reproduced the frozen-RM `_switch_chat_template` boundary logic OFFLINE
(fsdp_workers.py:1360-1467) on the **current on-disk E103 parquet** and, separately, on the **exact
real CoT captured from the failing run's log** (the restaurant "Cuervo Gold margarita" example). In
every case the path produces a **valid, non-empty scored region**:

- exact-real-CoT example: `thinking_length=351 < full_len=361`, `answer_start=349`,
  `response_token_count=13`, scored region decodes to
  `'>Our house special is our Cuervo Gold margarita .<end_of_turn>\n'` (== the ground truth).
- 5 sampled E103 rows: `thinking_length < full_len` in all, `response_token_count = 8–23`, scored
  region == ground_truth.

So on the real E103 data none of the invalid triggers fire offline:
- line 1373 (`</think>` absent) — responses ARE well-formed (`format_error_ratio=0.000` in the run;
  `</think>` present), so this does not fire.
- line 1434 (`thinking_length >= full_len`) — FALSE (fl > tl by ~ground_truth length; prompts are
  small, mean ~180-200 tok, max <600, so [prompt+CoT+GT] is ~350-750 tok « max_length 2560).
- line 1318/1837 (`response_token_count==0` → empty mask) — does NOT happen; the answer region is
  correctly masked (8-23 tokens).

**Correction to the earlier diagnosis:** the −45 collapse is therefore **NOT** an
empty-ground-truth / truncation / stitching bug in the data. That hypothesis is falsified by the
offline reproduction on the exact failing inputs.

### The one hard structural difference (a latent inconsistency, but NOT this collapse's cause)
`answer_pp` column dtype governs where it loads (rl_dataset.py:183-187 `torch.tensor(v)` →
collate_fn tensors vs non_tensors):
- float64 scalar → `data.batch['answer_pp']` (tensor).
- object/None → `data.non_tensor_batch['answer_pp']`.

The frozen-RM path reads `data.batch['answer_pp']` **unguarded** (fsdp_workers.py:1468) while the
actor-RM path reads `non_tensor_batch.get('answer_pp',[0.0])` with a None guard (772-774). A build
that emits object/None answer_pp would therefore make the **frozen** path `KeyError`, not the actor
path. BUT: at *run* time both E102 and E103 had valid float answer_pp (E102's current all-NaN is a
later artifact — building the surprise config, which *extends* dcfg_mix_best3_gemma, overwrote the
base parquet's answer_pp). And `answer_pp` never feeds `rm_score` (subtract_baseline=False). So this
asymmetry is a real latent fragility to fix, but it is **not** the cause of the E102-vs-E103 divergence.

### What this leaves as the actual cause
Responses are well-formed, prompts are short, the stitching/masking is provably correct on the exact
failing inputs — yet the live frozen-RM Gemma run pins **every** rollout at the −45 invalid sentinel
from step 1 (advantages≡0). This is NOT reproducible from the stitching path offline, which points to
a **runtime forward-pass / data-loading interaction specific to frozen-RM + Gemma-2 on the new
S3-pipeline parquet** (e.g. NaN logits in the RM bf16/flash-attn/gemma2 soft-cap forward →
`nan_to_num`→invalid at fsdp_workers.py:1320, or a batch-assembly difference), rather than the
truncation bug originally suspected. E105 (another agent, h100-189-003) independently hit the
identical −45 collapse on a *different* filtered parquet (randlen), confirming it is a shared
frozen-RM+Gemma+new-parquet issue, not surprise-selection-specific.

### Proposed next actions (need user decision — touches shared verl/ code)
1. **Minimal diagnostic re-run:** add per-sample instrumentation to the frozen-RM path (count invalid
   by trigger: no-`</think>` vs tl>=fl vs response_token_count==0 vs NaN-logits) and re-run E103 for
   ~5 steps to capture *which* line actually fires at runtime. Smallest safe step; requires a
   temporary verl/ print (shared code) → needs approval.
2. **Harden the frozen path** to mirror the actor path (None/dtype-guard `answer_pp`; it already has
   the response_token_count guard) — defensive, cheap, but may not be the fix.
3. **If NaN logits:** inspect the RM forward for gemma2 attn logit soft-capping under bf16+flash-attn
   in the frozen RewardModelWorker vs the actor-as-RM path (which works on the same data for Qwen).

Deferring the shared-code change to the user (golden rule #4). Setting E103 → Awaiting-input,
aligned with E105's parallel escalation of the same shared-code bug.
### Attempt r2 — 2026-07-17T00:11:17+00:00

- **RUN_NAME:** `data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r2`
- **Host:** h100-076-003   **git:** `51194e4`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_surprise_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r2 _(paste link after launch)_
- **Log path:** `logs/20260717/data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r2.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_surprise_gemma.parquet \
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
    trainer.experiment_name=data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r2 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
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

**How to rerun:** `EXP_ID=data-recipe-P0g_filter_surprise DATA_NAME=dcfg_mix_best_surprise_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_surprise_gemma.parquet RUN_INDEX=2 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r2 — 2026-07-17T00:11:29+00:00

- **RUN_NAME:** `data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r2`
- **Host:** h100-076-003   **git:** `51194e4`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_surprise_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r2 _(paste link after launch)_
- **Log path:** `logs/20260717/data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r2.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_surprise_gemma.parquet \
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
    trainer.experiment_name=data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r2 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
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

**How to rerun:** `EXP_ID=data-recipe-P0g_filter_surprise DATA_NAME=dcfg_mix_best_surprise_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_surprise_gemma.parquet RUN_INDEX=2 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r2 — 2026-07-17T00:11:45+00:00

- **RUN_NAME:** `data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r2`
- **Host:** h100-076-003   **git:** `51194e4`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_surprise_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r2 _(paste link after launch)_
- **Log path:** `logs/20260717/data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r2.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_surprise_gemma.parquet \
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
    trainer.experiment_name=data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r2 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
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

**How to rerun:** `EXP_ID=data-recipe-P0g_filter_surprise DATA_NAME=dcfg_mix_best_surprise_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_surprise_gemma.parquet RUN_INDEX=2 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


### Attempt r2 — 2026-07-17 (RESUME on shared fix)
- **Trigger:** Shared fix committed by another agent: `8769467` "model-aware invalid check — tag-free
  Gemma2/Qwen3 rollouts no longer force-invalid on missing `</think>`". This resolves the blocker I
  escalated (the frozen-RM Gemma `-45` invalid-sentinel collapse). Sibling E107 (filter_predictable)
  relaunched healthy on the same fix (reward mean 0.005→0.064 with variance, non-zero advantages).
  Corrected root cause (supersedes my answer_pp/runtime-forward hypothesis): the unconditional
  `THINK_CLOSE not in response` gate flagged ~100% of tag-free Gemma rollouts invalid; the fix gates
  it on `REQUIRE_ANSWER_TAGS` via `is_invalid_response`.
- **Run:** `...-r2`, WandB `6mwal1n0`, log `logs/20260717/...-r2.log`. Node h100-076-003.
- **Knobs unchanged** from r1 (frozen RM, power k=5 ll_min=-4, kl=0.05, lr=5e-7, fp=5, ec=0.001,
  ctx 2048/512, require_answer_tags=False, fold_system_prompt=True). Running on committed fixed code.
- **Watch:** reward/mean should show variance (not pinned at -45); advantages ≠ 0; kl climbing.

- **r2 healthy start CONFIRMED (steps 1–6):** the `-45` invalid-sentinel collapse is GONE.
  `critic/rewards/min` no longer pinned at -45; `reward/format_error_ratio=0.000`; response_length
  ~150–170 tok (min 6–17, max ~380, no clip); by step 6 within-group variance appears
  (`critic/advantages/max=3.694 min=-0.260`), `actor/pg_loss` 0.000→0.001, `grad_norm` 0.014→0.068,
  `kl_coef=0.050`. reward/mean displays ~0.000 (power reward near top of range; z-scored advantages
  amplify tiny per-group differences) — same healthy signature as E107. Training proceeding; ~172
  steps/epoch expected. WandB 6mwal1n0.

- **r2 UPDATE — reward FLOORS at 0 (no -45, but no gradient either):** the shared fix removed the
  invalid-sentinel collapse, but across steps 1–11 `critic/score/max = critic/rewards/max = 0.000`
  (only step 6 shows a tiny advantages blip, max 3.694/min -0.260; every other step advantages≈0).
  Interpretation: the power reward maps normalized ll∈[ll_min,0]→[0,1]^k; with `ll_min=-4` on the
  SURPRISE-filtered data (deliberately the highest-surprisal / least-predictable turns, whose
  frozen-Gemma-2 ground-truth log-likelihood is well below -4) essentially every rollout clamps to
  the valid floor 0 ⇒ zero within-group variance ⇒ zero GRPO advantage ⇒ no learning. This is NOT a
  code bug — it's an inherent interaction between "select hardest turns" and a shallow `ll_min=-4`.
  Sibling E107 (predictable / low-surprisal) is healthy at the SAME `ll_min=-4` because its targets
  sit near ll≈0. Decision escalated to user (set tracker → Awaiting-input): deepen `ll_min` for the
  surprise arm (breaks knob-parity with E104/E105/E107) vs accept as a documented no-signal null.
  Run left running on h100-076-003 pending the decision.
### Attempt r1 — 2026-07-17T01:08:43+00:00

- **RUN_NAME:** `data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-8-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-076-003   **git:** `22e3d41`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_surprise_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-8 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-8-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260717/data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-8-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_surprise_gemma.parquet \
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
    trainer.experiment_name=data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-8-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=google/gemma-2-2b-it \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-8 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-8 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=data-recipe-P0g_filter_surprise DATA_NAME=dcfg_mix_best_surprise_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_surprise_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-17T01:08:55+00:00

- **RUN_NAME:** `data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-8-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-076-003   **git:** `22e3d41`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_surprise_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-8 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-8-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260717/data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-8-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_surprise_gemma.parquet \
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
    trainer.experiment_name=data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-8-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=google/gemma-2-2b-it \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-8 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-8 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=data-recipe-P0g_filter_surprise DATA_NAME=dcfg_mix_best_surprise_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_surprise_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


### Attempt r1 @ ll_min=-8 — 2026-07-17 (user-approved deeper floor)
- **Decision:** User chose to kill the floored r2 (ll_min=-4) and rerun with a deeper floor
  `ll_min=-8` so the surprise-filtered (high-surprisal) targets are not all clamped to the power
  valid-floor. Explicitly accepted trade-off: this breaks knob-parity with siblings E104/E105/E107
  (which stay at ll_min=-4).
- **Run:** `...-power-k5-llmin-8-...-r1`, WandB `ufkbmjfw`,
  log `logs/20260717/...-llmin-8-...-r1.log`. Node h100-076-003. All other knobs unchanged
  (frozen RM, power k=5, kl=0.05, lr=5e-7, fp=5, ec=0.001, ctx 2048/512, require_answer_tags=False,
  fold_system_prompt=True). Running on committed fixed code 8769467.
- **Watch:** reward should now lift off the floor (critic/score/max > 0 with variance),
  advantages non-zero, gradient flowing — vs r2 which pinned at 0.

- **ll_min=-8 HEALTHY start CONFIRMED (steps 1–9):** reward lifted off the floor — critic/score/mean
  ~5–16 (r2@ll_min=-4 was pinned at 0), critic/rewards/max=40 (some rollouts saturate the +40
  MAX_REWARD clamp but the mean sits well below → distribution has spread), advantages healthy
  (max ~3.7, min ~-2 to -3.75), non-zero gradient, no Traceback/OOM. Confirms the r2 flooring was a
  shallow-`ll_min` × surprise-filter interaction, not a code bug. Deeper floor -8 restores signal.
  WandB ufkbmjfw. Training to completion.

- **ll_min=-8 BREVITY-HACK COLLAPSE (killed @ step13):** deeper floor overshot the other way. Reward
  saturates the +40 MAX_REWARD clamp (critic/rewards/max=40 every step; sibling healthy E107 peaked
  at ~2.77), and within ~7 steps the policy brevity-hacks: response_length collapses 108→85→25→18→16
  →~15 tok (min 3–5), KL blows up 0.022→0.70 (kl_coef=0.05 can't contain a reward of this
  magnitude), score_mean stays high ~7–16. Sampled rollouts are degenerate — a single short
  `<think>"<one plausible human line>"` with an empty answer, i.e. the model directly echoes a
  short human-style utterance to maximize the frozen-RM likelihood of the (also short) true next
  turn, doing NO reasoning. Not ToM learning.
- **Knob bracketed:** ll_min=-4 → reward floors at 0 (no gradient); ll_min=-8 → reward saturates +40
  (brevity-hack collapse). The surprise arm needs an intermediate floor and/or stronger containment.
  Escalated to user (tracker → Awaiting-input). Candidate fixes: (i) intermediate ll_min≈-6 (matches
  the healthy Qwen power runs' k=3/ll_min=-6); (ii) reduce steepness k=5→3; (iii) raise KL and/or add
  explicit length control; (iv) accept surprise-arm as not-learnable at these knobs. Run killed, node
  freed.
### Attempt r1 — 2026-07-17T16:30:45+00:00

- **RUN_NAME:** `data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-076-003   **git:** `02ab658`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_surprise_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260717/data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_surprise_gemma.parquet \
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
    trainer.experiment_name=data-recipe-P0g_filter_surprise-dcfg_mix_best_surprise_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=google/gemma-2-2b-it \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=data-recipe-P0g_filter_surprise DATA_NAME=dcfg_mix_best_surprise_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best_surprise_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


### Attempt r1 @ ll_min=-6 — 2026-07-17 (user decision 2: intermediate floor)
- **Decision:** After ll_min=-4 floored (no signal) and ll_min=-8 brevity-hacked (saturated +40),
  user chose the intermediate `ll_min=-6` (keep k=5) — matches the healthy Qwen power runs' floor.
- **Run:** `...-power-k5-llmin-6-...-r1`, WandB `aa64z8gx`, log `logs/20260717/...-llmin-6-...-r1.log`.
  Node h100-076-003. All other knobs unchanged (frozen RM, power k=5, kl=0.05, lr=5e-7, fp=5,
  ec=0.001, ctx 2048/512, require_answer_tags=False, fold_system_prompt=True). Fixed code 8769467.
- **Watch:** the sweet spot — reward should have variance WITHOUT saturating the +40 clamp
  (critic/rewards/max well below 40, like E107's ~2.77 peak), response_length stable (~100–180, not
  collapsing to ~15), KL contained (not blowing past ~0.3). If it saturates+brevity-hacks again →
  escalate; if it floors → escalate.

- **ll_min=-6 HEALTHY start (steps 1–8):** the sweet spot vs the two failure modes. Reward has real
  variance WITHOUT saturating — critic/score/mean 0.007–0.83, critic/rewards/max mostly 2.9–17 (only
  step 7 touched the +40 clamp, not pinned; contrast ll_min=-8 = 40 every step, ll_min=-4 = 0 every
  step). KL contained 0.001–0.031 (vs 0.7 blow-up at ll_min=-8). resp_len ~90–167 (drifting down from
  167→92, watching for brevity drift but KL tiny + reward unsaturated ⇒ looks like normal settling),
  advantages non-zero, format_err=0, no crash. WandB aa64z8gx. Monitoring to completion via schedule
  (kill+escalate if it later saturates+brevity-hacks or floors).

- **ll_min=-6 ALSO COLLAPSED (killed @ step15, ~15h wall-clock):** the healthy steps 1–8 were
  transient. By step ~9–10 the policy brevity-drifted (resp_len 167→39, min 3–5), `critic/rewards/min`
  floored to 0, `critic/score/max` intermittently saturated the +40 clamp (steps 9,11,13), and the
  `[GRPO group rewards]` dump went all-zero at step 11 → noisy/unstable, not learning. SECONDARY OP
  ISSUE: once the policy degenerated, the shared verbose per-rollout + `[GRPO group rewards]` console
  printing (core_algos.py:143, long-standing debug print) exploded and became the bottleneck — GPUs
  fell to 1% util and only ~15 steps completed in ~15h (timing_s/step stayed ~27s, so the wall-clock
  went entirely into log I/O). This slowness is a SYMPTOM of the collapse (degenerate outputs → print
  backlog), not independent; a stable run (k=3?) should stay fast like the healthy first steps.
- **CONCLUSION — k=5 power fails on the surprise arm at every floor tried:** ll_min=-4 → reward floors
  at 0 (no gradient); ll_min=-6 and ll_min=-8 → brevity-hack collapse (reward saturates +40, resp_len
  crashes, KL climbs). The steepness of `k=5` (reward reaches the +40 clamp) is the likely culprit.
  Escalated to user (Awaiting-input); recommend option (ii) k=3, ll_min=-6 (less steep, matches the
  healthy Qwen power runs' k=3/ll_min=-6), or raising KL.
