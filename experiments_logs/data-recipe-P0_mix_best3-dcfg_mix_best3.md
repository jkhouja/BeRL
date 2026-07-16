# E024 — data-recipe-P0_mix_best3 (Qwen2.5-3B, best-3 domain mixture)

- **RUN_NAME_BASE:** data-recipe-P0_mix_best3-dcfg_mix_best3
- **Exp #:** E024  | **Exp ID:** P0_mix_best3
- **Run name (full):** P0_mix_best3-dcfg_mix_best3-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1
- **Attempts:** r1 (authoritative)
- **Owner_host:** h100-013-002
- **WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/xirhpvd1
- **Log:** logs/20260715/P0_mix_best3-dcfg_mix_best3-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1.log
- **Start:** 2026-07-15 ~15:06 UTC

## Hypothesis / question
**Which domain mixture is best (Qwen2.5 arm)?** Phase-0 S2 mixture run. S1 single-domain ranking
(d_avg primary + d_cavg format-controlled cross-check, clean runs only) selected the best-3 =
**casino + empathetic + dailydialog** (casino #1 d_avg & best d_cavg +0.071; empathetic best
combined rank; dailydialog high d_avg + highest gsm8k transfer). This trains on the balanced mix
(casino 3000 + empathetic 4000 + dailydialog 4000 = 11000 rows) and asks whether mixing the top
singles beats the best single (E017 casino: ToM HM(l5)=0.472 vs base 0.417 = +5.5pp, avg +2.5pp,
mmlu +16.8pp on Qwen2.5). Expected: mixture ≥ best single, with broader/robust ToM transfer.

## Implementation details (resolved knobs)
- Model: Qwen/Qwen2.5-3B-Instruct (tag-based; require_answer_tags=True).
- Reward: power, k=5, ll_min=-6, actor-as-RM (USE_ACTOR_AS_RM=True), nobaseline (SUBTRACT_BASELINE=False).
- KL=0.05 (low_var_kl), LR=5e-7, format_penalty=5, entropy_coeff=0.0.
- Batch=32, mini_batch=128, rollout_n=16, total_epochs=2, val_suite=subsample300.
- Gen ctx: max_prompt=2048, max_response=512 (verified in driver .out).
- Data: dcfg_mix_best3 → data/dcfg_mix_best3.parquet (11000 rows, answer_pp baseline computed, 0 nulls).
- Locked Qwen2.5 Phase-0 recipe (matches E016–E030 arm).

### Exact launch command
```
EXP_ID=P0_mix_best3 DATA_NAME=dcfg_mix_best3 DATA_TRAIN=data/dcfg_mix_best3.parquet \
  REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-6 USE_ACTOR_AS_RM=True SUBTRACT_BASELINE=False \
  KL=0.05 LR=5e-7 FORMAT_PENALTY=5 ENTROPY_COEFF=0.0 MAX_PROMPT=2048 MAX_RESP=512 \
  bash experiments/train_behavior_qwen2.5.sh
```

## Debugging / issues
- 2026-07-15: dcfg_mix_best3 parquet built fresh (config newly added). build_dataset.py wrote 11000
  rows then computed answer_pp on GPU (~678s / ~11min). Verified 0 nulls. Dry-run confirmed all knobs.
  Launched clean (driver PID 3497300, main_task 3504800, ppo 3497329). max_response=512 verified.

## Findings
(pending — run in progress; ~11000 rows / 32 * 2 epochs ≈ 688 steps, ~5-6h wall)

## How to rerun
See exact launch command above (data parquet already built at data/dcfg_mix_best3.parquet).

## FINAL FINDINGS (E024, r1) — 2026-07-15

**Verdict: STRONG POSITIVE, but mixture MATCHES (does not beat) best single-domain (casino).**

Scorer (`scripts/score_run.py --last 5`, 24 ToM benchmarks excl gsm8k/mmlu):
- ToM HM(last5)=**0.4705** vs base(step0)=0.4176 → **+5.3pp**
- ToM avg(last5)=**0.526** vs base=0.5033 → **+2.3pp**
- mmlu=0.6366 vs step0 0.46 → **+17.7pp**
- gsm8k=0.6602 vs 0.657 → +0.3pp (neutral)
- Health(final): kl=0.055, entropy=1.344, resp_len=105.5, reward=39.0, parseable=1.0, max_resp=512

**Comparison vs E017 casino single-domain** (HM +5.5pp / avg +2.5pp / mmlu +16.8pp):
- Mixture (casino+empathetic+dailydialog) ≈ casino single: HM +5.3 vs +5.5, avg +2.3 vs +2.5, mmlu +17.7 vs +16.8.
- **Conclusion:** the best-3 mixture does NOT beat the best single domain (casino) on Qwen2.5-3B — it essentially matches it. No mixture synergy observed; casino alone captures the gain. mmlu transfer slightly higher.

**Health:** fully stable throughout. resp_len flat ~97-107 (no fp5 collapse — consistent with E017), kl bounded ~0.05-0.07, format_error=0, no crash. fp5+actor-RM well-tolerated on Qwen2.5.

**HM trajectory:** rises 0.418→~0.47 by step 90, plateaus flat to step 686 (no late erosion). Robust.

Run: step 685/686, WandB xirhpvd1, driver PID 3497300 / ppo 3497329. 2 epochs, ~5h wall.
### Attempt r1 — 2026-07-15T23:16:41+00:00

- **RUN_NAME:** `data-recipe-P0_mix_best3-dcfg_mix_best3-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-189-003   **git:** `d203deb`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best3.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=data-recipe-P0_mix_best3-dcfg_mix_best3-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260715/data-recipe-P0_mix_best3-dcfg_mix_best3-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best3.parquet \
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
    trainer.experiment_name=data-recipe-P0_mix_best3-dcfg_mix_best3-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
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

**How to rerun:** `EXP_ID=data-recipe-P0_mix_best3 DATA_NAME=dcfg_mix_best3 MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best3.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** POSITIVE + STABLE (r1, WandB 33l79vjl, 343/343 steps, eval iters 70).
- **ToM HM(last5)=0.4609 HM(last3)=0.4608** (baseline step0=0.4148) => **+4.61pp**.
- **ToM avg/AM(last5)=0.5191 avg(last3)=0.52** (baseline step0=0.5036) => **+1.55pp**.
- gsm8k=0.7128 (step0=0.653, **d+6.0pp**); mmlu=0.6302 (step0=0.48, **d+15.0pp**).
- health(final): kl=0.059 entropy=1.231 resp_len=97.5 reward=38.97 parseable=1.0 max_resp=512 -- no collapse, no length inflation.
- **Stability:** HM rises 0.415->~0.47 by step20 and holds flat 0.46-0.48 across the whole run; last-5 eval iters (325:0.458 330:0.462 335:0.458 340:0.46 343:0.463) all normal, resp_len ~97 (not inflated). **NOT contaminated** -- no late divergence. KL steady 0.05-0.09.
- **Verdict:** best-3 mix (casino+empathetic+dailydialog, 11000 rows) transfers to held-out ToM (+4.61pp HM) with clean stability and general-capability gains (gsm8k/mmlu both up). Slightly below the 10-domain E025 mix_all (HM +5.45pp, AM +2.32pp, 26128 rows) on both ToM metrics => for Qwen2.5 the broader 10-domain mix edges the curated best-3 subset; mix_all remains the stronger dcfg_default candidate. Both POSITIVE+STABLE.

### Attempt r1 — 2026-07-15T23:16:57+00:00

- **RUN_NAME:** `data-recipe-P0_mix_best3-dcfg_mix_best3-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-189-003   **git:** `d203deb`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best3.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=data-recipe-P0_mix_best3-dcfg_mix_best3-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260715/data-recipe-P0_mix_best3-dcfg_mix_best3-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best3.parquet \
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
    trainer.experiment_name=data-recipe-P0_mix_best3-dcfg_mix_best3-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
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

**How to rerun:** `EXP_ID=data-recipe-P0_mix_best3 DATA_NAME=dcfg_mix_best3 MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_mix_best3.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** POSITIVE + STABLE (r1, WandB 33l79vjl, 343/343 steps, eval iters 70).
- **ToM HM(last5)=0.4609 HM(last3)=0.4608** (baseline step0=0.4148) => **+4.61pp**.
- **ToM avg/AM(last5)=0.5191 avg(last3)=0.52** (baseline step0=0.5036) => **+1.55pp**.
- gsm8k=0.7128 (step0=0.653, **d+6.0pp**); mmlu=0.6302 (step0=0.48, **d+15.0pp**).
- health(final): kl=0.059 entropy=1.231 resp_len=97.5 reward=38.97 parseable=1.0 max_resp=512 -- no collapse, no length inflation.
- **Stability:** HM rises 0.415->~0.47 by step20 and holds flat 0.46-0.48 across the whole run; last-5 eval iters (325:0.458 330:0.462 335:0.458 340:0.46 343:0.463) all normal, resp_len ~97 (not inflated). **NOT contaminated** -- no late divergence. KL steady 0.05-0.09.
- **Verdict:** best-3 mix (casino+empathetic+dailydialog, 11000 rows) transfers to held-out ToM (+4.61pp HM) with clean stability and general-capability gains (gsm8k/mmlu both up). Slightly below the 10-domain E025 mix_all (HM +5.45pp, AM +2.32pp, 26128 rows) on both ToM metrics => for Qwen2.5 the broader 10-domain mix edges the curated best-3 subset; mix_all remains the stronger dcfg_default candidate. Both POSITIVE+STABLE.

