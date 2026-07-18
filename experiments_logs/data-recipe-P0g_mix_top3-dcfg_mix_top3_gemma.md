### Attempt r1 — 2026-07-17T22:03:14+00:00

- **RUN_NAME:** `data-recipe-P0g_mix_top3-dcfg_mix_top3_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-082-004   **git:** `558dc42`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `data/dcfg_mix_top3_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=data-recipe-P0g_mix_top3-dcfg_mix_top3_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260717/data-recipe-P0g_mix_top3-dcfg_mix_top3_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/dcfg_mix_top3_gemma.parquet \
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
    trainer.experiment_name=data-recipe-P0g_mix_top3-dcfg_mix_top3_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
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

**How to rerun:** `EXP_ID=data-recipe-P0g_mix_top3 DATA_NAME=dcfg_mix_top3_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=data/dcfg_mix_top3_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-17T22:03:27+00:00

- **RUN_NAME:** `data-recipe-P0g_mix_top3-dcfg_mix_top3_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-082-004   **git:** `558dc42`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `data/dcfg_mix_top3_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=data-recipe-P0g_mix_top3-dcfg_mix_top3_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260717/data-recipe-P0g_mix_top3-dcfg_mix_top3_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/dcfg_mix_top3_gemma.parquet \
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
    trainer.experiment_name=data-recipe-P0g_mix_top3-dcfg_mix_top3_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
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

**How to rerun:** `EXP_ID=data-recipe-P0g_mix_top3 DATA_NAME=dcfg_mix_top3_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=data/dcfg_mix_top3_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


---

## Findings (r1 — COMPLETED, POSITIVE but does NOT beat mix_best3)

**Run:** WandB `ldyyw1c7`, 343/343 steps, Gemma-2-2B-it, frozen base-LM RM, tag-free, power k5 ll_min-4, KL0.05, LR5e-7, fp5 ec0.001, 2048/512.

**Canonical scorer output (`scripts/score_run.py`):**
```
eval iters: 70 (step 0..343); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.1649  HM(last3)=0.1663  (baseline step0=0.0929)
ToM avg(last5)=0.405  avg(last3)=0.4111  (baseline step0=0.315)
gsm8k (separate): 0.386 (step0=0.277, delta vs step0=+0.109)
mmlu (separate): 0.4432 (step0=0.387, delta vs step0=+0.056)
health(final): kl=0.008 entropy=1.415 resp_len=169.666 reward=0.018 parseable=1.0 max_resp=512
```

**Verdict: POSITIVE transfer, clean health, but the honest d_cavg re-ranking did NOT improve the mixture.**
- ToM avg(last5) 0.315→0.405 (**+9.0pp**), avg(last3) +9.6pp; HM(last5) 0.0929→0.1649 (+7.2pp) — HM is noise-prone for Gemma (near-zero baseline), avg is the reliable metric.
- Capability IMPROVED: gsm8k Δ+10.9pp, mmlu Δ+5.6pp.
- Health is the cleanest of the Gemma mixes: final **kl=0.008** (vs E101 mix_all's KL 6.9 drift), resp_len 170 (bounded, no 512-cap length inflation), parseable 1.0, no collapse. HM trajectory rises to a ~0.25 peak around step 270–285, then settles ~0.18.

**Comparison (Gemma domain-mixture picture):**
| Run | Mix | ToM avg Δ | ToM HM Δ | gsm8k Δ | mmlu Δ | health |
|---|---|---|---|---|---|---|
| E101 mix_all | 10 domains | +9.73pp | +7.99pp | +9.2pp | +1.7pp | KL 6.9 drift, HM declines late |
| E102 mix_best3 | craigslist+dailydialog+empathetic | **+11.85pp** | **+16.2pp** | +12.7pp | +6.6pp | KL healthy, 24/24 up |
| **E108 mix_top3 (this)** | p4g+empathetic+dailydialog | +9.0pp | +7.2pp | +10.9pp | +5.6pp | **KL 0.008 cleanest** |

**Interpretation:** The honest-rerank top3 (which SWAPPED OUT craigslist — negative honest d_cavg −0.081 — for p4g) **does not beat** E102 mix_best3 (which INCLUDED craigslist) on final ToM (avg +9.0pp < +11.85pp; HM +7.2pp < +16.2pp), and is roughly on par with E101 mix_all (+9.73pp avg). So the **format-controlled single-domain reranking (d_cavg) does NOT predict mixture ToM gain better than the raw d_avg selection** in this Gemma family — the domain whose isolated honest gain was negative (craigslist) still contributed to the strongest mix. E108's one clear edge is dramatically cleaner training dynamics (KL 0.008 vs E101's 6.9). Net: behavior-prediction RL transfers robustly across all three Gemma mixes, but honest-rerank top3 is not the winning recipe — mix_best3 remains the strongest Gemma mixture.

**How to rerun:** `EXP_ID=data-recipe-P0g_mix_top3 DATA_NAME=dcfg_mix_top3_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=data/dcfg_mix_top3_gemma.parquet REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-4 USE_ACTOR_AS_RM=False KL=0.05 LR=5e-7 FORMAT_PENALTY=5 ENTROPY_COEFF=0.001 MAX_PROMPT=2048 MAX_RESP=512 TEST_FREQ=5 TOTAL_EPOCHS=1 RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`
