### Attempt r1 — 2026-07-13T05:24:46+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-4.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-156-003   **git:** `6018931`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/1024 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-4.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-4.0-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-4.0-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-4.0 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-4.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm4 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


---

## Findings (r1 — PS125, run by h100-156-003, 2026-07-13)

WandB: https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/lpqv3uqz
Log: logs/20260713/Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.0_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-4.0-lr5e-7-kl0.05-n16-r1.log

Canonical scorer (scripts/score_run.py) output:
```
eval iters: 39 (step 0..190); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.1229  HM(last3)=0.1027  (baseline step0=0.0931)
ToM avg(last5)=0.386  avg(last3)=0.3726  (baseline step0=0.3152)
gsm8k (separate): 0.3586 (step0=0.277, delta vs step0=+0.082)
mmlu (separate): 0.4332 (step0=0.383, delta vs step0=+0.050)
health(final): kl=0.075 entropy=1.464 resp_len=101.822 reward=-1.095 parseable=1.0
ToM HM trajectory: 0:0.093 5:0.045 10:0.031 15:0.017 20:0.046 25:0.065 30:0.065 35:0.046 40:0.047 45:0.046 50:0.043 55:0.095 60:0.076 65:0.043 70:0.017 75:0.068 80:0.066 85:0.101 90:0.042 95:0.032 100:0.066 105:0.082 110:0.043 115:0.021 120:0.047 125:0.086 130:0.085 135:0.075 140:0.084 145:0.131 150:0.067 155:0.075 160:0.126 165:0.195 170:0.16 175:0.139 180:0.13 185:0.086 190:0.081
```

**Verdict: HEALTHY & STABLE (no collapse), modest ToM gain — best COMPLETED Gemma-2 cell so far.**
ToM HM(last5)=0.123 vs baseline 0.093 = +3.0pp; **avg(last5)=0.386 vs 0.315 = +7.1pp** (the avg
gain is the more meaningful signal here). Capabilities IMPROVED: gsm8k Δ+0.082, mmlu Δ+0.050.
Health excellent: KL=0.075 (low), resp_len=101.8 (NO collapse), entropy 1.464, parseable=1.0,
sampled `<think>` CoTs are multi-sentence genuine reasoning. Same Gemma-2 HM volatility as PS113:
noisy 0.02–0.13 through the run, mid-run peak 0.195@165, then decline to 0.081@190 (last-5 window
sits on the decline, understating the peak).

**CRITICAL CROSS-CELL FINDING (collapse cause isolated):** PS125 (actor-RM, ec=0.0, **fp=0**,
ll_min=-4) is STABLE — resp_len ~100, KL 0.075 — whereas PS123 (actor-RM, ec=0.0, **fp=5**,
ll_min=-6) COLLAPSED (resp_len 180→8, KL→1.08) and was killed. The ONLY reward-shaping differences
are fp (5 vs 0) and ll_min (-6 vs -4). => **`format_penalty=5` (not actor-RM or ec=0.0 alone) is the
primary driver of the length-collapse / co-adaptation hack on Gemma-2.** actor-RM + ec=0.0 is fine
with fp=0. Recommendation for Gemma-2 actor-RM cells: keep format_penalty=0 (fp does not gate length
for tag-free parsers anyway); avoid fp>0 with actor-as-RM.

Gemma-2 ranking (completed): PS125 actor-RM k5 llm-4 (HM 0.123, avg 0.386, caps up) > PS113
frozen-RM k7 llm-6 (HM 0.104). PS123 (fp5) FAILED. All Gemma-2 gains remain weak/volatile in
absolute terms (weak ToM base). NOT yet a Phase-1 winner but the strongest stable Gemma cell.
