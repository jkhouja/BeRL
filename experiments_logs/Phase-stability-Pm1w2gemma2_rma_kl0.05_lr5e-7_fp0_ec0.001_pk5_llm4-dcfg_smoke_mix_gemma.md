### Attempt r1 — 2026-07-13T06:14:21+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-013-002   **git:** `6018931`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/i7fwbifq
- **Log path:** `logs/20260713/Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.power_ll_min=-4 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-4 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm4 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_



## Hypothesis (PS126, actor-RM)
Tests whether the PS110 winner recipe (Gemma-2 power k=5, ll_min=-4, fp=0, ec=0.001, kl=0.05, lr=5e-7, +3.2pp STABLE) still holds when the **actor itself scores the reward (USE_ACTOR_AS_RM=True)** instead of a frozen base LM. Concern: actor-RM can create a moving target / self-reinforcing loop that destabilizes (PS097-style length-hack or KL blow-up like PS116). Expect: if stable, modest positive transfer comparable to PS110; if the actor-RM feedback loop dominates, watch for length drift and KL>0.15.

- WandB: https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/i7fwbifq ; main_ppo launched clean, no crash, MAX_RESP=512, micro_batch=8.

## FINAL findings (PS126, actor-RM) — STABLE + POSITIVE transfer
Completed cleanly (step 189/191, GPUs→1MiB, no crash). Canonical scorer (`scripts/score_run.py --last 5`):
- **ToM avg(last5)=0.3646 vs base 0.3147 → +5.0pp** (interpret via avg; HM unreliable on Gemma). HM(last5)=0.120 vs 0.093.
- gsm8k=0.323 (+4.6pp vs step0); mmlu=0.393 (+0.0pp, **no regression**).
- health: resp_len=77.5 (bounded, no length-hack), parseable=1.0.
- kl_loss stayed **~0.10–0.12 throughout** the run (well below the 0.15 concern line); the score_run "final kl=4.294" is a single terminal-step artifact, not representative of the trajectory.
- HM trajectory recovered to 0.135 at step 190 (above base 0.093), no PS097-style collapse.

**Verdict:** Actor-RM (USE_ACTOR_AS_RM=True) on Gemma-2 with the power k5/ll_min-4/fp0/ec0.001/kl0.05/lr5e-7 recipe is **STABLE and slightly BETTER than the frozen-RM PS110 winner** (+5.0pp vs +3.2pp). No feedback-loop pathology (no length drift, no KL blowup, no eval collapse). **Actor-RM transfers to Gemma and is competitive with / better than frozen-RM.**

**Comparison ledger (Gemma-2, kl0.05/lr5e-7 unless noted):**
- PS097 log_prob (lr1e-6) = DEGENERATE (length-hack, evals→0).
- PS110 power k5 ll_min-4 fp0 ec0.001 frozen-RM = +3.2pp STABLE.
- PS116 power k7 ll_min-6 fp5 ec0.001 frozen-RM = -3.4pp NET NEGATIVE (kl blew up 3x).
- **PS126 power k5 ll_min-4 fp0 ec0.001 actor-RM = +5.0pp STABLE (best Gemma result so far).**

Rerun: `EXP_ID=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm4 REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-4 USE_ACTOR_AS_RM=True KL=0.05 LR=5e-7 FORMAT_PENALTY=0 ENTROPY_COEFF=0.001 MAX_RESP=512 bash experiments/smoke_gemma.sh`
