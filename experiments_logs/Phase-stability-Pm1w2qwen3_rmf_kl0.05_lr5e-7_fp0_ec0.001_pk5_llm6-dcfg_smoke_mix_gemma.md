### Attempt r1 — 2026-07-13T12:56:48+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-013-002   **git:** `855d28f`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/7sbtx603
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1.log`

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
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=999 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen3-1.7B \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_



## Hypothesis (PS146, Qwen3-1.7B power k5 ll_min-6 frozen-RM)
Wave-2 transfer of the Qwen2.5 power winner PS074 (kl0.05 lr5e-7) onto Qwen3-1.7B; frozen-RM, power k=5, ll_min=-6, fp=0, ec=0.001. The Qwen3 analogue of the Gemma PS110 winner (+3.2pp). Tag-free shared parquet, require_answer_tags=False, Gen ctx enforced 2048/512.

Key question: does POWER reward rescue Qwen3 where log_prob (PS139) was NET NEGATIVE (-4.0pp ToM, mmlu -14.5pp)? On Gemma, power (PS110/PS126) clearly beat log_prob (PS097 degenerate). Expect: if the power/log_prob gap holds cross-family, PS146 should be stable + modestly positive vs PS139's decline. Watch resp_len (Qwen3 thinking ~470 near cap), kl, and the ToM/mmlu/gsm8k trajectory.

- WandB: https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/7sbtx603 ; main_ppo clean, no crash, max_response=512, require_answer_tags=False, micro_batch=8.

## FINAL findings (PS146, Qwen3-1.7B power k5 ll_min-6 frozen-RM) — FLAT/NEUTRAL but STABLE (power >> log_prob)
Completed cleanly (step 189/191, GPUs→1MiB, no crash). Canonical scorer (`scripts/score_run.py --last 5`):
- **ToM avg(last5)=0.2712 vs base 0.2719 → −0.1pp (FLAT/neutral)**. HM(last5)=0.151 vs base 0.161 (−1.0pp).
- **gsm8k=0.489 vs base 0.487 → +0.2pp (PRESERVED)**; **mmlu=0.251 vs base 0.243 → +0.8pp (PRESERVED)**.
- health: resp_len=466 (Qwen3 thinking, stable from start, no hack), parseable=1.0, kl=0.003 (rock-stable the ENTIRE run), entropy=0.283.
- HM trajectory: **perfectly flat ~0.15–0.17 the whole run**, no decline (minor final dip 0.125@190).

**Verdict:** power reward on Qwen3-1.7B (k5/ll_min-6/fp0/ec0.001/kl0.05/lr5e-7, frozen-RM = PS074-winner knobs) is **STABLE and NEUTRAL** — it does NO harm and fully preserves capabilities (mmlu +0.8, gsm8k +0.2) but produces no ToM gain on Qwen3 (flat).

**KEY A/B (power vs log_prob on Qwen3, same model/kl0.05, both frozen-RM):**
- **PS139 log_prob (lr1e-6 fp5 ec0): NET NEGATIVE** — ToM −4.0pp, gsm8k −11.3pp, mmlu −14.5pp, HM monotonic decline.
- **PS146 power (lr5e-7 fp0 ec0.001): NEUTRAL/STABLE** — ToM −0.1pp, gsm8k +0.2pp, mmlu +0.8pp, KL 0.003 rock-stable.
=> **Power DECISIVELY beats log_prob on Qwen3** (preserves vs erodes). This mirrors Gemma (power PS110/PS126 positive vs log_prob PS097 degenerate). Cross-family takeaway: **power reward is the robust choice; log_prob damages non-Qwen2.5 families.**

Caveat: power gave +3–5pp ToM on Gemma (PS110/PS126) but only FLAT on Qwen3. Possible causes: 512 response cap truncating the Qwen3 thinking trace, Qwen3 already stronger (harder to lift), or frozen-RM signal too weak for a thinking model — worth testing actor-RM (see PS-series) and/or a larger resp budget in a follow-up.

Rerun: `EXP_ID=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm6 REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-6 USE_ACTOR_AS_RM=False KL=0.05 LR=5e-7 FORMAT_PENALTY=0 ENTROPY_COEFF=0.001 MAX_PROMPT=2048 MAX_RESP=512 bash experiments/smoke_qwen3.sh`
