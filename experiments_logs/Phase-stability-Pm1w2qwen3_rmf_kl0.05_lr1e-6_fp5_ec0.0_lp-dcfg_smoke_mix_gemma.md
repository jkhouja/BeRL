### Attempt r1 — 2026-07-13T09:51:27+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr1e-6_fp5_ec0.0_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1`
- **Host:** h100-013-002   **git:** `8a8f0e2`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=frozen baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr1e-6_fp5_ec0.0_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1 https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/swseokts
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr1e-6_fp5_ec0.0_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr1e-6_fp5_ec0.0_lp-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1 \
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
    +reward_model.reward_type=log_prob \
    +reward_model.power_k=2.0 \
    +reward_model.power_ll_min=-8.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=log_prob \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr1e-6_fp5_ec0.0_lp DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_



## Hypothesis (PS139, Qwen3-1.7B log_prob frozen-RM)
Wave-2 transfer of the Qwen2.5 log_prob winner (PS013: kl0.05 lr1e-6 fp5 ec0.0) onto **Qwen3-1.7B** (a hybrid thinking model). Frozen-RM, log_prob reward, fp=5 (format penalty), ec=0.0. Data is the shared **tag-free** dcfg_smoke_mix_gemma parquet; MODEL_FAMILY=qwen3 applies the Qwen3 chat template with require_answer_tags=False. Gen ctx enforced at 2048/512 (MAX_RESP=512 overrides the smoke_qwen3 default of 2048, per launch-skill mandate 3a — qwen3 rows must NOT silently run at 2048/4096).

Key question: does log_prob (which DEGENERATED on Gemma as PS097, but was the WINNER on Qwen2.5) transfer to Qwen3? Watch for: (1) thinking-model behavior with a 512 response cap — possible think-trace truncation; (2) length-hack / eval collapse (PS097-style); (3) KL blowup. Compare to Qwen2.5 PS013 winner and Gemma outcomes.

- WandB: https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/swseokts ; main_ppo launched clean, no crash, max_response_length=512, max_prompt=2048, require_answer_tags=False, micro_batch=8.

## FINAL findings (PS139, Qwen3-1.7B log_prob frozen-RM) — NET NEGATIVE transfer
Completed cleanly (step 189/191, GPUs→1MiB, no crash). Canonical scorer (`scripts/score_run.py --last 5`):
- **ToM avg(last5)=0.2323 vs base 0.2723 → −4.0pp** (NEGATIVE). HM(last5)=0.070 vs base 0.161 (−9.1pp).
- **gsm8k=0.374 vs base 0.487 → −11.3pp (heavy regression)**; **mmlu=0.098 vs base 0.243 → −14.5pp (heavy regression)**.
- health: resp_len=470 (pinned near the 512 cap — Qwen3 thinking trace, stable from step 1, NOT a hack), parseable=1.0, kl=0.012 (stable, no blowup), entropy=0.276.
- HM trajectory: peaks ~0.168 @step5 then **monotonically declines** to 0.075 @step190 — slow steady erosion, not a collapse-to-zero.

**Verdict:** log_prob reward on Qwen3-1.7B with the PS013-winner knobs (fp5/ec0/kl0.05/lr1e-6, frozen-RM) is **NET NEGATIVE** — mechanically stable (no length-hack explosion, KL controlled, 100% parseable) yet steadily degrades ToM AND badly regresses general capabilities (mmlu −14.5pp, gsm8k −11.3pp). The thinking model spends ~470 tokens/response without benefit. **log_prob does NOT transfer to Qwen3** — echoes the Gemma PS097 degeneration (less catastrophic: decline not full collapse), reinforcing that log_prob is a poor cross-family reward vs power.

**Cross-family log_prob ledger:** Qwen2.5 = winner in some cells (PS013) but reward-hacks in others (PS001/PS002 collapsed); Gemma PS097 = DEGENERATE (length-hack, evals→0); **Qwen3 PS139 = NET NEGATIVE (−4.0pp ToM, heavy mmlu/gsm8k regression)**. Power reward (Gemma PS110/PS126) remains the more robust cross-family choice.

Rerun: `EXP_ID=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr1e-6_fp5_ec0.0_lp REWARD_TYPE=log_prob USE_ACTOR_AS_RM=False KL=0.05 LR=1e-6 FORMAT_PENALTY=5 ENTROPY_COEFF=0.0 MAX_PROMPT=2048 MAX_RESP=512 bash experiments/smoke_qwen3.sh`
