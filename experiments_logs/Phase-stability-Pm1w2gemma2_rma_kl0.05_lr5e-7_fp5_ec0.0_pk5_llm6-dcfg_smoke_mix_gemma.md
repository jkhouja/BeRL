### Attempt r1 — 2026-07-13T04:42:39+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-156-003   **git:** `ac8969f`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/1024 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


---

## Findings (r1 — PS123, run by h100-156-003, 2026-07-13) — FAILED (health collapse, killed at step 36)

WandB: https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/hilqkfxd
Log: logs/20260713/Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1.log

**Outcome: KILLED at step 36/191 due to severe response-length collapse + KL explosion (actor-as-RM co-adaptation).**

Response-length trajectory (per step, collapsed monotonically):
```
step 1:180  2:143  3:147  4:148  5:122  6:126  7:72  8:38  9:28  10:24  ... 20:19  ... 30:15  32:10.5  33:9.2  34:10  35:9.2  36:8.4
```
The CoT collapsed from ~180 tokens to ~8-10 tokens in 36 steps while reward/mean stayed positive
(~5-13) and **KL exploded to 1.08** (healthy runs ~0.05-0.2). Classic **actor-as-RM co-adaptation
collapse**: the reward model IS the (updating) actor, so its own log-likelihood estimates drift
upward as the policy degenerates to trivial near-empty CoTs — a self-referential reward-hacking loop.
`entropy_coeff=0.0` removed the entropy floor that would resist the collapse; `format_penalty=5`
did not prevent it (tag-free parser does not require `</think>`, so ultra-short responses are not
format-penalised).

Partial canonical scores (evals at steps 0..35 only, run did not finish):
```
eval iters: 8 (step 0..35); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.1808  HM(last3)=0.1933  (baseline step0=0.0929)
ToM avg(last5)=0.4062  avg(last3)=0.4112  (baseline step0=0.3149)
gsm8k: 0.336 (step0=0.28, Δ+0.056);  mmlu: 0.4332 (step0=0.38, Δ+0.053)
health(final): kl=1.08 entropy=1.791 resp_len=10.27 reward=8.635 parseable=1.0
ToM HM trajectory: 0:0.093 5:0.046 10:0.075 15:0.156 20:0.163 25:0.166 30:0.204 35:0.205
```
**IMPORTANT — the transient eval-HM rise (0.093→0.205) is NOT trustworthy:** it coincides with the
response collapse and KL=1.08 drift, i.e. the policy is degenerating, not learning robust ToM. A
model emitting 10-token CoTs is not doing genuine Theory-of-Mind reasoning; the metric gain is a
co-adaptation artifact that would not be robust/OOD-stable (the whole point of BeRL). Compare PS113
(frozen-RM, ec=0.0, fp=0): resp_len stayed ~100-155, KL 0.21, no collapse.

**Verdict: FAILED — degenerate config.** actor-RM + `entropy_coeff=0.0` + `format_penalty=5` on
Gemma-2-2B drives response-length collapse and KL explosion. Killed early to free the node.
**Recommendation:** actor-RM on Gemma-2 needs an entropy floor (ec>0) and/or tighter KL, and
format_penalty does not gate length for tag-free parsers. Do NOT adopt. If re-run is desired, add
entropy_coeff>=0.001 and consider a length-aware penalty.
