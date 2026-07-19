### Attempt r1 — 2026-07-12T23:52:28+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-013-002   **git:** `d5f3d58`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 → https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/y7pnhxva
- **Log path:** `logs/20260712/Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm4-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.use_actor_as_rm=False \
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

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk5_llm4 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Hypothesis:** Wave-2 cross-family transfer of the Qwen2.5 **power** winner (PS074, kl0.05 lr5e-7,
HM-last5 0.462) to Gemma-2-2B. Power reward has valid_floor=0 (no -80 sentinel) and was the
strongest Qwen2.5 family; ec=0.001 entropy bonus intended to counter the length collapse→explosion
that made the log_prob winner (PS097) **degenerate** on Gemma-2 (all ToM evals peaked then collapsed
to ~0). Key control vs PS097: MAX_RESP=512 (matches Gen ctx 2048/512) to bound the length-hack.
Watch: response_length trajectory + whether evals hold or peak-then-collapse like PS097.

**Findings:** _(fill on completion via log-results skill)_


---
**Monitor note (2026-07-13 00:27 UTC, ~step 41/191):** STABLE, opposite failure mode from PS097.
Response length bounded ~120-160 across all 31 rollouts (NO collapse→explosion, NO length-hack,
NO OOM; GPU ~18GB). Throughput healthy (step 41 in ~37min; will finish, not the ~1step/30min stall
PS097 hit). Rewards varying (~ -6 to +0.6). BUT ToM evals are essentially FLAT at base level with a
tiny early bump then slight decay: tomi 0.597→0.563 (pk 0.597), bigtom_fwd_belief 0.753→0.713
(pk 0.787), bigtom_bwd_belief 0.527→0.510, bigtom_fwd_action 0.600→0.567 (pk 0.640),
fantom_belief_mc 0.223→0.220 (pk 0.250), fantom_ans_binary 0.253→0.227, fantom_info_binary
0.423→0.387; gsm8k 0.277→0.310, mmlu 0.383→0.383. Preliminary read: power+entropy at lr5e-7 is
STABLE on Gemma-2 (fixes PS097's degeneration) but UNDER-DRIVEN — no meaningful ToM gain over base.
Letting it run to completion for the full HM-last5; not terminating (healthy throughput).

---
## Findings (FINAL, 2026-07-13) — authoritative attempt: -r1 (completed step 0..190, 39 evals)

**Verdict: STABLE + MODEST POSITIVE transfer.** The Qwen2.5 power winner recipe (PS074) DOES
transfer to Gemma-2-2B in a stable way — the opposite of the log_prob sibling PS097, which was
degenerate (length-hacked to 512, evals collapsed to ~0). Power reward (k=5, ll_min=-4, valid_floor=0)
+ entropy(0.001) + the conservative lr5e-7 gives bounded-length, non-collapsing training with a small
real ToM gain.

Canonical scorer (`scripts/score_run.py --last 5`):
- **ToM avg(last5)=0.3467, avg(last3)=0.3597 vs base 0.3147 → +3.2pp (last5) / +4.5pp (last3).**
  Arithmetic mean is the fair Gemma aggregator (per PS104: 3 near-zero fantom_*_list benches make HM
  unreliable on Gemma).
- ToM HM(last5)=0.0573, HM(last3)=0.0674 vs base 0.0929 (HM *down* — an artifact of the near-zero
  list benches, NOT a real regression; the arith-mean and per-bench trajectories are up).
- gsm8k 0.277→0.345 (**+6.8pp**), mmlu 0.383→0.414 (**+3.1pp**) — NO capability regression; both
  improved.
- health(final): kl=0.062, entropy=1.604, **resp_len=112 (bounded — no length-hack)**, reward=0.006,
  **parseable=1.0**.

Per-benchmark (base → best/last): bigtom_forward_belief 0.753 → **0.863** (peak) / 0.820,
bigtom_forward_action 0.600 → 0.723 / 0.680, bigtom_backward_belief 0.527 → 0.600 / 0.543,
tomi 0.597 → 0.633 / 0.613, explore_tom 0.363 → 0.447 / 0.417, fantom_belief_mc 0.223 → 0.283,
fantom_answerability_binary 0.253 → 0.293, fantom_info_binary 0.423 → 0.450. The fantom_*_list
benches stay ~0 (base can't do them; unchanged). Warmup: evals were ~flat for steps 0-13 then
climbed from ~step 14; bigtom/explore_tom carry most of the gain.

**Length trajectory:** stable ~110-175 the entire run — NEVER approached the 512 cap. No
collapse-then-explosion. This is exactly the stability PS097 lacked. Throughput healthy (~1 step/min,
finished in ~3h).

**Comparisons:**
- vs PS097 (log_prob, same model/kl, lr1e-6): PS097 DEGENERATE (evals → ~0, length-hack, throughput
  collapse). PS110 STABLE +3.2pp. → reward family + lower lr + entropy matter for Gemma stability.
- vs PS104 (log_prob Gemma, arith 0.316→0.345, +2.9pp): PS110 (power) +3.2pp — comparable, and
  cleaner (also lifts gsm8k/mmlu).
- vs PS074 (Qwen2.5 power winner, HM 0.462): not HM-comparable across families (Gemma HM unreliable),
  but the transfer is positive and stable.

**Implication:** For Gemma-2 Wave-2 selection, use arith-mean (not HM). Power+entropy at lr5e-7 is a
viable stable Gemma regime; the remaining PS111+ variants (fp, higher k, ll_min, actor-RM) test
whether any push the +3.2pp further.

**Rerun:** identical command (knobs in the launcher stub above); MAX_RESP=512, MICRO_BATCH=8.
