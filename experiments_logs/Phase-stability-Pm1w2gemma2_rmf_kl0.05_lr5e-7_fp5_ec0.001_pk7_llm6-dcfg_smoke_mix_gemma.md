### Attempt r1 — 2026-07-13T03:04:15+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-013-002   **git:** `c6c4ef5`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7 ll_min=-6 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_ → https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/brxhralc

- **Hypothesis:** Wave-2 Gemma-2 power variant — higher curvature (k=7) + wider valid window
  (ll_min=-6) + format penalty (fp=5) + entropy (ec=0.001), lr5e-7. Building on PS110 (power k5
  ll_min-4 fp0 ec0.001) which was STABLE with +3.2pp ToM avg. Tests whether the format penalty +
  higher k push the gain further or destabilise (fp can suppress the length-hack but may over-constrain).
  Watch same PS097-degeneration signals (length→512 cap, eval collapse); expect PS110-like stability.
- **Log path:** `logs/20260713/Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.power_k=7 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=7 \
    +actor_rollout_ref.power_ll_min=-6 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr5e-7_fp5_ec0.001_pk7_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


---
**Monitor note (2026-07-13 03:38 UTC, ~step 38/191):** CONCERNING — early peak-then-DECLINE, but
a DIFFERENT mechanism than PS097. Length is BOUNDED ~150 (dipped to ~31-50 at steps 7-20, then
recovered to ~150-163 — NO explosion toward the 512 cap; fp=5 is suppressing length-hack as intended).
BUT evals are declining monotonically after ~step 10: tomi 0.597→0.613(pk)→0.263, bigtom_fwd_belief
0.753→0.787(pk)→0.380, bigtom_fwd_action 0.617→0.300, explore_tom 0.373→0.097, fantom_belief_mc
0.263→0.093. Reward is high+POSITIVE (~2.8 to 15.5, larger from k=7/ll_min=-6) WHILE ToM evals
crater → looks like REWARD-HACKING (the k=7 power reward is decoupling from answer correctness),
not the length-hack PS097 showed. Throughput healthy (step 38, ~1 step/min, not stalling), so NOT
terminating yet — PS110 also had an early flat/dip phase before recovering, so giving one more tick
to distinguish a transient dip from a true collapse. If evals keep collapsing next tick → verdict
DEGENERATE (k=7+fp=5 over-constrains / reward-hacks on Gemma).

**Monitor update (2026-07-13 04:08 UTC, ~step 64/191):** NOT a collapse — RECOVERING (V-shaped).
The decline reversed: tomi 0.263→0.417, bigtom_fwd_belief 0.380→0.527, explore_tom 0.083→0.147,
fantom_belief_mc 0.093→0.140. So the earlier peak-then-decline was a deep TRANSIENT dip (deeper
than PS110's), now climbing back. Length stabilized ~115-123 (GPU dropped to ~9GB). But recovery is
PARTIAL/uneven: tomi & bigtom heading back toward base, while explore_tom (base 0.363) and
fantom_belief_mc (base 0.223) remain well below base at ~0.15/0.14. More VOLATILE than PS110. Not
terminating (healthy throughput, still evolving) — the final last-5 window will decide vs PS110's
+3.2pp. Provisional read: k=7+fp=5 induces a violent early transient then partial recovery; likely
weaker/less stable than PS110's k=5 fp=0.

---
## Findings (FINAL, 2026-07-13) — authoritative attempt: -r1 (completed step 0..190, 39 evals)

**Verdict: NET NEGATIVE / mildly destabilizing. The k=7 + fp=5 + ll_min=-6 combination HURTS on
Gemma-2** — the opposite of the sibling PS110 (k=5, ll_min=-4, fp=0), which gained +3.2pp. Not
degenerate in the PS097 sense (bounded length, parseable=1.0, clean completion, no length-hack), but
it ends BELOW base and regresses mmlu.

Canonical scorer (`scripts/score_run.py --last 5`):
- **ToM avg(last5)=0.2802, avg(last3)=0.292 vs base 0.3145 → -3.4pp (last5) / -2.3pp (last3).** NET
  NEGATIVE (arith-mean is the fair Gemma metric).
- ToM HM(last5)=0.0389 vs base 0.0853 (down; HM unreliable on Gemma as usual).
- gsm8k 0.277→0.327 (+5.0pp) but **mmlu 0.383→0.353 (-3.0pp, REGRESSED)** — a capability cost.
- health(final): **kl=0.193 (HIGH — ~3x PS110's 0.062 → the policy drifted far from base and
  destabilized)**, entropy=1.927, resp_len=142 (bounded — fp=5 did prevent the length-hack),
  reward=7.451 (high positive, k=7/ll_min=-6 scaling), parseable=1.0.

**Trajectory:** violent V-shape. Evals peaked ~step 10 (tomi 0.613, bigtom_fwd_belief 0.787), then a
DEEP transient dip to ~step 30 (tomi 0.263, explore_tom 0.083, fantom_belief_mc 0.093), then a
PARTIAL recovery that plateaued BELOW base (tomi ~0.48, bigtom_fwd_belief ~0.56, explore_tom ~0.25,
fantom_belief_mc ~0.13). Length: dipped ~31-50 (steps 7-20) then stabilized ~110-160, never near the
512 cap.

**Comparison (isolating the knobs on the SAME model/lr/kl):**
- PS110 (k=5, ll_min=-4, fp=0, ec=0.001): **avg +3.2pp, kl=0.062 — STABLE WINNER.**
- PS116 (k=7, ll_min=-6, fp=5, ec=0.001): **avg -3.4pp, kl=0.193 — NET NEGATIVE.**
→ On Gemma-2, the gentler power curvature (k=5) with NO format penalty is markedly better; higher
k=7 + fp=5 + wider ll_min=-6 pushes KL up ~3x, induces a violent transient, and lands below base.
For Gemma Wave-2 selection, prefer low-k / no-fp power configs.

**Do NOT rerun** — reproducibly net-negative. Vary toward PS110's gentler settings instead.
