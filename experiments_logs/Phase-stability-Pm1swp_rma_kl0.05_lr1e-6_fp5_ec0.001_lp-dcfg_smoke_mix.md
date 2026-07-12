### Attempt r1 — 2026-07-11T19:59:03+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp5_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1`
- **Host:** h100-077-004   **git:** `059f8dd`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=actor baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp5_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260711/Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp5_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.think_only_pg=True \
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
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp5_ec0.001_lp-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=999 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-3B-Instruct \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=log_prob \
    +reward_model.power_k=2.0 \
    +reward_model.power_ll_min=-8.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=log_prob \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.05_lr1e-6_fp5_ec0.001_lp DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


---

### Findings (attempt r1) — completed 2026-07-11T21:34 (h100-077-004)

**Hypothesis:** Does actor-as-RM behavior-LL (log_prob) at kl=0.05, lr=1e-6, fp=5, ec=0.001 climb
stably (HM over last-X ≥ baseline) without collapse? (Actor-RM analogue of the stable frozen-RM
cells PS009/PS010.)

**Verdict: STABLE — no collapse, marginally above baseline.** Completed full 191-step epoch (evals
every 10). `format_error_ratio=0` at every step; KL contained ~0.20–0.31 (kl_coef=0.05); entropy
stable ~2.0–2.4; response_length steady ~105–116 (no length blow-up / degenerate repetition).

**Config-selection metric (HM over ToM subtypes, excl. mmlu/gsm8k):**
- Baseline (step 0) HM_tom = **0.417**
- HM(last-3 evals) = **0.433**, HM(last-5 evals) = **0.423** (+0.6pp vs baseline)
- Peak HM_tom = **0.459 @ step 20**

**Per-benchmark (step0 → step190):**
| Benchmark | step0 | step190 | note |
|---|---|---|---|
| simpletom_mental | 0.853 | 0.910 | held/up |
| bigtom_forward_action | — | 0.770 | strong |
| tomi | 0.587 | 0.570 | dipped 0.47@140, recovered |
| bigtom_forward_belief | 0.767 | 0.640 | declines |
| tombench | — | 0.680 | stable |
| explore_tom | — | 0.620 | stable |
| gsm8k (gen-cap) | 0.660 | 0.447 | mild drift down |
| mmlu (gen-cap) | 0.473 | 0.597 | up |
| hi_tom | — | 0.313 | low (hard) |

**Health/hacking:** No reward-hacking or collapse signature (contrast PS002/PS005/PS007 which
crashed all evals). Actor-as-RM does not destabilize at kl=0.05/lr=1e-6 — behaves like the stable
frozen cells. Mild general-capability drift (gsm8k) is the main cost.

**Downstream implication:** actor-RM @ kl=0.05/lr=1e-6/fp=5/ec=0.001/log_prob is a *stable but weak*
Phase-1 candidate — on par with frozen PS009 (HM-last5=0.425) / PS010 (0.433), not a clear winner.
Confirms actor-as-RM is a viable stable regime at kl=0.05, but log_prob reward only holds ~baseline
(does not climb) — consistent with the power-reward family being the more promising direction.

**How to rerun:** `ONLY_IDX="32" bash experiments/phase_stability_sweep.sh` (from ~/repo/BeRL, conda env `tom`).
