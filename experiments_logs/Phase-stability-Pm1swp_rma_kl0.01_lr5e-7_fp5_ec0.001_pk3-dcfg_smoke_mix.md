### Attempt r1 — 2026-07-12T01:04:31+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1`
- **Host:** h100-076-003   **git:** `bc2bb8b`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=3 ll_min=-6.0 rm_mode=actor baseline=False kl=0.01 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1.log`

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
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
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
    algorithm.kl_ctrl.kl_coef=0.01 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.001_pk3-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k3-llmin-6.0-lr5e-7-kl0.01-n16-r1 \
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
    +reward_model.reward_type=power \
    +reward_model.power_k=3 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=3 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.01_lr5e-7_fp5_ec0.001_pk3 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** WandB `igiqjo53`. Ran to completion (step 190, final validation logged); clean exit, GPUs freed to 1 MiB (no teardown hang). **Verdict: HEALTHY / STABLE — SURPRISE: actor-RM keeps the steep power reward stable even at kl=0.01, where the frozen-RM twin (PS036) collapsed.**

Config: **actor RM**, kl=0.01, lr=5e-7, fp=5, ec=0.001, reward=power (k=3, ll_min=−6).

Training dynamics (healthy despite low KL):
- `critic/rewards/mean`: −36.6 → **+32.1** (power reward climbs strongly positive — same steep gradient as the frozen twin PS036).
- `response_length/mean`: **47.0 → 104.1** — HEALTHY band (~104). Dipped to ~56 mid-run (step 66) then recovered to 104. **No length-inflation, no brevity collapse.** This is the key surprise: the frozen-RM twin PS036 (power, kl0.01, same fp5) length-inflated to 146–155, and actor-RM log_prob twins collapsed too (PS018 inflation→90, PS022 brevity→57). Actor-RM + power at kl0.01 is the ONLY kl0.01 cell so far that stayed length-healthy.
- `format_error_ratio`: 0.000.

Eval (subsample300), step0 → final:
| Benchmark | step0 | final | Δpp |
|---|---|---|---|
| **aggregate mean (~23–26)** | 0.509 | ~0.506 | **~−0.2 (flat)** |
| explore_tom | 0.480 | 0.610 | +13.0 |
| tomi | 0.597 | 0.677 | +8.0 |
| simpletom_behavior | 0.557 | 0.590 | +3.3 |
| opentom_attitude | 0.440 | 0.420 | −2.0 |
| fantom_answerability_binary | 0.207 | 0.183 | −2.3 |
| bigtom_forward_action | 0.737 | 0.647 | −9.0 |
| bigtom_forward_belief | 0.770 | 0.663 | −10.7 |

Interpretation & hypothesis test: The monitor expected a collapse (kl=0.01 has been unstable everywhere). Instead PS052 stayed HEALTHY. Mechanism: the **actor-RM self-scores the policy's own (updating) outputs**, so the power reward's length-inflation pressure is self-limiting — the scorer moves with the policy rather than statically rewarding ever-longer sequences (as a frozen RM does in PS036). This is a genuinely new observation: **actor-as-RM provides a second, reward-side stabilizer that can substitute for high KL** on the length axis. Aggregate eval remains FLAT (~0.506) on the smoke_mix probe — no net accuracy gain (expected for this tiny stability dataset), with notable idiosyncratic gains (explore_tom +13.0, tomi +8.0). Caveat: this is stability-only on smoke_mix; the actor-RM reward-drift risk (actor rewarding its own drift, as seen in PS018) did not bite here at fp5, but should be watched on real data. **Refines the Phase −1 conclusion: kl=0.05 is the primary/robust stabilizer, but actor-RM can stabilize length even at kl=0.01 with the power reward + fp5.** PS011 (log_prob kl0.05+fp5) and PS044 (power kl0.05+fp5) remain the safest co-leading `Pm1swp_best` candidates; PS052 is an interesting low-KL actor-RM alternative to revisit on real data.

