### Attempt r1 — 2026-07-13T16:19:12+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-076-003   **git:** `b36bdee`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7 ll_min=-6 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/2048 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1.log`

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
    data.max_response_length=2048 \
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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm6-dcfg_smoke_mix_gemma-Qwen3-1.7B-frozenRM-nobaseline-power-k7-llmin-6-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.power_k=7 \
    +reward_model.power_ll_min=-6 \
    reward_model.format_penalty=0 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=7 \
    +actor_rollout_ref.power_ll_min=-6 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (r1 — authoritative, completed 2026-07-13 21:11)

**Verdict: SAFE but ~NEUTRAL — power reward does not collapse on Qwen3 (unlike log_prob), but with
frozen-RM + k=7 it does not improve downstream either.** Confirms power is the *safe* transferable
reward family; the improvement seen on Gemma-2 (PS122) did not reproduce here.

**Training health:** power reward positive throughout (~0–4.6, mostly ~1–3, never pinned at the −40
invalid sentinel → tag-free fix OK), format_error_ratio 0.000, advantages non-zero, response_length
stable ~640–800 (no brevity collapse, no inflation). Full 1 epoch (step 189 + final val step:190),
clean exit, GPU → 1 MiB.

**Downstream eval (26 × *_sub300, step:0 → step:190):**
- **MEAN 0.455 → 0.443 = −1.2pp** (essentially flat).
- **11 worsened (>2pp), 6 improved (>2pp), 0 collapsed-to-zero.**
- Mild losses (opentom_multihop_fo −9.0, opentom_multihop_so −6.4, simpletom_mental −5.4) offset by
  mild gains (simpletom_judgment +7.0, fantom_answerability_binary +4.6, gsm8k +3.7,
  bigtom_forward_belief +3.7). Net wash; capability preserved.

**Cross-family × reward × RM summary (all tag-free gemma data, kl0.05, fp=0):**

| Run   | Model      | Reward   | RM     | k | LR    | Base→Final  | Mean Δ      | Verdict |
|-------|------------|----------|--------|---|-------|-------------|-------------|---------|
| PS101 | Gemma-2-2B | log_prob | actor  | – | 1e-6  | 0.316→0.081 | **−23.5pp** | FAILED (collapse) |
| PS141 | Qwen3-1.7B | log_prob | actor  | – | 1e-6  | 0.455→0.345 | **−11.0pp** | FAILED (collapse) |
| PS122 | Gemma-2-2B | power    | actor  | 5 | 5e-7  | 0.316→0.413 | **+9.7pp**  | PASSED (gains) |
| PS154 | Qwen3-1.7B | power    | frozen | 7 | 5e-7  | 0.455→0.443 | **−1.2pp**  | SAFE (flat) |

**Interpretation:** the dominant, consistent signal across families is **reward family**: log_prob
collapses downstream on both non-Qwen2.5 families; power never collapses. Power's *magnitude* of
benefit differs (Gemma +9.7pp vs Qwen3 −1.2pp), but PS154 confounds three variables vs PS122
(family Qwen3-vs-Gemma, RM frozen-vs-actor, k 7-vs-5) plus a much stronger Qwen3 base (0.455 with
gsm8k/bigtom already ~0.85 → ceiling effects), so the flat result is not attributable to any single
knob. Candidate drivers of the missing gain: (a) frozen-RM gives a weaker/less-aligned LL signal
than actor-as-RM; (b) higher k=7 sharpens the reward and may dampen useful gradient; (c) Qwen3's
higher starting accuracy leaves less headroom. **Headline for the paper: power reward is the safe,
transferable family (0/4 collapse) whereas bare log_prob is not (2/2 collapse); actor-RM on Gemma
additionally yields real downstream gains.** A clean follow-up would hold family fixed and vary only
RM (frozen vs actor) and k to isolate why Qwen3 power is flat rather than positive.

**Rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rmf_kl0.05_lr5e-7_fp0_ec0.001_pk7_llm6 DATA_NAME=dcfg_smoke_mix_gemma REWARD_TYPE=power POWER_K=7 POWER_LL_MIN=-6 USE_ACTOR_AS_RM=False KL=0.05 LR=5e-7 FORMAT_PENALTY=0 ENTROPY_COEFF=0.001 bash experiments/smoke_qwen3.sh`
