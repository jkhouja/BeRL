### Attempt r1 — 2026-07-19T21:14:35+00:00

- **RUN_NAME:** `s2-ST14-Phase-stability-v2_g2_reward_logprob-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-117-001   **git:** `e841f0e`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=1.0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-ST14-Phase-stability-v2_g2_reward_logprob-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260719/s2-ST14-Phase-stability-v2_g2_reward_logprob-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1.log`

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
    actor_rollout_ref.actor.format_penalty=0.0 \
    actor_rollout_ref.actor.format_penalty_std_coef=1.0 \
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
    trainer.experiment_name=s2-ST14-Phase-stability-v2_g2_reward_logprob-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=google/gemma-2-2b-it \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=log_prob \
    +reward_model.power_k=2.0 \
    +reward_model.power_ll_min=-8.0 \
    reward_model.format_penalty=0.0 \
    reward_model.format_penalty_std_coef=1.0 \
    +actor_rollout_ref.reward_type=log_prob \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-v2_g2_reward_logprob DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-19T21:14:49+00:00

- **RUN_NAME:** `s2-ST14-Phase-stability-v2_g2_reward_logprob-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-117-001   **git:** `e841f0e`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=frozen baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=1.0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-ST14-Phase-stability-v2_g2_reward_logprob-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260719/s2-ST14-Phase-stability-v2_g2_reward_logprob-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1.log`

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
    actor_rollout_ref.actor.format_penalty=0.0 \
    actor_rollout_ref.actor.format_penalty_std_coef=1.0 \
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
    trainer.experiment_name=s2-ST14-Phase-stability-v2_g2_reward_logprob-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=google/gemma-2-2b-it \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=False \
    +reward_model.reward_type=log_prob \
    +reward_model.power_k=2.0 \
    +reward_model.power_ll_min=-8.0 \
    reward_model.format_penalty=0.0 \
    reward_model.format_penalty_std_coef=1.0 \
    +actor_rollout_ref.reward_type=log_prob \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-v2_g2_reward_logprob DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


- **WandB link (resolved):** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/txbm9pza

---

## Findings — r1 (2026-07-20, Completed)

**Canonical score (`scripts/score_run.py`, 8 eval iters, steps 0..190; 24 ToM benchmarks excl gsm8k/mmlu):**
```
ToM HM(last5)=0.2941  HM(last3)=0.3160  (baseline step0=0.0929)
ToM avg(last5)=0.4653 avg(last3)=0.4851 (baseline step0=0.3148)
gsm8k (separate): 0.4286  (step0=0.277, Δ=+0.152)
mmlu  (separate): 0.4694  (step0=0.383, Δ=+0.086)
health(final): kl=0.321 entropy=1.676 resp_len=269.9 reward=-5.496 parseable=1.0 max_resp=512
ToM HM trajectory: 0:0.093 30:0.213 60:0.209 90:0.231 120:0.286 150:0.305 180:0.336 190:0.302
```

**Verdict:** Gemma-2-2B + frozen-RM + **log_prob** reward gives a large ToM lift over a very weak
base (step0 HM 0.093 — gemma base has poor format compliance on many benchmarks), climbing to
HM≈0.336 by step 180. BUT the run is **KL-volatile**: raw kl_loss transiently spiked to ~3.3
(~step 90), then settled elevated (~0.3–0.8), with entropy drifting up to ~2.2 at the spike and
grad_norm elevated. Response length oscillated (459→111→252→270) with a mid-run clip_ratio spike
(0.42). Format never hard-collapsed (parseable=1.0 at final), but end-of-run format_pass wobbled on
several benchmarks (fantom_answerability_list 0.17, fantom_info_list 0.55, hi_tom 0.57, dyntom_type_c
0.68), pulling HM down from 0.336@180 to 0.302@190.

**Implication:** log_prob reward on the gemma frozen-RM family is *effective but not clean* under
KL=0.05 / LR=5e-7 — the transient KL blowup + terminal format wobble mark it as less stable than the
qwen2.5 behavior runs. Compare against the gemma power anchor (ST13) and neg_perplexity (ST15) to
decide which reward family is the stable choice for the gemma family. Consider a KL/entropy-coeff
adjustment if log_prob is pursued for gemma.

**How to rerun:**
`RUN_STAGE=s2 EXP_NUM=ST14 EXP_ID=Phase-stability-v2_g2_reward_logprob DATA_NAME=dcfg_smoke_mix_gemma DATA_TRAIN=$HOME/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet REWARD_TYPE=log_prob USE_ACTOR_AS_RM=False SUBTRACT_BASELINE=False KL=0.05 LR=5e-7 TRAIN_BATCH=32 MINI_BATCH=128 MICRO_BATCH=8 ROLLOUT_N=16 MAX_PROMPT=2048 MAX_RESP=512 ENTROPY_COEFF=0.001 FORMAT_PENALTY=0.0 FORMAT_PENALTY_STD_COEF=1.0 NUM_GPUS=8 TP_SIZE=2 bash experiments/train_behavior_gemma.sh`
