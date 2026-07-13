### Attempt r1 — 2026-07-13T08:04:07+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-6.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-007-002   **git:** `bf3098b`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7 ll_min=-6.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-6.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-6.0-lr5e-7-kl0.05-n16-r1.log`

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
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.think_only_pg=True \
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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-6.0-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=999 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=google/gemma-2-2b-it \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=power \
    +reward_model.power_k=7 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=7 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(dry-run preview — NOT launched)_

### Attempt r1 — 2026-07-13T08:04:19+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-6.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-007-002   **git:** `bf3098b`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=7 ll_min=-6.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-6.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-6.0-lr5e-7-kl0.05-n16-r1.log`

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
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.think_only_pg=True \
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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k7-llmin-6.0-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=999 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=google/gemma-2-2b-it \
    reward_model.micro_batch_size=8 \
    +reward_model.subtract_baseline=False \
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=power \
    +reward_model.power_k=7 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=7 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.0_pk7_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings (PS131, r1, completed 190/191 steps):**

Canonical score (`scripts/score_run.py --last 5`, ToM excl gsm8k+mmlu):
- ToM **HM(last5)=0.019** / HM(last3)=0.019 (baseline step0=0.093) — degenerate/floored (Gemma fantom_*_list ~0).
- ToM **avg(last5)=0.247** / avg(last3)=0.247 (baseline step0=0.315) => **NEGATIVE transfer, -0.068**.
- gsm8k=0.368 (D +0.091 vs step0); mmlu=0.347 (D -0.040).
- Health(final): **kl=0.23 (excellent control), entropy=0.599**, resp_len=263.4, reward=12.8, parseable=1.0.
- HM trajectory: dropped by step30 and stayed floored ~0.02, no recovery (contrast PS121 which recovered to 0.221).

Verdict: **STABLE but NEGATIVE — the mirror image of PS121.** Both are Gemma-2 power actor-RM; the only diffs are
fp/k: **PS121 (fp0, k5)** gave +0.036 ToM but blew KL to 1.845; **PS131 (fp5, k7)** holds KL at 0.23 but loses ToM
(-0.068). Interpretation: the **fp=5 format penalty stabilizes drift but suppresses the exploration that produces the
ToM gain** — you can have the gain (PS121) or the stability (PS131) but this grid can't get both at once. Strong gsm8k
+0.091 either way. Not a candidate recipe (negative ToM). Best Gemma-2 so far remains PS121 (positive but needs KL
control / a follow-up with tighter KL or intermediate fp).

Rerun: see the "How to rerun" line above (smoke_gemma.sh, USE_ACTOR_AS_RM=True, FORMAT_PENALTY=5, POWER_K=7).

