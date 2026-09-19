### Attempt r1 — 2026-08-09T02:47:32+00:00

- **RUN_NAME:** `s2-QG-B2g-02-QG-B2g-tomrl_gemma_s2-ToM_HiEx_hint_v3_notags-gemma-2-2b-it-rulebased-lr5e-7-kl0.001-n16-r1`
- **Host:** h100-003-002   **git:** `892a3a3`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3_notags.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=n/a power_k=n/a ll_min=n/a rm_mode=frozen baseline=False kl=0.001 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/2048 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=0.0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/mt7cwen5
- **Log path:** `logs/20260809/s2-QG-B2g-02-QG-B2g-tomrl_gemma_s2-ToM_HiEx_hint_v3_notags-gemma-2-2b-it-rulebased-lr5e-7-kl0.001-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3_notags.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=8 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=google/gemma-2-2b-it \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.think_only_pg=False \
    actor_rollout_ref.actor.format_penalty=0.0 \
    actor_rollout_ref.actor.format_penalty_std_coef=0.0 \
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
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-QG-B2g-02-QG-B2g-tomrl_gemma_s2-ToM_HiEx_hint_v3_notags-gemma-2-2b-it-rulebased-lr5e-7-kl0.001-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=QG-B2g-tomrl_gemma_s2 DATA_NAME=ToM_HiEx_hint_v3_notags MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3_notags.parquet RUN_INDEX=1 bash experiments/train_tom_gemma.sh`

**Debugging / issues:**

- **2026-08-09T03:32Z:** Stopped r1 after step 136 due to emerging generation collapse.
  Response length was normally 6-10 tokens, then became unstable from step 120 onward:
  step 133 mean=551.8 with 13.3% clipped at 2048; step 135 mean=246.3 with 6.2%
  clipped; step 136 mean=208.2 with 6.2% clipped. Long sampled outputs repeated location
  tokens such as `green_treasure_chest` hundreds of times or emitted a literal `<eos>`
  followed by long `<pad>` runs. Format parseability remained 100%, so the rule scorer did
  not penalize this degeneration.
- Evaluation had not yet collapsed: step-120 ToM HM was 0.505 versus the 0.464 seed-2
  baseline, with gsm8k and mmlu also above baseline. Nevertheless, continued training was
  invalid because reward and format metrics masked increasingly degenerate generations.

**Findings:** r1 is incomplete and held `In-debug`; do not use it as a completed seed.
