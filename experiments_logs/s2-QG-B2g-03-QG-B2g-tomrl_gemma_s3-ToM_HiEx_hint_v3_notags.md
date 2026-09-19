### Attempt r1 — 2026-08-09T03:34:49+00:00

- **RUN_NAME:** `s2-QG-B2g-03-QG-B2g-tomrl_gemma_s3-ToM_HiEx_hint_v3_notags-gemma-2-2b-it-rulebased-lr5e-7-kl0.001-n16-r1`
- **Host:** h100-003-002   **git:** `892a3a3`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3_notags.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=n/a power_k=n/a ll_min=n/a rm_mode=frozen baseline=False kl=0.001 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/2048 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=0.0
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/ex5ll3rw
- **Log path:** `logs/20260809/s2-QG-B2g-03-QG-B2g-tomrl_gemma_s3-ToM_HiEx_hint_v3_notags-gemma-2-2b-it-rulebased-lr5e-7-kl0.001-n16-r1.log`

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
    trainer.experiment_name=s2-QG-B2g-03-QG-B2g-tomrl_gemma_s3-ToM_HiEx_hint_v3_notags-gemma-2-2b-it-rulebased-lr5e-7-kl0.001-n16-r1 \
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

**How to rerun:** `EXP_ID=QG-B2g-tomrl_gemma_s3 DATA_NAME=ToM_HiEx_hint_v3_notags MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3_notags.parquet RUN_INDEX=1 bash experiments/train_tom_gemma.sh`

**Debugging / issues:**

- **2026-08-09T04:42Z:** Stopped r1 after step 247 due to sustained generation collapse.
  Isolated literal `<eos>` plus generated `<pad>` runs first appeared at steps 152 and 156,
  then spread across 9 of steps 236-247, with up to 3.9% of responses clipped at 2048.
  At step 247, KL reached 9.554 and entropy fell to 0.003 while the rule scorer still
  reported 100% parseability. Samples included `file cabinet<eos><pad>...`,
  `red_envelope<eos><pad>...`, and `red_box <eos><pad>...`.
- Interim canonical scoring through step 240: ToM HM(last5)=0.5133,
  HM(last3)=0.5153 (baseline 0.4634); ToM avg(last5)=0.5723,
  avg(last3)=0.5756 (baseline 0.5142); gsm8k delta=-0.038; mmlu delta=+0.013;
  max_resp=2048. The eval gain does not make this incomplete, degenerating seed valid.

**Findings:** r1 failed and is not a completed seed.
