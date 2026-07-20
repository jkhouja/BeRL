### Attempt r1 — 2026-07-19T21:09:13+00:00

- **RUN_NAME:** `s2-paper-repl_rulebased_tom-tom_hiex_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1`
- **Host:** h100-076-003   **git:** `e841f0e`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=n/a power_k=n/a ll_min=n/a rm_mode=frozen baseline=False kl=0.001 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/2048 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=0.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-paper-repl_rulebased_tom-tom_hiex_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260719/s2-paper-repl_rulebased_tom-tom_hiex_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=8 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-paper-repl_rulebased_tom-tom_hiex_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=paper-repl_rulebased_tom DATA_NAME=tom_hiex_hint_v3 MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3.parquet RUN_INDEX=1 bash experiments/train_tom_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_

### Attempt r1 — 2026-07-19T21:09:27+00:00

- **RUN_NAME:** `s2-paper-repl_rulebased_tom-tom_hiex_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1`
- **Host:** h100-076-003   **git:** `e841f0e`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=n/a power_k=n/a ll_min=n/a rm_mode=frozen baseline=False kl=0.001 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/2048 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.001 think_only_pg=False format_penalty=0.0 format_penalty_std_coef=0.0
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=s2-paper-repl_rulebased_tom-tom_hiex_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260719/s2-paper-repl_rulebased_tom-tom_hiex_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1.log`

**Exact command:**
```bash
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3.parquet \
    data.val_files=[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet] \
    data.val_metric_suffix=_sub300 \
    data.train_batch_size=8 \
    data.val_batch_size=16 \
    data.prompt_is_text=False \
    +data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=s2-paper-repl_rulebased_tom-tom_hiex_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=paper-repl_rulebased_tom DATA_NAME=tom_hiex_hint_v3 MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/ToM_train_HiEx_hint_v3.parquet RUN_INDEX=1 bash experiments/train_tom_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


---

## Consolidated Findings (authoritative — attempt r1, WandB `p0rabtjb`)

- **Purpose:** Paper-replication **rule-based ToM RL baseline** (direct ToM, uses labels; plan §B2)
  requested by user "so we have that in the project wandb". Ad-hoc run — **no tracker row**.
- **WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/p0rabtjb
- **Log:** `logs/20260719/s2-paper-repl_rulebased_tom-tom_hiex_hint_v3-Qwen2.5-3B-Instruct-rulebased-lr5e-7-kl0.001-n16-r1.log`
- **Config:** Qwen2.5-3B-Instruct, task=tom_rulebased (no LM reward model; data_source→rule scoring),
  KL=0.001, LR=5e-7, train_batch=8, mini_batch=128, rollout_n=16, max_ctx=2048/2048, entropy_coeff=0.001,
  1 epoch (3200 rows → 400 steps), data=`ToM_train_HiEx_hint_v3.parquet`, eval=subsample300, require_answer_tags=True.

### Canonical scores (`scripts/score_run.py`)
```
eval iters: 15 (step 0..400); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.4781  HM(last3)=0.4800  (baseline step0=0.4162)
ToM avg(last5)=0.5435  avg(last3)=0.5462  (baseline step0=0.5026)
gsm8k (separate): 0.5226 (step0=0.657, delta vs step0=-0.134)
mmlu  (separate): 0.4580 (step0=0.487, delta vs step0=-0.029)
health(final): kl=0.203 entropy=0.721 resp_len=76.094 reward=2.594 parseable=1.0 max_resp=2048
ToM HM trajectory: 0:0.416 30:0.427 60:0.431 90:0.434 120:0.455 150:0.432 180:0.443 210:0.447 240:0.457 270:0.466 300:0.476 330:0.472 360:0.475 390:0.486 400:0.477
```

### Verdict
- **Stable, no collapse, no reward-hacking.** parseable=1.0 throughout; format_error→0.0; KL tiny
  (coef 0.001, kl_loss→0.20); entropy declines 1.05→0.72 as the policy sharpens onto correct answers;
  reward/mean (rule accuracy, range −1..3) rises 0.78→2.5, all_correct_ratio ~0.84 by late training.
- **Strong ToM transfer (label-supervised upper reference).** HM(last5)=0.478 = **+0.062 over step-0
  baseline** and a clean, near-monotone climb (0.416→0.486). This is the intended ToM-RL baseline and
  markedly outperforms the label-free neg_perplexity behavior run (ST03: HM +0.018 only).
- **Capability regression:** gsm8k −0.134, mmlu −0.029 (typical RL specialization cost).
- **Use:** Serves as the label-using ToM-RL reference for BeRL's central "label-competitive without
  labels" comparison. Now available in project WandB (`p0rabtjb`).
