### Attempt r1 — 2026-07-12T10:13:10+00:00

- **RUN_NAME:** `Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-077-004   **git:** `afc71ac`   **conda env:** tom
- **Model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen2.5-3B-Instruct)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=True entropy_coeff=0.0 think_only_pg=True format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1.log`

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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TOM_EXP \
    trainer.experiment_name=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5-dcfg_smoke_mix-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-6.0 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-6.0 \
    +reward_model.require_answer_tags=True \
    +actor_rollout_ref.require_answer_tags=True
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1swp_rma_kl0.05_lr5e-7_fp5_ec0.0_pk5 DATA_NAME=dcfg_smoke_mix MODEL_PATH=Qwen/Qwen2.5-3B-Instruct DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen2.5.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (r1) — h100-077-004

**Verdict: NEW TOP CELL. Best HM_tom of the sweep AND best general-cap preservation.**

### HM_tom (harmonic mean over ToM subtypes, excl mmlu/gsm8k), sub300
| step | 0 | 20 | 40 | 60 | 70 | 90 | 130 | 170 | 190 |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| HM_tom | 0.416 | 0.464 | 0.468 | 0.470 | **0.479** | 0.472 | 0.470 | 0.471 | 0.467 |

- **Baseline(step0)=0.416 · HM(last3)=0.468 · HM(last5)=0.4672 · PEAK=0.479@step70.**
- Δ vs baseline **+5.1pp** (last5) — the largest gain of this node's runs. Climbs to a broad peak by step~70 then holds a high plateau 0.465-0.479 through step190; no decay.

### Health
- format_error_ratio = 0.000 throughout.
- KL_loss ~0.05-0.09 (very tight — stable KL + low lr).
- entropy_loss ~1.33-1.39 (low/conservative).
- response_length/mean ~107 steady (no length hacking).

### Comparison (power-k5 cells)
| cell | KL | lr | fp | RM | HM(last5) | gsm8k end |
|------|-----|-----|-----|-----|-----------|-----------|
| **PS091 (this, TOP)** | 0.05 | 5e-7 | 5 | actor | **0.4672** | 0.657 (flat) |
| PS078 | 0.05 | 1e-6 | 0 | frozen | 0.4625 | 0.49 |
| PS083 | 0.01 | 5e-7 | 5 | actor | 0.456 | 0.62 |
| PS070 | 0.01 | 1e-6 | 0 | frozen | 0.443 | 0.24 (collapse) |

### General-capability note
- gsm8k 0.66 → 0.657 (essentially unchanged — best general-cap preservation of any cell).
- mmlu 0.48 → 0.637 (improved).

### Downstream implication
- **actor power-k5 kl=0.05 lr5e-7 fp5 ec0.0 is the current Phase-1 BEST config** — highest ToM transfer (+5.1pp), a broad robust plateau, tightest KL, and near-perfect general-cap retention.
- Refines the picture: the winning recipe is **stable KL (0.05) + LOW lr (5e-7) + format_penalty=5 + power-k5** — lower lr than the prior best-known (1e-6) both raises ToM and protects general capability. Recommend re-deriving Phase-0 default around this cell.

### Rerun one-liner
`setsid bash -c 'source ~/.bashrc; conda activate tom; cd ~/repo/BeRL; ONLY_IDX=91 bash experiments/phase_stability_sweep.sh' >/tmp/ps091_bootstrap.log 2>&1 &`
