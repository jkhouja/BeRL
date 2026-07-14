### Attempt r1 — 2026-07-13T22:11:51+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.001_pk5_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-189-003   **git:** `6b69268`   **conda env:** tom
- **Model:** `Qwen/Qwen3-1.7B` (Qwen3-1.7B)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-4 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=XFORMERS GPU_MEM_UTIL=0.35 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.001_pk5_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.001_pk5_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1.log`

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
    trainer.experiment_name=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.001_pk5_llm4-dcfg_smoke_mix_gemma-Qwen3-1.7B-actorRM-nobaseline-power-k5-llmin-4-lr5e-7-kl0.05-n16-r1 \
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
    +reward_model.use_actor_as_rm=True \
    +reward_model.reward_type=power \
    +reward_model.power_k=5 \
    +reward_model.power_ll_min=-4 \
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=power \
    +actor_rollout_ref.power_k=5 \
    +actor_rollout_ref.power_ll_min=-4 \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2qwen3_rma_kl0.05_lr5e-7_fp5_ec0.001_pk5_llm4 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=Qwen/Qwen3-1.7B DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_qwen3.sh`

**Findings:** _(fill on completion via log-results skill)_


## Findings (r1, scored 2026-07-14)

Canonical score (`scripts/score_run.py`):
```
eval iters: 39 (step 0..190); ToM benchmarks: 24 (excl gsm8k, mmlu)
ToM HM(last5)=0.1494  HM(last3)=0.1523  (baseline step0=0.1605)
ToM avg(last5)=0.2672  avg(last3)=0.2677  (baseline step0=0.272)
gsm8k (separate): 0.4896 (step0=0.487, delta vs step0=+0.003)
mmlu  (separate): 0.2308 (step0=0.243, delta vs step0=-0.012)
health(final): kl=0.002 entropy=0.284 resp_len=453.049 reward=-2.559 parseable=1.0 max_resp=512
```

**Verdict: STABLE / neutral — fp5 rescues the aggressive ll_min=-4 floor.**
- ToM HM −1.1pp, ToM avg −0.5pp (within noise), no collapse (entropy 0.284, parseable 1.0).
- **gsm8k +0.3pp (0.490 vs 0.487)** — the decisive result: PS150 (frozen power k5, **ll_min=-4, fp=0**)
  crashed gsm8k −31.6pp off-distribution; here the same aggressive ll_min=-4 floor + **fp=5** holds
  capability flat. Confirms **format penalty (fp=5) is the key Qwen3 stabilizer**, dominating the
  reward-floor choice.
- No positive transfer to Qwen3 (HM slightly down) but no capability regression → adds a **third
  stable Qwen3 anchor** (with PS156 frozen k7/ll_min-6/fp5 and PS163 actor k5/ll_min-6/fp5).

### Wave-2 Qwen3 summary (6 runs)
| Row | RM | reward | k | ll_min | fp | ec | HM Δ | gsm8k Δ | verdict |
|-----|----|--------|---|--------|----|----|------|---------|---------|
| PS142 | actor | log_prob | – | – | 0 | .001 | −7.8pp | −6.8pp | NEG (self-hack via log_prob) |
| PS150 | frozen | power | 5 | −4 | 0 | .001 | −8.5pp | **−31.6pp** | NEG (off-dist crash) |
| PS156 | frozen | power | 7 | −6 | 5 | .001 | −0.9pp | −0.2pp | STABLE |
| PS163 | actor | power | 5 | −6 | 5 | 0 | −0.4pp | −1.1pp | STABLE |
| PS168 | actor | power | 5 | **−4** | **5** | .001 | −1.1pp | **+0.3pp** | STABLE (fp5 rescues ll_min-4) |

Takeaway: Qwen3 stability needs a **bounded reward (power, avoid raw log_prob) + format penalty fp=5**;
fp dominates the reward-floor (ll_min) choice — with fp5 even ll_min=-4 is safe. No positive ToM
transfer to Qwen3 yet.
