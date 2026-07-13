### Attempt r1 — 2026-07-13T04:48:58+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1`
- **Host:** h100-082-004   **git:** `6ab92f1`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=power power_k=5 ll_min=-6.0 rm_mode=actor baseline=False kl=0.05 lr=5e-7 rollout_n=16 epochs=1 max_ctx=2048/1024 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260713/Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1.log`

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
    data.max_response_length=1024 \
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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.001_pk5_llm6-dcfg_smoke_mix_gemma-gemma-2-2b-it-actorRM-nobaseline-power-k5-llmin-6.0-lr5e-7-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    reward_model.type=lm \
    reward_model.enable=True \
    reward_model.model.path=google/gemma-2-2b-it \
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
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.001_pk5_llm6 DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


#### Findings (r1, 2026-07-13, h100-082-004) — COMPLETED (with OOM caveat)
**Canonical score (`scripts/score_run.py`, 38 eval iters step0..185, 24 ToM benches excl gsm8k/mmlu):**
- ToM **HM(last5)=0.1826** HM(last3)=0.1944 (baseline step0=0.093) — ~2× baseline
- ToM **avg(last5)=0.418** avg(last3)=0.4147 (baseline step0=0.315) — **+10.3pp**
- gsm8k (separate): 0.5252 (step0=0.277, **Δ=+0.248**)
- mmlu (separate): 0.4752 (step0=0.387, **Δ=+0.088**)
- health(final): kl=0.58 entropy=1.974 resp_len=897.4 reward=6.24 **parseable=1.0**

**Per-benchmark step0→final(step185, subsample300):** tomi 0.597→0.587 · bigtom_fb 0.753→0.830 ·
bigtom_bb 0.527→0.580 · hi_tom 0.130→0.210 · explore_tom 0.360→0.570 ·
simpletom_mental 0.463→0.597 · opentom_location_fo 0.510→0.710.

**Verdict: STRONG POSITIVE TRANSFER (Completed).** Actor-RM power reward (k=5, ll_min=−6) on Gemma-2
improves ToM broadly (avg +10.3pp; HM ~2×) with no reward-hacking (evals rose alongside reward,
parseable=1.0) and big capability gains (gsm8k +24.8pp, mmlu +8.8pp). **Direct contrast with the
frozen-RM sibling PS107** (same power/k/ll_min, frozen-RM) which gave NEGATIVE transfer (avg −7.3pp):
⇒ **for Gemma-2 power, RM mode is decisive — actor-RM works, frozen-RM fails.**

**OOM caveat (debugging):** the run crashed at step 188/191 with
`torch.OutOfMemoryError: CUDA out of memory (12.51 GiB)` in
`dp_actor.update_policy → loss.backward()`. Root cause: response length grew to ~897 tokens (near the
1024 cap) with kl drift to 0.58, so the actor backward activation memory spiked. 187/191 steps + 38
eval iters completed; canonical last-5 scoring (steps 165–185) is representative and the verdict is
unchanged by the missing final step-190 eval. Not resubmitted (result complete enough; a rerun risks
the same late OOM). **If a clean full-horizon r2 is desired,** relaunch with a memory guard such as a
trailing hydra override `actor_rollout_ref.actor.ppo_micro_batch_size=4` (halves backward activation
mem) — do NOT edit shared configs.

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rma_kl0.05_lr5e-7_fp5_ec0.001_pk5_llm6 DATA_NAME=dcfg_smoke_mix_gemma DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-6.0 USE_ACTOR_AS_RM=True KL=0.05 LR=5e-7 FORMAT_PENALTY=5 ENTROPY_COEFF=0.001 TEST_FREQ=5 TOTAL_EPOCHS=1 RUN_INDEX=2 bash experiments/train_behavior_gemma.sh actor_rollout_ref.actor.ppo_micro_batch_size=4`
