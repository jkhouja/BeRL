### Attempt r1 — 2026-07-12T17:28:54+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr1e-6_fp5_ec0.001_lp-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1`
- **Host:** h100-033-004   **git:** `16570d4`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=frozen baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/1024 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr1e-6_fp5_ec0.001_lp-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr1e-6_fp5_ec0.001_lp-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1.log`

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
    actor_rollout_ref.actor.optim.lr=1e-6 \
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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr1e-6_fp5_ec0.001_lp-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r1 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
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
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=log_prob \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr1e-6_fp5_ec0.001_lp DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=1 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


### Debugging note — 2026-07-12T18:xx (host h100-033-004, r1)
**BLOCKED — no learning signal (batch-wide bug).** At steps 1–38 every GRPO group reward is a
constant **-85.0** (`[GRPO group rewards]` all -85; critic/score max==min==-85; advantages/pg_loss=0).
-85 = `invalid_reward_value(log_prob, format_penalty=5) = valid_floor(-40) - fp(5) - INVALID_MARGIN(40)`
→ **every rollout is being marked INVALID**, so GRPO has zero advantage and cannot learn from the
behavior reward (only KL/entropy move the policy).

**Root cause:** the reward-side validity check marks a rollout invalid when `</think>` is absent —
`verl/workers/fsdp_workers.py:1373` (frozen RM `_switch_chat_template`) and `:707` (actor RM
`_build_rm_inputs`): `if self._parser.THINK_CLOSE not in response: invalid_response = True`, with **no
guard for tag-free parsers**. But Gemma/Qwen3 use `cot_eval_notags`
(`scripts/prompt_templates.py:72`), which instructs *"Please reason step by step, and then clearly
state your final answer"* — **no `<think>`/`</think>` tags**. So tag-free responses never contain
`</think>` → all invalid → constant -85. Log confirms: 13,689× "No </think> tag found",
`GemmaResponseParser.REQUIRE_ANSWER_TAGS=False`.

**Scope:** affects **every Wave-2 Gemma & Qwen3 log_prob/power row** (all siblings show the same -85;
multiple agents launched them concurrently). Wave-1 Qwen2.5 is unaffected (tagged data emits `</think>`).

**Proposed fix (needs user sign-off — shared verl/ code, Golden Rule #4):** make the `invalid_response`
check tag-free-aware — for parsers with `REQUIRE_ANSWER_TAGS=False`, treat a rollout as invalid only
when it has no scoreable answer content (empty response), not when `</think>` is missing. Then rerun
the Gemma/Qwen3 batch.

**Action:** row set to `Awaiting-input`; process left running (pid group) for live WandB inspection.
### Attempt r2 — 2026-07-12T21:32:49+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr1e-6_fp5_ec0.001_lp-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r2`
- **Host:** h100-033-004   **git:** `719757a`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=frozen baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/1024 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr1e-6_fp5_ec0.001_lp-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r2 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr1e-6_fp5_ec0.001_lp-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r2.log`

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
    actor_rollout_ref.actor.optim.lr=1e-6 \
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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr1e-6_fp5_ec0.001_lp-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r2 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
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
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=log_prob \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr1e-6_fp5_ec0.001_lp DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=2 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


### Monitoring note — 2026-07-12T22:08 (r2, host h100-033-004)
r2 (post-fix) has a healthy learning signal (reward varies, advantages nonzero) but shows an early
**response-length runaway**: resp_len/mean 146→22→13→113→865→939 over steps 1–21, clip_ratio
0.049→0.742→0.807 (81% of rollouts truncated at MAX_RESP=1024 by step 21). Eval trajectory:
tomi 0.597(s0)→0.620(s10)→0.393(s20); gsm8k 0.277→0.350→0.330 — tomi drops as eval answers get cut
off by the length cap. Looks like log_prob length-hacking → likely a POOR cell. Per full-horizon
methodology (no early-kill; evals can dip/recover; score HM-over-last-5), letting it run to step 190;
final verdict on completion.

### Debugging note — 2026-07-12T22:38 (r2 FAILED — CUDA OOM)
r2 crashed with `torch.OutOfMemoryError: CUDA out of memory` in the actor backward at **step 21**
(~22:00). Cause: **wrong MAX_RESP** — launched via `train_behavior_gemma.sh` which uses the gemma
common default `MAX_RESP=1024`, but the tracker Gen ctx for PS100 is **2048/512** (and
`phase_stability_sweep.sh` sets `MAX_RESP=512`). The 1024 cap let the log_prob length-runaway grow
responses to ~939 tokens (clip_ratio 0.81 by step 21); the long sequences OOM'd the FSDP backward.
**Fix:** rerun as r3 with `MAX_RESP=512` (matches tracker Gen ctx + sweep spec; smaller activations
avoid OOM). Note: sibling Wave-2 gemma runs launched via the per-family launcher likely share the
same 1024 vs 512 mismatch.
### Attempt r3 — 2026-07-12T22:41:07+00:00

- **RUN_NAME:** `Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr1e-6_fp5_ec0.001_lp-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r3`
- **Host:** h100-033-004   **git:** `2ba54e8`   **conda env:** tom
- **Model:** `google/gemma-2-2b-it` (gemma-2-2b-it)
- **Data (train):** `/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet`
- **Val files:** `[/mnt/home/judekhouja/repo/BeRL/data/cleaned_tom/eval_subsample_300.parquet]`
- **Knobs:** reward=log_prob power_k=2.0 ll_min=-8.0 rm_mode=frozen baseline=False kl=0.05 lr=1e-6 rollout_n=16 epochs=1 max_ctx=2048/512 cot_var=cot_eval require_answer_tags=False entropy_coeff=0.001 think_only_pg=False format_penalty=5
- **Env:** VLLM_ATTENTION_BACKEND=FLASH_ATTN GPU_MEM_UTIL=0.3 TP=2 n_gpus=8
- **WandB:** project=TOM_EXP run=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr1e-6_fp5_ec0.001_lp-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r3 _(paste link after launch)_
- **Log path:** `logs/20260712/Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr1e-6_fp5_ec0.001_lp-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r3.log`

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
    actor_rollout_ref.actor.optim.lr=1e-6 \
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
    trainer.experiment_name=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr1e-6_fp5_ec0.001_lp-dcfg_smoke_mix_gemma-gemma-2-2b-it-frozenRM-nobaseline-log_prob-k2.0-llmin-8.0-lr1e-6-kl0.05-n16-r3 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
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
    reward_model.format_penalty=5 \
    +actor_rollout_ref.reward_type=log_prob \
    +actor_rollout_ref.power_k=2.0 \
    +actor_rollout_ref.power_ll_min=-8.0 \
    +data.fold_system_prompt=True \
    +reward_model.require_answer_tags=False \
    +actor_rollout_ref.require_answer_tags=False
```

**How to rerun:** `EXP_ID=Phase-stability-Pm1w2gemma2_rmf_kl0.05_lr1e-6_fp5_ec0.001_lp DATA_NAME=dcfg_smoke_mix_gemma MODEL_PATH=google/gemma-2-2b-it DATA_TRAIN=/mnt/home/judekhouja/repo/BeRL/data/dcfg_smoke_mix_gemma.parquet RUN_INDEX=3 bash experiments/train_behavior_gemma.sh`

**Findings:** _(fill on completion via log-results skill)_


---

## 2026-07-13 — r3 OUTCOME: FAILED (CUDA OOM @ MAX_RESP=512)

**Result:** r3 (WandB sdrgulee, MAX_RESP=512) CUDA-OOM-crashed in the actor backward at
**~step 109/191** (log mtime 00:48, confirmed dead — 0 main_ppo/ray procs; 3 OutOfMemoryError in log).

**Failure trace:** `ray_trainer.fit → update_actor → dp_actor.update_policy → loss.backward()` →
`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.27 GiB. GPU 0 ... 11.11 GiB free.`

**Root cause:** log_prob reward length-hacking. Even with the intended MAX_RESP=512 (r2 had OOM'd
at 1024/step21), the policy progressively inflated CoT length to game the held-out-utterance
log-likelihood. Late-training response_length/mean climbed to ~468→456→**491**/512 with
clip_ratio 0.87→0.82→**0.92** — i.e. ~90% of rollouts hit the 512 cap. The near-cap batch made the
FSDP backward activation footprint exceed 79 GiB/H100.

**Health before crash (from prior checks):** learning signal fine (reward > -85, nonzero advantages),
parseable=1.0, kl/entropy sane; evals MIXED not collapsed (tomi 0.60→0.34 down, gsm8k 0.28→0.51 up,
bigtom_fwd_belief recovered ~0.85). So the failure is a memory/stability failure, NOT a no-signal or
collapse failure.

**Decision:** Per protocol (OOM/crash twice on same cell → do not keep retrying), PS100 set to
**Status=Failed**. Two independent OOMs (MAX_RESP=1024 r2, MAX_RESP=512 r3) show raw log_prob on
Gemma-2 at this batch×ctx is OOM-unstable.

**Mitigation guidance for sibling Wave-2 Gemma-2 log_prob rows:** either reduce GPU micro-batch /
increase grad-accum / lower response cap, enable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
or prefer the power reward family (bounded, less length-hacking) for Gemma-2 transfer.

**Rerun (if ever needed, NOT planned):** same launch as r3 but add
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and/or a smaller `ppo_micro_batch_size`.
