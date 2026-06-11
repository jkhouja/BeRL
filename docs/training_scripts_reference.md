# Training Scripts Reference

## Script Overview

| Script | Training Data | Reward Type | RM Mode | KL | Eval Data | Rounds |
|--------|--------------|-------------|---------|-----|-----------|--------|
| `empathic_dialogue_grpo.sh` | ToM 3.2k (`tom_train.parquet`) | Configurable (LL/rule-based) | Configurable (frozen/actor) | 0.001 | v2 | 11, 12 |
| `tom_grpo.sh` | ToM 3.2k (`ToM_train_HiEx_hint.parquet`) | Rule-based (no RM) | N/A | 0.001 | v3 | 11 (rerun) |
| `tom_grpo_power_reward.sh` | ToM 3.2k (`tom_train.parquet`) | Power LL (k=2, ll_min=-2) | Frozen | 0.05 | v2 | 13 |
| `dialogue_8k_grpo.sh` | Dialogue 16k (`merged_dialogue_datasets_16k.parquet`) | LL (neg_perplexity) | Frozen | 0.05 | v2 | 9, 14b |
| `dialogue_grpo_power_reward.sh` | Dialogue 16k ToM prompt (`merged_dialogue_datasets_16k_tom_prompt.parquet`) | Power LL (k=2, ll_min=-5) | Frozen | 0.05 | v2 | 15 |
| `dialogue_grpo_power_reward_actorRM.sh` | Dialogue 16k ToM prompt (`merged_dialogue_datasets_16k_tom_prompt.parquet`) | Power LL (k=2, ll_min=-3.5) | Actor-as-RM | 0.05 | v3 | 16, 16b, 16c |
| `negtom_grpo.sh` | NegotiationToM 800 (`NegotiationToM_...parquet`) | LL | Frozen | 0.001 | v1 (old) | — |

## Detailed Script Parameters

### `empathic_dialogue_grpo.sh`
- **Current config:** Rule-based or LL reward (configurable via `USE_LM_REWARD`)
- **Training data:** `data/tom_train.parquet` (3200 rows: hi_tom + explore_tom)
- **Eval data:** `data/cleaned_tom/ToM_test_HiExTi_hint_v2.parquet`
- **Model:** Qwen/Qwen2.5-3B-Instruct
- **Key params:**
  - `SUBTRACT_BASELINE=False`
  - `USE_ACTOR_AS_RM=False`
  - `REWARD_TYPE=log_prob`
  - `USE_LM_REWARD=True` (set to `False` for rule-based Round 11)
  - `kl_loss_coef=0.001`, `kl_ctrl.kl_coef=0.001`
  - `clip_ratio=0.2`, `grad_clip=1.0`
  - `ppo_mini_batch_size=128`
  - `batch_size=32`, `rollout_n=16`, `epochs=2`
  - `save_freq=30`, `test_freq=10`
- **Rounds used for:** 11 (rule-based, `USE_LM_REWARD=False`), 12 (LL, `USE_LM_REWARD=True`)
- **Notes:** The most versatile script — can run rule-based or LL reward by toggling `USE_LM_REWARD`

### `tom_grpo.sh`
- **Current config:** Rule-based reward (no RM enabled)
- **Training data:** `data/cleaned_tom/ToM_train_HiEx_hint.parquet`
- **Eval data:** `data/cleaned_tom/ToM_test_HiExTi_hint_v3.parquet`
- **Key params:**
  - No `reward_model.enable` flag (defaults to rule-based)
  - `kl_loss_coef=0.001`, `kl_ctrl.kl_coef=0.001`
  - `ppo_mini_batch_size=128`
  - `batch_size=32`, `rollout_n=16`, `epochs=2`, `GPUs=8`
  - `gpu_memory_utilization=0.35`
- **Notes:** Originally a 2-GPU script, updated to 8 GPUs + v3 eval. Uses different training parquet than `empathic_dialogue_grpo.sh`.

### `tom_grpo_power_reward.sh`
- **Current config:** Power LL reward on ToM data
- **Training data:** `data/tom_train.parquet` (3200 rows)
- **Eval data:** `data/cleaned_tom/ToM_test_HiExTi_hint_v2.parquet`
- **Key params:**
  - `REWARD_TYPE=power`, `POWER_K=2.0`, `POWER_LL_MIN=-2.0`
  - `SUBTRACT_BASELINE=False`, `USE_ACTOR_AS_RM=False`
  - `kl_loss_coef=0.05`, `kl_ctrl.kl_coef=0.05`
  - `clip_ratio=0.2`, `grad_clip=1.0`
  - `ppo_mini_batch_size=128`
- **Rounds used for:** 13

### `dialogue_8k_grpo.sh`
- **Current config:** Neg-perplexity LL reward on dialogue data
- **Training data:** `data/merged_dialogue_datasets_16k.parquet`
- **Eval data:** `data/cleaned_tom/ToM_test_HiExTi_hint_v2.parquet`
- **Key params:**
  - `REWARD_TYPE=neg_perplexity`
  - `SUBTRACT_BASELINE=False`, `USE_ACTOR_AS_RM=False`
  - `kl_loss_coef=0.05`, `kl_ctrl.kl_coef=0.05`
  - `clip_ratio=0.2`, `grad_clip=1.0`
  - `ppo_mini_batch_size=128`
- **Rounds used for:** 9, 14b
- **Notes:** Despite the name "8k", currently points to 16k dialogue data

### `dialogue_grpo_power_reward.sh`
- **Current config:** Power LL reward on dialogue data with ToM system prompt
- **Training data:** `data/merged_dialogue_datasets_16k_tom_prompt.parquet`
- **Eval data:** `data/cleaned_tom/ToM_test_HiExTi_hint_v2.parquet`
- **Key params:**
  - `REWARD_TYPE=power`, `POWER_K=2.0`, `POWER_LL_MIN=-5.0`
  - `SUBTRACT_BASELINE=False`, `USE_ACTOR_AS_RM=False`
  - `kl_loss_coef=0.05`, `kl_ctrl.kl_coef=0.05`
  - `ppo_mini_batch_size=128`
- **Rounds used for:** 15

### `dialogue_grpo_power_reward_actorRM.sh`
- **Current config:** Power LL reward with actor-as-RM on dialogue data
- **Training data:** `data/merged_dialogue_datasets_16k_tom_prompt.parquet`
- **Eval data:** `data/cleaned_tom/ToM_test_HiExTi_hint_v3.parquet`
- **Key params:**
  - `USE_ACTOR_AS_RM=True`
  - `REWARD_TYPE=power`, `POWER_K=2.0`, `POWER_LL_MIN=-3.5`
  - `SUBTRACT_BASELINE=False`
  - `kl_loss_coef=0.05`, `kl_ctrl.kl_coef=0.05`
  - `ppo_mini_batch_size=128`
  - Passes reward params to both `reward_model.*` and `actor_rollout_ref.*`
- **Rounds used for:** 16 (ll_min=-2), 16b (ll_min=-3.5), 16c (ll_min=-3.5 + v3 eval)

### `negtom_grpo.sh`
- **Current config:** LL reward on NegotiationToM data
- **Training data:** `data/NegotiationToM_Qwen-Qwen2.5-3B-Instruct_limit800.parquet` (800 rows)
- **Eval data:** `data/cleaned_tom/ToM_test_HiExTi_hint.parquet` (v1, old)
- **Key params:**
  - No `SUBTRACT_BASELINE` or `USE_ACTOR_AS_RM` flags
  - `kl_loss_coef=0.001`, `kl_ctrl.kl_coef=0.001`
  - `ppo_mini_batch_size=32` (smaller than others)
  - `epochs=4`
  - No wandb logging
- **Notes:** Experimental script, not used in main changelog rounds

## Common Parameters (shared across most scripts)

| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen2.5-3B-Instruct` |
| Learning rate | `5e-7` |
| GPUs | 8 |
| Rollout N | 16 |
| Max prompt length | 1024 |
| Max response length | 2048 |
| KL loss type | `low_var_kl` |
| Gradient checkpointing | True |
| FSDP param/grad/optimizer offload | True |
| Tensor parallel size | 2 |
| VLLM attention backend | XFORMERS |
| Test frequency | Every 10 steps |

## Eval Data Versions

| Version | File | Description |
|---------|------|-------------|
| v1 | `ToM_test_HiExTi_hint.parquet` | Original, raw format with embedded `<\|im_start\|>` tags |
| v2 | `ToM_test_HiExTi_hint_v2.parquet` | Fixed: proper `[{system}, {user}]` chat format |
| v3 | `ToM_test_HiExTi_hint_v3.parquet` | v2 + concise answer instruction ("output ONLY the key noun or object") |
