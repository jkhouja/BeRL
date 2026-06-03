# TomRL Progress Update — June 3, 2026

## Summary

Since the Round 17 breakthrough (dialogue data improving ToM reasoning without direct ToM training), we've made significant progress across three fronts: (1) scaling to 7B and adding new benchmarks, (2) extending to Qwen3 architecture, and (3) generalizing the training infrastructure to support new model families like Gemma2.

---

## 1. Round 19: 7B Scaling + FANToM Benchmark + New Training Data (May 24-25)

### New Benchmark: FANToM

Added FANToM as a multi-party conversation ToM benchmark (10,422 samples) with 5 question types: belief multiple-choice, answerability (binary/list), and information access (binary/list). This complements our existing tomi, explore_tom, and hi_tom benchmarks with a focus on tracking beliefs across multi-speaker conversations.

### New Training Data: Conversations Gone Awry (CGA)

Added Wikipedia Talk page debates from ConvoKit — adversarial/conflict conversations with longer responses (avg 85 words vs 22 for dialogue). CGA outperforms standard dialogue on FANToM (belief_mc: 39% vs 31%), suggesting that conflict-driven perspective-taking transfers more strongly to multi-party ToM.

### 3B Results: Data Source Comparison

| Benchmark | Baseline | Dialogue | CGA | Combined | Direct ToM |
|-----------|----------|----------|-----|----------|------------|
| tomi | 63.2% | **68.7%** | 66.2% | 67.4% | 69.0% |
| explore_tom | 47.0% | 61.6% | 63.0% | 74.8% | **85.5%** |
| hi_tom | 19.0% | 28.8% | 29.9% | 31.4% | **36.3%** |
| fantom_belief_mc | 7.2% | 31.2% | 39.0% | 39.2% | **49.7%** |
| fantom_info_binary | 4.6% | 18.2% | 25.3% | 29.3% | **51.8%** |

### 7B Scaling: Combined Dialogue+CGA

Scaling from 3B to 7B produced dramatic improvements, particularly on FANToM:

| Benchmark | 3B Combined | 7B Combined | Delta |
|-----------|-------------|-------------|-------|
| tomi | 67.4% | **69.4%** | +2.0pp |
| hi_tom | 31.4% | **44.0%** | +12.6pp |
| fantom_belief_mc | 39.2% | **62.3%** | +23.1pp |
| fantom_info_binary | 29.3% | **63.5%** | +34.2pp |
| explore_tom | 74.8% | **76.8%** | +2.0pp |

### Key 7B Finding: Dialogue Transfer Generalizes Better Than Direct ToM

At 7B, direct ToM training **fails to improve tomi** (stays at 67.8% baseline), while combined dialogue+CGA achieves 69.4% (+1.6pp). This is the strongest evidence yet that dialogue-based ToM transfer produces more generalizable reasoning than training directly on ToM examples. Direct ToM still dominates on hi_tom (72.6% vs 44.0%) — higher-order belief tracking benefits from explicit supervision.

---

## 2. Round 20: Qwen3 Compatibility (May 30)

### Infrastructure Changes

Adapted the training stack to support Qwen3's native `<think>` mode:
- Made `<answer>` tag extraction optional (`require_answer_tags=False`)
- Fixed double `<think>` tag parsing in eval and actor-as-RM stitching
- Handled Qwen3's longer response generation (~750 tokens vs ~290 for Qwen2.5)

### Qwen3-1.7B Results

| Benchmark | Baseline | Direct ToM | Combined | Qwen2.5-3B (ref) |
|-----------|----------|------------|----------|-------------------|
| tomi | 66.0% | **77.2%** | 76.6% | 69.0% |
| explore_tom | 48.5% | **76.5%** | 69.3% | 85.5% |
| hi_tom | 35.4% | **43.1%** | 29.0% | 36.3% |
| fantom_belief_mc | 46.4% | **54.0%** | 49.1% | 49.7% |

**Qwen3-1.7B surpasses Qwen2.5-3B on tomi** (77.2% vs 69.0%) with half the parameters. Native thinking mode provides much stronger baselines and larger training gains. However, unlike Qwen2.5, combined dialogue training underperforms direct ToM on Qwen3 across all metrics.

---

## 3. Round 21: Multi-Model Infrastructure — Runtime System Prompts + Gemma2 Support (June 3)

### Problem

Two infrastructure limitations were blocking model generalization:
1. **System prompts baked into parquet files** — changing prompts for ablations required regenerating datasets
2. **Gemma2 incompatibility** — Gemma2's chat template rejects `system` role messages, blocking training entirely

### Solution: Runtime System Prompt Injection + Folding

Added two new config parameters to the data pipeline:
- **`data.system_prompt`** — replaces or injects system messages at runtime without regenerating parquets
- **`data.fold_system_prompt`** — removes system messages and prepends content to the first user message (for models like Gemma2 that don't support system roles)

Also made prompt truncation configurable (`data.truncation=left|error`) to handle the slightly longer prompts that result from system prompt folding.

Changes touch a single point in the data pipeline (`RLHFDataset.__getitem__`), so all downstream consumers (tokenization, generation, actor-as-RM) automatically get the processed prompts.

### Verification

- **Qwen2.5-3B smoke test**: Full 375-step epoch completed, baseline scores match (tomi: 0.634→0.640, explore_tom: 0.464→0.637). No regression.
- **Gemma2-2b-it smoke test**: Successfully trained with `fold_system_prompt=True`. First-ever Gemma2 training run on this codebase. Baselines: tomi 0.615, explore_tom 0.351, hi_tom 0.132.

### Gemma2-2b-it: First Training Results

Early results from Gemma2-2b-it with combined dialogue+CGA data:

| Benchmark | Baseline (step 0) | Step 5 |
|-----------|-------------------|--------|
| tomi | 0.615 | **0.665** (+5.0pp) |
| explore_tom | 0.351 | **0.624** (+27.3pp) |
| hi_tom | 0.132 | **0.248** (+11.6pp) |

Training is stable through 10+ steps. A full run with LL reward and FANToM eval is currently in progress.

### Gemma2 Memory Considerations

Gemma2's 256K vocabulary + logit softcapping creates large intermediate tensors. Working config requires:
- `max_response_length=1024` (vs 2048 for Qwen)
- `gpu_memory_utilization=0.3`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

---

## Files Changed (Round 21)

| File | Change |
|------|--------|
| `verl/utils/dataset/rl_dataset.py` | `system_prompt`, `fold_system_prompt` params + `_process_system_prompt()` method |
| `verl/trainer/ppo/ray_trainer.py` | Pass through new config params; configurable `truncation` |
| `experiments/gemma2_dialogue_cga_combined.sh` | Gemma2 power-reward training script |
| `experiments/gemma2_dialogue_cga_ll_fantom.sh` | Gemma2 LL-reward + FANToM eval script |

---

## What's Next

- Complete Gemma2-2b-it full training runs (LL reward + FANToM eval in progress)
- Gemma2 scaling (9B, 27B) if 2B results are promising
- Cross-architecture comparison: Qwen2.5 vs Qwen3 vs Gemma2 on dialogue→ToM transfer
- Investigate why combined training underperforms direct ToM on Qwen3 (architecture-specific or thinking-mode effect?)
