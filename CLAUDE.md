# BeRL — TomRL: Dialogue→ToM Transfer via GRPO

## Project Goal

Demonstrate that **dialogue data improves Theory of Mind (ToM) reasoning** in small LLMs (Qwen2.5 0.5B-7B) through GRPO training — without ever training on ToM examples directly.

## Key Narrative

The core hypothesis: predicting what someone says next in conversation requires modeling their mental state (beliefs, emotions, knowledge). This implicit ToM signal in dialogue data can transfer to explicit ToM benchmarks.

Rounds 1-16 failed to show transfer. **Round 17d was the breakthrough**: the bottleneck was **system prompt mismatch** between training and evaluation, not the data itself. Once the training system prompt was aligned with the eval prompt (`cot_eval` style), dialogue training produced the first-ever positive ToM transfer.

## Round 17 Results (Dialogue→ToM Transfer)

| Experiment | Key Change | tomi | explore_tom | hi_tom | Outcome |
|-----------|------------|------|-------------|--------|---------|
| 17a | Baseline | 63.3% | 47.0% | 19.0% | Degradation |
| 17b | Filtered data | 63.3% | 46.8% | 17.9% | Slower degradation |
| 17c | ll_min=-8 | 63.3% | 47.3% | 19.3% | Slowest degradation |
| **17d** | **+ Eval prompt alignment** | **65.3%** | **56.0%** | **26.9%** | **Breakthrough** |
| 17e | + Wider clip (-40,40) | 68.6% | 60.6% | 32.5% | Best hi_tom |
| **17f** | **+ Actor as RM** | **69.4%** | **66.9%** | 29.8% | **Best tomi & explore** |

### Dialogue vs Direct ToM Training (17g)

| Metric | Baseline | Direct ToM (17g) | Dialogue 17f |
|--------|----------|-----------------|--------------|
| tomi | 63.1% | 68.9% | **69.4%** |
| explore_tom | 47.0% | **83.0%** | 66.9% |
| hi_tom | 19.1% | **34.7%** | 29.8% |

Dialogue training (17f) **surpasses** direct ToM training on tomi (generalization benchmark) — 69.4% vs 68.9% — without seeing a single ToM example. Achieves ~80% of direct ToM gains on tomi and ~55% on explore_tom purely through conversational reasoning transfer.

## Breakthrough Recipe (17f best config)

- **Dataset**: 6,214 filtered dialogue samples (DailyDialog + Empathetic Dialogues), min_turns=6, min_response_words=10
- **System prompt**: `cot_eval` — matches eval prompt exactly (generic CoT with `<think>` tags + ToM framing)
- **Reward**: Power-law LL reward, ll_min=-8, clip (-40, 40), actor-as-RM
- **GRPO**: KL=0.05, LR=5e-7, batch=32, mini_batch=128, rollout_n=16, 2 epochs
- **Model**: Qwen2.5-3B-Instruct

## System Prompt Ablations (17i-j)

- `cot_eval` (generic) works best — flexible enough to transfer between dialogue and ToM
- `cot_tom` (intents/beliefs/goals framework) degrades from higher baseline
- `cot_tom2` ("all parties perspective") causes catastrophic collapse

## Key Technical Insights

1. **Prompt alignment is critical** — training/eval system prompt mismatch was the primary transfer bottleneck
2. **Data quality > quantity** — filtered 6k beats noisy 16k (17f vs 17h)
3. **Dense reward matters** — ll_min=-8 gives ~70% of samples non-zero reward vs ~21% with ll_min=-5
4. **Actor-as-RM provides curriculum** — evolving reward signal keeps gradients fresh on repeated data
5. **Generic prompts transfer better** — overly specific ToM prompts hurt generalization

## Important Files

- `grpo_tuning_changelog.md` — full experiment log (17+ rounds)
- `experiments/empathic_dialogue_grpo.sh` — main flexible training script
- `experiments/dialogue_grpo_power_reward.sh` — dialogue GRPO with power reward
- `experiments/dialogue_grpo_fantom_eval.sh` — dialogue GRPO with FANToM eval
- `experiments/dialogue_grpo_power_reward_actorRM.sh` — actor-as-RM variant
- `experiments/tom_grpo_power_reward.sh` — direct ToM training
- `scripts/convert_dailydialog.py` — DailyDialog dataset converter
- `scripts/convert_empathetic_dialogues.py` — Empathetic Dialogues converter
- `scripts/convert_conversations_gone_awry.py` — ConvoKit CGA converter
- `scripts/prompt_templates.py` — system prompt styles (cot_eval, cot_tom, etc.)
- `scripts/configs/pipeline_config_filtered_eval_prompt.yaml` — best dataset config
- `scripts/configs/pipeline_config_conversations_gone_awry.yaml` — CGA dataset config
- `examples/data_preprocess/prepare_fantom.py` — FANToM eval benchmark preparation
- `verl/utils/reward_score/fantom.py` — FANToM scoring module
- `docs/skill_adding_dataset.md` — guide for adding new training datasets

## Eval Benchmarks

- **tomi**: 5,994 samples (Clever Hans) — generalization test, never in training
- **hi_tom**: 600 samples — multi-order belief reasoning
- **explore_tom**: 1,066 samples — 1,200 in training set for direct ToM
- **fantom**: 10,422 samples — multi-party conversation ToM (belief MC, answerability, info-access)
- **4th-order**: 600 samples — highest-order belief tracking

## Round 19: FANToM Eval + Conversations Gone Awry

Added FANToM as a new eval benchmark (multi-party conversation ToM) and Conversations Gone Awry (ConvoKit) as a new training data source.

### FANToM Benchmark

FANToM tests ToM reasoning in multi-party conversations — 5 question types:
- **belief_mc**: multiple-choice belief questions (1,540 samples)
- **answerability_binary/list**: who can answer a question (3,571 / 870 samples)
- **info_binary/list**: who has access to information (3,571 / 870 samples)

### Dialogue vs CGA vs Combined Training (3B)

| Benchmark | Baseline | Dialogue Peak | CGA Peak | Combined Peak | Direct ToM Peak |
|-----------|----------|--------------|----------|---------------|-----------------|
| tomi | 63.2% | **68.7%** | 66.2% | 67.4% | 69.0% |
| explore_tom | 47.0% | 61.6% | 63.0% | 74.8% | **85.5%** |
| hi_tom | 19.0% | 28.8% | 29.9% | 31.4% | **36.3%** |
| fantom_belief_mc | 7.2% | 31.2% | 39.0% | 39.2% | **49.7%** |
| fantom_answ_binary | 6.2% | 17.6% | 18.3% | 18.3% | **27.7%** |
| fantom_answ_list | 14.6% | 26.4% | **33.2%** | 31.1% | 34.1% |
| fantom_info_binary | 4.6% | 18.2% | 25.3% | 29.3% | **51.8%** |
| fantom_info_list | 14.8% | 28.9% | **34.3%** | 32.4% | 35.7% |

### 7B Scaling — Combined Data (Dialogue + CGA)

| Benchmark | 3B Combined Peak | **7B Combined Peak** | 7B vs 3B |
|-----------|-----------------|---------------------|----------|
| tomi | 67.4% | **69.4%** | +2.0pp |
| explore_tom | 74.8% | **76.8%** | +2.0pp |
| hi_tom | 31.4% | **44.0%** | +12.6pp |
| fantom_belief_mc | 39.2% | **62.3%** | +23.1pp |
| fantom_answ_binary | 18.3% | **23.6%** | +5.3pp |
| fantom_answ_list | **31.1%** | 17.8% | -13.3pp |
| fantom_info_binary | 29.3% | **63.5%** | +34.2pp |
| fantom_info_list | **32.4%** | 27.8% | -4.6pp |

### 7B: Direct ToM vs Combined (Dialogue+CGA)

| Benchmark | 7B Baseline | 7B Direct ToM Peak | 7B Combined Peak |
|-----------|-------------|-------------------|-----------------|
| tomi | 67.8% | 67.8% (no gain) | **69.4%** |
| explore_tom | 72.8% | **80.8%** | 76.8% |
| hi_tom | 42.5% | **72.6%** | 44.0% |
| fantom_belief_mc | 60.4% | 60.5% | **62.3%** |
| fantom_answ_binary | 20.1% | 21.8% | **23.6%** |
| fantom_answ_list | 14.5% | 14.5% (no gain) | **17.8%** |
| fantom_info_binary | 58.6% | 63.3% | **63.5%** |
| fantom_info_list | 26.7% | **28.0%** | 27.8% |

7B key results:
- **Direct ToM fails to improve tomi at 7B** — stays at 67.8% baseline, while combined achieves 69.4%. Dialogue transfer generalizes better at scale.
- **hi_tom: direct ToM dominates** — 72.6% vs 44.0%. Direct training produces massive hi_tom gains at 7B.
- **Combined wins on FANToM** — belief_mc 62.3% vs 60.5%, answ_binary 23.6% vs 21.8%, info_binary 63.5% vs 63.3%. Dialogue+CGA outperforms direct ToM on multi-party reasoning.
- **7B baselines are much stronger** — fantom_belief_mc starts at 60.4% (vs 7.2% at 3B), leaving less room for improvement.
- List question types (answ_list, info_list) degraded at 7B — possible formatting issue.

### Key Findings

1. **CGA outperforms dialogue on FANToM** — conflict/adversarial conversations transfer more strongly to multi-party ToM. belief_mc: 39% vs 31%.
2. **Combined data achieves best explore_tom** — 74.8% at 3B, 76.8% at 7B.
3. **7B scaling dramatically improves FANToM** — belief_mc 62.3% (+23pp), info_binary 63.5% (+34pp) over 3B.
4. **Dialogue transfer generalizes better at scale** — at 7B, direct ToM fails to improve tomi while combined dialogue+CGA achieves +1.6pp.
5. **No collapse** — scores remain stable through 750 steps at both scales.
6. **CGA characteristics**: Wikipedia Talk page debates, avg 85 words/response (vs 22 for dialogue), richer argumentation and perspective-taking.

### Wandb Runs

- Dialogue + FANToM (3B): `dialogue_filtered_eval_prompt-...-fantom` (run `m8vcmg1u`)
- CGA only (3B): `conversations_gone_awry-...-fantom` (run `c0w80kpt`)
- Combined (3B): `dialogue_cga_combined-...-fantom`
- Direct ToM (3B): `round19-tom3k-rulebased-fantom-...` (run `qhnoz5kh`)
- Combined (7B): `dialogue_cga_combined-Qwen2.5-7B-...-fantom` (run `nxg8ii0w`)
- Direct ToM (7B): `round19-tom3k-rulebased-fantom-Qwen2.5-7B-...` (run `1h44li3k`)

## Round 20: Qwen3 Compatibility + Scaling (2026-05-30)

Made `<answer>` tag extraction optional (`require_answer_tags=False`) for Qwen3 models that use native `<think>` mode without `<answer>` tags. Fixed double `<think>` tag parsing, eval answer extraction, and actor-as-RM stitching.

### Qwen3-1.7B Results

| Benchmark | Baseline | Direct ToM Peak (20a) | Combined Peak (20b) | Qwen2.5-3B Peak |
|-----------|----------|----------------------|---------------------|-----------------|
| tomi | 66.0% | **77.2%** | 76.6% | 69.0% |
| explore_tom | 48.5% | **76.5%** | 69.3% | 85.5% |
| hi_tom | 35.4% | **43.1%** | 29.0% | 36.3% |
| belief_mc | 46.4% | **54.0%** | 49.1% | 49.7% |
| answ_binary | 43.7% | 44.5% | 44.5% | 27.7% |
| info_binary | 65.2% | 65.7% | 65.8% | 51.8% |

Key: Qwen3-1.7B surpasses Qwen2.5-3B on tomi (77.2% vs 69.0%) with half the parameters. Native thinking mode provides much stronger baselines and training gains. Combined dialogue training underperforms direct ToM on Qwen3, unlike Qwen2.5 where they were competitive.

### Wandb Runs (Round 20)

- Direct ToM (1.7B): `round20-tom3k-rulebased-fantom-Qwen3-1.7B-5e-7-16`
- Combined (1.7B): `dialogue_cga_combined-Qwen3-1.7B-actorRM-nobaseline-lr5e-7-n16-power-reward-k2.0-llmin-8.0-fantom`

## Branch & Workflow

- Working branch: `jude/dev`
- Base model: Qwen2.5-3B-Instruct (also tested 0.5B, 7B), Qwen3-1.7B
- Training framework: verl (FSDP workers)
- Reward clipping constants in `verl/workers/fsdp_workers.py`
