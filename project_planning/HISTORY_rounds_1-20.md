# BeRL — Historical Results Archive (Rounds 1–20)

> Archived from `CLAUDE.md` on 2026-07-05 during the `jude/paper` cleanup. This is the pre-paper
> exploratory history that established the method and best-known config. The **forward-looking plan
> is `BeRL_paper_plan.md` + `BeRL_experiments_tracker.md`**; the detailed run-by-run log is
> `../grpo_tuning_changelog.md` (also archival). Rounds here are NOT the final paper experiments.

## Key narrative
Predicting what someone says next in conversation requires modeling their mental state (beliefs,
emotions, knowledge). This implicit ToM signal in dialogue data can transfer to explicit ToM
benchmarks. Rounds 1–16 failed to show transfer. **Round 17d was the breakthrough**: the bottleneck
was **system prompt mismatch** between training and evaluation. Once the training system prompt was
aligned with the eval prompt (`cot_eval` style), dialogue training produced the first positive ToM
transfer.

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

Dialogue training (17f) **surpasses** direct ToM training on tomi (69.4% vs 68.9%) without seeing a
single ToM example.

### Best config (17f)
- **Dataset**: 6,214 filtered dialogue samples (DailyDialog + Empathetic Dialogues), min_turns=6, min_response_words=10
- **System prompt**: `cot_eval` (matches eval prompt: generic CoT with `<think>` tags + ToM framing)
- **Reward**: power-law LL, ll_min=-8, clip (-40, 40), actor-as-RM
- **GRPO**: KL=0.05, LR=5e-7, batch=32, mini_batch=128, rollout_n=16, 2 epochs
- **Model**: Qwen2.5-3B-Instruct

### System prompt ablations (17i-j)
- `cot_eval` (generic) works best. `cot_tom` (intents/beliefs/goals) degrades. `cot_tom2`
  ("all parties perspective") causes catastrophic collapse.

### Key technical insights
1. Prompt alignment is critical (primary transfer bottleneck).
2. Data quality > quantity (filtered 6k beats noisy 16k).
3. Dense reward matters (ll_min=-8 → ~70% samples non-zero reward vs ~21% at ll_min=-5).
4. Actor-as-RM provides an evolving curriculum.
5. Generic prompts transfer better than overly specific ToM prompts.

## Round 19: FANToM Eval + Conversations Gone Awry
Added FANToM (multi-party conversation ToM: belief_mc, answerability_binary/list, info_binary/list)
and Conversations Gone Awry (ConvoKit) training data.

### Dialogue vs CGA vs Combined (3B)
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

### 7B scaling — Combined (Dialogue + CGA)
| Benchmark | 3B Combined | 7B Combined | Δ |
|-----------|-----------------|---------------------|----------|
| tomi | 67.4% | **69.4%** | +2.0pp |
| explore_tom | 74.8% | **76.8%** | +2.0pp |
| hi_tom | 31.4% | **44.0%** | +12.6pp |
| fantom_belief_mc | 39.2% | **62.3%** | +23.1pp |
| fantom_answ_binary | 18.3% | **23.6%** | +5.3pp |
| fantom_answ_list | **31.1%** | 17.8% | -13.3pp |
| fantom_info_binary | 29.3% | **63.5%** | +34.2pp |
| fantom_info_list | **32.4%** | 27.8% | -4.6pp |

### 7B: Direct ToM vs Combined
| Benchmark | 7B Baseline | 7B Direct ToM | 7B Combined |
|-----------|-------------|-------------------|-----------------|
| tomi | 67.8% | 67.8% (no gain) | **69.4%** |
| explore_tom | 72.8% | **80.8%** | 76.8% |
| hi_tom | 42.5% | **72.6%** | 44.0% |
| fantom_belief_mc | 60.4% | 60.5% | **62.3%** |
| fantom_answ_binary | 20.1% | 21.8% | **23.6%** |
| fantom_answ_list | 14.5% | 14.5% | **17.8%** |
| fantom_info_binary | 58.6% | 63.3% | **63.5%** |
| fantom_info_list | 26.7% | **28.0%** | 27.8% |

Findings: direct ToM fails to improve tomi at 7B (dialogue transfer generalizes better at scale);
direct ToM dominates hi_tom (72.6% vs 44.0%); combined wins on FANToM; 7B baselines much stronger;
list question types degraded at 7B (possible formatting issue). CGA = Wikipedia Talk page debates,
avg 85 words/response (vs 22 for dialogue).

### Wandb runs
- Dialogue + FANToM (3B): `dialogue_filtered_eval_prompt-...-fantom` (`m8vcmg1u`)
- CGA only (3B): `conversations_gone_awry-...-fantom` (`c0w80kpt`)
- Combined (3B): `dialogue_cga_combined-...-fantom`
- Direct ToM (3B): `round19-tom3k-rulebased-fantom-...` (`qhnoz5kh`)
- Combined (7B): `dialogue_cga_combined-Qwen2.5-7B-...-fantom` (`nxg8ii0w`)
- Direct ToM (7B): `round19-tom3k-rulebased-fantom-Qwen2.5-7B-...` (`1h44li3k`)

## Round 20: Qwen3 Compatibility + Scaling (2026-05-30)
Made `<answer>` tag extraction optional (`require_answer_tags=False`) for Qwen3 native `<think>`
mode. Fixed double `<think>` parsing, eval answer extraction, actor-as-RM stitching. (Framework
fixes: vLLM Qwen3 rollout support, QK-norm in the HF training forward, and an actor-as-RM 0/0 NaN
guard — see `docs/CHANGE_HISTORY.md` + `docs/ENVIRONMENT_SETUP.md`.)

### Qwen3-1.7B Results
| Benchmark | Baseline | Direct ToM Peak (20a) | Combined Peak (20b) | Qwen2.5-3B Peak |
|-----------|----------|----------------------|---------------------|-----------------|
| tomi | 66.0% | **77.2%** | 76.6% | 69.0% |
| explore_tom | 48.5% | **76.5%** | 69.3% | 85.5% |
| hi_tom | 35.4% | **43.1%** | 29.0% | 36.3% |
| belief_mc | 46.4% | **54.0%** | 49.1% | 49.7% |
| answ_binary | 43.7% | 44.5% | 44.5% | 27.7% |
| info_binary | 65.2% | 65.7% | 65.8% | 51.8% |

Also (this session) an **all-dialogue merged (9-source) Qwen3-1.7B** run reached tomi peak 77.6 /
explore_tom 74.9 — matching/exceeding direct-ToM tomi, strong positive transfer, no collapse.

Key: Qwen3-1.7B surpasses Qwen2.5-3B on tomi (77.2% vs 69.0%) with half the parameters; native
thinking mode gives stronger baselines and gains.

### Wandb runs (Round 20)
- Direct ToM (1.7B): `round20-tom3k-rulebased-fantom-Qwen3-1.7B-5e-7-16`
- Combined (1.7B): `dialogue_cga_combined-Qwen3-1.7B-actorRM-nobaseline-lr5e-7-n16-power-reward-k2.0-llmin-8.0-fantom`

## Eval benchmarks used in these rounds
- **tomi** (5,994; generalization, never in training), **hi_tom** (600; multi-order belief),
  **explore_tom** (1,066), **fantom** (10,422; multi-party), **4th-order** (600).
