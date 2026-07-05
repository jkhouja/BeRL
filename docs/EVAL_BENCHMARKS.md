# ToM Evaluation Benchmarks

This document catalogs every Theory-of-Mind (ToM) evaluation benchmark available in the
repo: its purpose, source, format, files, `data_source` tags, and how it is scored.

All eval parquets live under `data/cleaned_tom/` (git-ignored) and are produced by the
`examples/data_preprocess/prepare_*.py` scripts. Every eval row uses the shared
model-agnostic schema (see `examples/data_preprocess/tom_eval_common.py`), so a single
parquet works across Qwen2.5 / Qwen3 / Gemma2 (the chat template is applied at load time).

## How evals are scored

Scoring is dispatched by `data_source` in `verl/trainer/main_ppo.py::_select_rm_score_fn`:

| Scorer | Benchmarks | Notes |
|--------|-----------|-------|
| `explore_tom.compute_score` | `explore_tom`, `hi_tom`, `tomi` | in-distribution ToM (open-ended) |
| `fantom.compute_score` | `fantom*` | multi-party MC/binary/list |
| `tom_mc.compute_score` | `simpletom*`, `tombench`, `opentom*`, `bigtom*`, `dyntom*`, `mmlu`, `ullman*`, `exploretom_infilled` | generic MC/binary/exact/list |

`tom_mc` ground-truth is a JSON string:
`{"answer", "answer_text", "question_type" (mc|binary|exact|list), "wrong_answer", "choices"}`.
Score = **format (±1) + answer (±2)**, range **[-3, +3]** — consistent with the other ToM
scorers. `question_type="exact"` delegates to `explore_tom.check_answer_correctness`
(exact-normalized OR prediction-ends-with-gold) so open-ended answers match the
in-distribution explore_tom eval.

## Prompt alignment (train vs eval)

Every eval parquet bakes the **`cot_eval`** system prompt (identical text across the merged
tomi/hi_tom/explore_tom parquet, `prepare_fantom.py`, and the 8 `tom_eval_common`-based evals).
Training data built by the dialogue/ToM pipeline also bakes `cot_eval` (`system_prompt_style:
cot_eval`), so **by default training and every eval are aligned** — the Round-17d alignment that
unlocked ToM transfer.

There is **no runtime system-prompt override by default** (`data.system_prompt` unset), so each
dataset uses its baked prompt:

- **Default runs (`cot_eval`)**: aligned, nothing to do.
- **Experiments varying the CoT/system prompt** (RQ `cot-style`, or any `system_prompt_style` ≠
  `cot_eval`): you **MUST** pass `+data.system_prompt="<exact prompt text>"` at launch (**Option
  A**). It replaces the system message in **both** train and val loaders
  (`verl/utils/dataset/rl_dataset.py:127`, wired in `ray_trainer.py:390,406`), keeping the
  comparison fair. Otherwise evals stay frozen at `cot_eval` while training drifts → the Round-17
  train/eval mismatch reappears. See `launch-experiment` skill §4a. Caveats: only applies when
  `data.prompt_is_text=False` (all current parquets); it overwrites eval-specific system content
  including the tomi `hint_v3` room-witness note.

## Selecting the eval set (`VAL_SUITE`)

Launchers pick the validation parquet(s) via the `VAL_SUITE` env var in `experiments/lib/common.sh`
(an explicit `VAL_FILES=[...]` bypasses it):

| `VAL_SUITE` | Files | Prompts | Approx eval time (Qwen2.5-3B / 8×H100) | Use |
|---|---|---|---|---|
| `subsample300` (**default**) | `eval_subsample_300.parquet` | 7,500 (25 subtypes × 300) | ~1.6 min | Every-`TEST_FREQ` monitoring across all benchmarks |
| `full` | all 10 benchmark parquets | ~40,000 | ~8 min | Final headline reporting (run at start/end) |
| `core` | HiExTi_v3 + fantom_50pct | 13,272 | ~2.8 min | Legacy default (core benchmarks only) |
| `sanity` | `eval_suite_sanity.parquet` | 689 | ~20 s | Quick collapse/regression check (~40/subtype) |

Time model (Qwen2.5-3B, 8×H100, `response_length=512`): `≈ 11s + 0.0117s × N_prompts`. Qwen3/Gemma
native-thinking models are ~1.5× slower.

**Representative subset** (`subsample300`): stratified `min(300, available)` rows per `data_source`,
fixed seed 42, `ullman_perturbed` excluded (only 9 rows). At N=300 the worst-case 95% CI half-width on
per-subtype accuracy is ≈ `0.98/√300 ≈ 3.5pp` — far tighter than the 5–15pp transfer effects this
project reports, so the subset reflects true per-benchmark performance while running ~5× faster than
the full suite. Regenerate with:

```bash
python examples/data_preprocess/build_eval_subsample.py --n_per_subtype 300 --seed 42
```

**Metric names.** Each val metric is logged as `val/test_score/<data_source><suffix>`, where the
suffix is set per suite via `data.val_metric_suffix` (common.sh): `subsample300` → `_sub300`,
`core` → `_core`, `sanity` → `_sanity`, `full` → *empty* (canonical `val/test_score/tomi` etc.).
This keeps a subset run's WandB series from overwriting the full-suite series of the same benchmark.
Override with `VAL_METRIC_SUFFIX=...` (or `data.val_metric_suffix=...`).

## Benchmark catalog

### Pre-existing (already in repo)

| Benchmark | data_source | Rows | Format | File / prep | Purpose |
|-----------|-------------|------|--------|-------------|---------|
| ToMi | `tomi` | 5,994 | open-ended | `data/cleaned_tom/ToM_test_HiExTi_*.parquet` (`merge_tom.py`) | Generalization (never in training) |
| Hi-ToM | `hi_tom` | 600 | open-ended | same merged parquet | Multi-order belief reasoning |
| ExploreToM | `explore_tom` | 1,066 | open-ended | same merged parquet | In-distribution ToM (symbolic story) |
| FANToM | `fantom_*` | 10,422 | MC/binary/list | `prepare_fantom.py` / `reward_score/fantom.py` | Multi-party conversation ToM |

### Added in this work

#### SimpleToM — Applied
- **Purpose:** does the model *apply* mental-state inference (mental-state → behavior → judgment)? Exposes the documented gap: models infer mental states well but fail to judge behavior given those states.
- **Source:** `allenai/SimpleToM` (mental-state / behavior / judgment QA).
- **Format:** binary/2-option MC. **3,441 rows** (`mental` 1,147, `behavior` 1,147, `judgment` 1,147). File `simpletom_test.parquet` (1.0 MB).
- **data_source:** `simpletom_mental`, `simpletom_behavior`, `simpletom_judgment`.
- **Prep:** `examples/data_preprocess/prepare_simpletom.py`.

#### ToMBench — OOD
- **Purpose:** broad ToM ability coverage (false belief, faux pas, intention, …) on English-translated items.
- **Source:** `ycfNTU/tombench_merged` (`merged_dataset.csv`).
- **Format:** 2–4-way MC. **2,859 rows.** File `tombench_test.parquet` (0.9 MB).
- **data_source:** `tombench` (ability stored in `extra_info`).
- **Prep:** `examples/data_preprocess/prepare_tombench.py`.

#### OpenToM — OOD
- **Purpose:** first/second-order belief, attitude, and multihop reasoning on natural-language narratives.
- **Source:** `SeacowX/OpenToM` (`opentom.json`).
- **Format:** MC (attitude, multihop, fine-location) + binary (coarse-location). **13,707 rows** (`multihop_fo` 3,576, `multihop_so` 3,576, `location_fo` 3,575, `location_so` 2,384, `attitude` 596). File `opentom_test.parquet` (5.1 MB).
- **data_source:** `opentom_attitude`, `opentom_location_fo`, `opentom_location_so`, `opentom_multihop_fo`, `opentom_multihop_so`.
- **Prep:** `examples/data_preprocess/prepare_opentom.py`.

#### BigToM — OOD
- **Purpose:** procedurally-generated causal ToM; forward (belief/action) and backward (belief) reasoning under true/false-belief conditions.
- **Source:** GitHub `cicl-stanford/procedural-evals-tom` (`data/bigtom/bigtom.csv`). No HF data mirror; conditions reconstructed exactly as the official `generate_conditions.py` (initial belief hidden).
- **Format:** 2-choice MC. **1,200 rows** (400 per variable, tb+fb combined). File `bigtom_test.parquet` (0.3 MB).
- **data_source:** `bigtom_forward_belief`, `bigtom_forward_action`, `bigtom_backward_belief` (condition in `extra_info`).
- **Prep:** `examples/data_preprocess/prepare_bigtom.py`.

#### DynToM — Dynamic
- **Purpose:** track an evolving mental state across multi-scenario social stories (state-in-scenario, causal influence, and state trajectory).
- **Source:** `YangXiao-nlp/DynToM` (`DynToM.json`, ~83k questions over 1,161 stories).
- **Format:** MC. **2,000 rows** sampled (`type_d` 1,026, `type_a` 538, `type_c` 436). File `dyntom_test.parquet` (6.5 MB).
- **data_source:** `dyntom_type_a`, `dyntom_type_c`, `dyntom_type_d`.
- **Prep:** `examples/data_preprocess/prepare_dyntom.py` (`--limit` controls sample size).

#### MMLU — Guardrail
- **Purpose:** general-knowledge guardrail; check that ToM/dialogue training does not degrade broad reasoning.
- **Source:** `cais/mmlu` (subset `all`, split `test`).
- **Format:** 4-way MC. **2,000 rows** sampled (of 14,042). File `mmlu_test.parquet` (1.2 MB).
- **data_source:** `mmlu` (subject in `extra_info`).
- **Prep:** `examples/data_preprocess/prepare_mmlu.py` (`--limit`).

#### Ullman-perturbed — Robustness
- **Purpose:** canonical Ullman (2023) false-belief perturbations (transparent bag, cannot read, trusted-friend override, …) that break brittle pattern-matchers.
- **Source:** GitHub `cicl-stanford/procedural-evals-tom` (`data/expert_data/ullman.csv`).
- **Format:** 3-way MC. **9 rows.** File `ullman_perturbed_test.parquet` (<0.1 MB).
- **data_source:** `ullman_perturbed`.
- **Prep:** `examples/data_preprocess/prepare_ullman_perturbed.py`.

#### ExploreToM-Infilled — Robustness
- **Purpose:** robustness counterpart to the in-distribution `explore_tom` eval — same questions/answers but on the natural-language `infilled_story` instead of the symbolic story structure.
- **Source:** `facebook/ExploreToM` (train split, 13,309 rows).
- **Format:** open-ended (`exact`) + `binary` (yes/no). **1,500 rows** sampled. File `exploretom_infilled_test.parquet` (1.9 MB).
- **data_source:** `exploretom_infilled`.
- **Prep:** `examples/data_preprocess/prepare_exploretom_infilled.py` (`--limit`, `--seed`).
- **Note:** sampled from the same pool as explore_tom training data — potential train/eval overlap; treat as a surface-form robustness probe, not a held-out generalization test.

## Baseline scores (Qwen2.5-3B-Instruct)

From a sanity validation run (mean score, range [-3, +3]; 0 = chance-ish on binary):

| data_source | score | | data_source | score |
|-------------|-------|-|-------------|-------|
| simpletom_mental | 0.90 | | opentom_location_fo | 0.73 |
| simpletom_behavior | 0.60 | | opentom_location_so | 0.45 |
| simpletom_judgment | 0.18¹ | | opentom_attitude | 0.48 |
| tombench | 0.55 | | dyntom_type_a | 0.53 |
| bigtom_forward_belief | 0.73 | | dyntom_type_c | 0.55 |
| bigtom_forward_action | 0.68 | | dyntom_type_d | 0.43 |
| bigtom_backward_belief | 0.55 | | mmlu | 0.40 |
| opentom_multihop_fo | 0.75 | | ullman_perturbed | 0.44 |
| opentom_multihop_so | 0.48 | | exploretom_infilled | 0.58 |

¹ `simpletom_judgment` is below chance because the model judges behavior omnisciently
while the gold answer reflects the character's (mistaken) mental state — this is the
**intended** SimpleToM finding, not a scoring bug (verified: `mental` 0.90 vs `judgment` 0.18).

Eval pipeline validated end-to-end on live RL rollouts: **0 parsing/scoring bugs**, ~3%
genuine model tag-misses (same rate as existing evals), ~22 s per 689-sample validation.

## Not implemented (flagged for review)

| Benchmark | Why skipped |
|-----------|-------------|
| **DialToM** | No locatable HF/GitHub dataset source. |
| **Mental-State-Representation probe** | Requires hidden-state linear probing, not a rule-based QA scorer. |
| **Behavior-PPL** | Held-out dialogue perplexity/log-likelihood metric using training reward machinery; needs a separate harness, not a QA parquet+scorer. |

## Tests

- `tests/reward_score/test_tom_mc.py` — unit checks of the scorer over letter / `(X)` / option-text / binary / exact answer formats.
- `tests/reward_score/test_eval_parquets.py` — integration: for every present parquet, gold answers score > 0 and obviously-wrong answers score ≤ 0.

## Regenerating the parquets

```bash
conda activate tom
cd examples/data_preprocess
python prepare_simpletom.py            --out ../../data/cleaned_tom/simpletom_test.parquet
python prepare_tombench.py             --out ../../data/cleaned_tom/tombench_test.parquet
python prepare_opentom.py              --out ../../data/cleaned_tom/opentom_test.parquet
python prepare_bigtom.py               --out ../../data/cleaned_tom/bigtom_test.parquet
python prepare_dyntom.py               --out ../../data/cleaned_tom/dyntom_test.parquet --limit 2000
python prepare_mmlu.py                 --out ../../data/cleaned_tom/mmlu_test.parquet --limit 2000
python prepare_ullman_perturbed.py     --out ../../data/cleaned_tom/ullman_perturbed_test.parquet
python prepare_exploretom_infilled.py  --out ../../data/cleaned_tom/exploretom_infilled_test.parquet --limit 1500
```

The in-distribution merged ToM eval (`ToM_test_HiExTi_hint_v3.parquet` = tomi + hi_tom +
explore_tom, 8060 rows) is regenerated from `merge_tom.py` (needs the source parquets under
`data/cleaned_tom/merge/`). The `v3` variant = `cot_eval` + room-witness hint + concise-answer
instruction, produced by:

```bash
python merge_tom.py --add_hint --concise_answer   # writes ToM_test_HiExTi_hint_v3.parquet
```
