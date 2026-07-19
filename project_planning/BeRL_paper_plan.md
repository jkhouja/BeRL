# BeRL Paper — Experiment Plan (v2)

**Thesis (single falsifiable claim):**
> Reinforcing *behavior prediction* — the likelihood a model assigns to the real next human
> utterance given [context + CoT] — induces Theory of Mind that (a) **transfers OOD**,
> (b) is **robust / applied** (not surface shortcuts), and (c) is **label-competitive with
> direct ToM-supervised RL**, using **no ToM labels**.

BeRL = GRPO on a reward = length-normalized log-likelihood / neg-perplexity of the held-out
human utterance (optionally baseline-subtracted), scorer = frozen base LM or the actor.

**This is the v2 plan.** The pre-v2 design + results are archived in
`old_BeRL_paper_plan.md`, `old_BeRL_experiments_tracker.md`, and summarised in
`BeRL_findings.md`. v2 re-derives the stable config **under the merged fixes** (§"Merged fixes
since v1") on a **leaner, collapse-informed** stability grid, then rebuilds Phase 0+ on top of
the newly-chosen recipe. Live status → `BeRL_experiments_tracker.md` (41-column v2 tracker).

---

## Merged fixes since v1 (why we re-run stability)
These landed on `jude/paper` after the v1 sweep and change training/measurement enough that the
locked recipe must be re-confirmed, not assumed:
- **Std-gated format-compliance penalty** (`format_penalty_std_coef`, Options 3+2; default `0.0`
  = legacy flat penalty). When `>0`, format-violating-but-valid responses are hard-gated below
  every well-formed response, with a margin scaled to the batch reward spread — a reward-type-
  symmetric alternative to the flat `fp` subtraction. **Phase −1 tests flat-fp vs std-gated.**
- **Gemma tag-free invalid-gate fix** (`8769467`): the invalid-detector no longer hard-requires a
  literal `</think>` the tag-free Gemma recipe never emits (v1 zeroed the gradient on some cells).
- **Rule/eval message-format alignment** (`ToM_train_HiEx_hint_v3.parquet`, native `[system,user]`,
  no `<think>` prefill) + startup train/val prompt-consistency guard — relevant to the Q1-B2
  rule-based ToM baseline (train now optimizes the exact format it is scored under).
- **Single-epoch guard** (`TOTAL_EPOCHS=1` default) — v1 saw KL-drift/length-inflation at
  full-epoch on some Gemma corpora.
- **frozenRM naming + `EXP_NUM` run-name prefix** — bookkeeping only; every run/log/WandB now
  starts with its tracker short id.
- **Format-controlled honest metric** (`d_cavg` = Δ mean `P(correct|parseable)`; `d_cavg_peak`;
  pooled `d_cond_acc`) is now the **primary selection metric** — raw `d_hm`/`d_avg` are
  format-confounded (v1's biggest raw gains were reward-hacks). See `BeRL_findings.md` legend.

---

## Repo knobs (what exists vs. needs building)
**Exists:** `reward_type ∈ {log_prob, neg_perplexity, power}`, `power_k`/`power_ll_min`,
`subtract_baseline`, `actor_as_rm` vs frozen RM; `format_penalty` (flat) + **`format_penalty_std_coef`
(std-gated, NEW)**; rule-based ToM reward (`experiments/train_tom_*.sh`); `prompt_style`,
`generation_prefix` (`<think>`), ToM template tags; `limit_turn`, `turn_order`; format-controlled
val metrics (`val/format_pass/*`, `val/answer_acc_cond/*`).
**Needs building `[build]`:** control reward (shuffled/mismatched utterance, length/register-matched);
surprise sampling (select turns by baseline-PPL / info-asymmetry) + length-matched random control;
training-time curriculum scheduler (turn_order is only a data sort); mixed reward (behavior +
rule-based, Q1-B3); SFT trainer entrypoint (Q0-A2, `train_sft_*` is a stub).

---

## Evaluation suite (define once; run on every checkpoint)
Default eval = **`subsample300`** (25 subtypes × 300 = 7,500 prompts) for all search phases; escalate
to `full` (~40k) only for headline reporting. Score every run vs its own step-0 baseline; select on
the honest **`d_cavg`** (format-controlled), cross-check raw `d_avg` (use `d_avg` for weak-base Gemma).

| Bucket | Benchmarks | Purpose |
|---|---|---|
| ID | Hi-ToM, ToMi, ExploreToM | in-family |
| OOD | ToMBench, OpenToM, BigToM, FANToM | contamination-safe generalization |
| Applied | SimpleToM, DialToM | explicit→applied transfer (key claim) |
| Robustness | ExploreToM-Infilled, perturbed/Ullman-style | shortcut detection |
| Dynamic | DynToM | multi-turn belief tracking |
| Probe | Benchmarking Mental State Representations | mechanism (belief reps) |
| Guardrail | MMLU, GSM8K | catastrophic-forgetting check (never folded into ToM) |
| Behavior | held-out utterance PPL | did the trained objective actually improve? |

---

## Phase −1 (v2) — Reduced, collapse-informed stability re-derivation (run FIRST)
**Goal: re-confirm a non-collapsing, non-reward-hacking config per family under the merged fixes** —
a prerequisite for trusting any data comparison. Instead of the v1 96-cell/family full factorial, v2
**anchors on the v1 winner and varies one axis at a time** (OFAT), keeping the axes that are still
uncertain (or newly-relevant) and **excluding the cells v1 proved collapse/hack.**

**Model families (optimize each independently — a winning recipe does NOT generalize across families):**
Qwen2.5-3B (workhorse), Qwen3-1.7B, Gemma-2-2B. **Data:** fixed reasonable small mix
(`dcfg_smoke_mix{,_gemma}` or the v1 mix), 1 seed. **`ll_min` stays deferred to Q2** (corpus-relative).

**Anchor (per family)** = v1 locked winner:
- Qwen2.5-3B: `power · k=5 · ll_min=−6 · actor-RM · fp=5(flat) · ec=0.0 · kl=0.05 · lr=5e-7`
- Qwen3-1.7B: `power · k=5 · ll_min=−6 · actor-RM · fp=5(flat) · ec=0.001 · kl=0.05 · lr=5e-7`
- Gemma-2-2B: `power · k=5 · ll_min=−4 · frozen-RM · fp=5(flat) · ec=0.001 · kl=0.05 · lr=5e-7`

### Excluded (v1 collapse/hack evidence — do NOT re-run at scale)
| Excluded cell | v1 evidence | Treatment in v2 |
|---|---|---|
| `power k=7` (esp. actor) | reward-hack: KL→9.85, rollouts→~1.5 tok; honest `d_cavg`≈+0.004 (~82% format) | dropped from reward-family axis |
| `actor-RM` at `kl=0.01` | actor-RM blew KL in **12/72** runs vs **0/80** frozen; worst at low KL | actor tested **only at kl=0.05**; frozen may carry the kl=0.01 probe |
| `fp=0` (no format penalty) | raw gains are format-only; honest signal collapses | kept **only** as a single diagnostic reference cell (PD), not swept |

### v2 stability sub-runs (per family; the **anchor cell is shared** across stages, counted once)
| Stage | Axis varied (all other knobs = family anchor) | Cells | Picks / question |
| --- | --- | ---: | --- |
| **PA. reward family** | `log_prob` / `power k3` / `power k5` (**k7 dropped**) | 3 | stable reward family |
| **PB. RM mode** | `frozen` / `actor` (both @kl=0.05) | +1 | re-confirm actor-RM KL-blowup risk under fixes |
| **PC. KL × LR stability** (re-added) | `kl∈{0.01,0.05}` × `lr∈{5e-7,1e-6}` | +3 | does kl=0.01 / doubled-lr still drift under merged fixes? |
| **PD. format-fix** (NEW) | flat `fp=5` / **std-gated `fp_std_coef=1.0`** / `fp=0` ref | +2 | does the std-gated gate beat flat fp on honest `d_cavg`? |
| **PE. entropy check** | `ec∈{0.0,0.001}` | +1 | family-specific entropy coefficient |
| **Per-family total** | anchor(1) + 3 + 1 + 3 + 2 + 1 | **11** | best stable, honest config |
| **Phase −1 (v2) total** | 11 × 3 families | **33** | → v2 "stable default config" per family |

**Selection = honest `d_cavg`** (late window), gated **clean** (final KL < 1.0 AND min rollout
resp_len ≥ 30); cross-check `d_cavg_peak` (peak, upward-biased) and `d_cond_acc` (pooled).
**Collapse/health checks:** reward not saturating at clamp; utterance-PPL improving without length
blowup/degeneration; entropy not crashing; KL controlled.

**Output:** the v2 locked "stable default config" per family → feeds Phase 0.

---

## Phase 0 (v2) — Training-data recipe search  ⬜ EMPTY (to populate on the Phase −1 winner)
Fixes the corpus used by A1/Q0/Q1, built **on top of the v2 Phase −1 winning recipe** (not the v1
recipe). **Left intentionally empty until Phase −1 (v2) locks the config** — populate the sub-runs
(S1 singletons → S2 mixes → S3 turn filtering → S6 Gemma confirm) once the recipe is chosen, so the
data search runs under the exact stable/honest config it will ship with.

Search on Qwen2.5-3B (workhorse), single seed, select on held-out honest `d_cavg`; confirm the
winner on Gemma-2-2B. Data hygiene: exclude first/last turns; held-out utterances from other
conversations or strictly-later turns (no leakage).

**v1 archived result (context, NOT locked):** recipe = `dcfg_mix_best3{,_gemma}`, **no turn filter**;
no mixture synergy over the best single domain (Qwen); turn-surprisal filtering gave no benefit
(quantity > quality); honest gain small (+0.07–0.15 cond-acc). See `BeRL_findings.md` Phase 0.
Re-evaluate whether these hold under the v2 recipe before reusing them.

---

## Q0 — Identifiability / causal controls (the spine — front-load)
**Reviewers will attack "accuracy up ≠ ToM." Prove the gain comes from predicting the *correct*
human utterance *via reasoning*, not any-RL-signal or lexical shortcuts.** Model: **3B**, ≥3 seeds on A1.

| Run | Setup | Isolates |
| --- | --- | --- |
| A0 | base, no training. Run all evals | reference |
| A1 | **BeRL**: best v2 config on the v2 recipe | main effect |
| A2 | SFT on gold utterances (no RL) `[build SFT entrypoint]` | RL vs SFT |
| A3 | reward = likelihood of shuffled/mismatched utterance `[build]` | correct human vs any sentence? |
| A4 | no-CoT (generation_prefix off) | does the CoT carry it? |

**Gate:** if A1 does not clearly beat A2/A3/A4 on honest `d_cavg`, the thesis is unsupported — stop.

---

## Q1 — Generalization vs. direct ToM (headline)
Model: **3B**; repeat B1/B2 at **0.5B & 7B-1M** for a scaling curve.

| Run | Reward | Note |
| --- | --- | --- |
| B1 | behavior (= A1) | **no ToM labels** |
| B2 | rule-based ToM (`train_tom_*.sh`, message-format v3 data) | direct ToM (uses labels) = ToM-RL baseline |
| B3 | behavior + rule-based (combined) `[build mixed reward]` | does behavior add on top of labels? |

Headline: OOD/Applied/Robustness vs signal type; ID–OOD gap; **label-efficiency** (B1 uses zero ToM
labels); **reverse transfer** (behavioral/applied training → explicit & higher-order ToM).

---

## Q2 — Reward-shaping refinement (after Phase 0; on the chosen corpus)
Reward *family* fixed in Phase −1; here **refine** on the locked corpus. Set `power_ll_min` **from the
chosen corpus's** utterance-LL percentiles (not arbitrary constants). Search @Qwen2.5-3B, confirm on
7B (Qwen2.5) and Gemma-2B.

| Stage | Runs |
| --- | ---: |
| A. Frozen-model search: `power_k∈{1.5,2,3,4}`@fixed ll_min, then `ll_min∈{p25,p50,p75}`@best k | 7 |
| B. Repeat A on actor-as-RM | 7 |
| confirm on 7B Qwen2.5 + 2B Gemma | 2 |
| **Q2 total** | 16 |

Diagnostics: reward-hacking (utterance-PPL↓ while honest ToM flat, length inflation, degeneration).
Global RL HPs (lr, KL β, rollout-N) fixed in Phase −1.

---

## Q3 — Scale and interaction type / data (hypothesis)
**Hypothesis: ToM transfer scales with the ToM-dependence of the training signal and model size.** Model: **3B**.

| Run | Variable | Hypothesis |
| --- | --- | --- |
| D1 (×5–6) | domain: Empathetic / CaSiNo+Craigslist / CGA / Diplomacy / P4G / DailyDialog (smalltalk) | info-asymmetric domains > smalltalk |
| D2 (×2) | long vs short (`limit_turn`) | context-length effect |
| D3 (×3) | curriculum `turn_order`: early→late / late→early / random `[build sched]` | expect null → report as negative |
| D4 (×2) | **surprise sampling** vs length-matched random `[build]` | ToM-dependent turns drive gains |
| D5 (×4) | #conversations = 1k / 5k / 10k / all | data-scaling law |
| D6 (×3) | model scale: best hypers @3B, 7B, Gemma-2B | model-scaling law |

---

## Q4 — Thinking style (mechanism)
Model: **3B**.

| Run | Variable |
| --- | --- |
| E1 | freeform CoT (= A1) |
| E2 | template ToM (emotion+belief+intent+strategy) |
| E3 (×3) | tag ablation: −emotion / −belief / −intent |

Plus **trace analysis**: probe / LLM-judge whether CoTs contain belief content — explains any
template-regression (likely format-tax / over-constraint).

---

## QG — Cross-family headline replication (Gemma-2B) — **core**
Reproduce the **Q0 causal identity + Q1 comparison** on a different family. Frozen scorer = Gemma
base; Gemma chat template; tag-free data. Uses the recipe/config locked + Gemma-confirmed in
Phase −1 / Phase 0 / Q2.

| Run | Setup | Isolates |
| --- | --- | --- |
| A1g (×3) | BeRL behavior (best config) | main effect (cross-family) |
| A3g | reward = shuffled/mismatched utterance `[build]` | correct-human vs any sentence |
| A4g | no-CoT (generation_prefix off) | does the CoT carry it? |
| B2g (×2) | rule-based ToM (uses labels) | vs direct ToM (cross-family) |
| **QG total** | 7 (A1g shares 1 seed w/ Q3-D6-Gemma → net +6) | |

**Gate (cross-family):** A1g should beat A3g/A4g, mirroring the Q0 gate — else the causal claim is Qwen-specific.

---

## Cross-cutting rigor
- **Seeds:** ≥3 for headline (A0/A1, B1/B2); 1–2 for ablations. Report bootstrap CIs over eval items.
- **Selection metric = honest `d_cavg`** (format-controlled), clean-gated (KL<1, resp_len≥30); raw
  `d_hm`/`d_avg` reported alongside but never as the sole selector (v1 lesson: raw gains were hacks).
- **Optimize each family independently** — Phase −1 re-derives per family (Qwen2.5, Qwen3, Gemma-2);
  no cross-family recipe transfer assumed.
- **Search = workhorse family** for Phase 0/Q2 (Qwen2.5-3B), Gemma/Qwen3 confirmations per phase.
- **Sizes / scaling:** 3B workhorse; scaling curve at 0.5B & 7B-1M for Q1 and Q3-D6.
- **Hygiene:** one variable per run; freeze KL/lr/rollout-N within a question; exclude first/last
  turns; no held-out leakage; verify Hi-ToM train does not leak into ID eval.

## Total run count (training runs; A0 is eval-only)
**Order: Phase −1 (v2) → Phase 0 → Q2 → Q0 → Q1 → Q3 → Q4 → QG.**

| Block | Runs |
|---|--:|
| Phase −1 (v2) reduced stability: 11 × 3 families | 33 |
| Phase 0 data-recipe search (TBD on Phase −1 winner) | ~15 |
| Q2 reward refinement | 16 |
| Q0 identifiability (3B): A1×3, A2, A3, A4 | 6 |
| Q1 gen vs direct: B1@0.5B/7B, B2@3B×3/0.5B/7B, B3@3B | 8 |
| Q3 scale/interaction (3B): D1×6, D2×2, D3×3, D4×2, D5×4, D6×3 | 20 |
| Q4 thinking style (3B): E2, E3×3 | 4 |
| QG cross-family (Gemma-2B): A1g×3, A3g, A4g, B2g×2 | 6 |
| **Grand total** | **~108** |

Front-load Phase −1 (stop if nothing trains stably), then Phase 0; gate downstream on the Q0
A1-vs-controls check (honest `d_cavg`).

## Two strengthening analyses
- **Behavior↔ToM correlation across checkpoints:** does held-out utterance-PPL improvement *track*
  ToM-benchmark improvement? Coupling = mechanism evidence; decoupling = shortcut finding.
- **CoT-transfer:** can feeding learned RL CoTs into another LLM improve *its* ToM performance?
- **Concurrent-work positioning:** *What's On My Mind* (2025, latent-thought GRPO) and *thought-tracing*
  (2025, SMC likelihood weighting) — cite as parallel; BeRL differs by rewarding the *observed
  utterance* (behavioral, label-free) vs modeling *thoughts*.

## `[build]` checklist
- [ ] SFT trainer entrypoint (Q0-A2; `train_sft_qwen2.5.sh` is a stub)
- [ ] control reward: shuffled / mismatched utterance (length/register-matched) — Q0-A3 / QG-A3g
- [ ] surprise sampling (turn selection by baseline PPL / info-asymmetry) + length-matched random control
- [ ] training-time curriculum scheduler (beyond turn_order data sort) — Q3-D3
- [ ] mixed reward (behavior + rule-based) for Q1-B3
- [ ] trace-content probe / LLM-judge for belief content — Q4
- [ ] wire OOD scorers: ToMBench, OpenToM, BigToM, SimpleToM, DynToM (+ MMLU/GSM8K guardrail)
- [ ] held-out dev set for model/recipe selection (disjoint from final test suite)
