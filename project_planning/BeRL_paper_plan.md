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
Qwen2.5-3B (workhorse), Gemma-2-2B. (Qwen3-1.7B dropped from Phase −1 v2 per user 2026-07-19.)
**Data:** fixed reasonable small mix
(`dcfg_smoke_mix{,_gemma}` or the v1 mix), 1 seed. **`ll_min` stays deferred to Q2** (corpus-relative).

**Anchor (per family)** — the v1 winner region, **minimally perturbed to guarantee novelty vs the
old stage** (see "Novelty" below): reward = **`power k=4`** (the old sweep only used k∈{3,5,7}) with
the **new std-gated format gate** (`format_penalty_std_coef=1.0`, which never existed pre-v2):
- Qwen2.5-3B: `power · k=4 · ll_min=−6 · actor-RM · std-gated-fp(coef 1.0) · ec=0.0 · kl=0.05 · lr=5e-7`
- Gemma-2-2B: `power · k=4 · ll_min=−4 · frozen-RM · std-gated-fp(coef 1.0) · ec=0.001 · kl=0.05 · lr=5e-7`

**Novelty vs the old stage (verified against `old_BeRL_experiments_tracker.md`).** The old Phase −1
grid swept only `reward∈{log_prob, power k3/k5/k7}` with the **flat** format penalty
(`fp_std_coef` did not exist) — `k=4`, `k=6`, `neg_perplexity`, and the std-gated gate were **never
run** (0 completed cells). Anchoring v2 on **`power k=4` + std-gated fp** therefore makes **every one
of the 24 cells a genuinely new configuration**, not a re-run: the two non-power reward probes
(`log_prob`, `neg_perplexity`) appear only under the new std-gated gate, and the two flat-penalty
comparison cells (PD) use the never-run `k=4` exponent.

### Excluded (v1 collapse/hack evidence — do NOT re-run at scale)
| Excluded cell | v1 evidence | Treatment in v2 |
|---|---|---|
| `power k=7` (esp. actor) | reward-hack: KL→9.85, rollouts→~1.5 tok; honest `d_cavg`≈+0.004 (~82% format) | dropped from reward-family axis |
| `actor-RM` at `kl=0.01` | actor-RM blew KL in **12/72** runs vs **0/80** frozen; worst at low KL | actor tested **only at kl=0.05**; frozen may carry the kl=0.01 probe |
| `fp=0` (no format penalty) | raw gains are format-only; honest signal collapses | kept **only** as a single diagnostic reference cell (PD), not swept |

### v2 stability sub-runs (per family; the **anchor cell is shared** across stages, counted once)
| Stage | Axis varied (all other knobs = family anchor) | Cells | Picks / question |
| --- | --- | ---: | --- |
| **PA. reward family / shape** | `log_prob` / `neg_perplexity` / `power k6` (anchor = `power k4`; **k7 dropped** as reward-hack) | 3 | stable reward family + exponent sensitivity |
| **PB. RM mode** | `frozen` / `actor` (both @kl=0.05) | +1 | re-confirm actor-RM KL-blowup risk under fixes |
| **PC. KL × LR stability** (re-added) | `kl∈{0.01,0.05}` × `lr∈{5e-7,1e-6}` | +3 | does kl=0.01 / doubled-lr still drift under merged fixes? |
| **PD. format-fix** (NEW) | **std-gated `fp_std_coef=1.0`** (anchor) / flat `fp=5` / `fp=0` ref | +2 | does the std-gated gate beat flat fp on honest `d_cavg`? |
| **PE. entropy check** | `ec∈{0.0,0.001}` | +1 | family-specific entropy coefficient |
| **PF. batch size** (NEW) | `train_batch=64` (anchor = 32) | +1 | larger-batch stability/throughput |
| **Per-family total** | anchor(1) + 3 + 1 + 3 + 2 + 1 + 1 | **12** | best stable, honest config |
| **Phase −1 (v2) total** | 12 × 2 families (Qwen2.5, Gemma-2) | **24** | → v2 "stable default config" per family |

**Selection = honest `d_cavg`** (late window), gated **clean** (final KL < 1.0 AND min rollout
resp_len ≥ 30); cross-check `d_cavg_peak` (peak, upward-biased) and `d_cond_acc` (pooled).
**Collapse/health checks:** reward not saturating at clamp; utterance-PPL improving without length
blowup/degeneration; entropy not crashing; KL controlled.

**Output:** the v2 locked "stable default config" per family → feeds Phase 0.

**Tracker rows:** these 24 cells are populated as **`ST01`–`ST24`** in `BeRL_experiments_tracker.md`
(`Phase-stability` RQ tag; `ST01`–`ST12` Qwen2.5-3B, `ST13`–`ST24` Gemma-2-2B; each family's
anchor = the `..._anchor_pk4_stdgate` row — Qwen2.5 `ST01`, Gemma-2 `ST13`; `ST12`/`ST24` = the PF
`train_batch=64` cells). The Qwen3-1.7B block was removed and the remaining rows renumbered
contiguously (2026-07-19).

---

## Phase 0 (v2) — Training-data recipe search  ✅ COMPLETE (P0-01…14) — **VERDICT: null; keep `smoke_mix`**
**Result (2026-07-20, see `BeRL_findings.md` Phase 0):** data composition is a **null lever**. On the
stable `d_avg` metric every corpus (mix_all, mix_best3, all 8 single domains) lands at the same
~+0.02 as the `smoke_mix` baseline (+0.0215±0.001); all conditional-metric gaps are within the N=3
noise band (SD ~0.08). **Decision: `smoke_mix` is the fixed corpus for A1/Q0/Q1/Q2/…**; S3 turn-filter
and the S1-top2 3-seed replicates are **skipped** (confirmed null). Design below kept for the record.

Fixes the corpus used by A1/Q0/Q1, built **on top of the v2 Phase −1 winning recipe**.

**Locked v2 recipe carried in (from Phase −1, `BeRL_findings.md`):**
- **Qwen2.5-3B (workhorse):** `power k4 · ll_min−6 · actor-RM · fp=0 · ec=0.0 · kl=0.05 · lr=5e-7 ·
  train_batch=32 · rollout16 · 1 epoch`, tagged data (`cot_eval`). *(fp=0, NOT fp=5 — v2 reversed v1.)*
- **Gemma-2-2B (confirm):** `power k4 · ll_min−4 · frozen-RM · fp=0 · ec=0.001 · kl=0.05 · lr=5e-7`,
  tag-free (`cot_eval_notags`).

### ⚠️ Design change forced by the Phase −1 error-bar (read first)
Phase −1's N=3 anchor replicate (ST01≡ST10≡ST28, identical config) measured the **true run-to-run
noise**: **SD(d_cavg) ≈ 0.083** (2σ band ±0.17) but **SD(d_avg) ≈ 0.001**. The v1 Phase-0 data search
was **single-seed, ranked on d_cavg** — so its honest deltas (mix picks separated by ~0.04–0.08
d_cavg) were **entirely within the noise floor** and are NOT trustworthy. The v2 data search therefore:
1. **Primary metric = `d_avg`** (stable, better-powered) on Qwen2.5, where format is ~100%-saturated so
   d_avg is *not* format-confounded. `d_cavg`/`d_cond_acc` are secondary and used as the honest metric
   **only for Gemma** (format not saturated there). Never rank a data config on a d_cavg gap <≈0.17.
2. **≥3 seeds per shortlisted data config** — report mean ± SD; a config only "wins" if its d_avg (or
   Gemma d_cond_acc) mean clears the baseline by more than the seed SD. **This is the single biggest
   change vs v1** and the reason v1's mixture/turn-filter conclusions must be re-derived, not reused.
3. **Reuse the free N=3 smoke_mix baseline** (ST01/ST10/ST28: d_avg +0.0215±0.001, d_cavg +0.067±0.083)
   as the reference corpus — every candidate is compared against it with error bars.

### Structure (Qwen2.5 search → Gemma confirm); prefix `P0##`, `RUN_STAGE=s2`
- **S1 — single-domain screen (1 seed each, screen on d_avg).** The 8 built domains
  (`dcfg_{casino,cga,craigslist,dailydialog,diplomacy,empathetic,p4g,thoughttrace}`). Cheap screen to
  rank domains by raw d_avg; **shortlist the top ~2**.
- **S2 — mixtures vs baseline (3 seeds each).** `dcfg_smoke_mix` (=baseline, have N=3) vs `dcfg_mix_all`
  vs `dcfg_mix_best3` vs the S1 top-2 singletons. **Question: does *any* corpus beat smoke_mix on d_avg
  beyond seed SD?** (v1 said mixtures don't compose — re-test with power.)
- **S3 — quantity vs quality / turn filtering (GATED, low priority).** `dcfg_mix_best_{surprise,randlen,
  predictable}`. **Gate:** only run if S2 produced a corpus with honest signal above noise to
  concentrate; v1's S3 was null ("quantity > quality"). Skip by default.
- **S6 — Gemma confirm (3 seeds).** Re-run the S2 winner + smoke_mix on Gemma; judge on d_cond_acc.

**Selection:** the corpus whose **d_avg mean − baseline** is largest AND exceeds the seed SD; ties →
prefer the simplest/cleanest (smoke_mix) and largest stable corpus (v1: quantity > quality). Feeds A1.

**v1 archived result (context, NOT locked — likely within-noise, see above):** recipe =
`dcfg_mix_best3{,_gemma}`, no turn filter; no mixture synergy; turn-surprisal null; honest gain small.
See `old_BeRL_findings.md` Phase 0.

Data hygiene (unchanged): exclude first/last turns; held-out utterances from other conversations or
strictly-later turns (no leakage).

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

## Q1 — Generalization vs. direct ToM (headline)  🟡 B2 POPULATED (2026-07-21; Q1-B2-01…03)
Model: **3B**; repeat B1/B2 at **0.5B & 7B-1M** for a scaling curve.

| Run | Reward | Note |
| --- | --- | --- |
| B1 | behavior (= A1) | **no ToM labels** — = anchor ST01/10/28 (N=3, already run; reuse) |
| B2 | rule-based ToM (`train_tom_*.sh`, message-format v3 data) | direct ToM (uses labels) = ToM-RL baseline — **🟡 populated Q1-B2-01…03 (3B ×3 seeds, kl0.001)** |
| B3 | behavior + rule-based (combined) `[build mixed reward]` | does behavior add on top of labels? |

Headline: OOD/Applied/Robustness vs signal type; ID–OOD gap; **label-efficiency** (B1 uses zero ToM
labels); **reverse transfer** (behavioral/applied training → explicit & higher-order ToM).

---

## Phase SS — Scale-stability re-derivation (motivated by Q3-D6)  🟡 ACTIVE (2026-07-21; SS-01…09)
Q3-D6 found **no positive model-scaling** because the 3B-locked recipe **blows up KL off-3B** (0.5B 2.28,
7B 1.57, Gemma 2.63; 0.5B/7B used **actor RM** = known destabiliser, Gemma frozen but hot). Before any
headline scaling curve, re-stabilise the recipe **per scale/family** on `smoke_mix`:
- **0.5B & 7B (Qwen2.5):** swap actor→**frozen RM** (isolate), + a stronger-KL (kl0.1) and lower-LR (2e-7) arm.
- **Gemma-2B (already frozen):** stronger-KL / lower-LR / both.
9 rows (3 per scale, 1 seed). **Selection:** KL_max<1.0 (stable) AND honest d_cavg (format not saturated
off-3B). Winner per scale → re-run a clean D6 curve. Global reward family fixed (power k4).

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

## Q3 — Scale and interaction type / data (hypothesis)  ✅ D6+D5 DONE (2026-07-20/21); D1 done via Phase-0
**Hypothesis: ToM transfer scales with the ToM-dependence of the training signal and model size.** Model: **3B**.
**Metric protocol (from Phase −1 error bar): rank on stable `d_avg` (SD 0.001); d_cavg noise-dominated.**

| Run | Variable | Hypothesis | Status |
| --- | --- | --- | --- |
| D1 (×5–6) | domain: Empathetic / CaSiNo+Craigslist / CGA / Diplomacy / P4G / DailyDialog (smalltalk) | info-asymmetric domains > smalltalk | ✅ **DONE via Phase-0 S1 (P0-01…08)** → **hypothesis NOT supported on d_avg**: dailydialog (smalltalk) +0.0209 ties top negotiation domains; diplomacy near bottom (+0.0071). Domain is a **null lever** (see `BeRL_findings.md` Phase 0). |
| D6 (×3) | model scale: best hypers @ **0.5B / 3B(=A1) / 7B** (Qwen2.5) + **Gemma-2B** | model-scaling law | ✅ **DONE (Q3-01…06)** → **NO positive scaling under the 3B-locked recipe**: 3B best (+0.0215 honest); **7B flat/neg** (−0.02, saturated, no headroom); **0.5B & Gemma format-confounded + KL blew up (2.3–2.6)** → recipe is 3B-tuned, must re-stabilise per scale before a scaling claim. |
| D5 (×4) | #conversations ≈ **1k / 5k / 10k** (+ free endpoints **6.1k=smoke_mix**, **26k=mix_all**) | data-scaling law | ✅ **DONE (Q3-07…12)** → **clean saturating curve on d_avg**: +0.0095(1k)→+0.019(5k)→+0.0215(6.1k)→+0.017(10k)→+0.022(26k). **Gain plateaus by ~5–6k**; smoke_mix well-sized. **Data quantity IS a lever below ~5k** (unlike size/composition). |
| D2 (×2) | long vs short (`limit_turn`) | context-length effect | ⏸ deferred (interaction-type; Phase-0 suggests within-noise) |
| D3 (×3) | curriculum `turn_order`: early→late / late→early / random `[build sched]` | expect null → report as negative | ⏸ deferred (`[build]`; expected null) |
| D4 (×2) | **surprise sampling** vs length-matched random `[build]` | ToM-dependent turns drive gains | ⏸ gated (`[build]`; old-stage null) |

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
- **Optimize each family independently** — Phase −1 re-derives per family (Qwen2.5, Gemma-2);
  no cross-family recipe transfer assumed.
- **Search = workhorse family** for Phase 0/Q2 (Qwen2.5-3B), Gemma confirmations per phase.
- **Sizes / scaling:** 3B workhorse; scaling curve at 0.5B & 7B-1M for Q1 and Q3-D6.
- **Hygiene:** one variable per run; freeze KL/lr/rollout-N within a question; exclude first/last
  turns; no held-out leakage; verify Hi-ToM train does not leak into ID eval.

## Total run count (training runs; A0 is eval-only)
**Order: Phase −1 (v2) → Phase 0 → Q2 → Q0 → Q1 → Q3 → Q4 → QG.**

| Block | Runs |
|---|--:|
| Phase −1 (v2) reduced stability: 12 × 2 families (Qwen2.5, Gemma-2) | 24 |
| Phase 0 data-recipe search (TBD on Phase −1 winner) | ~15 |
| Q2 reward refinement | 16 |
| Q0 identifiability (3B): A1×3, A2, A3, A4 | 6 |
| Q1 gen vs direct: B1@0.5B/7B, B2@3B×3/0.5B/7B, B3@3B | 8 |
| Q3 scale/interaction (3B): D1×6, D2×2, D3×3, D4×2, D5×4, D6×3 | 20 |
| Q4 thinking style (3B): E2, E3×3 | 4 |
| QG cross-family (Gemma-2B): A1g×3, A3g, A4g, B2g×2 | 6 |
| **Grand total** | **~99** |

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
