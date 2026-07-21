# BeRL Paper — Findings Log (v2 stage)

**Purpose:** Living record of *results* for the **v2 experiment stage** (`s2-` runs, tracker
`BeRL_experiments_tracker.md`, plan `BeRL_paper_plan.md`). Pre-v2 results (Rounds 1–20, the old
`PS###`/`E###` stage, Phase 0 S1–S3) are archived in **`old_BeRL_findings.md`** — consult-only.

**Last updated:** 2026-07-21 — **Phase −1 (v2)** ✅ · **Phase 0 data ablation** ✅ (null → `smoke_mix`
locked) · **Q3 scale** ✅ (Q3-01…12): **D5 data-quantity matters below ~5k then plateaus** (smoke_mix
well-sized); **D6 model-size shows NO positive scaling under the 3B-locked recipe** (3B best; 7B
flat/neg; 0.5B & Gemma format-confounded + KL-unstable → recipe needs per-scale re-tuning). **Headline
metric =
`d_cavg`, not HM** — the HM ranking was overturned once format was controlled (see Qwen2.5).
**⚠️ See §0 (read first):** the std-gated format gate was INACTIVE all stage (fp=0 no-op); ST01≡ST10
is an accidental replicate giving SD(d_cavg)≈0.057 → most Qwen2.5 gaps are within noise. First active
gate = ST26/ST27; **both done → active gate HURTS both families (Qwen2.5 −0.073, Gemma −0.056);
fp=0 confirmed best**. Reproducibility ST28 done → **N=3 anchor SD(d_cavg)=0.083** ⇒ d_cavg is
noise-dominated on Qwen2.5; **rank format-saturated models on stable d_avg (anchor +0.021)** instead.

**Each section is organised as:** **RQ** (the question) · **Runs** (kicked off + status) ·
**Findings** · **Key runs** (WandB links). WandB project = `jkhouja-oxford/TOM_EXP`.

---

## Metric legend (read first)
Every run is scored vs **its own step-0 baseline** on the `subsample300` eval suite (25 subtypes).
- **HM / avg** — harmonic / arithmetic mean over the 24 ToM benchmarks (raw accuracy). `gsm8k`/`mmlu`
  are capability guardrails, never folded into ToM. **For the weak Gemma-2 base HM is unreliable**
  (near-0 floor benchmarks make HM jump on format alone) → prefer **avg** for Gemma.
- **d_hm** — HM(last5) − step-0 baseline. **Raw / format-confounded and biased toward lifting
  near-zero benchmarks** — a run can gain HM purely by learning to emit a parseable answer on
  floor benches. **Do NOT rank on this.** Kept only as a secondary column to expose the bias.
- **d_cavg** — Δ of the *arithmetic mean* of `P(correct | parseable)` across the 24 ToM benches
  (late window − early). **The headline honest signal**: strips the "learned to emit a parseable
  answer" confound and does **not** over-weight bottom benches. Cross-check with `d_cond_acc`
  (pooled, sample-weighted — more conservative) and `d_fmt_pass` (if the gain rode a format-pass
  jump, `d_cavg` discounts it). **d_cavg and d_hm diverge sharply on this stage** (see Qwen2.5).
- **HM_peak** — best single-eval HM during training (upward-biased; read next to KL/resp_len).
- **KL_f** — final KL to the frozen reference (health; "clean" = KL<1.0 AND min resp_len≥30).
- All ToM numbers from the tracker `Results` cells / per-run `experiments_logs/*.md`; SE ≈ 0.03–0.05
  on the 300/subtype eval — **direction trustworthy, magnitude an estimate.**

**Standard per-run table (every section):** `id · config-delta-vs-anchor · d_cavg (headline honest) ·
d_cavg_peak · d_cond_acc · d_hm (format-confounded, do-not-rank) · d_fmt_pass · Δmmlu · KL_f`.
All numbers from `analysis/reassess_v2_q25.csv` (`scripts/reassess_runs.py`). **Noise:** conditional
metrics come from 24-sample/step val blocks → **SE ≈ 0.05**; direction of the large gaps
(anchor +0.15 vs neg_ppl −0.12) is trustworthy, small ±0.04 differences are within noise.

**v2 anchors (all other knobs held at these):**
- **Qwen2.5-3B (ST01):** `power k=4 · ll_min=−6 · actor-RM · fp=0 (⚠️ gate INACTIVE, see §0) · ec=0.0 ·
  kl=0.05 · lr=5e-7 · train_batch=32 · rollout16 · 1 epoch` on `dcfg_smoke_mix` (tagged, `cot_eval`).
- **Gemma-2-2B (ST13):** `power k=4 · ll_min=−4 · frozen-RM · fp=0 (⚠️ gate INACTIVE, see §0) · ec=0.001 ·
  kl=0.05 · lr=5e-7 · train_batch=32` on `dcfg_smoke_mix_gemma` (tag-free, `cot_eval_notags`).

---

## 0. ⚠️ CRITICAL CORRECTION + reproducibility / error bars (READ FIRST)

**(a) The std-gated format gate was INACTIVE across the entire v2 stage (fp=0 no-op bug).**
Every row labelled "std-gated" (ST01–08, ST11–12, ST25 Qwen2.5; ST13–20, ST23 Gemma) set
`format_penalty=0` with `format_penalty_std_coef=1.0`. But `apply_format_penalty` short-circuits
`if not format_penalty: return rm_score` (`verl/workers/fsdp_workers.py:108`) — with **fp=0 the gate
never fires regardless of std_coef**. So those runs had **no format penalty at all**. Consequences:
- The anchors **ST01/ST13 did NOT use a std-gated gate**; their "std-gated" label is wrong.
- **ST01 (anchor) and ST10 (fp=0 reference) are byte-for-byte identical configs** (`fp_std` is unread
  when `fp=0`) → an **accidental replicate**, not two conditions. The earlier claim *"std-gated gate
  helps"* is **unsupported** — the gate never fired.
- The **first genuinely-active gate**, **ST26** (Qwen2.5, `fp=5 + fp_std_coef=1.5`), is **COMPLETE
  and HONESTLY NEGATIVE: d_cavg = −0.073**. Format pass was **already 100% at step 0** (early=late=1.0),
  so the gate had nothing to fix and only injected reward noise that hurt honest ToM. **Both
  active-penalty runs are negative** (flat fp5 ST09 −0.042, gated ST26 −0.073) while every positive run
  has **fp=0**. The Gemma active-gate **ST27** is also **COMPLETE and does NOT help** (honest
  d_cond_acc +0.020 vs anchor ST13 +0.103; d_cavg −0.056) *and* destabilised (kl_max spiked to 1.36),
  even though Gemma had real format headroom (56%→62%). → **Resolved for BOTH families: the format
  penalty (flat or gated) does NOT help; fp=0 is best.**

**(b) Run-to-run error bar — now N=3 (anchor replicate ST01 ≡ ST10 ≡ ST28, all identical fp=0 config;
same default seed=1 → variance is pure vLLM/FSDP nondeterminism at temp=1.0):**

| metric | ST01 | ST10 | ST28 | mean | **SD (n=3)** | SE |
|---|---|---|---|---|---|---|
| **d_cavg** (conditional) | 0.149 | 0.069 | −0.017 | **+0.067** | **0.083** | 0.048 |
| d_avg (raw mean acc) | 0.0211 | 0.0207 | 0.0227 | **+0.0215** | **0.0011** | 0.0006 |

**🔴 Implication — the headline d_cavg is DOMINATED by noise; raw d_avg is the reliable signal on
Qwen2.5.** The identical anchor config, run 3×, spans d_cavg **−0.017 → +0.149** (SD **0.083**, 2 SD
band **±0.17**) — that covers *almost the entire* Phase −1 Qwen2.5 spread (+0.149 → −0.115). So the
"winner" ST01 (+0.149) was a lucky draw; the anchor's *true* d_cavg ≈ **+0.07 ± 0.08** (barely > 0).
**Almost NO Qwen2.5 d_cavg ranking is significant** — even the worst runs (neg_ppl −0.115, kl0.01
−0.097) are only ~2 SD below the anchor mean (borderline). In contrast **raw d_avg is rock-stable at
+0.021 ± 0.001** and, because Qwen2.5 format is ~100% throughout (no format headroom), d_avg is **not**
format-confounded here → it is the **better-powered honest metric for this family**, showing a **small
but real +0.02 ToM gain** from BeRL. **Revised rule:** on format-saturated models, rank on **d_avg**;
treat d_cavg gaps <≈0.17 as noise. (d_cavg remains the right honest metric where format is *not*
saturated — e.g. Gemma at 56% — because there d_avg *is* format-confounded.)

**(c) Reproducibility run ST28 — DONE** (d_cavg −0.017, d_avg +0.023): folded into the N=3 error bar
above. *(Caveat: ST01 launched just before the `s2-` prefix/std_coef logging landed — same code branch,
valid replicate.)*

---

# Phase −1 (v2): non-collapsing, non-hacking stability recipe per family

**Novelty vs old stage:** anchored on **`power k=4` + std-gated format gate** — neither `k=4` nor the
std-gated gate was ever run in the old stage, so every v2 cell is a new configuration (see plan).

## Qwen2.5-3B — Phase −1 v2 (ST01–ST12) ✅ COMPLETE

**Runs:** 12/12 complete (190 steps / 1 epoch each). All ran **stable** — no KL blowup, no length
collapse, parseable≈1.0, fmt_err=0 throughout. Metrics from `analysis/reassess_v2_q25.csv`.

**⚠️ HM vs honest d_cavg disagree — rank on d_cavg.** `d_hm` is compressed into a narrow
+0.04–0.06 band for **every** run (it rewards lifting near-zero benches by format alone), whereas the
honest **`d_cavg` spans +0.149 → −0.115**. Runs that top the HM table (ST04 `power k6`, ST05 frozen,
ST09 flat-fp5) have **flat-to-negative honest ToM** — their HM gain is bottom-bench format lift, not
reasoning. This is exactly the HM bias we avoid.

### Master ranking — honest `d_cavg` (headline), all knobs = anchor unless noted
| id | axis Δ vs anchor | **d_cavg** | d_cavg_peak | d_cond_acc | d_hm | d_fmt_pass | Δmmlu | KL_f |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| **ST01** | **anchor** (pk4·actor·std-gated·kl0.05·lr5e-7·ec0) | **+0.149** | +0.149 | +0.147 | +0.043 | 0.000 | +0.138 | 0.11 |
| ST11 | ec=0.001 | +0.094 | +0.094 | +0.089 | +0.041 | +0.010 | +0.159 | 0.05 |
| ST10 | fp=0 (no penalty) | +0.069 | +0.069 | +0.057 | +0.054 | +0.031 | +0.155 | 0.10 |
| ST12 | train_batch=64 | +0.062 | +0.083 | +0.062 | +0.039 | 0.000 | +0.156 | 0.13 |
| ST07 | lr=1e-6 (2×) | +0.024 | +0.066 | +0.034 | +0.051 | +0.042 | +0.156 | 0.10 |
| ST04 | reward **power k6** | −0.035 | +0.059 | −0.039 | +0.055 | +0.031 | +0.166 | 0.06 |
| ST05 | RM **frozen** | −0.035 | +0.038 | −0.042 | +0.049 | +0.042 | +0.123 | 0.09 |
| ST09 | flat fp=5 | −0.042 | 0.000 | −0.047 | +0.046 | +0.031 | +0.133 | 0.09 |
| ST02 | reward **log_prob** | −0.059 | +0.004 | −0.057 | +0.025 | +0.042 | +0.138 | 0.28 |
| ST06 | kl=0.01 | −0.097 | 0.000 | −0.095 | +0.041 | −0.021 | +0.134 | 0.26 |
| ST03 | reward **neg_perplexity** | −0.115 | 0.000 | −0.126 | +0.014 | +0.021 | +0.126 | 0.26 |

_(ST08 kl0.01/lr1e-6 still Processing — excluded.)_

**Findings (Qwen2.5-3B), read on `d_cavg` — ⚠️ but see §0: gaps <≈0.11 are within run-to-run noise (SD 0.057):**
1. **Nominal top run = the anchor (ST01):** `power k4 · actor-RM · fp=0 · kl0.05 · lr5e-7 ·
   ec0.0`, **d_cavg +0.149** with **d_fmt_pass = 0.000** → gain is *pure reasoning*, zero format-pass
   shift. **BUT ST10 is the identical config** (fp=0 makes fp_std irrelevant) and scored only +0.069 →
   the +0.149 is one draw of a noisy metric, not a robust win (see §0).
2. **Reward: `power k4` is honest-best; higher/other rewards format-lift without ToM.** `power k6`
   (ST04) and `log_prob` (ST02) / `neg_perplexity` (ST03) are all honestly **negative** (−0.035 /
   −0.059 / −0.115) despite positive HM. k6 overcooks the exponent; the non-power rewards are the
   worst honest performers. **Keep k4** (neg_ppl/log_prob exceed ~2 SD below anchor → real).
3. **⚠️ Format penalty does NOT help Qwen2.5 — RESOLVED (see §0).** The "std-gated" anchor rows had
   fp=0 (no-op), so ST01 ≡ ST10 and their gap is noise. The two runs with an **active** penalty are
   **both honestly negative**: flat fp5 (ST09, −0.042) and the genuinely-active gate fp5/std1.5 (ST26,
   **−0.073**, ~3 SD below the fp=0 cluster). Qwen2.5 format-pass is ~100% from step 0, so any penalty
   has nothing to fix and only injects reward noise. → **Keep fp=0 for Qwen2.5.** (Gemma ST27 still
   Training — its base has real format loss so the gate may still help there.)
4. **RM: actor > frozen on honest ToM** at kl0.05 (anchor +0.149 vs frozen ST05 −0.035; gap ~0.18 >
   2 SD → plausibly real). Both stable.
5. **KL/LR: keep kl0.05 / lr5e-7; the KL cliff is between 0.03 and 0.01.** kl=0.03 (**ST25, d_cavg
   +0.118**) is indistinguishable from the kl0.05 anchor cluster (mean +0.109±0.057), while kl=0.01
   (ST06, −0.097) clearly hurts honest ToM and runs hotter (KL_f 0.26). Doubled lr=1e-6 (ST07, +0.024)
   is within noise of the anchor. → kl0.03–0.05 safe, kl0.01 is the regression.
6. **Neutral knobs (within noise of anchor):** entropy ec=0.001 (ST11, +0.094) and batch=64 (ST12,
   +0.062) — both indistinguishable from the anchor → pick batch=64 for throughput (~2× fewer steps).

**Caveat:** run-to-run **SD(d_cavg) ≈ 0.057** (§0) ⇒ only gaps >≈0.11 are trustworthy. Robust
separations: **frozen / neg_ppl −0.115 / kl0.01 −0.097 / log_prob −0.059 (bottom)** clearly below the
anchor cluster; everything in the +0.06→+0.15 band is one indistinguishable group. Peak-window
(`d_cavg_peak`) is ~0 for every negative run → no honest-gain window at all.

**→ Qwen2.5-3B locked Phase-0 config (honest):** = the **ST01 anchor** —
`power k4 · actor-RM · fp=0 (no format penalty — CONFIRMED best; active gate ST26 −0.073) · kl=0.05 · lr=5e-7 ·
ec=0.0 · batch 32(or 64)`.
**Top honest cluster (statistically tied):** ST01 (+0.149) ≈ ST11 (+0.094) ≈ ST10 (+0.069) ≈ ST12 (+0.062).

**Key runs:** ST01 anchor [`wo4edr5s`] · ST11 ec0.001 [`996mhdd9`] · ST10 fp0 [`d4u59d98`];
**honest-negative counter-examples** (HM-positive but d_cavg<0): ST04 k6 [`4sv4b4mg`], ST03 neg_ppl
[`axc03qdr`].


## Gemma-2-2B — Phase −1 v2 (ST13–ST24) 🏃 IN PROGRESS

**Runs:** ST13–ST19 **Training** (anchor + reward family + RM + KL/LR); ST20–ST24 **Not-started**
(kl0.01/lr1e-6, format-fix ×2, ec0.0, batch64). No results yet — reminder: **use `avg` not HM for
Gemma** (weak base inflates HM on format alone).

| id | axis (Δ vs Gemma anchor ST13) | status | WandB |
|---|---|---|---|
| ST13 | **anchor** power k4 · frozen · ec0.001 | Training | `nix3o6rs` |
| ST14 | log_prob | Training | `txbm9pza` |
| ST15 | neg_perplexity | Training | `og7l8pf1` |
| ST16 | power k6 | Training | `ioh4iwei` |
| ST17 | actor RM | Training | `gv7139bp` |
| ST18 | kl=0.01 | Training | `odtha91c` |
| ST19 | lr=1e-6 | Training | `91fpaz5i` |
| ST20 | kl=0.01 · lr=1e-6 | Not-started | – |
| ST21 | flat fp=5 | Not-started | – |
| ST22 | fp=0 | Not-started | – |
| ST23 | ec=0.0 | Not-started | – |
| ST24 | train_batch=64 | Not-started | – |

**Findings:** _pending run completion._ Score on **`d_cavg`** (and `d_cond_acc`), **not HM** —
Gemma's weak base makes HM especially misleading (near-zero benches lift on format alone). Watch:
(a) does actor-RM (ST17) blow KL on the weak Gemma base (v1 risk); (b) whether `power k4` still
honest-beats `log_prob`/`neg_ppl` and `k6` (as it did for Qwen2.5); (c) does the std-gated gate lift
honest `d_cavg` here (Gemma's tag-free format was the v1 invalid-gate failure point).

---

# Phase 0 (v2): training-data recipe search (P0-01…P0-14) ✅ COMPLETE

## Qwen2.5-3B — data ablation — **VERDICT: data composition is a null lever; keep `smoke_mix`.**

**Design:** all 14 rows clone the locked Phase −1 Qwen2.5 recipe (ST01 anchor: `power k4 · ll_min−6 ·
actor-RM · fp=0 · ec=0.0 · kl=0.05 · lr=5e-7 · batch32 · rollout16 · 1 epoch`, tagged `cot_eval`) and
vary **only** the data config. **Primary metric = `d_avg`** (stable; SD 0.001) because Qwen2.5 format is
~100%-saturated so d_avg is not format-confounded; `d_cavg`/`d_cond_acc` are **noise-dominated**
(SD 0.083, 2σ ±0.17) and used only as a secondary cross-check. Baseline = the free N=3 `smoke_mix`
anchor **d_avg +0.0215±0.001** (d_cavg +0.067±0.083).

**S1 — single-domain screen (1 seed each), ranked on `d_avg`:**

| id | domain | **d_avg** | d_cavg | d_cond_acc | fmt_late | kl_max | note |
|---|---|---|---|---|---|---|---|
| P0-03 | craigslist | +0.0214 | −0.038 | **−0.040** | 1.00 | 0.92 | top d_avg but **neg cond** (LL↑, acc↓) |
| P0-07 | p4g | +0.0213 | +0.072 | **+0.072** | 0.98 | **0.20** | best clean single (pos cond, low KL) |
| P0-04 | dailydialog | +0.0209 | +0.028 | +0.027 | 1.00 | 0.94 | runs hot |
| P0-01 | casino | +0.0203 | −0.011 | −0.012 | 0.99 | 0.91 | |
| P0-06 | empathetic | +0.0196 | +0.010 | +0.008 | 1.00 | 1.09 | |
| P0-02 | cga | +0.0161 | −0.083 | −0.074 | 1.00 | 0.10 | |
| P0-05 | diplomacy | +0.0071 | −0.048 | −0.018 | 1.00 | 0.10 | low d_avg |
| P0-08 | thoughttrace | −0.0001 | +0.125 | +0.097 | 0.93 | 0.07 | ToM data; d_avg~0, **fmt dropped** → d_cavg format-confounded |

**S2 — mixtures (N=3 seeds), `d_avg` mean ± SD:**

| corpus | **d_avg mean±SD** | d_cavg mean±SD | d_cond_acc mean±SD |
|---|---|---|---|
| **`smoke_mix` (baseline, ST01/10/28)** | **+0.0215 ± 0.001** | +0.067 ± 0.083 | ~+0.05 ± 0.08 |
| `mix_all` (P0-09/10/11) | +0.0224 ± 0.0035 | −0.026 ± 0.029 | −0.025 ± 0.027 |
| `mix_best3` (P0-12/13/14) | +0.0168 ± 0.0008 | +0.009 ± 0.053 | +0.007 ± 0.050 |

**Findings:**
1. **No data configuration reliably beats `smoke_mix` on the stable `d_avg` metric.** Every corpus —
   `mix_all`, `mix_best3`, and the best single domains (p4g, craigslist, dailydialog) — lands at the
   **same ~+0.02 d_avg** as the baseline. `mix_all` ties baseline (+0.0224±0.0035); `mix_best3` is
   marginally *below* (+0.0168±0.0008). The **behavior-prediction gain is essentially data-agnostic.**
2. **All conditional-metric (d_cavg/d_cond_acc) differences sit inside the N=3 noise band (SD ~0.08)** —
   they cannot separate corpora. The single apparent standouts (thoughttrace d_cavg +0.125, p4g +0.072)
   are <1.5 SD from zero, and thoughttrace's is format-confounded (fmt_pass fell to 0.93).
3. **Reward-hacking watch:** craigslist has the *top* d_avg (+0.0214) but a **negative** d_cond_acc
   (−0.040) — perplexity-avg improved while conditional accuracy fell. d_avg alone can be gamed by a
   domain; the conditional cross-check catches it. (This is why craigslist is **not** a valid pick
   despite topping d_avg.)
4. **Decision (user, 2026-07-20): keep `smoke_mix` as the fixed corpus for all later phases**
   (Q0/Q1/Q2/…). It is the simplest, is already characterised with an N=3 error bar, and no candidate
   clears it beyond noise. **S3 (turn-filter) and the S1-top2 3-seed replicates are SKIPPED** — the
   ablation is a confirmed null, matching v1's "quantity > quality / mixtures don't compose" once
   error bars are applied.

**Consistency w/ v1:** v1 Phase-0 also found no mixture synergy and a small honest gain; v2 confirms
this is a **null**, and additionally shows v1's single-seed d_cavg mixture rankings were within-noise
(as predicted by the Phase −1 error bar). **Net: v1 and v2 agree — data recipe is not a lever.**

---

# Q3 (v2): scale & interaction type (Q3-01…Q3-12) ✅ COMPLETE

**Design:** locked per-family recipe, fixed `smoke_mix` corpus (data = null lever, Phase 0), 2 seeds
each. **Metric per model = whichever is honest given format saturation:** `d_avg` where format is
~100% (3B, 7B — not format-confounded), `d_cavg`/`d_cond_acc` where format is NOT saturated (0.5B,
Gemma). D1 (domain) already answered by Phase-0 S1 (null). Reassessment CSVs: `/tmp/q3reassess/*`.

## D6 — model-scaling law → **NO positive scaling; honest gain is a 3B-family phenomenon (recipe is 3B-tuned)**

| model | fmt early→late | metric used | Δ (mean±SD, n=2) | kl_max | verdict |
|---|---|---|---|---|---|
| **0.5B** (Q3-01/02) | 0.14→0.45 / 0.18→0.31 | d_cavg (format NOT sat.) | **+0.082 ± 0.137** | 2.28 / 0.62 | **format-limited + unstable**; d_avg +0.085 is format-confounded; honest cavg inconsistent (0.29→0.51 vs 0.50→0.44) → **inconclusive** |
| **3B** (ST01/10/28) | ~1.0 (saturated) | **d_avg** | **+0.0215 ± 0.001** | ~0.1 | **clean, stable, honest small gain ✅ (the anchor)** |
| **7B** (Q3-03/04) | 0.96→1.0 (saturated) | **d_avg** | **−0.021 ± 0.005** | 1.57 / 0.33 | format saturated; honest conditional flat/down (0.53→0.49) → **NO gain (slightly negative)** |
| **Gemma-2B** (Q3-05/06) | 0.55→0.68 (NOT sat.) | d_cavg (honest) | **−0.079 ± 0.0002** | 2.63 / 2.13 | d_avg +0.10 is format-confounded; **honest d_cavg NEGATIVE + KL blew up** → unstable + honest-negative |

**🔴 Findings:**
1. **The honest BeRL gain does NOT follow a positive model-scaling law.** Only **3B** shows a clean,
   stable, format-saturated positive gain (+0.0215). **7B** (also saturated) shows **no gain**
   (−0.02, conditional accuracy flat-to-down) — the strong 7B base has little headroom and RL only
   perturbs it. **0.5B** and **Gemma-2B** are **format-unsaturated**, so their large positive **d_avg
   is format-confounded** (learned to emit parseable answers), while the honest metric is flat (0.5B,
   noisy) or **negative** (Gemma −0.079).
2. **⚠️ Confound — the recipe was locked at 3B.** KL **blew up at every non-3B scale** (0.5B 2.28,
   7B 1.57, Gemma 2.6) → the 3B-tuned `lr=5e-7 · kl=0.05 · actor-RM` (Qwen) / ST13 recipe (Gemma) is
   **mis-calibrated off-3B**. So D6 currently measures *"the 3B recipe does not transfer across scale"*
   as much as *"the effect doesn't scale."* **A clean model-scaling claim requires re-stabilising the
   recipe per scale first** (Phase −1-style mini-sweep) — this is the motivation for the next phase.
3. **Practical takeaway:** 3B is the sweet spot for the Qwen2.5 family under the current recipe; do
   **not** assume the effect grows with size, and **re-tune before** running headline scale points.

## D5 — data-scaling law → **clean saturating curve; gain plateaus by ~5–6k (smoke_mix is well-sized)**

Qwen2.5-3B, format ~100% throughout (d_avg honest). Composition-controlled ladder (same 9-source ratio):

| #examples | d_avg (mean±SD) | note |
|---|---|---|
| 1k (Q3-07/08) | **+0.0095 ± 0.002** | clearly below plateau |
| 5k (Q3-09/10) | **+0.0192 ± 0.002** | near plateau |
| **6.1k = smoke_mix** (ST01/10/28) | **+0.0215 ± 0.001** | plateau knee (baseline) |
| 10k (Q3-11/12) | **+0.0169 ± 0.0002** | plateau (KL a touch hot 1.15/1.25) |
| 26k = mix_all (P0-09/10/11) | **~+0.022** | plateau |

**Finding:** a **real, above-noise data-scaling effect** — the honest gain **rises steeply 1k→5k
then saturates by ~5–6k**; more data (10k, 26k) gives **no further gain**. `smoke_mix` (6.1k) sits
right at the knee → **well-sized; no reason to enlarge the corpus.** (Contrast with D6: data quantity
*is* a lever below ~5k, whereas model size and data composition are not.)

## Q3 summary
- **D1 domain:** null (Phase-0 S1). **D5 data-quantity:** matters below ~5k, plateaus at ~6k.
  **D6 model-size:** no positive scaling under the 3B-locked recipe (3B best; 7B flat/neg; 0.5B &
  Gemma format-confounded + KL-unstable).
- **The one actionable lever found = data quantity (need ≥5k).** Model scale and data composition are
  **not** levers; recipe stability off-3B is an open problem gating the scale story.

---

## Open TODOs
- ✅ **Done:** format-controlled `d_cavg` reassessment of the 11 completed Qwen2.5 runs
  (`analysis/reassess_v2_q25.csv`; rerun: `python scripts/reassess_runs.py --logs logs/20260719/*ST0*q25*.log logs/20260719/*ST1[012]*q25*.log --out analysis/reassess_v2_q25.csv`).
  **Result overturned the HM ranking** — see Qwen2.5 section.
- Re-run reassess including ST08 (kl0.01/lr1e-6) once it completes.
- Complete Gemma-2-2B ST13–ST24; reassess on `d_cavg` and lock the Gemma stable default.
- ✅ **Done:** Phase 0 (v2) data ablation (P0-01…14) — **null; `smoke_mix` locked** for later phases.
- ✅ **Done:** Q3 (v2) scale (Q3-01…12) — D5 data-scale saturating curve (plateau ~5–6k); D6
  model-scale no positive scaling under 3B recipe (KL blowups off-3B).
- **Next (ACTIVE, populated 2026-07-21):**
  - **Phase SS** (SS-01…09) — per-scale recipe re-stabilisation on smoke_mix (frozen RM + KL/LR tune
    for 0.5B/7B/Gemma) to fix the off-3B KL blowups **before** a clean D6 model-scaling curve.
  - **Q1-B2** (Q1-B2-01…03) — direct rule-based ToM-RL at 3B (×3 seeds) = the labeled baseline for the
    headline label-free (B1=anchor) vs labeled comparison.
