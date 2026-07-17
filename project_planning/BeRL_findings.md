# BeRL Paper — Findings Log

**Purpose:** Living record of *results* per phase, mirroring the phase structure of
`BeRL_paper_plan.md`. Populated by inspecting per-run summaries (`experiments_logs/*.md`), the
tracker `Results` column (`BeRL_experiments_tracker.md`), and the independent format-controlled
re-assessment `analysis/reassess.csv` (`scripts/reassess_runs.py`). Companion to the plan (design)
and the tracker (live status).

**Last updated:** 2026-07-17 — Phase −1 ✅ · Phase 0: S1 ✅ · S2 ✅ (mix_best3) · S3 Qwen ✅ / Gemma 3-4
(**surprisal filtering gives no benefit; recipe = mix_best3, no filter**) · format-controlled
(`d_cavg`) cross-run analysis added.

**Each section below is organised as:** **RQ** (the question) · **Runs** (what was kicked off +
status) · **Findings** · **Key runs** (WandB links to track). WandB project = `jkhouja-oxford/TOM_EXP`.

---

## Metric legend (read first)
Every run is scored vs **its own step-0 baseline** on the `subsample300` eval suite (25 subtypes).
- **HM / avg** — harmonic / arithmetic mean over the 24 ToM benchmarks (raw accuracy). `gsm8k`/`mmlu`
  are capability guardrails, never folded into ToM. **For the weak Gemma-2 base HM is unreliable**
  (near-0 floor benchmarks make HM jump on format alone) → prefer **avg** for Gemma.
- **d_hm / d_avg** — raw ToM gain (last-3 window − step 0). **Format-confounded.**
- **d_cavg** — Δ of *arithmetic-mean* `P(correct | parseable)` across benchmarks, late window vs
  early. The **format-controlled** honest ToM signal (strips "learned to emit a parseable answer").
- **d_cavg_peak** — same, but at the **best** k-step training window (not end-of-training). Captures
  peak honest ToM before any late collapse. ⚠️ It is a **max over ~20 windows → upward-biased**; use
  as an optimistic upper bound, and always read next to `kl_final`/`resp_len_min`.
- **d_cond_acc** — *pooled* (sample-weighted) `P(correct|parsed)` gain; more conservative than d_cavg.
- **Noise:** all conditional metrics come from the 24-sample/step val-debug blocks → **SE ≈ 0.05**.
  Direction is trustworthy; magnitude is an estimate.
- **"Clean"** = final KL < 1.0 AND min rollout resp_len ≥ 30 (excludes reward-hacked / collapsed runs).

---

## Phase −1 — Setup & stability  ✅ COMPLETE

**RQ.** What reward family + stability HPs give a non-collapsing, non-reward-hacking config per model
family, *before* any data comparison — and once format is controlled for, does ToM improve at all?

**Runs.** Rows `PS001–PS182` (~182 sweep cells across Qwen2.5-3B, Qwen3-1.7B, Gemma-2-2B; reward ∈
{log_prob, power k3/k5/k7}, RM ∈ {frozen, actor}, kl, lr, fp, ec grid). All Completed/assessed;
full re-assessment in `analysis/reassess.csv` (171 scored logs) + `notebooks/BeRL_run_reassessment.ipynb`.

**Findings.**
- **Reward family + stability:** clean per-family winners share **kl=0.05, lr=5e-7, fp=5**. Raw-HM
  ranking is **format-confounded** — the biggest raw gains are **reward-hacks**. Canonical example:
  Gemma **actor-k7** (PS129-class) shows `d_hm +0.344` yet is a hack — KL→9.85, rollouts collapse to
  ~1.5 tokens; format-controlled `d_cavg` is only **+0.004** (~82% of the raw gain is format-pass, not
  reasoning). **actor-RM blew up KL in 12/72 runs vs 0/80 frozen.**
- **Format penalty (`fp`):** format-controlled ranking favors a **training `fp=5`** — suppresses
  format-only gains, most defensible genuine-reasoning signal.
- **Cross-family:** a winning recipe does **not** generalize across families; each needs its own recipe.
- **Known Gemma tag-free gate bug (fixed `8769467`):** the invalid-detector hard-required a literal
  `</think>` the Gemma tag-free recipe never emits → some cells 100% INVALID (zero gradient). Fixed;
  Phase-0 Gemma **power** arm runs fine.
- **★ Does format-controlled ToM actually improve? (d_cavg / d_cavg_peak over 171 runs.)** YES, but the
  honest effect is **modest, noisy, and concentrated in the kl0.05/lr5e-7/fp5 recipe.** On the
  **unbiased late** metric only a **minority** of clean runs clear the SE≈0.05 floor: Qwen2.5 **4/89**,
  Qwen3 **7/36**, Gemma-2 **6/24**. (`d_cavg_peak>0.05` hits many more — 11/89, 20/36, 17/24 — but that
  is max-selection bias, not evidence.) The **trustworthy positives** (clean AND late>+0.05 AND
  late≈peak, i.e. a *stable* gain): 3/89 Qwen2.5, 6/36 Qwen3, 4/24 Gemma-2. Best per family:
  - **Gemma-2 `rmf·kl0.05·lr5e-7·fp5`: d_cavg +0.148, cond_acc +0.091, KL 0.08, rlmin 84, late==peak**
    — the single most convincing honest ToM gain (clean, stable, strong on *both* cavg and pooled
    cond_acc; even d_hm +0.075). Matches the locked default.
  - **Qwen3 `rma·kl0.05·lr5e-7·fp5`: d_cavg +0.118** (but pooled cond_acc only +0.014 — cavg rides a
    few easy benchmarks; weaker than it looks).
  - **Qwen2.5 `rmf·kl0.05·lr5e-7·fp5`: d_cavg +0.097, cond_acc +0.093, KL 0.22** — modest but real.
  Bottom line: **BeRL's honest signal is real (not purely format) but small (+0.07–0.15 conditional
  accuracy)**; the winners across all three families converge on the locked stable recipe.

**→ Locked "stable default config" carried into Phase 0** (selected on format-controlled `d_cavg`):
- **Qwen2.5-3B:** `power · k=5 · ll_min=−6 · actor-RM · fp=5 · ec=0.0 · kl=0.05 · lr=5e-7`
- **Gemma-2-2B:** `power · k=5 · ll_min=−4 · frozen-RM · fp=5 · ec=0.001 · kl=0.05 · lr=5e-7`

**Key runs.**
- Reward-hack exemplar (raw looks great, honest ~0): Gemma actor-k7 — https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/dhdtyhev (`d_hm +0.344`, `d_cavg +0.004`, KL 9.85).
- Best honest gain / Gemma default: https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/5re3xzap (`rmf·kl0.05·lr5e-7·fp5`, `d_cavg +0.148`).
- Best Qwen3 honest: https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/omduwrua (`rma·kl0.05·lr5e-7·fp5`, `d_cavg +0.118`).
- Best Qwen2.5 honest: https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/5ikvh817 (`rmf·kl0.05·lr5e-7·fp5`, `d_cavg +0.097`).
- Reproduce leaderboard: `python scripts/reassess_runs.py --rank d_cavg` (or `--rank d_cavg_peak`).

---

## Phase 0 — Training-data recipe search  ◀ IN PROGRESS
Fixes the corpus for A1/Q0/Q1. Search on Qwen2.5-3B (workhorse), confirm on Gemma-2. Both arms run
the same recipes with their locked family config. Stages: S1 (singletons) → S2 (mixes) → S3 (turn
filtering) → S6 (Gemma confirm).

### S1 — single-domain singletons  ✅ COMPLETE (both arms)

**RQ.** Which single dialogue domain transfers best to ToM (label-free), and does *any* domain
transfer at all?

**Runs.** 8 domains × 2 families = 16 rows. Qwen2.5 E016–E023; Gemma-2 E093–E100. All Completed
(thoughttrace Qwen E023 excluded — degenerate, below reward floor).

**Findings.** Qwen2.5 (base HM/avg ~0.417/0.503) and Gemma-2 (base avg ~0.315):

| Domain | Qwen row | Qwen ToM HM Δ | Gemma row | Gemma ToM avg Δ | Note |
|---|---|---|---|---|---|
| empathetic | E016 | **+12.1pp** | E093 | **+14.7pp** | clearest positive both families; best Gemma d_cavg +0.013 |
| casino | E017 | **+5.5pp** | E094 | +11.9pp ⚠ | best Qwen d_cavg +0.071 but resp_len 26.6 (non-clean); Gemma late KL-blowup→excluded |
| p4g | E021 | +5.45pp | E098 | **+12.8pp** | best Gemma single (clean); Qwen d_cavg −0.084 (format-confounded) |
| craigslist | E018 | +4.9pp | E095 | +10.2pp | rock-stable; no fp5 collapse |
| dailydialog | E022 | +4.8pp | E099 | +6.9pp | **smalltalk NOT a null** — on par with structured-ToM domains |
| cga | E019 | +4.8pp | E096 | **−1.6pp** | positive on Qwen, **negative on Gemma** |
| diplomacy | E020 | +5.0pp | E097 | +1.1pp (underpowered) | Gemma only 612 rows→19 steps |
| thoughttrace | E023 | degenerate (excl) | E100 | **−17.0pp** | poor source domain |

- **Nearly every dialogue domain transfers positively with no ToM labels** (core BeRL claim) —
  strongest on Qwen2.5; larger but noisier on Gemma.
- **Domain ranking is family-dependent** and partly reverses (cga +Qwen/−Gemma; thoughttrace bad on both).
- **Smalltalk-baseline (dailydialog) is NOT a null** (revisited & resolved in S3).
- **Format caveat:** raw gains partly format-compliance; casino (Qwen) the only clearly-positive
  d_cavg single, and it reward-hacks on Gemma → genuine format-controlled improvement not proven at S1.
- **Reward-hacking is dataset-dependent:** identical Gemma config is stable on dailydialog/craigslist
  but KL-blows-up on casino at full-epoch length.

**Key runs.** Qwen empathetic E016, casino E017; Gemma empathetic E093, p4g E098 (search WandB by run
name `data-recipe-P0[g]_single_*`; full links in the tracker rows).

### S2 — greedy mixes (best-3, all)  ✅ COMPLETE (both arms)

**RQ.** Does mixing the best single domains beat the best singleton (is there mixture synergy), and
which mix is the corpus for downstream phases?

**Runs.** 4 rows: Qwen mix_all E025 / mix_best3 E024; Gemma mix_all E101 / mix_best3 E102. All Completed.
Best-3 chosen on `d_avg` (primary) + `d_cavg` (cross-check), clean runs only:
Qwen = casino+empathetic+dailydialog (`dcfg_mix_best3`); Gemma = craigslist+dailydialog+empathetic
(casino excluded — KL exploded 9.65) (`dcfg_mix_best3_gemma`).

**Findings.** Independent reassessment (`scripts/reassess_runs.py`):

| Arm | mix | d_avg | d_cavg | d_cond_acc | kl_final | resp_len_min |
|---|---|---|---|---|---|---|
| Qwen | mix_all (E025) | 0.024 | 0.053 | 0.046 | 0.107 | 38.9 |
| Qwen | **mix_best3 (E024)** | 0.023 | 0.046 | 0.046 | **0.055** | 45.8 |
| Gemma | mix_all (E101) | 0.099 | 0.030 | −0.057 | **6.9 ⚠** | 47.6 |
| Gemma | **mix_best3 (E102)** | **0.106** | −0.035 | −0.080 | **0.117** | 85.2 |

- **`mix_best3` wins BOTH arms → `dcfg_mix_best := dcfg_mix_best3{,_gemma}`.**
- **Qwen:** best3 ≈ mix_all on ToM gain but **cleaner** (kl 0.055 vs 0.107) → **no mixture synergy**;
  best3 also ≈ casino single (E017) → casino alone captures the gain.
- **Gemma:** best3 **strictly better** — higher d_avg AND stable, vs mix_all's KL-blowup 6.9 (its
  headline HM gain is a late-collapse artifact, not trustworthy).
- **Format caveat:** d_cavg ≈0 on Gemma, small-positive (+0.046) on Qwen — honest effect stays modest.

**Key runs.** Gemma mix_best3 **E102** — https://wandb.ai/jkhouja-oxford/TOM_EXP (d_avg +0.106, stable);
Qwen mix_best3 **E024**; Gemma mix_all **E101** (KL-blowup cautionary).

### S3 — turn filtering (surprise / off / randlen / predictable)  ◀ Qwen ✅ · Gemma 3-4 (surprise rerunning)

**RQ.** Does selecting the most *ToM-dependent* (highest-surprisal, lowest scorer `answer_pp`) human
turns beat using the whole corpus (quantity) or random selection — i.e. is it turn *quality* or
*exposure* that drives transfer? (Direct test of the S1 "smalltalk-not-null" puzzle.)

**Runs.** 4 modes × 2 families = 8 rows, all 1-epoch, extending the S2 winner `dcfg_mix_best3`,
keeping 50% of turns except `off` (full mix). Locked recipes (Qwen actor·k5·ll_min−6; Gemma
frozen·k5·ll_min−4). **Status: Qwen E026/E027/E028/E106 all Completed; Gemma E104 off / E105 randlen /
E107 predictable Completed; E103 surprise Training** (ll_min−6 rerun after a knob-bracketing saga).

**Findings.**

*Qwen2.5 (HM reliable, base 0.416):*
| Filter | Row | ToM HM Δ | Verdict |
|---|---|---|---|
| predictable | E106 | **+5.4pp** | strongest |
| off (full mix) | E027 | +5.0pp | baseline |
| randlen | E028 | +4.8pp | ≈ off |
| surprise | E026 | +4.1pp | **weakest** |

*Gemma-2 (avg, base ~0.315):*
| Filter | Row | ToM avg Δ (last3/5) | Verdict |
|---|---|---|---|
| off (full mix) | E104 | **+0.059/+0.064** (+18.8%) | **strong** |
| predictable | E107 | +0.052/+0.031 | modest+ |
| randlen | E105 | +0.024/+0.013 | modest+ |
| surprise | E103 | pending (see below) | — |

- **Surprisal-SELECTION does NOT help — it slightly HURTS.** Qwen ordering: predictable ≥ off ≥
  randlen > surprise. Gemma: off (full 11k-turn mix) far the strongest, well above halved sets.
  Both arms agree → **ToM gain is driven by corpus exposure / quantity, not turn surprisal.**
- **Resolves the S1 "smalltalk-not-null" puzzle:** turn quality is not the driver, so dailydialog
  transferring as well as structured-ToM domains is expected.
- **Gemma-surprise (E103) — knob-BRACKETED reward-shaping problem (open).** Highest-surprisal turns
  have very negative frozen-Gemma LL, so the power reward is knife-edge in `ll_min`: **−4 floors all
  rollouts at 0** (no gradient; NOT the old −45 bug — fixed by `8769467`); **−8 saturates the +40 clamp
  → brevity-hack collapse** (resp_len 108→15, KL 0.02→0.70). An intermediate **`ll_min=−6` rerun** is
  in progress (healthy pre-crash: resp_len ~112, score 0.17, nonzero advantages — escaped both failure
  modes), but note it **breaks knob-parity** with E104/E105/E107 (they stay −4), so it's a sensitivity
  point, not a strict filter comparison. If it fails, report Gemma-surprise as a **documented
  reward-shaping null**.
- **→ Phase-0 data recipe stays `dcfg_mix_best3{,_gemma}` with NO turn filter (`off`).**

**Key runs.**
- Qwen predictable E106 (best) — https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/hwld1qqz
- Qwen surprise E026 (weakest) — https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/ui2xm36n
- Qwen off E027 (baseline, the recipe we keep) — https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/x3xz6htr
- Gemma off E104 (strong) — https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/4d3po0yz
- Gemma surprise E103 (ll_min−6 rerun, watch) — https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/aa64z8gx

### S6 — confirm chosen recipe on Gemma (top-2)  ⬜ PENDING

**RQ.** Does the chosen Phase-0 recipe replicate on Gemma-2 (locks the A1 corpus)?
**Runs.** Rows E029/E030 — Backlog; unblocked once the S3 winner is finalised.
**Findings.** — (not yet run). **Key runs.** — (TBD).

**Phase 0 bottom line.** BeRL's central claim (label-free behavior-prediction reward → positive ToM
transfer) **holds broadly across domains and both families**, but (1) gains are modest and partly
format-driven, (2) no mixture synergy over the best single domain on Qwen, (3) turn-surprisal
filtering gives no benefit (quantity > quality), and (4) the honest format-controlled effect is small
(+0.07–0.15 cond-acc) and concentrated in the locked kl0.05/lr5e-7/fp5 recipe. Data recipe =
`dcfg_mix_best3{,_gemma}`, no filter; final lock awaits Gemma-surprise resolution + S6 confirm.

---

## Q0–QG — later phases  ⬜ NOT STARTED

**RQ.** Q0 identifiability/causal controls · Q1 generalization vs direct ToM (headline) · Q2
reward-shaping refinement (deferred `power_k`/`ll_min` grid) · Q3 scale/interaction · Q4 thinking
style · QG Gemma cross-family headline.
**Runs.** None yet — all gated on the Phase-0 default recipe.
**Findings / Key runs.** — (TBD).

---

## Open questions / watch-items
1. **Format vs. reasoning (partly answered):** the honest `d_cavg` effect is real but small and near
   the SE≈0.05 val-subset floor. A full-eval in-training format-pass would tighten it; the subset
   estimate can't prove large genuine gains. `d_cavg_peak` helps spot best-in-training but is
   upward-biased.
2. **Gemma late instability:** KL-drift/length-inflation at full-epoch on some datasets (casino,
   mix_all). `TOTAL_EPOCHS` default now 1 (commit `8629133`) — watch whether 1-epoch Gemma stays clean.
3. **Smalltalk not null — RESOLVED (S3):** surprisal is not the driver; corpus quantity/exposure is.
4. **No mixture synergy (Qwen S2):** the "best recipe" may be a single strong domain (casino/p4g)
   rather than a broad mix — carry into Q-phase ablations.
5. **Gemma-surprise reward floor (S3):** selecting lowest-probability turns pushes targets below any
   usable power reward floor; `ll_min` is knife-edge (−4 floors, −8 hacks, −6 in test).
