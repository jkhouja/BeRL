# BeRL Paper — Findings Log

**Purpose:** Living record of *results* per phase, mirroring the phase structure of
`BeRL_paper_plan.md`. Populated by inspecting per-run summaries (`experiments_logs/*.md`) and the
tracker `Results` column (`BeRL_experiments_tracker.md`). Companion to the plan (which holds design)
and the tracker (which holds live status).

**Last updated:** 2026-07-15 (Phase 0: S1 ✅ both arms · S2 ✅ both arms — mix_best3 wins · S3 ⛔ build-blocked).

**Reading the metrics.** Each run is scored vs **its own step-0 baseline** on the `subsample300`
eval suite. Two ToM aggregates over the 24 ToM benchmarks: **HM** (harmonic mean, min-dominated) and
**avg** (arithmetic mean). `gsm8k`/`mmlu` are reported separately as capability guardrails, never
folded into ToM. **For the weak Gemma-2 base, HM is unreliable** (near-0 floor benchmarks make HM
jump on format-compliance alone) — prefer **avg** for Gemma. A parallel **format-controlled** read
(`d_cavg` = Δ conditional accuracy `P(correct|parsed)`, from `scripts/reassess_runs.py`) strips out
"learned to emit parseable tags" gains; small/noisy (val-subset, SE≈0.05) but the honest ToM signal.
"Clean" = final KL < 1.0 AND min rollout resp_len ≥ 30 (excludes reward-hacked / collapsed runs).

---

## Phase −1 — Setup & stability  ✅ COMPLETE
**Goal:** a non-collapsing, non-reward-hacking config (reward family + stability HPs) before any data
comparison. (Rows `PS001–PS182`; full re-assessment in `notebooks/BeRL_run_reassessment.ipynb`.)

**Findings**
- **Reward family + stability:** clean per-family winners share **kl=0.05, lr=5e-7**. Raw-HM
  ranking is **format-confounded** — the biggest raw gains (e.g. Gemma actor-k7 PS129, dHM +0.344)
  are **reward-hacks** (KL→9.85, rollout collapses to ~1.5 tokens) with real eval gains that are
  *not adoptable*. **actor-RM blew up KL in 12/72 runs vs 0/80 frozen.**
- **Format penalty (`fp`):** format-controlled (`d_cavg`) ranking favors a **training format penalty
  `fp=5`** — it suppresses format-only gains and is the most defensible genuine-reasoning signal.
- **Cross-family (memory):** a winning recipe does **not** generalize across families; each family
  needs its own recipe (Qwen2.5 ≠ Qwen3 ≠ Gemma-2).
- **Known Gemma tag-free gate bug:** the invalid-detector hard-requires a literal `</think>`, which
  the Gemma tag-free native-thinking recipe never emits → some `log_prob`/`power` Gemma sweep cells
  100% INVALID (zero gradient). Isolated; the Phase-0 Gemma **power** arm runs fine. (3 stale cells
  PS099/103/106 skipped=Failed.)

**→ Locked "stable default config" carried into Phase 0** (selected on format-controlled `d_cavg`):
- **Qwen2.5-3B:** `power · k=5 · ll_min=−6 · actor-RM · fp=5 · ec=0.0 · kl=0.05 · lr=5e-7`
- **Gemma-2-2B:** `power · k=5 · ll_min=−4 · frozen-RM · fp=5 · ec=0.001 · kl=0.05 · lr=5e-7`

---

## Phase 0 — Training-data recipe search  ◀ IN PROGRESS (S1 ✅ · S2 ✅ mix_best3 wins · S3 ⛔ build-blocked)
Fixes the corpus for A1/Q0/Q1. Search on Qwen2.5-3B (workhorse), confirm on Gemma-2. Both arms run
the same recipes with their locked family config.

### S1 — single-domain singletons  ✅ COMPLETE (both arms)

**Qwen2.5-3B** (metric: ToM HM/avg pp gain vs base ~0.417/0.503; all clean unless noted):

| Domain | Row | Verdict | ToM HM Δ | ToM avg Δ | gsm8k/mmlu Δ | Notes |
|---|---|---|---|---|---|---|
| casino | E017 | **STRONG+** | +5.5pp | +2.5pp | +3.3/+16.8 | Best raw + best format-controlled (d_cavg +0.071), but short responses (resp_len 26.6) → flagged non-clean |
| p4g | E021 | **STRONG+** | +5.45pp | +2.5pp | +3.0/+14.0 | Fast rise, holds; d_cavg −0.084 (format-confounded) |
| craigslist | E018 | POS | +4.9pp | +2.0pp | +6.2/+17.7 | Rock-stable full epoch; Qwen fp5 immune to Gemma fp-collapse |
| cga | E019 | POS | +4.8pp | +1.8pp | +0.0/+15.0 | Smooth, no collapse |
| dailydialog | E022 | POS | +4.8pp | +2.4pp | +9.4/+18.0 | On par w/ diplomacy → **smalltalk-baseline hypothesis NOT supported** |
| diplomacy | E020 | POS | +5.0pp | +1.4pp | +5.6/+15.7 | Best d_cavg (0.000, neutral); only 38 steps |
| empathetic | E016 | POS | +12.1pp | +4.4pp | +7.6/+35.9 | First clearly-positive P0 cell |
| thoughttrace | E023 | **DEGENERATE** | — | — | — | All targets below ll_min=−6 floor → 0 gradient (`Awaiting-input`, excluded) |

**Gemma-2-2B** (metric: **avg** — HM unreliable for Gemma; base avg ~0.315):

| Domain | Row | Verdict | ToM avg Δ | gsm8k/mmlu Δ | Notes |
|---|---|---|---|---|---|
| p4g | E098 | **BEST+** | +12.8pp | +11.5/+8.7 | Best Gemma single; clean full epoch |
| casino | E094 | POS-but-UNSTABLE | +11.9pp | +16.2/+8.6 | **Late KL-blowup→~10 + length→512 cap** from step~362; last-5 contaminated → **excluded (unclean)** |
| craigslist | E095 | STRONG+ | +10.2pp | +11.0/+10.4 | No fp5 collapse; avg robust |
| dailydialog | E099 | POS | +6.9pp | +12.3/+4.4 | Stable full epoch (1723 steps); positive d_cavg (+0.005); instability is dataset-dependent |
| empathetic | E093 | POS | +14.7pp | −0.6/+2.7 | Cross-family confirm of E016; best positive d_cavg (+0.013) |
| diplomacy | E097 | NEUTRAL | +1.1pp | +0.8/+0.9 | Underpowered: only 612 rows→19 steps; needs upsampling |
| cga | E096 | **NEGATIVE** | −1.6pp | +5.3/−0.6 | Does not transfer on weak Gemma base (not a training pathology) |
| thoughttrace | E100 | **NEGATIVE** | −17.0pp | +9.0/−7.1 | Poor source domain for Gemma |

**S1 cross-cutting findings**
- **Nearly every dialogue domain transfers positively to ToM with no ToM labels** (the core BeRL
  claim) — strongest on Qwen2.5. Magnitudes are modest (~+2–5pp avg) on Qwen, larger but noisier on
  Gemma.
- **Domain ranking is family-dependent** and partly reverses: casino/p4g top both families;
  **cga is positive on Qwen but negative on Gemma**; thoughttrace is bad on both.
- **Smalltalk-baseline (dailydialog) is NOT a null** — it transfers on par with structured-ToM
  domains, weakening the "ToM-dependence of the corpus drives the gain" story (revisit in S3 filter).
- **Format vs reasoning caveat:** raw ToM gains are partly format-compliance. Format-controlled
  `d_cavg` is small/noisy; casino (Qwen) is the only domain with a clearly positive d_cavg, and it's
  the one that reward-hacks on Gemma — so genuine, robust, format-controlled ToM improvement is
  **not yet strongly proven** and is the key open measurement question.
- **Reward-hacking is dataset-dependent, not just config-dependent:** identical Gemma config is
  stable on dailydialog/craigslist but KL-blows-up on casino at full-epoch length.

### S2 — greedy mixes (best-3, all)  ✅ COMPLETE (both arms, 686/686)

Best-3 chosen on `d_avg` (primary) + `d_cavg` (cross-check), **clean runs only**:
- **Qwen2.5:** casino + empathetic + dailydialog  (`dcfg_mix_best3`)
- **Gemma:** craigslist + dailydialog + empathetic  (casino **excluded** — KL exploded 9.65)  (`dcfg_mix_best3_gemma`)

| Mix | Row | Status | Verdict |
|---|---|---|---|
| Qwen mix_all | E025 | Completed | (results not yet logged to tracker) |
| Qwen mix_best3 | E024 | ✅ Completed | **+5.3pp HM / +2.3pp avg** — MATCHES but does **not beat** casino single (E017). **No mixture synergy**; casino alone captures the gain. |
| Gemma mix_all | E101 | Completed | **POSITIVE — ~doubles ToM HM** (+8.0pp HM / +9.7pp avg); no capability regression. Caveat: peaked ~step620 then declined; final KL 6.9 (high, no collapse). |
| Gemma mix_best3 | E102 | ✅ Completed (686/686) | **best3 WINS Gemma arm** — d_avg **+0.106** (> mix_all +0.099) and **STABLE (kl 0.117)** vs mix_all's KL-blowup **6.9**. d_cavg −0.035 (format-controlled ≈ null). |

**Head-to-head reassessment (independent, `scripts/reassess_runs.py` on the 686/686 logs):**

| Arm | mix | d_avg | d_cavg | d_cond_acc | kl_final | resp_len_min |
|---|---|---|---|---|---|---|
| Qwen | mix_all (E025) | 0.024 | 0.053 | 0.046 | 0.107 | 38.9 |
| Qwen | **mix_best3 (E024)** | 0.023 | 0.046 | 0.046 | **0.055** | 45.8 |
| Gemma | mix_all (E101) | 0.099 | 0.030 | −0.057 | **6.9 ⚠** | 47.6 |
| Gemma | **mix_best3 (E102)** | **0.106** | −0.035 | −0.080 | **0.117** | 85.2 |

**S2 decision — `mix_best3` is the winning mix for BOTH arms → `dcfg_mix_best := dcfg_mix_best3{,_gemma}`.**
- **Qwen:** best3 ≈ mix_all on ToM gain (d_avg/d_cavg/cond_acc all within noise) but **cleaner** (kl 0.055 vs 0.107, longer responses) → **no mixture synergy**; best3 is the more principled, healthier pick.
- **Gemma:** best3 is **strictly better** — higher d_avg AND stable, whereas `mix_all` KL-exploded to 6.9 (a reward-hack / late-collapse), so mix_all's headline HM gain is not trustworthy.
- **Format caveat unchanged:** format-controlled gain (d_cavg) is ≈0 on Gemma and small-positive (+0.046) on Qwen — the honest ToM effect remains modest.

### S3 — turn filtering (surprise / off / random-length)  ⛔ BLOCKED ON `[build]`
Rows E026 (`filter_surprise`), E027 (`filter_off` = `dcfg_mix_best`), E028 (`filter_randlen`) — Backlog.
**The surprise-filter data feature is NOT yet implemented** (no `surprise`/`baseline_ppl`/`turn_filter`
knobs anywhere in `build_dataset.py`/`scripts/`). S3 needs a new data-gen stage: (1) score each
candidate human turn by frozen-base-LM PPL / info-asymmetry, (2) select high-surprisal turns,
(3) build a length-matched random control. This is a methodology-defining `[build]` item, not a
config edit — **cannot be auto-advanced by the monitor**; needs implementation + validation.
`filter_off` (E027) is launchable now (it is just `dcfg_mix_best` = the S2 winner), but is only
meaningful as the control arm once surprise/randlen exist.

### S6 — confirm chosen recipe on Gemma (top-2)  ⬜ PENDING
Rows E029/E030 — Backlog; locks the A1 corpus once the winner is chosen.

**Phase 0 provisional bottom line:** BeRL's central claim (label-free behavior-prediction reward
induces positive ToM transfer) **holds broadly across dialogue domains and both families**, but
(1) gains are modest and partly format-driven, (2) mixtures show no synergy over the best single
domain on Qwen, and (3) the honest, format-controlled effect size is still to be nailed down. Final
"BeRL default recipe" not yet locked (awaiting S2 head-to-head + S3 + Gemma confirm).

---

## Q0–QG — later phases  ⬜ NOT STARTED
Q0 (identifiability/causal controls), Q1 (generalization vs direct ToM — headline), Q2 (reward-shaping
refinement incl. the deferred `power_k`/`ll_min` grid), Q3 (scale/interaction), Q4 (thinking style),
QG (Gemma cross-family headline). All gated on the Phase-0 default recipe; no runs yet.

---

## Open questions / watch-items
1. **Format vs. reasoning:** need a full-eval format-pass logged in-training to measure `d_cavg` at
   low noise; current subset SE (~0.05) is too large to prove the genuine ToM effect.
2. **Gemma late instability:** KL-drift/length-inflation at full-epoch on some datasets (casino,
   mix_all). The **`TOTAL_EPOCHS` default is now 1** (commit `8629133`) which both speeds iteration
   and may sidestep the late-epoch blowup — watch whether 1-epoch Gemma runs stay clean.
3. **Smalltalk not null (S1):** dailydialog transfers as well as structured-ToM domains — S3 surprise
   filter is the designed test of whether turn *quality* (not just dialogue exposure) drives ToM.
4. **No mixture synergy (Qwen S2):** if confirmed after Gemma E102, the "best recipe" may be a single
   strong domain (casino/p4g) rather than a broad mix.
