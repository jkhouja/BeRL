# BeRL Paper — Experiment Plan

**Thesis (single falsifiable claim):**
> Reinforcing *behavior prediction* — the likelihood a model assigns to the real next human
> utterance given [context + CoT] — induces Theory of Mind that (a) **transfers OOD**,
> (b) is **robust / applied** (not surface shortcuts), and (c) is **label-competitive with
> direct ToM-supervised RL**, using **no ToM labels**.

BeRL = GRPO on a reward = length-normalized log-likelihood / neg-perplexity of the held-out
human utterance (optionally baseline-subtracted), scorer = frozen base LM or the actor.

---

## Repo knobs (what exists vs. needs building)
**Exists:** `reward_type ∈ {log_prob, neg_perplexity, power}`, `power_k`/`power_ll_min`,
`subtract_baseline`, `actor_as_rm` vs frozen RM; rule-based ToM reward
(`experiments/train_tom_*.sh` + `explore_tom*.py`); `prompt_style`, `generation_prefix`
(`<think>`), ToM template tags (intents/beliefs/intent/strategy); `limit_turn`, `turn_order`.
**Needs building `[build]`:** control reward (shuffled/mismatched utterance, length/register-matched);
surprise sampling (select turns by baseline-PPL / info-asymmetry) + length-matched random control;
training-time curriculum scheduler (turn_order is only a data sort); mixed reward (behavior + rule-based, Q1-B3).

---

## Evaluation suite (define once; run on every checkpoint)
| Bucket | Benchmarks | Purpose |
|---|---|---|
| ID | Hi-ToM, ToMi, ExploreToM | in-family |
| OOD | ToMBench, OpenToM, BigToM, FANToM | contamination-safe generalization |
| Applied | SimpleToM, DialToM | explicit→applied transfer (key claim) |
| Robustness | ExploreToM-Infilled, perturbed/Ullman-style | shortcut detection |
| Dynamic | DynToM | multi-turn belief tracking |
| Probe | Benchmarking Mental State Representations | mechanism (belief reps) |
| Guardrail | MMLU, GSM8K | catastrophic-forgetting check |
| Behavior | held-out utterance PPL | did the trained objective actually improve? |

---

## Phase −1 — Setup & stability (run FIRST, before data search)
**Goal: a non-collapsing, non-reward-hacking training config** — a prerequisite for trusting
any data comparison. 

Data: fixed reasonable data sample (small mix), 1 seed.
Eval: A 50% split of each of the current evals
Do NOT run the fine `power_k`/`power_ll_min` grid here — `ll_min` is corpus-relative, so it is
deferred to Q2. Phase −1 only picks the reward **family** + stability HPs (kl, clamp, baseline).

| Permutations / stages                                                                             | Options | Picks                                    |
| ------------------------------------------------------------------------------------------------- | ------: | ---------------------------------------- |
| Model: **Qwen2.5-3B** (same family/size as the workhorse)                                         |       1 |                                          |
| P-1a. reward family @default HPs (kl, clamp, power_k=2, ll_min=p50): log_prob / neg_perplexity / power | 3 | stable family                       |
| P-1b. stability HPs on best family: 2 kl × 2 clamp/invalid-penalty × 2 baseline(on/off)           |       8 | no collapse; baseline on/off decision    |
| Total P-1 search (Qwen2.5-3B)                                                                     |      11 | best stable config                       |
| P-1c. Confirm on Gemma-2B: rerun top-2 + worst-2 configs                                          |       4 | cross-family transfer                    |
| **Phase −1 total**                                                                                |      15 | → "stable default config" (Qwen + Gemma) |

Collapse/health checks: reward not saturating at clamp, utterance-PPL improving without length
blowup or degeneration, entropy not crashing, KL controlled.

## Phase 0 — Training-data recipe search (run after Phase −1; output = default corpus)
Fixes the corpus used by A1/Q0/Q1. Uses the **stable config from Phase −1**. **Search on
Qwen2.5-3B (workhorse family), single seed.** Select on a **held-out dev set** (dev subsets of
ToMBench/BigToM/SimpleToM + held-out utterance-PPL) that is **disjoint from the final test
suite** (no recipe-overfitting). Confirm the winner at the other model class (e.g. Gemma)

Data: Always ensure first and last turns are not included.
Held-out evals: Ensure held-out utterances are either from other conversations or if from a conversation that might have been included in training, the the held-out turn happens after training.

| Stage                                                                                                                                                                                 | Runs | Picks                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---: | --------------------------------------------- |
| S1. single-domain singletons: empathetic, casino, craigslist, CGA, diplomacy, P4G, dailydialog(smalltalk baseline), Thought-tracing                                                   |    8 | domain ranking by ToM-dependence              |
| S2. greedy mixes in rank order: best-3, all                                                                                                                                           |    2 | best composition                              |
| S3. Turn filtering: ToM-dependence/surprise filter on vs off vs random sample (off = a S2 data), random: sample same size as surprise filter with same length statistics of response) |    3 | quality, vs quantity and which turns to score |
| S6. confirm chosen recipe on Gemma By running top 2 data recipes                                                                                                                      |    2 | lock A1 corpus                                |
| **Phase 0 total**                                                                                                                                                                     |   15 | → "BeRL default recipe"                       |


## Q0 — Identifiability / causal controls (the spine — front-load this)
**Reviewers will attack "accuracy up ≠ ToM." Prove the gain comes from predicting the
*correct* human utterance *via reasoning*, not any-RL-signal or lexical shortcuts.**
Model: **3B**, ≥3 seeds on A1.

| Run | Setup                                                          | Isolates                                            |
| --- | -------------------------------------------------------------- | --------------------------------------------------- |
| A0  | base, no training. Run all evals                               | reference                                           |
| A1  | **BeRL**: Run best setup of behavior prediction                | main effect                                         |
| A2  | SFT on gold utterances (no RL)                                 | RL vs SFT                                           |
| A3  | reward = likelihood of shuffled/mismatched utterance `[build]` | is it the *correct* human behavior? or any sentece? |
| A4  | no-CoT (generation_prefix off)                                 | does the CoT carry it?                              |

**Gate:** if A1 does not clearly beat A2/A3/A4, the thesis is unsupported — stop and rethink.

---

## Q1 — Generalization vs. direct ToM (headline)
Model: **3B**; repeat B1/B2 at **0.5B & 7B-1M** for a scaling curve.

| Run | Reward                            | Note                                                                           |
| --- | --------------------------------- | ------------------------------------------------------------------------------ |
| B1  | behavior (= A1)                   | **no ToM labels**                                                              |
| B2  | rule-based ToM (`train_tom_*.sh`) | direct ToM (uses labels) = TOM-RL baseline                                     |
| B3  | behavior + rule-based (combined)  | does behavior add on top of labels? Might need implementation of mixed rewards |

Headline results: OOD/Applied/Robustness vs signal type; ID–OOD gap; **label-efficiency**
(B1 uses zero ToM labels); **reverse transfer** (behavioral/applied training → explicit &
higher-order ToM) — the novel direction.

---

## Q2 — Reward-shaping refinement (after Phase 0; on the chosen corpus)
Reward *family* was fixed in **Phase −1**; here we **refine** on the locked corpus. Set
`power_ll_min` **from the chosen corpus's** utterance-LL percentiles (not arbitrary constants).
Search on Qwen2.5-3B, confirm on 7B (Qwen2.5) and Gemma-2B.

| Stage                                                                                                                 | Runs |
| --------------------------------------------------------------------------------------------------------------------- | ---: |
| A. Frozen model search: `power` grid @Qwen2.5-3B: power_k∈{1.5,2,3,4}@fixed ll_min, then ll_min∈{p25,p50,p75}@best k  |    7 |
| B: Repeat A on Actor model as reward                                                                                  |    7 |
| confirm on 7B Qwen2.5 and 2B Gemma                                                                                    |    2 |
| **Q2 total**                                                                                                          |   16 |

Metrics beyond accuracy: reward-hacking diagnostics (utterance-PPL↓ while ToM flat, length
inflation, degeneration). Global RL HPs (lr, KL β, rollout-N) were fixed in Phase −1.

---

## Q3 — Scale and Interaction type / data (reframed as a hypothesis)
**Hypothesis: ToM transfer scales with the ToM-dependence of the training signal and model size.**
Model: **3B**. 


| Run       | Variable                                                                                                                                                      | Hypothesis                              |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| D1 (×5–6) | domain: Empathetic / CaSiNo+Craigslist (negotiation) / CGA (argument) / Diplomacy (deception) / P4G (persuasion) / DailyDialog (smalltalk baseline)           | info-asymmetric domains > smalltalk     |
| D2 (×2)   | long vs short (`limit_turn`)                                                                                                                                  | context-length effect                   |
| D3 (×3)   | curriculum `turn_order`: early→late / late→early / random `[build sched]`                                                                                     | expect null → report as negative result |
| D4 (×2)   | **surprise sampling** (high baseline-PPL Turn selection / info-asymmetric turns) vs random that matches turn length distribution of suprise sampling`[build]` | ToM-dependent turns drive gains         |
| D5 (×4)   | #conversations = 1k / 5k / 10k / all                                                                                                                          | data-scaling law                        |
| D6 (x3)   | Model Scale: best hypers repeat at @3B, 7B and Gemma 2B                                                                                                       | model-scaling law                       |

---

## Q4 — Thinking style (mechanism)
Model: **3B**.

| Run     | Variable                                      |
| ------- | --------------------------------------------- |
| E1      | freeform CoT                                  |
| E2      | template ToM (emotion+belief+intent+strategy) |
| E3 (×3) | tag ablation: −emotion / −belief / −intent    |


Plus **trace analysis**: probe / LLM-judge whether CoTs contain belief content — this
*explains* the template-regression finding (likely format-tax / over-constraint).

---

## QG — Cross-family headline replication (Gemma-2B) — **core**
Reproduce the **Q0 causal identity + Q1 comparison** on a different family (not just config
transfer). Frozen scorer = Gemma base; Gemma chat template. Uses the recipe/config already
locked + Gemma-confirmed in Phase −1 / Phase 0 / Q2.

| Run          | Setup                                          | Isolates                       |
| ------------ | ---------------------------------------------- | ------------------------------ |
| A1g (×3)     | BeRL behavior (best config)                    | main effect (cross-family)     |
| A3g          | reward = shuffled/mismatched utterance `[build]` | correct-human vs any sentence  |
| A4g          | no-CoT (generation_prefix off)                 | does the CoT carry it?         |
| B2g (×2)     | rule-based ToM (uses labels)                   | vs direct ToM (cross-family)   |
| **QG total** | 7 (A1g shares 1 seed with Q3-D6-Gemma → net +6) |                                |

**Gate (cross-family):** A1g should beat A3g/A4g, mirroring the Q0 gate — otherwise the causal
claim is Qwen-specific.

---

## Cross-cutting rigor
- **Seeds:** ≥3 for headline (A0/A1, B1/B2); 1–2 for ablations. Report bootstrap CIs over eval items.
- **Search = workhorse family.** All search phases (Phase −1 stability, Phase 0 data recipe, Q2
  reward) run on **Qwen2.5-3B** — the same family/size used for the headline runs — so the locked
  config/recipe is validated on the workhorse, not a different family.
- **Sizes / scaling:** 3B workhorse everywhere; scaling curve at 0.5B & 7B-1M for Q1, and Q3-D6
  (3B / 7B / Gemma-2B).
- **Cross-family (model-independence) is woven in AND replicated:** each search phase confirms its
  choice on **Gemma-2B** (Phase −1 top-2/worst-2; Phase 0 top-2 recipes; Q2 best config), Q3-D6
  runs the best config on Gemma-2B, and **QG (core)** reproduces the full Q0 causal identity
  (A1g vs A3g/A4g) + Q1 comparison (B2g) on Gemma-2B.
- **Hygiene:** one variable per run; freeze KL/lr/rollout-N within a question; exclude first/last
  turns; held-out utterances from other conversations or strictly later turns (no leakage); verify
  Hi-ToM train does not leak into ID eval.

## Total run count (training runs; A0 is eval-only)
Seed policy: headline runs (A1, B1, B2) = 3 seeds; all others = 1 seed.
**Order: Phase −1 → Phase 0 → Q2 → Q0 → Q1 → Q3 → Q4 → QG** (stability → data → reward refinement → claims → cross-family).
All search phases run on Qwen2.5-3B; Gemma-2B confirmations are included per phase.

| Block | Runs |
|---|--:|
| Phase −1 setup/stability: 11 @Qwen2.5-3B + 4 Gemma confirm | 15 |
| Phase 0 data-recipe search: 13 @Qwen2.5-3B + 2 Gemma confirm | 15 |
| Q2 reward refinement: frozen grid (7) + actor grid (7) @3B + confirm 7B/Gemma (2) | 16 |
| Q0 identifiability (3B): A1×3, A2, A3, A4 (A0 eval-only) | 6 |
| Q1 gen vs direct: B1@0.5B, B1@7B (B1@3B=A1), B2@3B×3, B2@0.5B, B2@7B, B3@3B | 8 |
| Q3 scale/interaction (3B): D1×6, D2×2, D3×3, D4×2, D5×4, D6×3 | 20 |
| Q4 thinking style (3B): E2, E3×3 (E1=A1) | 4 |
| QG cross-family (Gemma-2B): A1g×3, A3g, A4g, B2g×2 (A1g shares 1 seed w/ D6) | 6 |
| **Grand total** | **~90** |

Compute note: Phase −1/Phase 0/Q2 search (~40 runs) is at 3B, not a smaller model — heavier than
the earlier 1.5B plan, but removes the cross-family transfer risk. Front-load Phase −1 (stop if
nothing trains stably), then Phase 0; gate downstream on the Q0 A1-vs-controls check.

## Two strengthening analyses
- **Behavior↔ToM correlation across checkpoints:** does held-out utterance-PPL improvement
  *track* ToM-benchmark improvement? Tight coupling = mechanism evidence; decoupling = shortcut finding.
- Can feeding learned CoTs from RL into another LLM improve that LLM's TOM performance? 
- **Concurrent-work positioning:** *What's On My Mind* (2025, latent-thought signal, GRPO) and
  *thought-tracing* (2025, SMC likelihood weighting) — cite as parallel; BeRL differs by
  rewarding the *observed utterance* (behavioral, label-free) vs modeling *thoughts*.

## `[build]` checklist
- [ ] data-recipe search harness: mixing/weighting, ToM-dependence filter, composition ratios
- [ ] held-out dev set for model/recipe selection (disjoint from final test suite)
- [ ] control reward: shuffled / mismatched utterance (length/register-matched)
- [ ] surprise sampling (turn selection by baseline PPL / info-asymmetry) + length-matched random control
- [ ] training-time curriculum scheduler (beyond turn_order data sort)
- [ ] mixed reward (behavior + rule-based) for Q1-B3
- [ ] trace-content probe / LLM-judge for belief content
- [ ] wire OOD scorers: ToMBench, OpenToM, BigToM, SimpleToM, DynToM (+ MMLU/GSM8K guardrail)
- [ ] Gemma-2-2B path (chat template + same-family frozen scorer; scripts exist)
