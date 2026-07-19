# E028 — data-recipe-P0_filter_randlen (dcfg_mix_best_randlen)

- **RUN_NAME_BASE:** data-recipe-P0_filter_randlen-dcfg_mix_best_randlen
- **Exp #:** E028 · **Exp ID:** data-recipe-P0_filter_randlen
- **Attempts:** r1 (authoritative)
- **Model:** Qwen2.5-3B-Instruct · **Owner_host:** h100-013-002
- **WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/52uf3h0i
- **Log path:** logs/20260715/E028-P0_filter_randlen-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1.log
- **Driver out:** logs/20260715/E028_driver_r1.out
- **Start:** 2026-07-15 ~23:28 UTC

## Hypothesis / question
Phase-0 S3 turn-filtering ablation — **random length-matched control**. Keeps 50% of the best-mix
corpus (casino+empathetic+dailydialog) drawn at RANDOM but matched to the surprise set's
response-length distribution (NOT surprisal-selected). This is the quantity/length-controlled
comparison for the `surprise` arm (E026): if `surprise` (high-baseline-PPL / ToM-dependent turns)
beats `randlen`, the gain is from *selecting info-asymmetric turns*, not merely from using fewer /
shorter turns. Expected: randlen ≈ full-mix baseline (E024), below `surprise` if surprise selection
matters.

## Implementation details
- **Data:** dcfg_mix_best_randlen (5500/11000 rows kept, keep_fraction=0.5, seed=42, length-matched
  to surprise set). Built via `python build_dataset.py --config scripts/configs/dcfg_mix_best_randlen.yaml`.
  answer_pp: 0 nulls. Sources: dailydialog 2115 / empathetic 2101 / casino 1284. Filter report:
  surprisal(-log_prob) kept mean=5.866 vs corpus 5.302; resp-words mean kept=9.9 / corpus 14.6.
- **Locked Qwen2.5 recipe:** reward=power k=5 ll_min=-6, actor-RM, baseline=False, kl=0.05, lr=5e-7,
  fp=5, ec=0.0, rollout_n=16, max_ctx=2048/512, require_answer_tags=True, **total_epochs=2**.
- Batch=32, mini_batch=128, val_suite=subsample300. ~5500 rows / 32 * 2 epochs ≈ 344 steps.

### Exact launch command
```
setsid bash -c 'conda activate tom; \
EXP_ID=E028 DATA_NAME=P0_filter_randlen DATA_TRAIN=data/dcfg_mix_best_randlen.parquet \
REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-6 USE_ACTOR_AS_RM=True SUBTRACT_BASELINE=False \
KL=0.05 LR=5e-7 FORMAT_PENALTY=5 ENTROPY_COEFF=0.0 TOTAL_EPOCHS=2 MAX_PROMPT=2048 MAX_RESP=512 \
bash experiments/train_behavior_qwen2.5.sh' > logs/20260715/E028_driver_r1.out 2>&1 < /dev/null &
```

## Debugging / issues
- Launcher (train_behavior_qwen2.5.sh via lib/common.sh) defaults TOTAL_EPOCHS=1 — MUST export
  TOTAL_EPOCHS=2 for Phase-0. Verified in driver out: total_epochs=2, max_response_length=512.

## Findings
(pending — run in progress)

## How to rerun
See exact launch command above; rebuild data first if `data/dcfg_mix_best_randlen.parquet` missing.

## FINAL FINDINGS (E028, r1) — 2026-07-16

**Verdict: POSITIVE control. Random 50% length-matched ≈ full-mix (barely below).**

Scorer (`scripts/score_run.py --last 5`, 24 ToM benchmarks excl gsm8k/mmlu):
- ToM HM(last5)=**0.4663** vs base(step0)=0.4183 → **+4.8pp**
- ToM avg(last5)=**0.524** vs base=0.5038 → **+2.0pp**
- mmlu=0.6166 vs step0 0.473 → **+14.4pp**
- gsm8k=0.667 vs 0.66 → +0.7pp (neutral)
- Health(final): kl=0.061, entropy=1.277, resp_len=97.0, reward=39.46, parseable=1.0, max_resp=512

**Comparison vs E024 full-mix** (11000 rows, HM +5.3pp / avg +2.3pp / mmlu +17.7pp):
- randlen (5500 rows, random 50% length-matched) ≈ full-mix, marginally below (HM +4.8 vs +5.3,
  avg +2.0 vs +2.3, mmlu +14.4 vs +17.7). **Halving the corpus via a random length-matched draw
  barely hurts** — data quantity is not the binding constraint at this scale.

**Decisive comparison (surprise SELECTION effect) — PENDING E026.** This randlen run is the
quantity/length-controlled CONTROL for the `surprise` arm (E026, still Processing by another agent
at time of writing). The S3 conclusion (does selecting high-baseline-PPL / ToM-dependent turns beat
a random length-matched draw?) requires E026's score:
  - If E026 surprise > E028 randlen → surprise SELECTION matters (info-asymmetric turns are the
    active ingredient).
  - If E026 surprise ≈ E028 randlen → the gain is not from surprisal selection per se.
(Log this delta once E026 completes.)

**Health:** fully stable — resp_len flat ~95-107 (no fp5 collapse), kl bounded ~0.05-0.10,
format_error=0, no crash. fp5+actor-RM well-tolerated on Qwen2.5 (consistent with E017/E024).

**HM trajectory:** rises 0.418→~0.47 by step 30, plateaus flat to step 342 (no late erosion).

Run: step 341/342, WandB 52uf3h0i, 5500 rows, total_epochs=2, ~3h wall.
