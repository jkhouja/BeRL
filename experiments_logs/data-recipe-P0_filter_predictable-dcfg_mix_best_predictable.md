# E106 — data-recipe-P0_filter_predictable (dcfg_mix_best_predictable)

- **RUN_NAME_BASE:** data-recipe-P0_filter_predictable-dcfg_mix_best_predictable
- **Exp #:** E106 · **Exp ID:** data-recipe-P0_filter_predictable
- **Attempts:** r1 (authoritative)
- **Model:** Qwen2.5-3B-Instruct · **Owner_host:** h100-013-002
- **WandB:** https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/hwld1qqz
- **Log path:** logs/20260716/E106-P0_filter_predictable-Qwen2.5-3B-Instruct-actorRM-nobaseline-power-k5-llmin-6-lr5e-7-kl0.05-n16-r1.log
- **Driver out:** logs/20260715/E106_driver_r1.out
- **Start:** 2026-07-16 ~01:44 UTC

## Hypothesis / question
Phase-0 S3 turn-filtering ablation — **PREDICTABLE (least-surprising) pole**. Opposite of the
`surprise` arm (E026): keeps the 50% of best-mix turns with the **lowest surprisal / highest
answer_pp** (most predictable, least ToM-dependent / info-asymmetric). Question: does *removing* the
ToM-dependent (high-surprisal) turns HURT transfer? If BeRL's ToM gain comes from learning to
predict info-asymmetric turns, training only on predictable turns should underperform both `surprise`
(E026) and the random control (E028) — ideally the WORST of the S3 arms.

## Implementation details
- **Data:** dcfg_mix_best_predictable (5500/11000 rows, keep_fraction=0.5, MOST-predictable = highest
  answer_pp). Built via `python build_dataset.py --config scripts/configs/dcfg_mix_best_predictable.yaml`.
  answer_pp: 0 nulls. Sources: casino 2044 / empathetic 1892 / dailydialog 1564. Filter report:
  surprisal(-log_prob) kept mean=3.535 vs corpus 5.302 (much lower = predictable); resp-words mean
  kept=19.4 / corpus 14.6 (predictable turns are longer).
- **Locked Qwen2.5 recipe:** reward=power k=5 ll_min=-6, actor-RM, baseline=False, kl=0.05, lr=5e-7,
  fp=5, ec=0.0, rollout_n=16, max_ctx=2048/512, require_answer_tags=True, **total_epochs=2**.
- ~5500 rows / 32 * 2 epochs ≈ 344 steps.

### Exact launch command
```
setsid bash -c 'conda activate tom; \
EXP_ID=E106 DATA_NAME=P0_filter_predictable DATA_TRAIN=data/dcfg_mix_best_predictable.parquet \
REWARD_TYPE=power POWER_K=5 POWER_LL_MIN=-6 USE_ACTOR_AS_RM=True SUBTRACT_BASELINE=False \
KL=0.05 LR=5e-7 FORMAT_PENALTY=5 ENTROPY_COEFF=0.0 TOTAL_EPOCHS=2 MAX_PROMPT=2048 MAX_RESP=512 \
bash experiments/train_behavior_qwen2.5.sh' > logs/20260715/E106_driver_r1.out 2>&1 < /dev/null &
```

## Debugging / issues
- Must export TOTAL_EPOCHS=2 (launcher defaults 1). Verified: total_epochs=2, max_response_length=512.

## Findings
(pending — run in progress)

## How to rerun
See exact launch command above; rebuild data first if `data/dcfg_mix_best_predictable.parquet` missing.

## FINAL FINDINGS (E106, r1) — 2026-07-16

**Verdict: STRONG POSITIVE — and a NOTABLE NEGATIVE for the surprise-selection hypothesis.**
Keeping the MOST-predictable (least ToM-dependent) turns does NOT hurt; it matches full-mix.

Scorer (`scripts/score_run.py --last 5`, 24 ToM benchmarks excl gsm8k/mmlu):
- ToM HM(last5)=**0.4736** vs base(step0)=0.4197 → **+5.4pp**
- ToM avg(last5)=**0.5279** vs base=0.5025 → **+2.5pp**
- mmlu=0.6268 vs step0 0.477 → **+15.0pp**
- gsm8k=0.7006 vs 0.66 → **+4.1pp**
- Health(final): kl=0.082, entropy=1.315, resp_len=110.6, reward=40.0, parseable=1.0, max_resp=512

**Qwen2.5 S3 turn-filter comparison table:**
| Arm | Rows | HM Δ | avg Δ | mmlu Δ |
|-----|------|------|-------|--------|
| E024 full-mix        | 11000 | +5.3 | +2.3 | +17.7 |
| E028 randlen (rand50%)| 5500 | +4.8 | +2.0 | +14.4 |
| **E106 predictable**  | 5500 | **+5.4** | **+2.5** | +15.0 |
| E026 surprise         | 5500 | PENDING (other agent) | | |

- **The PREDICTABLE pole ≈ full-mix and ≥ randlen.** Removing the high-surprisal / info-asymmetric
  ("ToM-dependent") turns does NOT degrade transfer — if anything predictable turns (longer,
  resp-words mean 19.4) are marginally stronger on ToM. This is a **negative result for the
  hypothesis that the BeRL ToM gain is driven by selecting surprising/ToM-dependent turns.**
- **Interpretation (provisional, pending E026):** the ToM transfer appears **robust to turn
  selection** on Qwen2.5 at this scale — surprisal-based curation is not the active ingredient. If
  E026 (surprise) also lands ≈ +5pp, the S3 conclusion is "selection doesn't matter; the mix itself
  carries the signal." If E026 markedly exceeds predictable, then there is a genuine surprise effect
  that predictable happens to also capture (unlikely given predictable already matches full-mix).

**Health:** fully stable — resp_len flat ~101-110 (no fp5 collapse), kl bounded ~0.06-0.11,
format_error=0, no crash. fp5+actor-RM well-tolerated (consistent with E017/E024/E028).

**HM trajectory:** rises 0.42→~0.475 by step 90, plateaus flat to step 342 (no late erosion).

Run: step 341/342, WandB hwld1qqz, 5500 rows, total_epochs=2, ~3h wall.
