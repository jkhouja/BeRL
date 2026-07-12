---
name: log-results
description: Record BeRL experiment results into the tracker row and its self-contained summary. Use when the user asks to log results, update the tracker, document the experiment, or write up results.
allowed-tools: Bash Read Edit Grep Glob
---

## Purpose

Results live in **two** places (the old `grpo_tuning_changelog.md` round-format is retired):
1. **`project_planning/BeRL_experiments_tracker.md`** — the row's `Results summary` + `Status`
   (source of truth for coordination).
2. **`experiments_logs/<RUN_NAME_BASE>.md`** — the self-contained, reproducible experiment record
   (base stem `<RQ>-<expid>-<data_name>`, no `-r<N>`; exact command, knobs, env, WandB/log, findings,
   rerun line) written by the launcher; all resubmission attempts (`-r1`, `-r2`, …) append here.

## Canonical scoring (MANDATORY — use the shared script, do NOT hand-roll HM)

All rows must report the **same** metric so numbers are comparable across agents. Compute it with:

```bash
python scripts/score_run.py <Log path>            # human-readable block
python scripts/score_run.py <Log path> --json      # machine-readable
```

Convention enforced by the script (`scripts/score_run.py` is the single source of truth):
- **ToM HM** = harmonic mean over the **24 Theory-of-Mind benchmarks**, **excluding `gsm8k` and
  `mmlu`**. Aggregation = *avg-then-HM*: average each benchmark over the last N eval iters, then take
  the harmonic mean across benchmarks. Report **HM(last5)** (primary) and **HM(last3)**, plus the
  step-0 baseline HM.
- **gsm8k** and **mmlu** are reported **SEPARATELY** as capability-regression evals (math reasoning /
  general knowledge) with `delta vs step0`. They are **never** folded into the HM.
- **parseable rate** (= 1 − `reward/format_error_ratio`) and **health** (kl / entropy / resp_len) are
  tracked separately, never inside the HM.

Paste the script's output into the `experiments_logs/<RUN_NAME_BASE>.md` findings, and put
`ToM HM(last5)=… HM(last3)=… (base …); gsm8k Δ…; mmlu Δ…; health …` into the tracker `Results summary`.

## Instructions

1. **Identify the experiment** (`Exp #` / `Run name`) and read its tracker row + `Log path`.

2. **Compute metrics with `python scripts/score_run.py <Log path>`** — this yields the canonical
   ToM HM(last5)/HM(last3), the separate gsm8k/mmlu regression numbers, the HM trajectory, and the
   health snapshot. Do not eyeball or re-derive HM by hand.

3. **Verify before marking `Completed`** (per tracker protocol): eval ran on the target set, WandB
   link + Log path are present, and health checks pass (no collapse/hacking — see `check-training`).

4. **Write the tracker `Results summary`** — a tight 1–3 lines: metric vs baseline + health/hacking
   note + downstream implication (e.g. "power@ll_min=p50 stable to 800 steps; tomi +5.1pp vs A0; no
   collapse ⇒ adopt as `stable` for Phase 0"). Set `Status=Completed` (or `Failed`).

5. **Fill the findings section** of `experiments_logs/<RUN_NAME_BASE>.md` by appending: the metrics table
   (step vs benchmarks), verdict, and any health/hacking observations. Keep the earlier
   command/knobs/env/rerun sections intact — append, don't overwrite (the file may be edited
   concurrently).

6. **Notable phase outcomes only** (e.g. "Phase −1 stable config = X", "Phase 0 winning recipe = Y")
   → record the resolved value in the tracker `Notes`/`OUTPUT` cell and, if it changes the design,
   note it in `project_planning/BeRL_paper_plan.md`. **Do NOT append to `grpo_tuning_changelog.md`** —
   that file is frozen archival (pre-paper Rounds 1–20) and is not part of the paper workflow.

7. **Propagate winners**: if this row is an `OUTPUT` row that resolves a placeholder
   (`stable`/`dcfg_default`/`Q2best`), record the concrete resolved values so downstream `Backlog`
   rows can be promoted by the user.

## Never

- Never invent metrics — read them from the log.
- Never hand-roll HM or include gsm8k/mmlu in it — always use `scripts/score_run.py`.
- Never mark `Completed` without WandB link + Log path + passing health check.
