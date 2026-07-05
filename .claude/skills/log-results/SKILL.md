---
name: log-results
description: Record BeRL experiment results into the tracker row and its self-contained summary. Use when the user asks to log results, update the tracker, document the experiment, or write up results.
allowed-tools: Bash Read Edit Grep Glob
---

## Purpose

Results live in **two** places (the old `grpo_tuning_changelog.md` round-format is retired):
1. **`project_planning/BeRL_experiments_tracker.md`** — the row's `Results summary` + `Status`
   (source of truth for coordination).
2. **`experiments_logs/<RUN_NAME>.md`** — the self-contained, reproducible experiment record
   (exact command, knobs, env, WandB/log, findings, rerun line) written by the launcher.

## Instructions

1. **Identify the experiment** (`Exp #` / `Run name`) and read its tracker row + `Log path`.

2. **Extract final/peak metrics** from the log (`val/test_score/` lines) for every benchmark in the
   row's `Target evals`. Compute deltas vs the model's baseline (`A0`/baseline row or
   `project_planning/HISTORY_rounds_1-20.md`).

3. **Verify before marking `Completed`** (per tracker protocol): eval ran on the target set, WandB
   link + Log path are present, and health checks pass (no collapse/hacking — see `check-training`).

4. **Write the tracker `Results summary`** — a tight 1–3 lines: metric vs baseline + health/hacking
   note + downstream implication (e.g. "power@ll_min=p50 stable to 800 steps; tomi +5.1pp vs A0; no
   collapse ⇒ adopt as `stable` for Phase 0"). Set `Status=Completed` (or `Failed`).

5. **Fill the findings section** of `experiments_logs/<RUN_NAME>.md` by appending: the metrics table
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
- Never mark `Completed` without WandB link + Log path + passing health check.
