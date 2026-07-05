---
name: claim-experiment
description: Atomically claim a BeRL experiment row from the tracker before running it. Use when the user asks to pick up the next experiment, claim a run, or start work on the experiment queue.
allowed-tools: Bash Read Edit Grep Glob
---

## Purpose

`project_planning/BeRL_experiments_tracker.md` is the single source of truth for the ~92 BeRL
training runs executed by multiple agents on a shared cluster. This skill implements the **claim +
dependency-gate protocol** so two agents never run the same row. Read
`project_planning/BeRL_experiments_tracker.md` §"Agent protocol" before acting.

## Instructions

1. **Find a claimable row.** Only rows with `Status=Not-started` are claimable. **Never touch
   `Backlog` rows** — the user gates those. Grep the tracker table for `Not-started`:
   ```
   grep -nE '\| Not-started \|' project_planning/BeRL_experiments_tracker.md
   ```
   Prefer the lowest `Exp #` whose dependencies are `Completed` (see the execution-order phases and
   the `=stable(Pm1)`/`=Q2best(Q2)`/`=Pm1a_best(...)` placeholders in the row).

2. **Check dependencies.** If the chosen row references a placeholder that resolves from an upstream
   experiment (e.g. `=stable(Pm1)`), confirm those upstream rows are `Completed`. If not, the row is
   not ready — pick another or stop and report.

3. **Claim atomically.** In a single edit, set `Status=Processing` and `Owner_host=<hostname>`
   (`hostname` output) for that row. If two agents collide on the same row, **lowest `Exp #` wins**;
   the other reverts its edit and picks a different row.

4. **Report** the claimed `Exp #`, `Exp ID`, `Run name`, `Data config name`, and resolved knobs.
   Hand off to the `launch-experiment` skill to resolve placeholders, generate data, and launch.

## Lifecycle

`Backlog → Not-started → Processing → Training → (In-debug ↔ Training) → Completed` (or `Failed`).
Use `Awaiting-input` (with a note in `Notes`) when a user decision is needed — e.g. the causal gate:
if A1 does not beat the shuffled/no-CoT controls, pause downstream and set A1 to `Awaiting-input`.
