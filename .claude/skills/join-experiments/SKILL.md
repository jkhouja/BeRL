---
name: join-experiments
description: Onboarding + coordination protocol for a Claude joining the BeRL experiment-execution pool as one agent among many. Use when a new agent is asked to help run/execute experiments, pick up the queue, or join the experimentation effort.
allowed-tools: Bash Read Edit Write Grep Glob
---

## You are one agent among many

Multiple Claude agents execute BeRL training runs concurrently on a shared cluster, coordinating
**only** through two shared, concurrently-edited files:
- `project_planning/BeRL_experiments_tracker.md` — the **v2** live queue (source of truth for who
  owns what and each row's `Status`).
- `project_planning/BeRL_paper_plan.md` — the **v2** forward-looking design/rationale (read for
  context; don't edit unless replanning). `old_BeRL_paper_plan.md` is the archived pre-v2 stage.

Your prime directive: **make progress on experiments without disrupting or corrupting shared state
that other agents depend on.** When in doubt, do less to the shared files, and re-read before you
write.

## Step 0 — Read before you touch anything

1. Read the tracker's §"Agent protocol", §"Execution order", and §"Column meanings".
2. Read this repo's companion skills — `claim-experiment`, `launch-experiment`, `check-training`,
   `log-results` — they implement the individual lifecycle steps. This skill is the umbrella that
   ties them together for a *newly joining* agent.
3. `hostname` → this is your `Owner_host`. `conda activate tom` (transformers 4.51.3, vLLM 0.6.3,
   torch 2.4.0).

## Golden rules (non-negotiable)

1. **Only claim `Not-started` rows.** Never touch `Backlog` rows (user-gated), and never run a row
   owned by another agent (`Owner_host` set, `Status` in
   `Processing/Training/In-debug/Awaiting-input`). Pick the lowest `Exp #` that is `Not-started` and
   whose dependencies are `Completed`.

2. **Consult-and-update the tracker BEFORE running any command tied to an experiment.** The tracker
   `Status` must always reflect reality *before* you act, not after:
   - Before launching → set `Status=Processing` + `Owner_host` (claim), then `Training` once it
     starts.
   - **Before asking the user anything** → first set `Status=Awaiting-input` and put the question in
     `Notes`, *then* ask. After the user answers → first update `Status` back to the relevant state
     (`Training`/`In-debug`/…), *then* continue execution.
   - On failure → `Failed` (+ Notes). On finish+verify → `Completed`.
   This ordering means that if you crash or another agent looks mid-action, the shared state is never
   lying about what is happening.

3. **Assume concurrent edits — re-read immediately before every write.** Another agent may have
   edited the tracker between your read and your write. Right before editing a row:
   - This is a shared working tree, not a merge workflow: **re-read the exact row** you are about to
     change and confirm it still says what you expect. If it changed (e.g. someone else claimed your
     row, or a lowest-`Exp #` collision — lowest wins), **abandon and pick another row**.
   - Prefer **append-only / single-row edits** over rewrites. Change only your own row's cells; never
     reflow, reorder, or bulk-edit the table. Never hold the file "open" across long operations.
   - Keep each tracker edit tiny and self-contained so it can't clobber a neighbor's concurrent edit.

4. **Never mutate shared code/config that other runs depend on.** Do not edit `verl/`, converters,
   or existing `dcfg_*` configs mid-flight — another agent's live run may load them. If a shared fix
   is genuinely needed, set your row to `Awaiting-input` and raise it with the user rather than
   silently changing shared code. Generate data once per `dcfg_*` (shared parquet); if it already
   exists, reuse it — don't regenerate over a file another run is reading.

5. **One node, one experiment at a time — and only *your own* node.** Each agent uses **exactly one
   compute node** (the one you are running on: `hostname` = your `Owner_host`) and runs **exactly one
   experiment at a time**. Claim a single `Not-started` row, launch it on your single
   node, and **monitor it to `Completed`/`Failed` before claiming the next row** — never claim or
   launch a second row while your current run is still `Processing`/`Training`. Do not acquire a
   second allocation to parallelize; parallelism across the queue comes from *other* agents each
   holding their own single node, not from one agent grabbing multiple nodes.
   - **Never launch, SSH into, `srun`/`ssh` onto, or otherwise run work on any node other than the
     one you are on, without explicit user permission.** Stay on your own host. If a run needs a
     different/bigger node, set your row to `Awaiting-input` and ask the user — do not hop nodes.

6. **Never override `TOTAL_EPOCHS` — every run is 1 epoch.** The launcher default is
   `TOTAL_EPOCHS=1` (`experiments/lib/common.sh:186`); it is the enforced project-wide policy for
   cross-run comparability, **not** a suggestion. Do **not** export or pass `TOTAL_EPOCHS` (do not
   copy a pre-2026-07-15 command template that carried `TOTAL_EPOCHS=2`). After launch, **verify**
   the run is single-epoch: the launcher aborts unless `TOTAL_EPOCHS=1` (override only with an
   explicit `ALLOW_MULTI_EPOCH=1`, which must be user-approved and noted in the row + repro-md), and
   the run log / repro-md must read `'total_epochs': 1` / `epochs=1`. A run that trained 2 epochs is
   **invalid for comparison** — mark its row `Not-started` (note why) so it is relaunched at 1 epoch.

7. **Keep every tracker row column-aligned (41 cells, v2 schema) — the #1 source of silent
   corruption.** The **v2** table has **exactly 41 columns** (see the tracker header row / its
   §"Column meanings" and §"Row hygiene" — do not hand-copy the list here, read it from the file so
   this skill can't go stale). Every data row must have **42 `|` characters / 41 cells**, start and
   end with `|`, with `Status` in its documented column. A misaligned row is **silently skipped** by
   every pipe-splitting status scanner (so it never shows up in queue/status counts). When you edit
   your row:
   - **Never put a raw `|` inside any cell** (Results/Notes free-text especially). Escape it as `\|`,
     or wrap the fragment in backticks. **Special model tokens with pipes** (`<|im_end|>`,
     `<|endoftext|>`) MUST be escaped (`` `<\|im_end\|>` ``) or they split the row into extra cells.
   - **Never paste a full multi-field launcher run-name into a single cell** — the `Run name` column
     holds only the base stem `<Exp #>-<Exp ID>-<Data config>` (no `-<model>-<params>-r<N>` suffix; that
     lives in `Log path`). Duplicating the params-suffixed name adds a stray cell and shifts every
     column right.
   - **Fill all structural columns before marking `Completed`:** `WandB link` = full
     `https://wandb.ai/jkhouja-oxford/TOM_EXP/runs/<id>` URL (never a bare run-id), `Log path` =
     `logs/<YYYYMMDD>/<RUN_NAME>.log`, `Summary doc` = `experiments_logs/<stem>.md`. Use `TBD` only
     as a temporary placeholder in a still-running row, never in a `Completed` one.
   - **After editing, verify the row is 41 cells** (e.g. `awk -F'|' 'NR==<line>{print NF-2}'` must
     print `41`, or count that the line has 42 `|`). Do not leave a row you touched malformed.

9. **Always announce which run you own — especially while waiting or asking.** The user watches
   several agents in parallel and needs to tell them apart. **Whenever you pause on a waiting loop,
   poll a running job, end a turn while a run is in flight, or ask the user anything**, begin that
   message with a one-line banner naming the run you are currently responsible for (or most recently
   finished): e.g. `[owning ST07-Phase-stability-v2_g2_anchor_pk4_stdgate on h100-013-002 — Training,
   step 120/380]`. If you have just finished and are idle, say so explicitly
   (`[idle — last finished ST07; scanning for next Not-started row]`). Never leave the user guessing
   which experiment a given agent window maps to.

10. **Use the v2 tracker only; the old tracker is read-only history.** All claiming/launching/status
    updates happen in `project_planning/BeRL_experiments_tracker.md` (v2).
    `project_planning/old_BeRL_experiments_tracker.md` (the archived `PS001–PS182` / `E016–E108`
    stage) **must not be edited or used to pick up work** — you may *consult* it for historical
    config/results context, but never claim, relaunch, or write to a row there.

8. **Keep the header row-count claim in sync.** The `## Experiments (<N> rows; …)` heading states the
   total. When you **add** a new row, bump the count in the same edit — stale counts (they
   drifted 175→275 once in the old tracker) mislead capacity planning and audits.
## Per-run log file (mandatory)

Every experiment gets its own markdown file at **`experiments_logs/<RUN_NAME_BASE>.md`**, where
`RUN_NAME_BASE` = the tracker `Run name` stem = `s2-<Exp #>-<Exp ID>-<data_name>` (stage tag
`RUN_STAGE=s2` + short row id `EXP_NUM` prefixed for WandB search; RQ tag baked into `Exp ID`;
**`test`** for smoke runs). The full WandB run name adds `-<model>-<params>-r<N>` (run index) — but the md file
is keyed by the **base stem without `-r<N>`** so every *attempt* (r1, r2, … after a crash/resubmit)
**appends** to the same file (append-friendly so concurrent tooling never truncates it). It must
summarize:

- **Entry name / RUN_NAME_BASE** and `Exp #` / `Exp ID` (the handle in the tracker), plus the list of
  attempted run indices (`-r1`, `-r2`, …) and which one is authoritative.
- **Hypothesis / question** — what this run tests and the expected outcome.
- **Implementation details** — resolved knobs (reward family, KL, clamp, baseline, RM mode, LR,
  `dcfg_*` data, CoT prompt var, model/size), the **exact launch command**, and the env.
- **Debugging / issues** — anything that went wrong and how it was resolved (append timestamped
  notes; don't overwrite).
- **Findings** — final/peak metrics vs baseline, health/hacking observations, verdict, downstream
  implication.
- **Useful metadata** — WandB link, `Log path`, `Owner_host`, start/end time, "how to rerun" line.

This file is self-contained: a future agent (or the paper) should be able to understand and reproduce
the run from it alone.

## The loop (what "join and execute" means)

1. Claim the next ready `Not-started` row (`claim-experiment`).
2. Resolve placeholders + build data if missing + launch the generic launcher in the background;
   create `experiments_logs/<RUN_NAME>.md`; set `Status=Training`, paste WandB/log (`launch-experiment`).
   **Launcher choice:** the `ONLY_IDX=<n>` shortcut through `experiments/phase_stability_sweep.sh`
   works **only** for the original Wave-1 Qwen2.5 cells `PS001–PS096` (n = `PS` number). The expanded
   Wave-2 rows `PS097–PS176` (Gemma-2/Qwen3) and the Qwen2.5 power-extension rows `PS177–PS182` are
   **not** in the sweep's 96-cell enumeration (they add `k=7`, `ll_min=-4`, a winner-anchored `fp×ec`
   grid), so `ONLY_IDX` will not launch them — use the per-family launcher with explicit knobs
   (`EXP_ID=… REWARD_TYPE=… POWER_K=… POWER_LL_MIN=… USE_ACTOR_AS_RM=… KL=… LR=… FORMAT_PENALTY=…
   FORMAT_PENALTY_STD_COEF=… ENTROPY_COEFF=… bash experiments/smoke_{gemma,qwen3,qwen2.5}.sh`; see
   `launch-experiment` and the tracker §"Launch mechanism").

   **Format compliance during training (optional gate).** The flat `FORMAT_PENALTY` (e.g. `5`) is
   reward-type-asymmetric after GRPO normalisation (≈1–2σ for `log_prob`, only ≈0.3σ for the 0–40
   `power` reward), so low-KL / high-LR runs can still collapse format. To **enforce** format
   compliance, additionally set `FORMAT_PENALTY_STD_COEF` (recommended **1.5–2.0**, default `0.0`=OFF):
   it hard-gates every format-violating rollout to `≈std_coef·σ` below all well-formed responses
   (lexicographic, reward-type-symmetric). Enable via `FORMAT_PENALTY=5 FORMAT_PENALTY_STD_COEF=1.5`.
   **Default OFF** — leave it unset for standard tracker rows so runs stay comparable to prior
   Completed rows; only enable when a row/user explicitly calls for a training-time format gate.
3. Monitor health; on any user-decision need, `Awaiting-input`-then-ask (`check-training`).
4. On completion, verify evals, append findings to `experiments_logs/<RUN_NAME>.md`, write the
   tracker `Results summary`, set `Status=Completed`; propagate any winner values that unblock
   `Backlog` rows for the user to promote (`log-results`).
5. Repeat with the next ready row **only after the current run has reached `Completed`/`Failed`** —
   one experiment on one node at a time (Golden rule #5).
6. **If there are no ready `Not-started` rows to pick up, do not stop — keep checking every 10
   minutes.** Stay idle on your free node and re-scan the tracker for a ready `Not-started` row on a
   recurring 10-minute schedule (some in-flight rows may `Fail` and revert to `Not-started`, and the
   user may promote `Backlog` rows). As soon as one appears, claim it (re-read first) and resume the
   loop. Only stop the 10-minute polling once **all** rows are terminal (`Completed`/`Failed`, none
   `Not-started`/`Training`/`Processing`) — then report the final summary.

## Never

- Never run more than one experiment at a time or hold more than one compute node as a single agent.
- **Never run work on, SSH into, or `srun`/`ssh` onto a node other than your own without explicit user permission.**
- **Never edit or claim rows in `old_BeRL_experiments_tracker.md`** — it is archived history (consult-only); all live work is in the v2 `BeRL_experiments_tracker.md`.
- **Never pause on a waiting loop, poll, or ask the user without first stating which run you currently own / last finished** (Golden rule #9).
- Never bulk-rewrite or reorder the tracker table; edit only your own row, and re-read first.
- Never claim `Backlog` rows or rows owned by others.
- Never ask the user without first flipping your row to `Awaiting-input`.
- Never regenerate/rename shared `dcfg_*` data or edit shared `verl/` code that live runs depend on.
- Never set `WANDB_MODE=offline` or drop the wandb logger (WandB online is mandatory).
- Never override `TOTAL_EPOCHS` or run more than 1 epoch (default `1`, `common.sh:186`) without an
  explicit user-approved `ALLOW_MULTI_EPOCH=1`; verify `'total_epochs': 1` in the log after launch.
