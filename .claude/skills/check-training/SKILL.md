---
name: check-training
description: Check the status/health of a running or completed BeRL experiment and update its tracker row. Use when the user asks to check training, check status, how's the run going, or check on the experiment.
allowed-tools: Bash Read Grep Glob Edit
---

## Instructions

1. **Locate the log.** Logs are written to `logs/<YYYYMMDD>/<RUN_NAME>.log` (RUN_NAME = tracker slug).
   If given an `Exp #`/`Run name`, read the `Log path` from
   `project_planning/BeRL_experiments_tracker.md`; otherwise list the most recent:
   ```
   ls -lt logs/$(date +%Y%m%d)/ 2>/dev/null | head -5
   ```

2. **Progress.** Current step = the last `step:N - global_seqlen` line (filter out `timing_s/step:N`
   false matches). Report `step / total`.

3. **Eval metrics.** Latest `val/test_score/` lines. For a completed run, prefer the canonical scorer
   `python scripts/score_run.py <log>` (ToM HM excl. gsm8k/mmlu, with gsm8k/mmlu reported separately;
   see `log-results`). For an in-flight run, parse the per-benchmark scores present in this run's
   `Target evals` (dev split, or the full suite: tomi / explore_tom / hi_tom / fantom_* / OOD /
   Behavior-PPL / guardrail). Present as a markdown table with the trend (improving / flat /
   degrading / collapsed).

4. **Health / hacking scan** (critical — this project watches for reward hacking & collapse):
   - Errors: `grep -cE 'Traceback|are not supported|OutOfMemory|CUDA error|must be real' <log>`
     (do NOT grep bare `nan` — it matches reasoning text).
   - Degenerate CoT: spot-check sampled generations for repetition, empty `<think>`, format drift,
     or answers copied from the prompt. Note reward-vs-eval divergence (reward up but eval flat/down
     ⇒ possible hacking).

5. **Process alive?** If it should be running, check GPU:
   `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader`.

6. **Update the tracker row** `Status`: `Training` (healthy, in progress), `In-debug` (issue found —
   note it), `Completed` (finished + evaluated), or `Failed`. Use `Awaiting-input` if a user decision
   is needed (e.g. the causal gate).

7. **Baselines for context** (report deltas vs the row's model/baseline, not a fixed number — read
   the relevant `A0`/baseline row or `project_planning/HISTORY_rounds_1-20.md` for reference points).

Do **not** rewrite results into the old changelog here — that is the `log-results` skill's job.
