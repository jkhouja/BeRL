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

   **CUDA OOM mid-run (esp. Gemma-2 behavior runs).** Intermittent `torch.OutOfMemoryError` in
   `update_actor → update_policy → loss.backward()` can kill a run mid-training on a rare long
   sequence (observed: ST21 gemma-2-2b crashed at step 68/190). Recovery, in order:
   - **Do NOT lower `MICRO_BATCH` to fix this.** The launcher's `MICRO_BATCH` is normalized per-GPU
     (`fsdp_workers.py:231/236/240` divide by the DP world size), so for an 8-GPU run it is already
     `8//8 = 1` sample/GPU — you cannot reduce the backward memory further. Worse, `MICRO_BATCH<8`
     floors `log_prob_micro_batch_size` to 0 → `ValueError: TensorDict.split: split_size must be
     positive, got 0` at `dp_actor.py:186` (instant crash on the first rollout).
   - **Correct lever: lower vLLM `GPU_MEM_UTIL`** (e.g. gemma default 0.3 → `GPU_MEM_UTIL=0.2`) to
     free ~8GB/GPU for the FSDP actor backward. This changes no science knob. Resubmit as the next
     `-r<N>` (launcher auto-increments); note the OOM + fix in the tracker row's Results/Notes cell.
   - Trade-off: less vLLM KV cache ⇒ slightly slower generation (~42–50s/step vs ~36s/step on
     gemma-2-2b), acceptable. All FSDP offloads (param/grad/optimizer) are already on by default.

6. **Update the tracker row** `Status`: `Training` (healthy, in progress), `In-debug` (issue found —
   note it), `Completed` (finished + evaluated), or `Failed`. Use `Awaiting-input` if a user decision
   is needed (e.g. the causal gate).

7. **Baselines for context** (report deltas vs the row's model/baseline, not a fixed number — read
   the relevant `A0`/baseline row or `project_planning/HISTORY_rounds_1-20.md` for reference points).

Do **not** rewrite results into the old changelog here — that is the `log-results` skill's job.
