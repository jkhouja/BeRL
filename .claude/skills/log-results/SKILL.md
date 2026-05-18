---
name: log-results
description: Log experiment results to the GRPO tuning changelog. Use when the user asks to log results, update the changelog, document the experiment, or write up results.
allowed-tools: Bash Read Edit Grep Glob
---

## Instructions

1. Read the current end of `grpo_tuning_changelog.md` to determine the next round/experiment number
2. Extract eval scores from the relevant log file(s) in `logs/`
3. Parse step-by-step metrics (tomi, explore_tom, hi_tom) into a table
4. Append a new section to the changelog following the existing format:
   - `### Experiment {round}{letter}: {title}`
   - **Purpose/Changes** section explaining what was tested
   - **Config** details (model, dataset, key hyperparams that differ from 17f)
   - Results table with step, tomi, explore_tom, hi_tom columns
   - **Result** line with verdict (checkmark/X) and summary
   - Comparison table against relevant baselines if applicable
5. If multiple experiments form a round, include a summary table at the end
6. Use the same emoji conventions: ✅ improvement, ❌ degradation/flat, ⚠️ mixed

## Existing changelog format reference

The changelog is at `grpo_tuning_changelog.md`. Read the last few sections to match style.
