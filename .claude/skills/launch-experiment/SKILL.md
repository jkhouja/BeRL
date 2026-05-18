---
name: launch-experiment
description: Launch a GRPO training experiment. Use when the user asks to run, launch, or start a training run or experiment. Handles script creation, venv activation, and background execution.
argument-hint: "[script-name or config description]"
allowed-tools: Bash Read Write Glob Grep
---

## Project Context

This is the BeRL/TomRL project training small LLMs (Qwen2.5) with GRPO for Theory of Mind reasoning via dialogue data.

## Current best config (17f on 3B)

- Dataset: `data/merged_dialogue_datasets_filtered_eval_prompt.parquet` (6,214 filtered samples)
- System prompt: `cot_eval`
- Reward: power-law LL, ll_min=-8, clip (-40, 40)
- Actor-as-RM, KL=0.05, LR=5e-7, batch=32, mini_batch=128, rollout_n=16, 2 epochs
- Eval: `data/cleaned_tom/ToM_test_HiExTi_hint_v3.parquet`

## Existing training scripts

!`ls -1 *.sh`

## Instructions

1. If the user specifies a config, create a **new script** rather than modifying existing ones
2. Always activate the venv before launching: `source /mnt/home/judekhouja/repo/BeRL/.venv/bin/activate`
3. Run in background so the user can continue working
4. After launching, confirm the experiment name and log file path
5. Check early output to verify training started (no import errors, model loading)
6. Name scripts descriptively based on the ablation (e.g., `dialogue_7b_frozenRM.sh`)
