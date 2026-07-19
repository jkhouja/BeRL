# BeRL — Behavior-prediction RL for Theory of Mind

## Project goal
Test whether reinforcing **behavior prediction** — the likelihood a model assigns to the real next
human utterance given [context + CoT] — induces Theory of Mind (ToM) that transfers OOD, is robust,
and is label-competitive with direct ToM-supervised RL, using **no ToM labels**.

BeRL = GRPO on reward = length-normalized log-likelihood / neg-perplexity of the held-out human
utterance (optionally baseline-subtracted); scorer = frozen base LM or the actor.

## Source of truth (READ FIRST)
The forward-looking plan and live execution state live in `project_planning/`:
- **`project_planning/BeRL_experiments_tracker.md`** — the **fresh v2** live coordination table
  (started 2026-07-19; every hyperparameter has an explicit column — no silent launcher defaults).
  Claim a `Not-started` row, run it, update status/WandB/log/results. Agent protocol + naming
  contract + `def`-default legend are defined at the top of the file.
- **`project_planning/old_BeRL_paper_plan.md`** — archived pre-v2 thesis/phases/eval-suite design
  (Phase −1 → Phase 0 → Q2 → Q0 → Q1 → Q3 → Q4 → QG); historical context, not the live plan.
- **`project_planning/old_BeRL_experiments_tracker.md`** — archived pre-v2 execution table
  (`PS001–PS182` sweep + `E016–E108`); historical results only.
- **`project_planning/HISTORY_rounds_1-20.md`** — archived pre-paper results (Rounds 1–20) and the
  best-known config. Historical only; not the final paper experiments.
- `grpo_tuning_changelog.md` — frozen archival run-by-run log (pre-paper Rounds 1–20); not part of
  the paper workflow, do not append.

## Best-known config (from Round 17f/20; starting point, to be re-derived in Phase −1/0/Q2)
Qwen2.5-3B-Instruct; `cot_eval` system prompt (matches eval); power-law LL reward, ll_min=-8,
clip (-40,40), actor-as-RM; KL=0.05, LR=5e-7, batch=32, mini_batch=128, rollout_n=16, 2 epochs.

## Naming contract (enforce in all scripts/configs)
- **Run name** = launcher `EXP_NAME` = WandB run name = log filename = tracker slug (lowercase),
  e.g. `pm1a_power`, `p0_single_cga`, `q0_a1_berl_s1`.
- **Data config** = `dcfg_*` = data-gen YAML stem AND output parquet stem (generate once, share).
- Each experiment auto-writes a self-contained summary `.md` (in `experiments_logs/`) with
  the exact command, all knobs, env, WandB link, log path, findings, and a rerun one-liner.

## Repo map
- `experiments/` — training launchers (being consolidated into generic, model-class-aware launchers
  + smoke tests; see plan). Scripts are dataset-agnostic; data is passed via `dcfg_*`.
- `scripts/` — dataset converters (`convert_*.py`: ~10 dialogue domains + ToM, on shared
  ConvoKit/dialogue base classes), `build_dataset.py` (CONVERTERS registry),
  `prompt_templates.py` (CoT/ToM prompt styles), `configs/` (`pipeline_config_*` → `dcfg_*`).
- `verl/` — training framework (FSDP). Reward computation + clipping in
  `verl/workers/fsdp_workers.py` (MIN/MAX_REWARD constants). Qwen3 support in
  `verl/models/transformers/qwen3.py` + `verl/third_party/vllm/` (see `docs/ENVIRONMENT_SETUP.md`).
- `examples/data_preprocess/prepare_fantom.py` — FANToM eval prep; `verl/utils/reward_score/fantom.py`.
- `docs/` — `ENVIRONMENT_SETUP.md`, `CHANGE_HISTORY.md`, `NEW_DATASETS.md`,
  `skill_adding_dataset.md`, `PIPELINE_DIAGRAM.md`.

## Conventions
- **Change history:** ALWAYS update `docs/CHANGE_HISTORY.md` on every code change (commit hash, date,
  title, what, why).
- **Change skills:** If during experiments you find out certain issues or gothcas that are generally important, update the relevant skills so that other agents pay attention.
- **WandB online always:** `WANDB_API_KEY` is in `~/.bashrc`; use `trainer.logger=['console','wandb']`.
  Do not set `WANDB_MODE=offline` or unset the key.
- **Secrets:** never hardcode tokens (HF_TOKEN etc.) in scripts — read from the environment.
- Branch: `jude/paper` (paper experiments). Base models: Qwen2.5-{0.5B,3B,7B}, Qwen3-1.7B, Gemma-2-2B.
