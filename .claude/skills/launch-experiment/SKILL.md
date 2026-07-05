---
name: launch-experiment
description: Launch a BeRL GRPO training experiment from the tracker. Use when the user asks to run, launch, or start a training run or experiment. Resolves knobs, generates data if needed, launches the generic launcher in the background, and records links back to the tracker.
argument-hint: "[Exp ID / Run name, e.g. E003 or pm1a_power]"
allowed-tools: Bash Read Write Edit Glob Grep
---

## Project Context

BeRL/TomRL trains small LLMs (Qwen2.5 / Qwen3 / Gemma-2) with GRPO to test whether a
**behavior-prediction reward** (log-likelihood of the real next human utterance given
[context + CoT]) induces Theory of Mind that transfers without ToM labels.

Source of truth: `project_planning/BeRL_experiments_tracker.md` (92-row execution table) +
`project_planning/BeRL_paper_plan.md` (design). Do **not** hardcode any single run's config here —
every knob comes from the claimed tracker row.

## Prerequisites

The row must already be **claimed** (`Status=Processing`, `Owner_host` set) — use the
`claim-experiment` skill first. Conda env `tom` (transformers 4.51.3, vLLM 0.6.3, torch 2.4.0):
`conda activate tom`.

## Instructions

1. **Read the claimed row** for the given `Exp #`/`Run name` and extract every knob:
   `Model/Size`, `Gen ctx`, `KL`, `Loss/reward type` (`log_prob|neg_perplexity|power|rule_based_tom|
   SFT|behavior+rule`), `Loss powers` (`power_k`/`power_ll_min`), `RM mode` (`frozen|actor`), `LR`,
   `Data sources`, `Data params`, `CoT prompt var`, `Target evals`, `Data config name` (`dcfg_*`),
   `Run name`. **Resolve placeholders** (`=stable(Pm1)`, `=Q2best(Q2)`, `=Pm1a_best(E001-03)`) by
   reading the referenced `Completed` rows and substituting concrete values.

2. **Ensure the data exists.** The `dcfg_*` name is the data-gen YAML stem AND the output parquet
   stem (generate once, shared across rows). If the parquet is missing, build it:
   `python build_dataset.py --config scripts/configs/<dcfg_name>.yaml`. See `docs/skill_adding_dataset.md`.

3. **Launch via the generic launcher** (per model class), in the background, passing knobs as
   env/args — never create a per-experiment script:
   - behavior reward → `experiments/train_behavior_grpo.sh`
   - direct rule-based ToM → `experiments/train_tom_rulebased.sh`
   - SFT baseline → `experiments/train_sft.sh`
   - or the dispatcher `experiments/run_experiment.sh <EXP_ID>` (parses the row → sets knobs → calls
     the right launcher).
   **Construct the WandB run name** (self-describing — see tracker §"Agent protocol" #4):
   ```
   <RQ>-<expid>-<data_name>-<model>-<params>-r<N>
   ```
   where `<RQ>-<expid>` = the tracker `Exp ID` (RQ tag already baked in), `<data_name>` = the
   `Data config name` (`dcfg_*`), `<model>` = family+size, `<params>` = key knobs (reward/lr/kl),
   and `-r<N>` = **run index**. Before launching, **pick the next free `N`**: scan
   `logs/<YYYYMMDD>/` and WandB for the base stem; if `-r1` already exists (e.g. a prior crash),
   use `-r2`, `-r3`, … so a resubmitted run never clashes with the crashed one. **Test/smoke runs
   use `RQ=test`** (e.g. `test-qwen3_smoke-…-r1`). `RUN_NAME` MUST equal WandB run name = log
   filename stem, and its base (`<RQ>-<expid>-<data_name>`) MUST match the tracker `Run name` cell.
   Model-family branches (attention backend, `require_answer_tags`, chat template) are handled inside
   the launcher (Qwen3: `+*.require_answer_tags=False`, `VLLM_ATTENTION_BACKEND=XFORMERS`,
   `gpu_memory_utilization=0.35`).

4. **WandB online is mandatory** — `WANDB_API_KEY` is in `~/.bashrc`; use `logger=[console,wandb]`.
   Never set `WANDB_MODE=offline`, never unset the key, never fall back to `[console]`.

   **4a. Prompt alignment (MANDATORY when the CoT/system prompt varies).** All eval parquets bake
   the `cot_eval` system prompt, and **no** eval prompt is re-aligned at runtime by default. So if
   your row varies the training system/CoT prompt — RQ `cot-style` (E082–E085), or **any** row whose
   `CoT prompt var` / `system_prompt_style` is not `cot_eval` — you MUST force **Option A**: pass
   ```
   +data.system_prompt="<exact system-prompt text for this run>"
   ```
   at launch. `RLHFDataset._process_system_prompt` (`verl/utils/dataset/rl_dataset.py:127`) then
   **replaces** the system message in **both** the train and val loaders (`ray_trainer.py:390,406`),
   so training *and* every eval use the identical system prompt and the comparison stays fair.
   Without it, the evals stay frozen at `cot_eval` while training drifts → train/eval mismatch (the
   Round-17 transfer bottleneck). Caveats: (1) the override only takes effect with
   `data.prompt_is_text=False` (message-format parquets — all current ones; the legacy
   `merge_tom.py` text-prefix format bypasses it); (2) it overwrites eval-specific system content
   **including** the tomi `hint_v3` room-witness note, so if that hint matters, append it to your
   override text. When the prompt is *not* varied (the default `cot_eval`), do **not** set
   `+data.system_prompt` — the baked prompts already match.

   **4b. Answer-tag alignment (MANDATORY — tag-free CoTs for Qwen3 / Gemma).** The scorer's
   `<answer>`-tag expectation is decided by **model type**, not by any flag: `main_ppo.py` picks a
   parser where Qwen2/2.5 require `<answer>` tags but **Qwen3 and Gemma/Gemma2 do NOT**
   (`response_parser.py`). The `+*.require_answer_tags=...` overrides are **no-ops** (the scorers
   read `parser.REQUIRE_ANSWER_TAGS` instead). Therefore the *data* must match the model:
   - **Qwen2/2.5** → tagged data: `system_prompt_style: cot_eval`, `add_response_tags: true`,
     `generation_prefix: "<think>"` (e.g. `pipeline_config_all_dialogue.yaml`,
     `data/merged_all_dialogue_eval_prompt.parquet`).
   - **Qwen3 / Gemma / Gemma2** → **tag-free** data: `system_prompt_style: cot_eval_notags`,
     `add_response_tags: false`, `generation_prefix: ""` (e.g.
     `pipeline_config_all_dialogue_notags.yaml`, `data/merged_all_dialogue_notags.parquet`).
   Never train Qwen3/Gemma on a `cot_eval`/`add_response_tags: true` dataset: the system prompt
   would command `<answer>` tags the target lacks and the parser ignores → mismatch. The data-gen
   pipeline (`build_dataset.py`) now **fails fast** if `system_prompt_style`, `add_response_tags`,
   and `generation_prefix` disagree (`validate_prompt_tag_consistency`).

5. **Verify start-up**: tail the log to confirm no import errors, model loads, first rollout begins.

6. **Record back to the tracker**: set `Status=Training`, paste `WandB link` and
   `Log path` (`logs/<YYYYMMDD>/<RUN_NAME>.log`, where `RUN_NAME` includes `-r<N>`) into the row.

7. **Reproducibility artifact**: create/append the self-contained per-run log at
   `experiments_logs/<RUN_NAME_BASE>.md` (base stem `<RQ>-<expid>-<data_name>`, **no** `-r<N>` — all
   attempts append to one file; record the exact command + all knobs + env + WandB/log + hypothesis +
   attempted run indices + a "how to rerun" line). See the `join-experiments` skill for the required
   sections. Findings are filled on completion by the `log-results` skill.

## Never

- Never hardcode the old "17f" config or any single run's hyperparameters.
- Never edit `Backlog` rows or launch a row you have not claimed.
- Never write a new one-off `.sh` per experiment — use the generic launchers.
