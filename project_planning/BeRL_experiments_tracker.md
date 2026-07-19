# BeRL Experiments — Live Coordination Tracker (v2)

**Purpose.** Single source of truth for multiple Claude agents executing BeRL training runs on a
shared cluster. Each agent **claims** a `Not-started` row (set `Status=Processing` + your
`Owner_host`), runs it, and updates `Status`, `WandB link`, `Log path`, and `Results summary`.
This is a **fresh v2 tracker** started 2026-07-19 after the merged fixes (tag-free invalid gate,
relative invalid sentinel, actor-RM reward-type wiring, std-gated format penalty, single-epoch
guard). The **complete pre-v2 history** (Phase −1 sweep `PS001–PS182` + `E016–E108`) is archived
verbatim in `old_BeRL_experiments_tracker.md`; the v2 design is `BeRL_paper_plan.md` (retired
pre-v2 design = `old_BeRL_paper_plan.md`).

> **Why v2 is wider.** Every hyperparameter that previously lived only as a *silent launcher
> default* (batch sizes, `max_prompt`/`max_resp`, `rollout_n`, `entropy_coeff`, `format_penalty`,
> `seed`, eval suite, …) now has its **own explicit column**. Fill every knob column — this is the
> whole point of v2: no run may rely on an unstated default. If a knob is genuinely the documented
> launcher default, write **`def`** (which resolves to the value in §"Launcher defaults" below),
> never leave it blank. `-` means "not applicable to this reward/task".

---

## Agent protocol (read before claiming)
1. **Claim atomically:** only rows with `Status=Not-started` are claimable — **never touch
   `Backlog` rows** (the user gates those). Set `Status=Processing` and `Owner_host=<hostname>` in
   the same edit before doing anything else. If two agents collide, lowest `Exp #` wins.
2. **Resolve every knob before launch.** Read the row and export each hyperparameter column
   explicitly to the launcher (never trust the launcher default silently — if the row says `def`,
   confirm the §"Launcher defaults" value is what you want and pass it). **Always export
   `EXP_NUM=<Exp #>`** (the short row id, e.g. `E034`/`PS2`) so the WandB run name / log / repro-md
   all start with that searchable handle. Resolve placeholders (`=<ExpID>best`) by reading the
   referenced `Completed` row and substituting concrete values.
3. **WandB online always** — `WANDB_API_KEY` is in `~/.bashrc`; `trainer.logger=[console,wandb]`.
   Never set `WANDB_MODE=offline`, never unset the key.
4. **Consult-and-update the tracker BEFORE acting.** `Status` must reflect reality *before* you act:
   `Not-started`→`Processing`(claim)→`Training`(launched)→`Completed`/`Failed`. Before asking the
   user anything, set `Status=Awaiting-input` + put the question in `Notes`, then ask.
5. **Assume concurrent edits — re-read immediately before every write.** Shared working tree: change
   only your own row's cells; never reflow/reorder/bulk-edit. `git pull --rebase --autostash` before
   push; **never `git add -A`** — add only the tracker + your own named files.
6. **Row hygiene (v2 contract):** every row must have **exactly 41 cells** (see the table
   header). Never put a raw `|` or special token (`<|im_end|>`) in a cell — escape as `\|` or
   backtick it. `Run name` holds only the base stem `<Exp #>-<Exp ID>-<Data config>` (the launcher
   appends `-<model>-<params>-r<N>`). Fill `WandB link` as a full URL, `Log path`, and `Summary doc` before
   marking `Completed`. Verify your row's cell count after editing.
7. **1 epoch, always.** Do not override `TOTAL_EPOCHS` (launcher aborts unless `=1`, override only
   with user-approved `ALLOW_MULTI_EPOCH=1`, noted in the row).
8. **One node, one experiment at a time.** Monitor to `Completed`/`Failed` before claiming the next.

---

## Launcher defaults (what `def` resolves to)
Source: `experiments/lib/common.sh` (behavior task). Confirm against the file before relying on it.

| Knob | Column | Default (`def`) | Notes |
|------|--------|-----------------|-------|
| `TRAIN_BATCH` | train_batch | `32` | rollout prompts per step |
| `MINI_BATCH` | mini_batch | `128` | PPO minibatch |
| `MICRO_BATCH` | micro_batch | `8` | per-GPU micro |
| `ROLLOUT_N` | rollout_n | `16` | samples per prompt (GRPO group) |
| `LR` | LR | `5e-7` | |
| `KL` | KL | `0.05` (behavior) / `0.001` (rule-based ToM) | |
| `MAX_PROMPT` | max_prompt | `2048` | |
| `MAX_RESP` | max_resp | `4096` qwen2.5/qwen3 · `1024` gemma · `2048` smoke_qwen | **family-dependent — always set explicitly** |
| `TOTAL_EPOCHS` | epochs | `1` | enforced |
| `ENTROPY_COEFF` | entropy_coeff | `0.001` | |
| `FORMAT_PENALTY` | format_penalty | `0.0` (OFF) | |
| `FORMAT_PENALTY_STD_COEF` | fp_std_coef | `0.0` (OFF) | >0 ⇒ std-gated hard format gate |
| `REWARD_TYPE` | reward_type | `power` (behavior) | `log_prob`/`neg_perplexity`/`power`/`rule_based_tom`/`SFT` |
| `POWER_K` | power_k | `2.0` | `-` unless reward=power |
| `POWER_LL_MIN` | ll_min | `-8.0` | `-` unless reward=power |
| `USE_ACTOR_AS_RM` | RM mode | `True`=actor (behavior) | ⚠ if unset, run-name mis-tags `frozenRM` (common.sh L79 bug) — set explicitly |
| `SUBTRACT_BASELINE` | baseline | `False` | |
| `NUM_GPUS` / `TP_SIZE` | gpus/TP | `8` / `2` | |
| `SAVE_FREQ` / `TEST_FREQ` | (fixed) | `50` / `30` | eval cadence (steps) |
| `VAL_SUITE` | eval suite | `subsample300` | `full`/`core`/`sanity` |
| `SEED` | seed | verl default `1` | set explicitly if varying |
| gen temperature | temp | verl default `1.0` | note in `temp` if varied |
| `REQUIRE_ANSWER_TAGS` | ans_tags | model-decided (Qwen2.5=yes; Qwen3/Gemma=no) | data must match model (tagged vs tag-free) |

---

## Column meanings
**Identity:** `Exp #` (global unique handle) · `Exp ID` (`<RQ>-<slug>`) · `RQ tag`
(`Phase-stability`/`data-recipe`/`reward-shaping`/`identifiability`/`generalization`/`interaction`/
`cot-style`/`model-independence`/`test`) · `Question` (one-line goal).
**Model:** `Model` family · `Size` · `RM mode` (`frozen`|`actor`).
**Reward:** `reward_type` · `power_k` · `ll_min` · `baseline` (subtract baseline `T`/`F`).
**RL/optim (all explicit — no silent defaults):** `KL` · `LR` · `train_batch` · `mini_batch` ·
`micro_batch` · `rollout_n` · `epochs` · `max_prompt` · `max_resp` · `entropy_coeff` ·
`format_penalty` · `fp_std_coef` · `temp` · `seed` · `gpus/TP`.
**Data/prompt:** `Data sources` · `Data params` (turn filter, mixture, caps, shuffle) ·
`Data config` (`dcfg_*` YAML+parquet stem) · `CoT prompt var` · `ans_tags` (require_answer_tags).
**Eval:** `eval suite` (`subsample300` default) · `Target evals`.
**Bookkeeping:** `Run name` (base stem **`<Exp #>-<Exp ID>-<Data config>`** — the launcher prepends
the short `Exp #` row id via `EXP_NUM` so every WandB run starts with e.g. `E034-…` for search, then
appends `-<model>-<params>-r<N>`) · `WandB link` (full URL) ·
`Log path` (`logs/<YYYYMMDD>/<RUN_NAME>.log`) · `Summary doc` (`experiments_logs/<stem>.md`) ·
`Owner_host` · `Status` (`Backlog`|`Not-started`|`Processing`|`Training`|`In-debug`|
`Awaiting-input`|`Completed`|`Failed`) · `Results summary` · `Notes`.

**Config-selection score (all rows, comparable):** `python scripts/score_run.py <Log path>` →
**ToM HM** = harmonic mean over the 24 ToM benchmarks **excluding `gsm8k`/`mmlu`** (report HM(last5)
primary + HM(last3) + step-0 baseline). `gsm8k`/`mmlu` reported separately (Δ vs step0); parseable
rate + health (KL, resp_len, entropy) tracked separately, never in HM. Prefer format-controlled
conditional accuracy (`d_cavg`, from `scripts/reassess_runs.py`) for cross-family recipe selection.

---

## Experiments (0 rows — v2 started 2026-07-19; add rows below as the new stage is planned)
| Exp # | Exp ID | RQ tag | Question | Model | Size | RM mode | reward_type | power_k | ll_min | baseline | KL | LR | train_batch | mini_batch | micro_batch | rollout_n | epochs | max_prompt | max_resp | entropy_coeff | format_penalty | fp_std_coef | temp | seed | gpus/TP | Data sources | Data params | Data config | CoT prompt var | ans_tags | eval suite | Target evals | Run name | WandB link | Log path | Summary doc | Owner_host | Status | Results summary | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
