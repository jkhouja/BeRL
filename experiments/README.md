# BeRL experiment launchers

All training runs go through **one** parameterised `main_ppo` code path. There are no
per-experiment scripts — instead a small matrix of thin launchers (task × model-class) that source
the shared engine `lib/common.sh`, plus a dispatcher that resolves a tracker row.

## Layout

| File | Purpose |
|------|---------|
| `lib/common.sh` | Shared engine: env setup (conda `tom`, **WandB online**), model-family branch, `RUN_NAME` build, reproducibility-md emit, arg assembly, `main_ppo` run. Not run directly. |
| `train_behavior_{qwen2.5,qwen3,gemma}.sh` | Behavior-reward GRPO (`log_prob`/`neg_perplexity`/`power`; frozen/actor RM). |
| `train_tom_{qwen2.5,qwen3,gemma}.sh` | Direct **rule-based** ToM GRPO (no LM reward model; `data_source`→rule scoring; KL=0.001). |
| `train_sft_qwen2.5.sh` | SFT baseline (Q0 A2). **[build] — not wired yet.** |
| `smoke_{qwen2.5,qwen3,gemma}.sh` | Short smoke runs (`EXP_ID=test-…`, 1 epoch, `test_freq=5`). |
| `phase_stability_sweep.sh` | Phase −1 one-shot HP+reward sweep driver: resolves each of the 96 Wave-1 cells' knobs + matching `EXP_ID` (tracker `PS001–PS096`). Cells are claimed/launched per-row (`ONLY_IDX="<n>"`) and parallelize across agents/nodes; sequential `IDX_START`/`IDX_END` chunking is a single-node convenience. See "Phase −1 sweep" below. |
| `run_experiment.sh <EXP_ID>` | Dispatcher: reads `project_planning/experiments.tsv`, maps knobs → env, calls the right launcher. |

## Usage

Direct launcher (env-driven; required: `EXP_ID`, `DATA_NAME`, `DATA_TRAIN`):

```bash
EXP_ID=Phase0-p0_power DATA_NAME=dcfg_default \
  DATA_TRAIN=$HOME/repo/BeRL/data/dcfg_default.parquet \
  bash experiments/train_behavior_qwen2.5.sh
```

Dispatcher (resolves a tracker row via the sidecar):

```bash
python scripts/tracker_to_sidecar.py          # regenerate sidecar when tracker knobs change
bash experiments/run_experiment.sh <EXP_ID>   # env vars you export first WIN over the sidecar
```

Preview without launching:

```bash
BERL_DRY_RUN=1 EXP_ID=... DATA_NAME=... DATA_TRAIN=... bash experiments/train_behavior_qwen2.5.sh
```

## Naming

`RUN_NAME = <EXP_ID>-<DATA_NAME>-<MODEL_TAG>-<PARAMS>-r<N>` = WandB run name = log filename stem
(`logs/<YYYYMMDD>/<RUN_NAME>.log`). The base stem `<EXP_ID>-<DATA_NAME>` matches the tracker
`Run name` cell. `common.sh` auto-picks the next free `-r<N>` (scan of `logs/`) or honors
`RUN_INDEX=`. Each attempt appends to the reproducibility record `experiments_logs/<EXP_ID>-<DATA_NAME>.md`.

## Key env knobs (see `lib/common.sh` for the full list + defaults)

`MODEL_PATH`, `REWARD_TYPE` (power|log_prob|neg_perplexity), `POWER_K`, `POWER_LL_MIN`,
`USE_ACTOR_AS_RM` (True|False), `SUBTRACT_BASELINE`, `KL`, `LR`, `ROLLOUT_N`, `MAX_PROMPT`,
`MAX_RESP`, `TOTAL_EPOCHS`, `SAVE_FREQ`, `TEST_FREQ`, `VAL_FILES`, `VAL_SUITE` (subsample300|full|core|sanity),
`VAL_METRIC_SUFFIX`, `DATA_TRAIN`, `DATA_NAME`,
`ENTROPY_COEFF` (actor entropy bonus, default 0.001), `THINK_ONLY_PG` (restrict PG+entropy to
`<think>…</think>`, default False), `FORMAT_PENALTY` (graded penalty for malformed responses,
default 0.0 — wired to both `actor.*` and `reward_model.*`),
`PROJECT_NAME` (WandB project, default **`TOM_EXP`**),
`EXP_ID`, `RUN_INDEX`, `GPU_IDS` (pin GPUs), `SYSTEM_PROMPT` (prompt-alignment Option A),
`BERL_DRY_RUN`, `BERL_NO_EXP_LOG` (skip the `experiments_logs/` reproducibility record for
verification/test runs; **WandB stays on**).

## Model-family specifics (set automatically per launcher)

- **qwen2.5** — `XFORMERS`, `require_answer_tags=True` (tagged data), gpu_mem 0.35, resp 4096.
- **qwen3** — `XFORMERS`, `require_answer_tags=False` (**tag-free** data), gpu_mem 0.35.
- **gemma** — `FLASH_ATTN` + `expandable_segments`, `+data.fold_system_prompt=True`,
  `require_answer_tags=False` (tag-free), gpu_mem 0.3, resp 1024, and the reward config is also
  passed on `actor_rollout_ref.*`.

`+data.truncation=left` is a **hard default** (fixes the `sequence_length>max_prompt_length` crash).

## Phase −1 sweep (`phase_stability_sweep.sh`)

The Phase −1 phase-stability sweep is a 96-cell factorial (`PS001–PS096`), and each cell is a
**normal, individually-claimable tracker row** — cells parallelize across agents/nodes via the
standard claim protocol, they do **not** run as one sequential batch. The driver
`phase_stability_sweep.sh` enumerates the 96 Wave-1 cells in the exact order of tracker rows
`PS001–PS096` (reward {log_prob, power k3/k5 ll_min=−6} × RM{frozen,actor} × KL{0.01,0.05} ×
LR{5e-7,1e-6} × format_penalty{0,5} × entropy_coeff{0,0.001}), sets each cell's env, and calls the
matching family launcher. Its job is to **resolve one cell's knobs + matching `EXP_ID`** given its
index — so the normal way to run a claimed row `PS<n>` is **single-cell mode `ONLY_IDX="<n>"`**. Fixed
per cell: `THINK_ONLY_PG=True`, `ROLLOUT_N=16`, ctx `2048/512`, `COT_VAR=cot_eval` (the tracker's
`COT_FREEFORM`), **1 epoch (191 steps), `TEST_FREQ=10`, `SAVE_FREQ=999`** (eval-only; score post-hoc
from WandB). Each generated `EXP_ID` matches the tracker's `Exp ID` column exactly.

```bash
# Standard per-row use: claim the lowest Not-started PS row, then launch just that cell.
ONLY_IDX="7" bash experiments/phase_stability_sweep.sh              # runs exactly PS007 (1 cell)
GPU_IDS=0,1 ONLY_IDX="7" bash experiments/phase_stability_sweep.sh  # pin 2 GPUs → several cells per node
FAMILY=gemma ONLY_IDX="7" bash experiments/phase_stability_sweep.sh # Wave 2 (gemma|qwen3)
BERL_DRY_RUN=1 ONLY_IDX="7" bash experiments/phase_stability_sweep.sh # preview EXP_ID/knobs (no launch)

# Convenience multi-cell modes (single-node; use only when NOT coordinating via the tracker):
IDX_START=1 IDX_END=32 bash experiments/phase_stability_sweep.sh    # log_prob block, sequential
bash experiments/phase_stability_sweep.sh                          # all 96 Qwen2.5, sequential
```

`FAMILY` (default `qwen2.5`) selects the launcher + data (`dcfg_smoke_mix` for qwen2.5,
`dcfg_smoke_mix_gemma` for gemma/qwen3). Prefer per-row claiming + `ONLY_IDX="<n>"` for parallel
execution across agents/nodes; `IDX_START`/`IDX_END` sequential chunking is a single-node convenience
for uncoordinated runs. A failed cell logs and (in multi-cell mode) the driver continues. Runs log to
WandB project `TOM_EXP`. Wave 2 runs only cells whose Qwen2.5 counterpart was not very poor (gated
post-Wave-1).

See `tests/launcher/test_launcher_knobs.sh` for dry-run assertions that the swept knobs propagate.

See the `launch-experiment` / `claim-experiment` / `check-training` / `log-results` skills for the
full tracker-driven protocol.
