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
`MAX_RESP`, `TOTAL_EPOCHS`, `SAVE_FREQ`, `TEST_FREQ`, `VAL_FILES`, `DATA_TRAIN`, `DATA_NAME`,
`EXP_ID`, `RUN_INDEX`, `GPU_IDS` (pin GPUs), `SYSTEM_PROMPT` (prompt-alignment Option A),
`BERL_DRY_RUN`.

## Model-family specifics (set automatically per launcher)

- **qwen2.5** — `XFORMERS`, `require_answer_tags=True` (tagged data), gpu_mem 0.35, resp 4096.
- **qwen3** — `XFORMERS`, `require_answer_tags=False` (**tag-free** data), gpu_mem 0.35.
- **gemma** — `FLASH_ATTN` + `expandable_segments`, `+data.fold_system_prompt=True`,
  `require_answer_tags=False` (tag-free), gpu_mem 0.3, resp 1024, and the reward config is also
  passed on `actor_rollout_ref.*`.

`+data.truncation=left` is a **hard default** (fixes the `sequence_length>max_prompt_length` crash).

See the `launch-experiment` / `claim-experiment` / `check-training` / `log-results` skills for the
full tracker-driven protocol.
