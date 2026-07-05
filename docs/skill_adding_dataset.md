# Adding a New Training Dataset

This guide covers the end-to-end process for adding a new training dataset to the BeRL dialogue→ToM pipeline.

## Pipeline Overview

```
Raw Dataset (HuggingFace / local)
    ↓
Converter Script (scripts/convert_*.py)
    ↓
YAML Data Config (scripts/configs/dcfg_*.yaml, extends dcfg_base.yaml)
    ↓
Build Script (python build_dataset.py --config ...)
    ↓
Final Training Parquet (data/*.parquet)
    ↓
GRPO Training (experiments/*.sh)
```

## Default Data Generation Parameters

Unless an experiment or grid-search deliberately overrides them, use these sensible defaults for every dataset entry in a pipeline config:

```yaml
min_turns: 6              # conversations need enough context for ToM signal
min_response_words: 3     # drop trivial one/two-word responses
max_response_words: 100   # cap over-long responses that dominate the LL reward
turn_order: "random"      # avoid positional bias across turns
```

Additional conditional defaults:

- **Human ↔ assistant/AI datasets** (e.g. ThoughtTrace, human–AI chat logs): only extract the **human** turns as prediction targets by setting `target_speaker: "user"`. We model the human's mental state, not the assistant's scripted output.
- **Datasets with splits**: use `split: "train"` for training data (reserve `validation`/`test` for eval-only benchmarks).

These are defaults, not hard rules — override any of them for a specific experiment or when sweeping in a grid-search, and document the deviation in the config.

## Step 1: Create a Converter

Create `scripts/convert_<name>.py` following the existing converter pattern. Your converter class must implement `download_dataset()` and `convert()`.

**Reference implementations:**
- `scripts/convert_dailydialog.py` — HuggingFace download + conversation parsing
- `scripts/convert_empathetic_dialogues.py` — Similar with emotion metadata
- `scripts/convert_theory_of_mind.py` — Reshapes existing parquet files

### Converter class skeleton

```python
class MyDatasetConverter:
    def __init__(self, config):
        self.config = config
        # config fields: split, sample_size, min_turns, max_turns,
        # min_response_words, max_response_words, prompt_style,
        # system_prompt_style, generation_prefix, add_response_tags,
        # include_system_in_prompt, multi_sample, turn_order, seed, etc.

    def download_dataset(self):
        """Download/load the raw dataset. Return raw data."""
        pass

    def convert(self):
        """Convert raw data to training parquet format.
        Returns a DataFrame with the required schema (see Step 4).
        """
        pass
```

### Key conversion logic

For dialogue datasets, the standard pattern is:
1. Filter conversations by `min_turns` / `max_turns` and `min_response_words` / `max_response_words`
2. For each conversation, create training examples by taking dialogue history up to turn `i` and predicting turn `i+1`
3. Apply `multi_sample` (all turns vs one random turn per conversation)
4. Apply `turn_order` sorting (`random`, `early_first`, `late_first`)
5. Format prompts using templates from `scripts/prompt_templates.py`

## Step 2: Register the Converter

In `build_dataset.py`, add your converter to the `CONVERTERS` registry:

```python
CONVERTERS = {
    "dailydialog": ("scripts.convert_dailydialog", "DailyDialogConverter"),
    "empathetic_dialogues": ("scripts.convert_empathetic_dialogues", "EmpatheticDialoguesConverter"),
    "theory_of_mind": ("scripts.convert_theory_of_mind", "TheoryOfMindConverter"),
    "my_dataset": ("scripts.convert_my_dataset", "MyDatasetConverter"),  # <-- add this
}
```

## Step 3: Create a YAML Pipeline Config (`dcfg_*` + `extends`)

Data configs live in `scripts/configs/` and are named `dcfg_<name>.yaml`. The config
**filename stem = `output_name` = the tracker's `Data config name` = the output parquet stem**
(`data/dcfg_<name>.parquet`). The `run_experiment.sh` dispatcher relies on this 1:1 mapping.

Rather than repeating the full parameter block, **inherit shared defaults from `dcfg_base.yaml`**
via `extends:` and declare only your `datasets:` list plus any per-source overrides:

```yaml
extends: dcfg_base.yaml          # shared pipeline + perplexity + dataset_defaults

pipeline:
  output_name: "dcfg_my_dataset" # == filename stem == data/dcfg_my_dataset.parquet

datasets:
  - source: "my_dataset"         # only override what differs from dcfg_base
    # sample_size: 6000          # null (base default) = use all
    # target_speaker: "user"     # set for human <-> AI datasets (predict human turns)
```

`extends` accepts a single path or a list (resolved relative to the config's directory) and
deep-merges parents first, then this file's keys (child wins). The `dataset_defaults:` block in
`dcfg_base.yaml` is applied to **every** dataset entry (below the hardcoded
`build_dataset.DATASET_DEFAULTS`, above the per-entry values). `dcfg_base.yaml` already encodes the
Default Data Generation Parameters above (`min_turns:6`, `min_response_words:3`,
`max_response_words:100`, `turn_order:random`, `simple`/`cot_eval`, `split:train`).

**Tag-free (Qwen3 / Gemma):** `extends: dcfg_base_notags.yaml` instead — it overrides
`system_prompt_style: cot_eval_notags`, `add_response_tags: false`, `generation_prefix: ""`.

The fully-explicit (non-`extends`) schema — every available key with inline docs — is
`scripts/configs/pipeline_config.yaml`, kept as the reference example:

```yaml
pipeline:
  output_dir: "data"
  output_name: "merged_dialogue_datasets_my_experiment"
  seed: 42
  shuffle: true

perplexity:
  enabled: true
  model_name: "Qwen/Qwen2.5-3B-Instruct"
  batch_size: 8

datasets:
  - source: "my_dataset"
    split: "train"
    sample_size: 6000          # null = use all
    min_turns: 6
    max_turns: null
    min_response_words: 3
    max_response_words: 100
    prompt_style: "simple"
    system_prompt_style: "cot_eval"    # use cot_eval for best transfer
    generation_prefix: "<think>"
    add_response_tags: true
    include_system_in_prompt: true
    multi_sample: true
    turn_order: "random"
    # target_speaker: "user"   # set for human <-> AI datasets (predict human turns)
```

### Config field reference

| Field | Type | Description |
|-------|------|-------------|
| `source` | str | Converter key from `CONVERTERS` registry |
| `split` | str | HuggingFace dataset split (`train`, `validation`, `test`) |
| `sample_size` | int/null | Subsample size (null = all data) |
| `min_turns` / `max_turns` | int/null | Filter conversations by turn count |
| `min_response_words` / `max_response_words` | int/null | Filter by response word count |
| `limit_turn` | int/null | Only use turns >= N |
| `prompt_style` | str | User prompt template (see `scripts/prompt_templates.py`) |
| `system_prompt_style` | str | System prompt template — **use `cot_eval` for best ToM transfer** |
| `generation_prefix` | str | Prefix for model generation (typically `<think>`) |
| `add_response_tags` | bool | Wrap ground truth in `<answer></answer>` tags |
| `include_system_in_prompt` | bool | Include system message in prompt dict |
| `multi_sample` | bool | True = all valid turns per conversation, False = one random turn |
| `turn_order` | str | `random`, `early_first`, or `late_first` |

### Prompt templates

Available styles are defined in `scripts/prompt_templates.py`:

**System prompts** (`system_prompt_style`):
- `cot_eval` — **Best for ToM transfer.** Generic CoT with `<think>` tags + ToM framing
- `cot_tom` — Intent/belief/goals framework (degraded performance in ablations)
- `cot_tom2` — All-parties perspective (caused catastrophic collapse)
- `default`, `research`, `empathy`, `cot` — Other variants

**User prompts** (`prompt_style`):
- `simple`, `detailed`, `baseline` — Dialogue prediction variants
- `cot_tom` — Theory-of-Mind reasoning prompt
- `empathy` — Includes `{emotion_label}` placeholder

## Step 4: Output Parquet Schema

The final training parquet must contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | list[dict] | `[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]` |
| `raw_system_prompt` | str | System prompt text |
| `raw_user_prompt` | str | User prompt text |
| `raw_prompt` | list[dict] | Same as `prompt` (explicit system + user messages) |
| `data_source` | str | Dataset identifier (e.g., `"dailydialog"`, `"my_dataset"`) |
| `ability` | str | Task category (e.g., `"conversation_generation"`) |
| `reward_model` | dict | `{"ground_truth": "response text", "style": "rule"}` |
| `metadata` | dict | `{conv_id, turn, total_turns, responding_speaker, dialogue_history, ...}` |
| `generation_prefix` | str | `"<think>"` |
| `response_with_tags` | str | `"<answer>ground truth response</answer>"` |
| `response_words` | int | Word count of ground truth response |
| `prompt_len` | int/None | Tokenized prompt length (filled by perplexity step) |
| `answer_pp` | float | Log-probability of response (filled by perplexity step) |
| `prompt_for_pp` | None | Reserved for perplexity preprocessing |

For **eval-only** datasets (like FANToM, tomi, hi_tom), the schema is simpler — see `examples/data_preprocess/prepare_fantom.py` and `examples/data_preprocess/merge_tom.py`. Eval parquets need: `prompt`, `data_source`, `ability`, `reward_model`, `extra_info`, `answer`, `question`, `story`.

## Step 5: Build the Dataset

```bash
python build_dataset.py --config scripts/configs/dcfg_my_dataset.yaml
```

This will:
1. Run each dataset's converter
2. Merge all outputs into one parquet
3. Shuffle with the configured seed
4. Compute perplexity scores (if enabled)
5. Save to `data/<output_name>.parquet`

You can also merge parquets manually:

```bash
python merge_parquet.py data/dataset1.parquet data/dataset2.parquet -o data/merged.parquet
```

## Step 6: Launch Training

Do **not** write a per-dataset training script. Use the generic launchers (task × model-class),
which read all knobs from env and reproduce a single `main_ppo` invocation. Point them at your
new parquet via `DATA_TRAIN` (and `DATA_NAME` = your dcfg/parquet stem):

```bash
# behavior-reward GRPO on your new dataset (Qwen2.5)
EXP_ID=<exp_id> DATA_NAME=<dcfg_stem> \
  DATA_TRAIN=/path/to/your/training.parquet \
  bash experiments/train_behavior_qwen2.5.sh

# preview the exact command without launching:
BERL_DRY_RUN=1 EXP_ID=... DATA_NAME=... DATA_TRAIN=... bash experiments/train_behavior_qwen2.5.sh
```

Or, if the dataset is tied to a tracker row, run it through the dispatcher:
`bash experiments/run_experiment.sh <EXP_ID>`. See `experiments/README.md` for the full launcher
reference and `experiments/lib/common.sh` for the env-knob surface.

## Best Practices

1. **Prompt alignment is critical** — training system prompt must match eval system prompt. Use `cot_eval` style.
2. **Data quality > quantity** — filtered 6k beats noisy 16k.
3. **Use `min_turns: 6`** — conversations need enough context for ToM signal.
4. **Use `min_response_words: 3` and `max_response_words: 100`** — drops trivial responses and caps over-long ones that dominate the LL reward.
5. **Use `turn_order: "random"`** — avoids positional bias across turns.
6. **For human ↔ AI datasets, set `target_speaker: "user"`** — predict the human's turns only.
7. **Keep `generation_prefix: "<think>"`** — aligns with CoT reasoning format (tag-free for Qwen3/Gemma; see §"Adding a New Training Dataset" answer-tag rules).
8. **Version control your configs** — configs in `scripts/configs/` are the reproducibility record.
9. **Name configs `dcfg_<name>.yaml`** — filename stem must equal `output_name` and the parquet stem; extend `dcfg_base.yaml` (or `dcfg_base_notags.yaml`) rather than duplicating defaults.

## Adding an Eval-Only Benchmark

For eval benchmarks (no training), follow the pattern in:
- `examples/data_preprocess/prepare_fantom.py` — Downloads, processes, outputs eval parquet
- `examples/data_preprocess/merge_tom.py` — Reshapes existing ToM datasets

You also need a scoring module in `verl/utils/reward_score/<name>.py` with a `compute_score(solution_str, ground_truth) -> float` function, and routing in `verl/trainer/main_ppo.py` `_select_rm_score_fn()`.

## File Reference

| File | Purpose |
|------|---------|
| `build_dataset.py` | Main pipeline orchestrator |
| `merge_parquet.py` | Parquet merge utility |
| `scripts/convert_*.py` | Dataset converters |
| `scripts/prompt_templates.py` | All prompt/system prompt templates |
| `scripts/configs/dcfg_*.yaml` | Data configs (extend `dcfg_base.yaml`) |
| `examples/data_preprocess/` | Eval dataset preprocessing |
| `verl/utils/reward_score/` | Scoring modules for eval |
| `verl/trainer/main_ppo.py` | Score function routing |
| `experiments/` | Training launch scripts |
