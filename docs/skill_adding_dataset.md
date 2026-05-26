# Adding a New Training Dataset

This guide covers the end-to-end process for adding a new training dataset to the BeRL dialogue→ToM pipeline.

## Pipeline Overview

```
Raw Dataset (HuggingFace / local)
    ↓
Converter Script (scripts/convert_*.py)
    ↓
YAML Pipeline Config (scripts/configs/pipeline_config_*.yaml)
    ↓
Build Script (python build_dataset.py --config ...)
    ↓
Final Training Parquet (data/*.parquet)
    ↓
GRPO Training (experiments/*.sh)
```

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

## Step 3: Create a YAML Pipeline Config

Create `scripts/configs/pipeline_config_<description>.yaml`:

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
    min_response_words: 10
    max_response_words: null
    prompt_style: "cot_tom"
    system_prompt_style: "cot_eval"    # use cot_eval for best transfer
    generation_prefix: "<think>"
    add_response_tags: true
    include_system_in_prompt: true
    multi_sample: true
    turn_order: "random"

  # Can include multiple datasets — they get merged
  - source: "dailydialog"
    split: "train"
    sample_size: 3000
    min_turns: 6
    min_response_words: 10
    prompt_style: "cot_tom"
    system_prompt_style: "cot_eval"
    generation_prefix: "<think>"
    add_response_tags: true
    include_system_in_prompt: true
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
python build_dataset.py --config scripts/configs/pipeline_config_my_experiment.yaml
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

## Step 6: Create an Experiment Script

Create `experiments/<experiment_name>.sh` following existing scripts. Key parameters to set:

```bash
data.train_files=/path/to/your/training.parquet
data.val_files=[/path/to/eval1.parquet,/path/to/eval2.parquet]
```

See `experiments/dialogue_grpo_fantom_eval.sh` for a complete example.

## Best Practices

1. **Prompt alignment is critical** — training system prompt must match eval system prompt. Use `cot_eval` style.
2. **Data quality > quantity** — filtered 6k beats noisy 16k.
3. **Use `min_turns: 6`** — conversations need enough context for ToM signal.
4. **Use `min_response_words: 10`** — filters out trivial responses.
5. **Keep `generation_prefix: "<think>"`** — aligns with CoT reasoning format.
6. **Version control your configs** — configs in `scripts/configs/` are the reproducibility record.
7. **Name configs descriptively** — `pipeline_config_[size]_[prompt_style]_[filter_desc].yaml`.

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
| `scripts/configs/pipeline_config_*.yaml` | Pipeline configurations |
| `examples/data_preprocess/` | Eval dataset preprocessing |
| `verl/utils/reward_score/` | Scoring modules for eval |
| `verl/trainer/main_ppo.py` | Score function routing |
| `experiments/` | Training launch scripts |
