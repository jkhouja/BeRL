# DailyDialog Dataset Conversion

This directory contains scripts to convert the [roskoN/dailydialog](https://huggingface.co/datasets/roskoN/dailydialog) dataset to the NegotiationToM format for RL training.

## Scripts

### `convert_dailydialog.py`

Main conversion script that downloads the DailyDialog dataset and converts it to match the NegotiationToM parquet format.

**Features:**
- Downloads dataset directly from HuggingFace
- Configurable filtering by conversation length and response word count
- Creates multiple training examples per conversation (one for each turn)
- Supports sampling for smaller datasets
- Outputs parquet file compatible with existing RL training pipeline

**Usage:**

```bash
# Basic usage - convert entire train split
python scripts/convert_dailydialog.py

# Convert with custom parameters
python scripts/convert_dailydialog.py \
    --output-dir data \
    --output-name DailyDialog_train_limit1000 \
    --min-turns 4 \
    --max-turns 15 \
    --min-response-words 5 \
    --max-response-words 100 \
    --sample-size 1000 \
    --split train \
    --seed 42

# Convert validation split
python scripts/convert_dailydialog.py \
    --split validation \
    --output-name DailyDialog_validation

# Convert test split with filtering
python scripts/convert_dailydialog.py \
    --split test \
    --output-name DailyDialog_test \
    --min-turns 6 \
    --max-response-words 50
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output-dir` | str | `data` | Output directory for converted dataset |
| `--output-name` | str | `DailyDialog` | Base name for output file |
| `--min-turns` | int | `4` | Minimum conversation turns to include |
| `--max-turns` | int | `None` | Maximum conversation turns (None for no limit) |
| `--min-response-words` | int | `5` | Minimum words in response |
| `--max-response-words` | int | `None` | Maximum words in response (None for no limit) |
| `--sample-size` | int | `None` | Number of examples to sample (None for all) |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--split` | str | `train` | Dataset split: train, validation, or test |

### `test_dailydialog_conversion.py`

Test script that validates the conversion process and output format.

**Usage:**

```bash
# Run tests
python scripts/test_dailydialog_conversion.py
```

**What it tests:**
- Downloads and converts a small sample (100 examples)
- Validates all required columns are present
- Checks data types and structures match NegotiationToM format
- Compares column sets with existing NegotiationToM dataset
- Displays sample examples for manual inspection

## Output Format

The converted dataset matches the NegotiationToM parquet format with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `prompt_len` | int/None | Length of prompt tokens (filled during tokenization) |
| `response_words` | int | Number of words in the response |
| `data_source` | str | Always "dailydialog" |
| `prompt` | list[dict] | Chat messages for training (user message only) |
| `raw_system_prompt` | str | System prompt text |
| `raw_user_prompt` | str | User prompt text |
| `prompt_for_pp` | None | Filled during perplexity calculation |
| `ability` | str | Always "conversation_generation" |
| `reward_model` | dict | Contains `ground_truth` and `style` |
| `metadata` | dict | Contains `conv_id`, `turn`, `total_turns`, `responding_speaker` |
| `answer_pp` | None | Filled during perplexity calculation |
| `raw_prompt` | list[dict] | Full chat messages (system + user) |

## Training Examples

For each conversation, the script creates multiple training examples:

```
Conversation: [Turn1, Turn2, Turn3, Turn4, Turn5]

Example 1:
  History: [Turn1]
  Predict: Turn2

Example 2:
  History: [Turn1, Turn2]
  Predict: Turn3

Example 3:
  History: [Turn1, Turn2, Turn3]
  Predict: Turn4

Example 4:
  History: [Turn1, Turn2, Turn3, Turn4]
  Predict: Turn5
```

This maximizes the training data extracted from each conversation.

## Dataset Statistics

The original roskoN/dailydialog dataset contains:
- **Train split**: ~11k conversations
- **Validation split**: ~1k conversations
- **Test split**: ~1k conversations

After conversion with default settings (`min_turns=4`, `min_response_words=5`), you can expect:
- Approximately 80-100k training examples from the train split
- Filtering removes very short conversations
- Each conversation generates multiple examples (one per turn after the first)

## Integration with RL Training

The converted dataset can be used with the existing RL training pipeline:

1. **Convert the dataset:**
   ```bash
   python scripts/convert_dailydialog.py --sample-size 1000 --output-name DailyDialog_limit1000
   ```

2. **Use in training:**
   The parquet file can be loaded directly by the existing data loaders since it matches the NegotiationToM format.

3. **Mix with other datasets:**
   You can combine DailyDialog with NegotiationToM for multi-domain RL training.

## Examples

### Create a small test dataset
```bash
python scripts/convert_dailydialog.py \
    --sample-size 500 \
    --output-name DailyDialog_test_500 \
    --min-turns 4 \
    --max-turns 10
```

### Create full training dataset with quality filters
```bash
python scripts/convert_dailydialog.py \
    --split train \
    --output-name DailyDialog_train_filtered \
    --min-turns 6 \
    --min-response-words 8 \
    --max-response-words 80
```

### Create validation set
```bash
python scripts/convert_dailydialog.py \
    --split validation \
    --output-name DailyDialog_validation \
    --min-turns 4
```

## Notes

- The script uses `tqdm` for progress bars during conversion
- All examples use alternating speakers: `Speaker_1` and `Speaker_2`
- The `answer_pp` and `prompt_len` fields are left as `None` and will be calculated by the training pipeline (similar to how NegotiationToM is processed)
- Random sampling is deterministic when using the same seed
- The conversion preserves the original dialogue text without modification
