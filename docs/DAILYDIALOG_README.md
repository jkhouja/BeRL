# DailyDialog Dataset Converter

A highly configurable script to convert the DailyDialog dataset to NegotiationToM format for RL training. This script is inspired by `verl/utils/dataset/tom_dataset.py` and provides extensive customization options for prompts, filtering, and output formatting.

## Features

- **Flexible Prompt Configuration**: Multiple prompt styles and templates
- **Custom System Prompts**: Choose from predefined styles or provide your own
- **Response Formatting**: Add tags (e.g., `<answer>...</answer>`) for structured outputs
- **Generation Prefixes**: Support for chain-of-thought style prompts
- **Advanced Filtering**: Filter by turns, word count, and more
- **Config File Support**: Load all settings from YAML or JSON files
- **Turn-Level Control**: Only include specific turns (e.g., skip early conversation)

## Quick Start

### Basic Usage

```bash
# Convert with default settings
python scripts/convert_dailydialog.py

# Limit to 1000 examples
python scripts/convert_dailydialog.py --output-name DailyDialog_train_limit1000 --sample-size 1000

# Use validation split
python scripts/convert_dailydialog.py --split validation --output-name DailyDialog_val
```

### Using Configuration Files

```bash
# Use a config file (recommended for complex setups)
python scripts/convert_dailydialog.py --config scripts/dailydialog_config_example.yaml
```

See `dailydialog_config_example.yaml` for configuration options and examples.

## Configuration Options

### Filtering Options

- `--min-turns`: Minimum conversation turns (default: 4)
- `--max-turns`: Maximum conversation turns (default: no limit)
- `--min-response-words`: Minimum words per response (default: 5)
- `--max-response-words`: Maximum words per response (default: no limit)
- `--limit-turn`: Only include turns >= this number (default: include all)

### Prompt Configuration

#### System Prompts

**Predefined Styles:**
- `default`: Basic conversational assistant prompt
- `research`: Research-oriented linguist and communication expert

**Custom:**
```bash
python scripts/convert_dailydialog.py --system-prompt "Your custom system prompt here"
```

#### User Templates

**Predefined Styles:**
- `simple`: Basic prediction prompt
- `detailed`: More elaborate conversation continuation style
- `baseline`: Minimal baseline prompt

**Custom:**
```bash
python scripts/convert_dailydialog.py --user-template "Custom template with {dialogue_history} and {responding_speaker} placeholders"
```

**Template Placeholders:**
- `{dialogue_history}`: The conversation history string
- `{responding_speaker}`: The speaker who will respond next (e.g., "Speaker_1")

### Response Formatting

```bash
# Add response tags (useful for structured outputs)
python scripts/convert_dailydialog.py \
    --add-response-tags \
    --response-tag-open "<answer>" \
    --response-tag-close "</answer>"

# Add generation prefix (e.g., for chain-of-thought)
python scripts/convert_dailydialog.py --generation-prefix "<think>"
```

### Output Options

```bash
# Specify output directory and name
python scripts/convert_dailydialog.py \
    --output-dir data \
    --output-name DailyDialog_custom

# Set dataset metadata
python scripts/convert_dailydialog.py \
    --ability conversation_generation \
    --data-source dailydialog
```

## Common Use Cases

### 1. Research-Style with Tags

Mimics the NegotiationToM format with research-oriented prompts:

```bash
python scripts/convert_dailydialog.py \
    --system-prompt-style research \
    --prompt-style detailed \
    --add-response-tags \
    --output-name DailyDialog_research
```

### 2. Chain-of-Thought Style

Encourages reasoning before response:

```bash
python scripts/convert_dailydialog.py \
    --generation-prefix "<think>" \
    --add-response-tags \
    --response-tag-open "<answer>" \
    --response-tag-close "</answer>" \
    --output-name DailyDialog_cot
```

### 3. Later Turns Only

Focus on more developed conversations:

```bash
python scripts/convert_dailydialog.py \
    --limit-turn 2 \
    --min-turns 6 \
    --output-name DailyDialog_later_turns
```

### 4. Using Config File

Create a YAML file (`my_config.yaml`):

```yaml
# Output
output_dir: "data"
output_name: "DailyDialog_custom"

# Filtering
min_turns: 6
limit_turn: 2
sample_size: 1000

# Prompts
system_prompt_style: "research"
prompt_style: "detailed"
add_response_tags: true

# Metadata
ability: "conversation_generation"
data_source: "dailydialog"
```

Then run:

```bash
python scripts/convert_dailydialog.py --config my_config.yaml
```

## Output Format

The script generates a Parquet file with the following structure:

| Column | Description |
|--------|-------------|
| `prompt_len` | Length of the prompt (to be filled during tokenization) |
| `response_words` | Number of words in the ground truth response |
| `data_source` | Dataset identifier (default: "dailydialog") |
| `prompt` | Chat-formatted prompt (list of dicts) |
| `raw_system_prompt` | System prompt string |
| `raw_user_prompt` | User prompt string |
| `prompt_for_pp` | Full prompt for perplexity calculations |
| `ability` | Ability tag (default: "conversation_generation") |
| `reward_model` | Dict with ground truth and style |
| `metadata` | Dict with conv_id, turn, total_turns, responding_speaker, dialogue_history |
| `answer_pp` | Answer perplexity (to be calculated) |
| `raw_prompt` | Full chat template including system prompt |
| `generation_prefix` | Optional generation prefix (if specified) |
| `response_with_tags` | Optional tagged response (if tags enabled) |

## Comparison with tom_dataset.py

This script mirrors the configurability of `tom_dataset.py`:

| Feature | tom_dataset.py | convert_dailydialog.py |
|---------|----------------|------------------------|
| Config dict support | ✅ | ✅ |
| Custom system prompts | ✅ | ✅ |
| Custom user templates | ✅ | ✅ |
| Generation prefix | ✅ | ✅ |
| Response tags | ✅ | ✅ |
| Turn filtering | ✅ | ✅ |
| Multiple prompt styles | ✅ | ✅ |
| YAML/JSON config files | ❌ | ✅ |
| CLI argument support | ❌ | ✅ |

## Dependencies

```bash
pip install pandas pyarrow tqdm huggingface_hub pyyaml
```

## Examples

See `scripts/dailydialog_config_example.yaml` for a comprehensive configuration example with multiple preset configurations commented out.

## Tips

1. **Start Simple**: Begin with default settings and gradually add customization
2. **Use Config Files**: For reproducibility, save your settings in a YAML file
3. **Inspect Output**: Use the inspection notebook to verify the format matches your expectations
4. **Iterate**: Try different prompt styles to see what works best for your model
5. **Sample First**: Use `--sample-size` to quickly test configurations before processing the full dataset

## Troubleshooting

**Issue**: Config file not loading
- **Solution**: Ensure the file has a `.yaml`, `.yml`, or `.json` extension

**Issue**: Template formatting errors
- **Solution**: Make sure your custom template includes `{dialogue_history}` and `{responding_speaker}` placeholders

**Issue**: Output file too large
- **Solution**: Use `--sample-size` to limit the number of examples

## License

This script is part of the BeRL project and follows the same license.
