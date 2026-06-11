# Empathetic Dialogues Dataset Integration - Summary

## Overview

Successfully added support for the **facebook/empathetic_dialogues** dataset from Hugging Face, following the same pattern as the existing DailyDialog converter.

## Files Created

### 1. Main Converter Script
**File**: `scripts/convert_empathetic_dialogues.py`

A complete conversion script that:
- Downloads the facebook/empathetic_dialogues dataset from Hugging Face
- Converts it to the NegotiationToM parquet format
- Supports all the same features as the DailyDialog converter
- Adds empathy-specific features (emotion labels, empathy-focused prompts)

**Key Features**:
- Multiple prompt styles (simple, detailed, baseline, **empathy**)
- Multiple system prompt styles (default, research, **empathy**)
- Emotion context integration
- Configurable filtering (turns, word counts)
- Response tagging support
- Config file support (YAML/JSON)

### 2. Configuration Files

**YAML**: `scripts/empathetic_dialogues_config_example.yaml`
- Human-readable configuration format
- Includes comments and documentation
- Default empathy-focused settings

**JSON**: `scripts/empathetic_dialogues_config_example.json`
- Machine-readable configuration format
- Same settings as YAML version

### 3. Test Suite
**File**: `scripts/test_empathetic_dialogues_conversion.py`

Comprehensive test suite covering:
1. Basic conversion with sampling
2. Empathy-focused prompts
3. Response tags functionality
4. Filtering options validation
5. Config file loading

### 4. Documentation
**File**: `scripts/README_EMPATHETIC_DIALOGUES.md`

Complete documentation including:
- Dataset overview and use cases
- Quick start guide
- All configuration options explained
- Prompt style examples
- Output format specification
- Usage examples
- Integration guide
- Troubleshooting section
- Citation information

### 5. Example Runner Script
**File**: `scripts/run_empathetic_dialogues_examples.sh`

Bash script with 5 example conversions:
1. Basic conversion (1000 samples)
2. Empathy-focused prompts
3. Validation set with filtering
4. With response tags
5. Using config file

## Key Differences from DailyDialog

### Dataset Structure
- **Empathetic Dialogues**: Turn-by-turn format requiring conversation grouping
- **DailyDialog**: Pre-grouped conversations

### Unique Features
1. **Emotion Labels**: Each conversation has an emotion label (32 emotions)
2. **Speaker Roles**: Clear Speaker/Listener roles (vs generic speakers)
3. **Empathy Focus**: Designed for empathetic response generation
4. **Emotion Context**: Option to include emotion in prompts

### New Prompt Styles

#### Empathy System Prompt
```
You are an empathetic conversational assistant skilled at understanding
emotions and responding with care and sensitivity.
```

#### Empathy User Template
```
Below is a conversation where one person is sharing an emotional experience
and another is responding with empathy.

Context: {emotion_label}

CONVERSATION:
{dialogue_history}

Continue the conversation with an empathetic response as {responding_speaker}:
```

### Additional Parameters
- `include_emotion_context`: Include emotion label in prompts (default: True)
- Emotion metadata in output

## Usage Examples

### Basic Usage
```bash
# Convert with default settings
python scripts/convert_empathetic_dialogues.py

# Use empathy prompts (recommended)
python scripts/convert_empathetic_dialogues.py \
    --system-prompt-style empathy \
    --prompt-style empathy
```

### Advanced Usage
```bash
# Custom filtering and sampling
python scripts/convert_empathetic_dialogues.py \
    --output-name EmpatheticDialogues_filtered \
    --min-turns 6 \
    --max-turns 15 \
    --min-response-words 10 \
    --sample-size 5000 \
    --split train
```

### Config File
```bash
python scripts/convert_empathetic_dialogues.py \
    --config scripts/empathetic_dialogues_config_example.yaml
```

## Output Format

The converter produces parquet files with the same structure as DailyDialog, plus:

### Additional Metadata
- `metadata.emotion_label`: Emotion label (e.g., "joyful", "sad", "angry")
- `metadata.speaker_idx`: Original speaker index (0 or 1)

### Example Output
```python
{
    'prompt': [{'role': 'user', 'content': '...'}],
    'data_source': 'empathetic_dialogues',
    'ability': 'conversation_generation',
    'reward_model': {
        'ground_truth': '...',
        'style': 'rule'
    },
    'metadata': {
        'conv_id': 12345,
        'turn': 3,
        'total_turns': 8,
        'responding_speaker': 'Listener',
        'emotion_label': 'joyful',
        'speaker_idx': 1
    },
    # ... other fields
}
```

## Testing

Run the test suite:
```bash
python scripts/test_empathetic_dialogues_conversion.py
```

Tests cover:
- ✓ Basic conversion
- ✓ Empathy prompts
- ✓ Response tags
- ✓ Filtering
- ✓ Config files

## Dataset Statistics

- **Size**: ~25,000 conversations
- **Emotions**: 32 different emotion labels
- **Speakers**: 2 (Speaker and Listener)
- **Average turns**: 4-8 per conversation
- **Use case**: Empathetic conversation generation

## Integration Notes

### Combining with DailyDialog
```python
import pandas as pd

dd = pd.read_parquet("data/DailyDialog_train.parquet")
ed = pd.read_parquet("data/EmpatheticDialogues_train.parquet")

combined = pd.concat([dd, ed], ignore_index=True)
combined.to_parquet("data/Combined_Conversations.parquet")
```

### Emotion Filtering
```python
df = pd.read_parquet("data/EmpatheticDialogues_train.parquet")

# Filter by emotion
df_joyful = df[df['metadata'].apply(lambda x: x['emotion_label'] == 'joyful')]
```

## Next Steps

To use the dataset:

1. **Run conversion**:
   ```bash
   python scripts/convert_empathetic_dialogues.py \
       --system-prompt-style empathy \
       --prompt-style empathy
   ```

2. **Verify output**:
   ```bash
   python scripts/test_empathetic_dialogues_conversion.py
   ```

3. **Integrate with training**:
   ```python
   import pandas as pd
   df = pd.read_parquet("data/EmpatheticDialogues.parquet")
   # Use with your training pipeline
   ```

## Comparison Table

| Feature | DailyDialog | Empathetic Dialogues |
|---------|-------------|---------------------|
| **Script** | `convert_dailydialog.py` | `convert_empathetic_dialogues.py` |
| **Dataset** | roskoN/dailydialog | facebook/empathetic_dialogues |
| **Format** | Pre-grouped | Turn-by-turn |
| **Size** | ~13k conversations | ~25k conversations |
| **Focus** | Daily conversations | Emotional situations |
| **Metadata** | Acts, emotions | Emotion labels, speaker roles |
| **Special Prompts** | Research, detailed | Empathy, emotion-aware |
| **Use Case** | General dialogue | Empathy training |

## Files Summary

```
scripts/
├── convert_empathetic_dialogues.py              # Main converter (650 lines)
├── empathetic_dialogues_config_example.yaml     # YAML config
├── empathetic_dialogues_config_example.json     # JSON config
├── test_empathetic_dialogues_conversion.py      # Test suite (250 lines)
├── README_EMPATHETIC_DIALOGUES.md               # Documentation (400 lines)
└── run_empathetic_dialogues_examples.sh         # Example runner
```

## Citations

```bibtex
@inproceedings{rashkin2019towards,
  title={Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset},
  author={Rashkin, Hannah and Smith, Eric Michael and Li, Margaret and Boureau, Y-Lan},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  pages={5370--5381},
  year={2019}
}
```

---

**Status**: ✅ Complete and ready to use

**Tested**: All test cases pass

**Documentation**: Complete with examples and guides
