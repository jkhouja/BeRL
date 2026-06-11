# Empathetic Dialogues Dataset Conversion

This directory contains scripts to convert the Empathetic Dialogues dataset from Hugging Face to the NegotiationToM format for RL training.

**Note**: We use [Dong237/empathetic_dialogues_cleaned](https://huggingface.co/datasets/Dong237/empathetic_dialogues_cleaned), a cleaned version of the original [facebook/empathetic_dialogues](https://huggingface.co/datasets/facebook/empathetic_dialogues) dataset. The original uses deprecated loading scripts that are no longer supported by the `datasets` library.

## Dataset Overview

The Empathetic Dialogues dataset contains ~25k conversations grounded in emotional situations. Each conversation involves:
- A **Speaker** who describes an emotional experience
- A **Listener** who responds with empathy
- An **emotion label** describing the speaker's emotional state (e.g., "joyful", "sad", "angry")

This dataset is particularly valuable for training models to:
- Understand emotional context
- Generate empathetic responses
- Handle sensitive conversations
- Recognize and respond to different emotional states

## Quick Start

### Basic Usage

```bash
# Convert train split with default settings
python scripts/convert_empathetic_dialogues.py

# Convert with custom output name
python scripts/convert_empathetic_dialogues.py --output-name EmpatheticDialogues_train

# Convert validation split with sampling
python scripts/convert_empathetic_dialogues.py --split validation --sample-size 1000

# Use empathy-focused prompts
python scripts/convert_empathetic_dialogues.py \
    --system-prompt-style empathy \
    --prompt-style empathy
```

### Using Config File

```bash
# Use the example config
python scripts/convert_empathetic_dialogues.py \
    --config scripts/empathetic_dialogues_config_example.yaml
```

## Configuration Options

### Dataset Filtering

- `--min-turns`: Minimum number of turns in a conversation (default: 4)
- `--max-turns`: Maximum number of turns (default: None)
- `--min-response-words`: Minimum words in response (default: 5)
- `--max-response-words`: Maximum words in response (default: None)
- `--limit-turn`: Only include turns >= this number (default: None)

### Sampling

- `--sample-size`: Number of examples to sample (default: None, use all)
- `--seed`: Random seed for reproducibility (default: 42)
- `--split`: Dataset split to use: train, validation, test (default: train)

### Prompt Styles

#### System Prompt Styles (`--system-prompt-style`)

1. **default**: General-purpose conversation assistant
   ```
   You are a helpful assistant helping with conversation analysis and generation.
   ```

2. **research**: Academic/research-focused
   ```
   You are an expert linguist and communication assistant helping with research
   on the psychology of interactions.
   ```

3. **empathy** (Recommended for this dataset): Empathy-focused
   ```
   You are an empathetic conversational assistant skilled at understanding
   emotions and responding with care and sensitivity.
   ```

#### User Prompt Styles (`--prompt-style`)

1. **simple**: Basic conversation continuation
2. **detailed**: More structured continuation prompt
3. **baseline**: Minimal prompt
4. **empathy** (Recommended for this dataset): Includes emotional context
   ```
   Below is a conversation where one person is sharing an emotional experience
   and another is responding with empathy.

   Context: {emotion_label}

   CONVERSATION:
   {dialogue_history}

   Continue the conversation with an empathetic response as {responding_speaker}:
   ```

### Advanced Options

- `--system-prompt`: Custom system prompt (overrides styles)
- `--user-template`: Custom user template with placeholders:
  - `{dialogue_history}`: The conversation so far
  - `{responding_speaker}`: Who should respond (Speaker/Listener)
  - `{emotion_label}`: The emotion label for the conversation
- `--generation-prefix`: Prefix to add to generation prompts
- `--include-system-in-prompt`: Include system prompt in the prompt field
- `--include-emotion-context`: Include emotion label in prompts (default: True)

### Response Formatting

- `--add-response-tags`: Wrap responses in tags
- `--response-tag-open`: Opening tag (default: `<answer>`)
- `--response-tag-close`: Closing tag (default: `</answer>`)

### Metadata

- `--ability`: Ability tag for the dataset (default: conversation_generation)
- `--data-source`: Data source identifier (default: empathetic_dialogues)

## Output Format

The script creates a parquet file with the following structure:

### Main Fields

- `prompt`: Chat-formatted prompt (list of role/content dicts)
- `raw_system_prompt`: System prompt text
- `raw_user_prompt`: User prompt text
- `raw_prompt`: Full prompt with system and user messages
- `data_source`: "empathetic_dialogues"
- `ability`: "conversation_generation" (or custom)
- `response_words`: Number of words in ground truth response

### Reward Model

- `reward_model.ground_truth`: The actual next utterance
- `reward_model.style`: "rule"

### Metadata

- `metadata.conv_id`: Unique conversation identifier
- `metadata.turn`: Turn number in the conversation
- `metadata.total_turns`: Total turns in the conversation
- `metadata.responding_speaker`: Who should respond (Speaker/Listener)
- `metadata.dialogue_history`: Full dialogue history as string
- `metadata.emotion_label`: Emotion label (e.g., "joyful", "sad")
- `metadata.speaker_idx`: Speaker index (0 or 1)

## Examples

### Example 1: Basic Training Set

```bash
python scripts/convert_empathetic_dialogues.py \
    --output-name EmpatheticDialogues_train \
    --split train \
    --min-turns 4 \
    --min-response-words 5
```

### Example 2: Empathy-Focused Subset

```bash
python scripts/convert_empathetic_dialogues.py \
    --output-name EmpatheticDialogues_empathy \
    --system-prompt-style empathy \
    --prompt-style empathy \
    --include-emotion-context \
    --sample-size 5000
```

### Example 3: Research Dataset

```bash
python scripts/convert_empathetic_dialogues.py \
    --output-name EmpatheticDialogues_research \
    --system-prompt-style research \
    --prompt-style detailed \
    --min-turns 6 \
    --max-turns 15 \
    --min-response-words 10
```

### Example 4: With Response Tags

```bash
python scripts/convert_empathetic_dialogues.py \
    --output-name EmpatheticDialogues_tagged \
    --add-response-tags \
    --response-tag-open "<empathetic_response>" \
    --response-tag-close "</empathetic_response>"
```

### Example 5: Custom Prompts

```bash
python scripts/convert_empathetic_dialogues.py \
    --output-name EmpatheticDialogues_custom \
    --system-prompt "You are an AI trained to provide emotional support." \
    --user-template "Given this emotional context: {emotion_label}\n\nConversation:\n{dialogue_history}\n\nRespond as {responding_speaker}:"
```

## Testing

Run the test suite to verify the conversion:

```bash
# Run all tests
python scripts/test_empathetic_dialogues_conversion.py

# Make test script executable
chmod +x scripts/test_empathetic_dialogues_conversion.py
./scripts/test_empathetic_dialogues_conversion.py
```

The test suite covers:
1. Basic conversion with sampling
2. Empathy-focused prompts
3. Response tags
4. Filtering options
5. Config file loading

## Dataset Statistics

After conversion, the script prints useful statistics:

```
Dataset info:
  - Shape: (50000, 12)
  - Columns: [...]
  - Data source: ['empathetic_dialogues']
  - Ability: ['conversation_generation']

Configuration:
  - System prompt style: empathy
  - Prompt style: empathy
  - Generation prefix: ''
  - Response tags: False

Sample statistics:
  - Avg response words: 15.3
  - Min response words: 5
  - Max response words: 87
  - Unique conversations: 8234
  - Unique emotions: 32
```

## Emotion Labels

The dataset includes 32 different emotion labels:

- Positive: joyful, excited, proud, grateful, hopeful, impressed, confident, faithful, caring, trusting, content, prepared, anticipating
- Negative: sad, afraid, angry, annoyed, disappointed, embarrassed, ashamed, guilty, disgusted, jealous, lonely, terrified, anxious, devastated, furious
- Neutral: surprised, nostalgic, sentimental

## Comparison with DailyDialog

| Feature | Empathetic Dialogues | DailyDialog |
|---------|---------------------|-------------|
| Focus | Emotional situations | Daily conversations |
| Context | Emotion-grounded | Topic-based |
| Speakers | Speaker/Listener roles | Generic speakers |
| Metadata | Emotion labels | Act/emotion annotations |
| Size | ~25k conversations | ~13k conversations |
| Use case | Empathy training | General dialogue |

## Integration with Training

The converted dataset can be used directly with the RL training pipeline:

```python
import pandas as pd

# Load converted dataset
df = pd.read_parquet("data/EmpatheticDialogues_train.parquet")

# Use with training code
from your_training_module import train_rl_model

train_rl_model(
    dataset=df,
    ability="conversation_generation",
    data_source="empathetic_dialogues"
)
```

## Advanced Usage

### Combining Multiple Datasets

```python
import pandas as pd

# Load both datasets
dd = pd.read_parquet("data/DailyDialog_train.parquet")
ed = pd.read_parquet("data/EmpatheticDialogues_train.parquet")

# Combine
combined = pd.concat([dd, ed], ignore_index=True)

# Save
combined.to_parquet("data/Combined_Conversations.parquet")
```

### Filtering by Emotion

```python
# Load dataset
df = pd.read_parquet("data/EmpatheticDialogues_train.parquet")

# Filter for specific emotions
positive_emotions = ['joyful', 'excited', 'proud', 'grateful']
df_positive = df[df['metadata'].apply(
    lambda x: x['emotion_label'] in positive_emotions
)]

# Save filtered subset
df_positive.to_parquet("data/EmpatheticDialogues_positive.parquet")
```

## Troubleshooting

### Issue: Import errors

**Solution**: Make sure required packages are installed:
```bash
pip install pandas pyarrow datasets huggingface-hub tqdm pyyaml
```

### Issue: Memory errors with large datasets

**Solution**: Use sampling:
```bash
python scripts/convert_empathetic_dialogues.py --sample-size 10000
```

### Issue: Slow downloads

**Solution**: The first download may be slow. The dataset is cached by Hugging Face, so subsequent runs will be faster.

## Files Created

Running the conversion creates:
- `convert_empathetic_dialogues.py` - Main conversion script
- `empathetic_dialogues_config_example.yaml` - Example config file
- `test_empathetic_dialogues_conversion.py` - Test suite
- `README_EMPATHETIC_DIALOGUES.md` - This file
- `data/EmpatheticDialogues_*.parquet` - Output files

## References

- [Empathetic Dialogues Paper](https://arxiv.org/abs/1811.00207)
- [Cleaned Dataset (Used by this script)](https://huggingface.co/datasets/Dong237/empathetic_dialogues_cleaned)
- [Original HuggingFace Dataset](https://huggingface.co/datasets/facebook/empathetic_dialogues) (Uses deprecated loading scripts)
- Original repository: [ParlAI](https://github.com/facebookresearch/ParlAI)

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{rashkin2019towards,
  title={Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset},
  author={Rashkin, Hannah and Smith, Eric Michael and Li, Margaret and Boureau, Y-Lan},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  pages={5370--5381},
  year={2019}
}
```
