# Empathetic Dialogues Quick Start Guide

## 🚀 Quick Start (5 minutes)

### Step 1: Install Dependencies

```bash
pip install pandas pyarrow datasets huggingface-hub tqdm pyyaml
```

### Step 2: Run Basic Conversion

```bash
# Convert train split with 1000 samples (for testing)
python scripts/convert_empathetic_dialogues.py \
    --output-name EmpatheticDialogues_train_sample \
    --sample-size 1000
```

This creates: `data/EmpatheticDialogues_train_sample.parquet`

### Step 3: Verify Conversion

```bash
# Run test suite
python scripts/test_empathetic_dialogues_conversion.py
```

## 🎯 Recommended Usage (Empathy-Focused)

For best results with empathetic conversation training:

```bash
python scripts/convert_empathetic_dialogues.py \
    --output-name EmpatheticDialogues_train_empathy \
    --system-prompt-style empathy \
    --prompt-style empathy \
    --include-emotion-context \
    --split train
```

## 📋 Common Use Cases

### Use Case 1: Small Training Set (Quick Experiments)

```bash
python scripts/convert_empathetic_dialogues.py \
    --output-name EmpatheticDialogues_small \
    --sample-size 5000 \
    --min-turns 4 \
    --min-response-words 5
```

### Use Case 2: Full Training Set (Production)

```bash
python scripts/convert_empathetic_dialogues.py \
    --output-name EmpatheticDialogues_full \
    --system-prompt-style empathy \
    --prompt-style empathy \
    --split train
```

### Use Case 3: Validation Set

```bash
python scripts/convert_empathetic_dialogues.py \
    --output-name EmpatheticDialogues_validation \
    --split validation \
    --system-prompt-style empathy \
    --prompt-style empathy
```

### Use Case 4: High-Quality Subset

```bash
python scripts/convert_empathetic_dialogues.py \
    --output-name EmpatheticDialogues_quality \
    --min-turns 6 \
    --max-turns 15 \
    --min-response-words 10 \
    --max-response-words 50
```

## 📝 Using Config Files

### Option 1: Use Provided Config

```bash
python scripts/convert_empathetic_dialogues.py \
    --config scripts/empathetic_dialogues_config_example.yaml
```

### Option 2: Create Custom Config

Create `my_config.yaml`:

```yaml
output_dir: "data"
output_name: "EmpatheticDialogues_custom"
split: "train"
sample_size: 10000
system_prompt_style: "empathy"
prompt_style: "empathy"
min_turns: 4
min_response_words: 8
```

Run it:

```bash
python scripts/convert_empathetic_dialogues.py --config my_config.yaml
```

## 🔍 Inspecting Results

### Load and Inspect

```python
import pandas as pd

# Load dataset
df = pd.read_parquet("data/EmpatheticDialogues_train_sample.parquet")

# Basic info
print(f"Total examples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Check first example
example = df.iloc[0]
print("\nSystem Prompt:", example['raw_system_prompt'])
print("\nUser Prompt:", example['raw_user_prompt'][:200])
print("\nGround Truth:", example['reward_model']['ground_truth'])
print("\nMetadata:", example['metadata'])
```

### Check Emotion Distribution

```python
# Get emotion counts
emotions = df['metadata'].apply(lambda x: x['emotion_label'])
print(emotions.value_counts())
```

### Filter by Emotion

```python
# Get only joyful conversations
joyful = df[df['metadata'].apply(lambda x: x['emotion_label'] == 'joyful')]
print(f"Joyful conversations: {len(joyful)}")
```

## 🧪 Testing

```bash
# Run all tests
python scripts/test_empathetic_dialogues_conversion.py

# Run example conversions
bash scripts/run_empathetic_dialogues_examples.sh
```

## 📊 Compare with DailyDialog

```bash
# First, create both datasets
python scripts/convert_dailydialog.py --sample-size 1000
python scripts/convert_empathetic_dialogues.py --sample-size 1000

# Compare them
python scripts/compare_conversation_datasets.py \
    --dailydialog data/DailyDialog.parquet \
    --empathetic data/EmpatheticDialogues.parquet
```

## 🔧 Troubleshooting

### Problem: Import Error

```
ModuleNotFoundError: No module named 'datasets'
```

**Solution**:
```bash
pip install datasets huggingface-hub
```

### Problem: Memory Error

```
MemoryError: Unable to allocate array
```

**Solution**: Use sampling:
```bash
python scripts/convert_empathetic_dialogues.py --sample-size 5000
```

### Problem: Slow Download

**Solution**: The first download is cached. Subsequent runs will be faster.

## 📚 Next Steps

1. **Combine datasets**:
   ```python
   import pandas as pd

   dd = pd.read_parquet("data/DailyDialog.parquet")
   ed = pd.read_parquet("data/EmpatheticDialogues.parquet")

   combined = pd.concat([dd, ed], ignore_index=True)
   combined.to_parquet("data/Combined_Conversations.parquet")
   ```

2. **Use in training**:
   ```python
   from your_training_module import train_model

   df = pd.read_parquet("data/EmpatheticDialogues_train_empathy.parquet")
   train_model(df)
   ```

3. **Filter and customize**:
   - Filter by specific emotions
   - Adjust turn lengths
   - Customize prompts
   - Add response tags

## 📖 Full Documentation

For complete documentation, see:
- `scripts/README_EMPATHETIC_DIALOGUES.md` - Full guide
- `EMPATHETIC_DIALOGUES_SUMMARY.md` - Overview and comparison
- `scripts/empathetic_dialogues_config_example.yaml` - Config reference

## 🎓 Examples Reference

| Command | Description |
|---------|-------------|
| `--split train` | Use training set |
| `--split validation` | Use validation set |
| `--split test` | Use test set |
| `--sample-size 1000` | Take 1000 samples |
| `--min-turns 6` | Min 6 turns per conversation |
| `--min-response-words 10` | Min 10 words per response |
| `--system-prompt-style empathy` | Use empathy system prompt |
| `--prompt-style empathy` | Use empathy user template |
| `--add-response-tags` | Add `<answer>` tags |
| `--config file.yaml` | Load from config file |

## ✅ Checklist

- [ ] Dependencies installed
- [ ] Basic conversion runs successfully
- [ ] Test suite passes
- [ ] Output file created in `data/` directory
- [ ] Inspected sample examples
- [ ] Ready to integrate with training pipeline

---

**Need help?** Check the full documentation in `scripts/README_EMPATHETIC_DIALOGUES.md`
