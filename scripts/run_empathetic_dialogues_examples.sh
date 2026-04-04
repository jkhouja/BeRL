#!/bin/bash
# Example scripts for converting Empathetic Dialogues dataset

# Create output directory
mkdir -p data

echo "========================================"
echo "Empathetic Dialogues Conversion Examples"
echo "========================================"

# Example 1: Basic conversion with sampling
echo ""
echo "Example 1: Basic conversion (1000 samples)"
python scripts/convert_empathetic_dialogues.py \
    --output-name EmpatheticDialogues_train_sample1000 \
    --sample-size 1000 \
    --split train

# Example 2: Empathy-focused prompts
echo ""
echo "Example 2: Empathy-focused prompts"
python scripts/convert_empathetic_dialogues.py \
    --output-name EmpatheticDialogues_empathy \
    --system-prompt-style empathy \
    --prompt-style empathy \
    --sample-size 500

# Example 3: Validation set with filtering
echo ""
echo "Example 3: Validation set with filtering"
python scripts/convert_empathetic_dialogues.py \
    --output-name EmpatheticDialogues_validation \
    --split validation \
    --min-turns 6 \
    --min-response-words 10

# Example 4: With response tags
echo ""
echo "Example 4: With response tags"
python scripts/convert_empathetic_dialogues.py \
    --output-name EmpatheticDialogues_tagged \
    --add-response-tags \
    --response-tag-open "<empathetic_response>" \
    --response-tag-close "</empathetic_response>" \
    --sample-size 500

# Example 5: Using config file
echo ""
echo "Example 5: Using config file"
python scripts/convert_empathetic_dialogues.py \
    --config scripts/empathetic_dialogues_config_example.yaml

echo ""
echo "========================================"
echo "All examples completed!"
echo "Check the data/ directory for output files"
echo "========================================"
