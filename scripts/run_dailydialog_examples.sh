#!/bin/bash
# Example usage scripts for convert_dailydialog.py

set -e  # Exit on error

echo "=================================="
echo "DailyDialog Dataset Conversion"
echo "=================================="
echo

# Activate virtual environment
source .venv/bin/activate

# Example 1: Create a small test dataset (1000 samples)
echo "Example 1: Creating small test dataset (1000 samples)..."
python scripts/convert_dailydialog.py \
    --sample-size 1000 \
    --output-name DailyDialog_train_limit1000 \
    --min-turns 4 \
    --min-response-words 5 \
    --seed 42

echo
echo "✓ Created: data/DailyDialog_train_limit1000.parquet"
echo

# Example 2: Create validation set
echo "Example 2: Creating validation dataset..."
python scripts/convert_dailydialog.py \
    --split validation \
    --output-name DailyDialog_validation \
    --min-turns 4 \
    --min-response-words 5 \
    --seed 42

echo
echo "✓ Created: data/DailyDialog_validation.parquet"
echo

# Example 3: Create filtered high-quality training set
echo "Example 3: Creating high-quality filtered training set..."
python scripts/convert_dailydialog.py \
    --split train \
    --output-name DailyDialog_train_quality \
    --min-turns 6 \
    --max-turns 15 \
    --min-response-words 8 \
    --max-response-words 80 \
    --seed 42

echo
echo "✓ Created: data/DailyDialog_train_quality.parquet"
echo

# Example 4: Create test set (smaller, 500 samples)
echo "Example 4: Creating test dataset..."
python scripts/convert_dailydialog.py \
    --split test \
    --output-name DailyDialog_test \
    --sample-size 500 \
    --min-turns 4 \
    --seed 42

echo
echo "✓ Created: data/DailyDialog_test.parquet"
echo

echo "=================================="
echo "All datasets created successfully!"
echo "=================================="
echo
echo "Dataset files in data/:"
ls -lh data/DailyDialog*.parquet
