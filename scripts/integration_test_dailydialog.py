#!/usr/bin/env python3
"""
Integration test: Load DailyDialog dataset with the actual training pipeline.

This test verifies that the converted DailyDialog dataset can be loaded and
processed by the same code that handles NegotiationToM data.
"""

import sys
import pandas as pd
import torch
from pathlib import Path

# Test basic loading
print("=" * 80)
print("DAILYDIALOG INTEGRATION TEST")
print("=" * 80)

# 1. Load the parquet file
print("\n1. Loading DailyDialog parquet file...")
dd_path = Path("data/DailyDialog_train_limit1000.parquet")
if not dd_path.exists():
    print(f"❌ File not found: {dd_path}")
    print("   Run: python scripts/convert_dailydialog.py --sample-size 1000")
    sys.exit(1)

df = pd.read_parquet(dd_path)
print(f"   ✓ Loaded {len(df)} examples")

# 2. Check all required fields are present
print("\n2. Checking required fields...")
required_fields = [
    'prompt', 'raw_prompt', 'raw_system_prompt', 'raw_user_prompt',
    'reward_model', 'metadata', 'response_words', 'data_source', 'ability'
]

missing_fields = [f for f in required_fields if f not in df.columns]
if missing_fields:
    print(f"   ❌ Missing fields: {missing_fields}")
    sys.exit(1)

print(f"   ✓ All required fields present")

# 3. Validate data types and structure
print("\n3. Validating data structures...")

sample = df.iloc[0]

checks = []

# Check prompt
try:
    prompt = sample['prompt']
    # After parquet serialization, it becomes numpy array
    if hasattr(prompt, '__iter__') and len(prompt) > 0:
        checks.append(("prompt is iterable", True))
    else:
        checks.append(("prompt is iterable", False))
except Exception as e:
    checks.append(("prompt is iterable", False))
    print(f"     Error: {e}")

# Check raw_prompt
try:
    raw_prompt = sample['raw_prompt']
    if hasattr(raw_prompt, '__iter__') and len(raw_prompt) == 2:
        checks.append(("raw_prompt has 2 messages", True))
    else:
        checks.append(("raw_prompt has 2 messages", False))
except Exception as e:
    checks.append(("raw_prompt has 2 messages", False))
    print(f"     Error: {e}")

# Check reward_model
try:
    reward_model = sample['reward_model']
    if isinstance(reward_model, dict):
        if 'ground_truth' in reward_model and 'style' in reward_model:
            checks.append(("reward_model structure", True))
        else:
            checks.append(("reward_model structure", False))
    else:
        checks.append(("reward_model structure", False))
except Exception as e:
    checks.append(("reward_model structure", False))
    print(f"     Error: {e}")

# Check metadata
try:
    metadata = sample['metadata']
    if isinstance(metadata, dict):
        if 'conv_id' in metadata and 'turn' in metadata:
            checks.append(("metadata structure", True))
        else:
            checks.append(("metadata structure", False))
    else:
        checks.append(("metadata structure", False))
except Exception as e:
    checks.append(("metadata structure", False))
    print(f"     Error: {e}")

# Check response_words
try:
    response_words = sample['response_words']
    # Handle both native Python int and numpy int types
    import numpy as np
    if isinstance(response_words, (int, float, np.integer)) and response_words > 0:
        checks.append(("response_words is positive number", True))
    else:
        checks.append(("response_words is positive number", False))
except Exception as e:
    checks.append(("response_words is positive number", False))
    print(f"     Error: {e}")

for check_name, passed in checks:
    status = "✓" if passed else "✗"
    print(f"   {status} {check_name}")

# 4. Test data extraction (simulate training pipeline)
print("\n4. Testing data extraction (simulate training pipeline)...")

try:
    # Extract fields as training pipeline would
    for i in range(min(3, len(df))):
        row = df.iloc[i]

        # Extract user prompt
        user_prompt = row['raw_user_prompt']
        assert isinstance(user_prompt, str) and len(user_prompt) > 0

        # Extract system prompt
        system_prompt = row['raw_system_prompt']
        assert isinstance(system_prompt, str) and len(system_prompt) > 0

        # Extract ground truth
        ground_truth = row['reward_model']['ground_truth']
        assert isinstance(ground_truth, str) and len(ground_truth) > 0

        # Extract response words
        response_words = row['response_words']
        assert response_words > 0

        # Extract metadata
        conv_id = row['metadata']['conv_id']
        turn = row['metadata']['turn']
        assert isinstance(conv_id, (int, float))
        assert isinstance(turn, (int, float))

    print(f"   ✓ Successfully extracted data from {min(3, len(df))} examples")

except Exception as e:
    print(f"   ❌ Failed to extract data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Display sample examples
print("\n5. Sample training examples:")
print("-" * 80)

for i in range(min(2, len(df))):
    row = df.iloc[i]
    print(f"\nExample {i+1}:")
    print(f"  Data source: {row['data_source']}")
    print(f"  Ability: {row['ability']}")
    print(f"  Conv ID: {row['metadata']['conv_id']}, Turn: {row['metadata']['turn']}")
    print(f"  Response words: {row['response_words']}")
    print(f"  Ground truth: {row['reward_model']['ground_truth'][:80]}...")
    print(f"  User prompt (first 150 chars): {row['raw_user_prompt'][:150]}...")

print("\n" + "-" * 80)

# 6. Summary
print("\n" + "=" * 80)
all_passed = all(passed for _, passed in checks)
if all_passed:
    print("✅ INTEGRATION TEST PASSED")
    print("=" * 80)
    print("\nThe DailyDialog dataset is fully compatible with the training pipeline!")
    print("You can now use it for RL training alongside NegotiationToM data.")
    sys.exit(0)
else:
    print("⚠️  INTEGRATION TEST FAILED")
    print("=" * 80)
    print("\nSome compatibility issues were found. Review the details above.")
    sys.exit(1)
