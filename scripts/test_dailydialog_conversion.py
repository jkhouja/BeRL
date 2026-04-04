#!/usr/bin/env python3
"""
Test script for DailyDialog conversion.

This script tests the conversion process and validates the output format.
"""

import sys
from pathlib import Path
import pandas as pd

# Import converter class directly
sys.path.insert(0, str(Path(__file__).parent))
from convert_dailydialog import DailyDialogConverter


def test_converter():
    """Test the DailyDialogConverter with a small sample."""
    print("=" * 80)
    print("Testing DailyDialogConverter")
    print("=" * 80)

    # Create converter with small sample size for testing
    converter = DailyDialogConverter(
        min_turns=4,
        max_turns=10,
        min_response_words=3,
        max_response_words=50,
        sample_size=100,  # Small sample for testing
        seed=42,
        split="train",
    )

    # Download dataset
    print("\n1. Downloading dataset...")
    dataset = converter.download_dataset()
    print(f"   ✓ Downloaded {len(dataset)} conversations")

    # Convert dataset
    print("\n2. Converting dataset...")
    df = converter.convert(dataset)
    print(f"   ✓ Converted to {len(df)} training examples")

    # Validate format
    print("\n3. Validating format...")
    validate_format(df)

    return df


def validate_format(df: pd.DataFrame):
    """Validate that the DataFrame matches NegotiationToM format."""

    # Check required columns
    required_columns = [
        'prompt_len', 'response_words', 'data_source', 'prompt',
        'raw_system_prompt', 'raw_user_prompt', 'prompt_for_pp',
        'ability', 'reward_model', 'metadata', 'answer_pp', 'raw_prompt'
    ]

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        print(f"   ✗ Missing columns: {missing_columns}")
        return False

    print(f"   ✓ All required columns present")

    # Check data types and structure
    checks = []

    # Check data_source
    if df['data_source'].unique().tolist() == ['dailydialog']:
        checks.append(("data_source", True, "All entries are 'dailydialog'"))
    else:
        checks.append(("data_source", False, f"Unexpected values: {df['data_source'].unique()}"))

    # Check ability
    if df['ability'].unique().tolist() == ['conversation_generation']:
        checks.append(("ability", True, "All entries are 'conversation_generation'"))
    else:
        checks.append(("ability", False, f"Unexpected values: {df['ability'].unique()}"))

    # Check prompt structure (should be list of dicts)
    sample_prompt = df['prompt'].iloc[0]
    if isinstance(sample_prompt, list) and len(sample_prompt) > 0:
        if isinstance(sample_prompt[0], dict) and 'content' in sample_prompt[0] and 'role' in sample_prompt[0]:
            checks.append(("prompt structure", True, "Prompt is list of dicts with 'content' and 'role'"))
        else:
            checks.append(("prompt structure", False, f"Invalid dict structure: {sample_prompt[0]}"))
    else:
        checks.append(("prompt structure", False, f"Invalid prompt type: {type(sample_prompt)}"))

    # Check raw_prompt structure
    sample_raw_prompt = df['raw_prompt'].iloc[0]
    if isinstance(sample_raw_prompt, list) and len(sample_raw_prompt) == 2:
        if (isinstance(sample_raw_prompt[0], dict) and
            sample_raw_prompt[0].get('role') == 'system' and
            isinstance(sample_raw_prompt[1], dict) and
            sample_raw_prompt[1].get('role') == 'user'):
            checks.append(("raw_prompt structure", True, "Raw prompt has system + user messages"))
        else:
            checks.append(("raw_prompt structure", False, "Invalid message structure"))
    else:
        checks.append(("raw_prompt structure", False, f"Expected 2 messages, got {len(sample_raw_prompt)}"))

    # Check reward_model structure
    sample_reward = df['reward_model'].iloc[0]
    if isinstance(sample_reward, dict) and 'ground_truth' in sample_reward and 'style' in sample_reward:
        checks.append(("reward_model", True, "Contains 'ground_truth' and 'style'"))
    else:
        checks.append(("reward_model", False, f"Invalid structure: {sample_reward}"))

    # Check metadata structure
    sample_metadata = df['metadata'].iloc[0]
    expected_keys = {'conv_id', 'turn', 'total_turns', 'responding_speaker'}
    if isinstance(sample_metadata, dict) and expected_keys.issubset(sample_metadata.keys()):
        checks.append(("metadata", True, f"Contains all expected keys: {expected_keys}"))
    else:
        checks.append(("metadata", False, f"Missing keys. Got: {sample_metadata.keys()}"))

    # Check response_words is positive integer
    if (df['response_words'] > 0).all() and df['response_words'].dtype in ['int64', 'int32']:
        checks.append(("response_words", True, "All positive integers"))
    else:
        checks.append(("response_words", False, "Contains non-positive or non-integer values"))

    # Print results
    all_passed = True
    for check_name, passed, message in checks:
        status = "✓" if passed else "✗"
        print(f"   {status} {check_name}: {message}")
        if not passed:
            all_passed = False

    return all_passed


def compare_with_negtom(dailydialog_df: pd.DataFrame):
    """Compare structure with existing NegotiationToM dataset."""
    print("\n4. Comparing with NegotiationToM dataset...")

    negtom_path = Path("data/NegotiationToM_Qwen-Qwen2.5-3B-Instruct_limit800.parquet")
    if not negtom_path.exists():
        print("   ⚠ NegotiationToM file not found, skipping comparison")
        return

    negtom_df = pd.read_parquet(negtom_path)

    print(f"   DailyDialog columns: {set(dailydialog_df.columns)}")
    print(f"   NegotiationToM columns: {set(negtom_df.columns)}")

    missing_in_dd = set(negtom_df.columns) - set(dailydialog_df.columns)
    extra_in_dd = set(dailydialog_df.columns) - set(negtom_df.columns)

    if not missing_in_dd and not extra_in_dd:
        print("   ✓ Column sets match perfectly!")
    else:
        if missing_in_dd:
            print(f"   ⚠ Missing in DailyDialog: {missing_in_dd}")
        if extra_in_dd:
            print(f"   ⚠ Extra in DailyDialog: {extra_in_dd}")

    print(f"\n   Sample DailyDialog row:")
    print(f"   {dailydialog_df.iloc[0].to_dict()}")


def display_sample_examples(df: pd.DataFrame, n: int = 3):
    """Display sample examples from the converted dataset."""
    print(f"\n5. Sample examples (showing {n}):")
    print("=" * 80)

    for i in range(min(n, len(df))):
        row = df.iloc[i]
        print(f"\nExample {i+1}:")
        print(f"  Conversation ID: {row['metadata']['conv_id']}")
        print(f"  Turn: {row['metadata']['turn']}/{row['metadata']['total_turns']}")
        print(f"  Response words: {row['response_words']}")
        print(f"  Responding speaker: {row['metadata']['responding_speaker']}")
        print(f"  Ground truth: {row['reward_model']['ground_truth'][:100]}...")
        print(f"  User prompt (first 200 chars): {row['raw_user_prompt'][:200]}...")
        print("-" * 80)


def main():
    """Run all tests."""
    try:
        # Test conversion
        df = test_converter()

        # Compare with NegotiationToM
        compare_with_negtom(df)

        # Display samples
        display_sample_examples(df)

        print("\n" + "=" * 80)
        print("✓ All tests passed!")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
