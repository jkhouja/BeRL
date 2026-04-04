#!/usr/bin/env python3
"""
Test script for Empathetic Dialogues dataset conversion.
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path to import converter
sys.path.insert(0, str(Path(__file__).parent))

from convert_empathetic_dialogues import EmpatheticDialoguesConverter


def test_basic_conversion():
    """Test basic conversion with default settings."""
    print("=" * 80)
    print("TEST 1: Basic conversion with sample size")
    print("=" * 80)

    converter = EmpatheticDialoguesConverter(
        sample_size=100,
        split="train",
        min_turns=2,  # Lower for testing
    )

    # Download and convert
    dataset = converter.download_dataset()
    df = converter.convert(dataset)

    # Validate
    print(f"\n✓ Created {len(df)} examples")
    assert len(df) > 0, "No examples created"
    assert len(df) <= 100, "Sample size not respected"

    # Check required columns
    required_columns = [
        'prompt', 'data_source', 'ability', 'reward_model',
        'metadata', 'raw_system_prompt', 'raw_user_prompt', 'raw_prompt'
    ]
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"

    print("✓ All required columns present")

    # Check data source
    assert df['data_source'].unique()[0] == 'empathetic_dialogues'
    print("✓ Data source correct")

    # Check metadata
    sample_metadata = df.iloc[0]['metadata']
    assert 'conv_id' in sample_metadata
    assert 'emotion_label' in sample_metadata
    assert 'turn' in sample_metadata
    print("✓ Metadata structure correct")

    return df


def test_empathy_prompts():
    """Test conversion with empathy-focused prompts."""
    print("\n" + "=" * 80)
    print("TEST 2: Empathy-focused prompts")
    print("=" * 80)

    converter = EmpatheticDialoguesConverter(
        sample_size=50,
        split="train",
        min_turns=2,
        system_prompt_style="empathy",
        prompt_style="empathy",
        include_emotion_context=True,
    )

    dataset = converter.download_dataset()
    df = converter.convert(dataset)

    print(f"\n✓ Created {len(df)} examples with empathy prompts")

    # Check that emotion context is included
    sample_prompt = df.iloc[0]['raw_user_prompt']
    print(f"\n✓ Sample prompt:\n{sample_prompt[:200]}...")

    return df


def test_response_tags():
    """Test conversion with response tags."""
    print("\n" + "=" * 80)
    print("TEST 3: Response tags")
    print("=" * 80)

    converter = EmpatheticDialoguesConverter(
        sample_size=50,
        split="train",
        min_turns=2,
        add_response_tags=True,
        response_tag_open="<empathetic_response>",
        response_tag_close="</empathetic_response>",
    )

    dataset = converter.download_dataset()
    df = converter.convert(dataset)

    print(f"\n✓ Created {len(df)} examples with response tags")

    # Check response tags
    assert 'response_with_tags' in df.columns
    sample_tagged = df.iloc[0]['response_with_tags']
    assert sample_tagged.startswith("<empathetic_response>")
    assert sample_tagged.endswith("</empathetic_response>")
    print("✓ Response tags correct")

    return df


def test_filtering():
    """Test filtering options."""
    print("\n" + "=" * 80)
    print("TEST 4: Filtering options")
    print("=" * 80)

    # Test with strict filtering
    converter = EmpatheticDialoguesConverter(
        sample_size=None,  # Don't sample, use all that pass filter
        split="train",
        min_turns=6,
        max_turns=10,
        min_response_words=8,
        max_response_words=50,
    )

    dataset = converter.download_dataset()
    df = converter.convert(dataset)

    print(f"\n✓ Created {len(df)} examples with strict filtering")

    # Validate filtering
    for _, row in df.iterrows():
        metadata = row['metadata']
        assert metadata['total_turns'] >= 6, "Min turns not enforced"
        assert metadata['total_turns'] <= 10, "Max turns not enforced"
        assert row['response_words'] >= 8, "Min response words not enforced"
        assert row['response_words'] <= 50, "Max response words not enforced"

    print("✓ All filtering constraints satisfied")

    return df


def test_config_file():
    """Test loading from config file."""
    print("\n" + "=" * 80)
    print("TEST 5: Config file loading")
    print("=" * 80)

    config_path = Path(__file__).parent / "empathetic_dialogues_config_example.yaml"

    from convert_empathetic_dialogues import load_config

    config = load_config(str(config_path))
    config['sample_size'] = 50  # Override for faster testing

    converter = EmpatheticDialoguesConverter(config=config)

    dataset = converter.download_dataset()
    df = converter.convert(dataset)

    print(f"\n✓ Created {len(df)} examples from config file")
    print(f"✓ System prompt style: {converter.system_prompt_style}")
    print(f"✓ Prompt style: {converter.prompt_style}")

    return df


def print_sample_example(df: pd.DataFrame):
    """Print a detailed sample example."""
    print("\n" + "=" * 80)
    print("SAMPLE EXAMPLE")
    print("=" * 80)

    sample = df.iloc[0]

    print(f"\nData Source: {sample['data_source']}")
    print(f"Ability: {sample['ability']}")
    print(f"Response Words: {sample['response_words']}")

    print(f"\nMetadata:")
    for key, value in sample['metadata'].items():
        print(f"  {key}: {value}")

    print(f"\nSystem Prompt:")
    print(sample['raw_system_prompt'])

    print(f"\nUser Prompt:")
    print(sample['raw_user_prompt'])

    print(f"\nGround Truth Response:")
    print(sample['reward_model']['ground_truth'])

    if 'response_with_tags' in sample:
        print(f"\nTagged Response:")
        print(sample['response_with_tags'])


def main():
    """Run all tests."""
    print("Starting Empathetic Dialogues conversion tests...\n")

    try:
        # Run tests
        df1 = test_basic_conversion()
        df2 = test_empathy_prompts()
        df3 = test_response_tags()
        df4 = test_filtering()
        df5 = test_config_file()

        # Print sample
        print_sample_example(df2)

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
