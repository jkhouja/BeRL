#!/usr/bin/env python3
"""
Compare DailyDialog and NegotiationToM datasets side by side.
"""

import pandas as pd
from pathlib import Path


def compare_datasets():
    """Compare the two datasets."""

    # Load datasets
    negtom_path = Path("data/NegotiationToM_Qwen-Qwen2.5-3B-Instruct_limit800.parquet")
    dailydialog_path = Path("data/DailyDialog_train_limit1000.parquet")

    if not negtom_path.exists():
        print(f"❌ NegotiationToM file not found: {negtom_path}")
        return False

    if not dailydialog_path.exists():
        print(f"❌ DailyDialog file not found: {dailydialog_path}")
        return False

    negtom = pd.read_parquet(negtom_path)
    dailydialog = pd.read_parquet(dailydialog_path)

    print("=" * 80)
    print("DATASET COMPARISON")
    print("=" * 80)

    # Basic stats
    print("\n📊 Basic Statistics:")
    print(f"  NegotiationToM:  {len(negtom):,} examples")
    print(f"  DailyDialog:     {len(dailydialog):,} examples")

    # Column comparison
    print("\n📋 Columns:")
    negtom_cols = set(negtom.columns)
    dailydialog_cols = set(dailydialog.columns)

    if negtom_cols == dailydialog_cols:
        print(f"  ✓ Both datasets have the same columns ({len(negtom_cols)})")
    else:
        print(f"  ⚠ Column mismatch!")
        missing = negtom_cols - dailydialog_cols
        extra = dailydialog_cols - negtom_cols
        if missing:
            print(f"    Missing in DailyDialog: {missing}")
        if extra:
            print(f"    Extra in DailyDialog: {extra}")

    # Data types
    print("\n📝 Data Types:")
    for col in sorted(negtom.columns):
        negtom_type = str(negtom[col].dtype)
        dd_type = str(dailydialog[col].dtype) if col in dailydialog.columns else "N/A"
        match = "✓" if negtom_type == dd_type else "✗"
        print(f"  {match} {col:20s}: NegToM={negtom_type:10s} | DailyDialog={dd_type:10s}")

    # Sample comparison
    print("\n🔍 Sample Data Comparison:")
    print("\n--- NegotiationToM Sample ---")
    sample_negtom = negtom.iloc[0]
    print(f"  data_source:    {sample_negtom['data_source']}")
    print(f"  ability:        {sample_negtom['ability']}")
    print(f"  response_words: {sample_negtom['response_words']}")
    print(f"  prompt type:    {type(sample_negtom['prompt'])}")
    print(f"  prompt len:     {len(sample_negtom['prompt'])}")
    print(f"  raw_prompt len: {len(sample_negtom['raw_prompt'])}")
    print(f"  reward_model:   {sample_negtom['reward_model']}")
    print(f"  metadata keys:  {sample_negtom['metadata'].keys() if isinstance(sample_negtom['metadata'], dict) else 'N/A'}")

    print("\n--- DailyDialog Sample ---")
    sample_dd = dailydialog.iloc[0]
    print(f"  data_source:    {sample_dd['data_source']}")
    print(f"  ability:        {sample_dd['ability']}")
    print(f"  response_words: {sample_dd['response_words']}")
    print(f"  prompt type:    {type(sample_dd['prompt'])}")
    print(f"  prompt len:     {len(sample_dd['prompt'])}")
    print(f"  raw_prompt len: {len(sample_dd['raw_prompt'])}")
    print(f"  reward_model:   {sample_dd['reward_model']}")
    print(f"  metadata keys:  {sample_dd['metadata'].keys() if isinstance(sample_dd['metadata'], dict) else 'N/A'}")

    # Value range comparisons
    print("\n📈 Value Ranges:")
    print(f"\n  response_words:")
    print(f"    NegotiationToM:  min={negtom['response_words'].min():3d}, max={negtom['response_words'].max():3d}, avg={negtom['response_words'].mean():.1f}")
    print(f"    DailyDialog:     min={dailydialog['response_words'].min():3d}, max={dailydialog['response_words'].max():3d}, avg={dailydialog['response_words'].mean():.1f}")

    print(f"\n  data_source values:")
    print(f"    NegotiationToM:  {negtom['data_source'].unique().tolist()}")
    print(f"    DailyDialog:     {dailydialog['data_source'].unique().tolist()}")

    print(f"\n  ability values:")
    print(f"    NegotiationToM:  {negtom['ability'].unique().tolist()}")
    print(f"    DailyDialog:     {dailydialog['ability'].unique().tolist()}")

    # Structure validation
    print("\n✅ Structure Validation:")

    checks = []

    # Check prompt structure
    try:
        assert isinstance(sample_dd['prompt'], list)
        assert all(isinstance(msg, dict) for msg in sample_dd['prompt'])
        assert all('content' in msg and 'role' in msg for msg in sample_dd['prompt'])
        checks.append(("prompt structure", True))
    except AssertionError:
        checks.append(("prompt structure", False))

    # Check raw_prompt structure
    try:
        assert isinstance(sample_dd['raw_prompt'], list)
        assert len(sample_dd['raw_prompt']) == 2
        assert sample_dd['raw_prompt'][0]['role'] == 'system'
        assert sample_dd['raw_prompt'][1]['role'] == 'user'
        checks.append(("raw_prompt structure", True))
    except AssertionError:
        checks.append(("raw_prompt structure", False))

    # Check reward_model structure
    try:
        assert isinstance(sample_dd['reward_model'], dict)
        assert 'ground_truth' in sample_dd['reward_model']
        assert 'style' in sample_dd['reward_model']
        checks.append(("reward_model structure", True))
    except AssertionError:
        checks.append(("reward_model structure", False))

    # Check metadata structure
    try:
        assert isinstance(sample_dd['metadata'], dict)
        assert 'conv_id' in sample_dd['metadata']
        assert 'turn' in sample_dd['metadata']
        checks.append(("metadata structure", True))
    except AssertionError:
        checks.append(("metadata structure", False))

    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")

    print("\n" + "=" * 80)

    if all(passed for _, passed in checks) and negtom_cols == dailydialog_cols:
        print("✅ DailyDialog dataset is compatible with NegotiationToM format!")
        print("=" * 80)
        return True
    else:
        print("⚠️  Some compatibility issues found. Review the details above.")
        print("=" * 80)
        return False


if __name__ == "__main__":
    import sys
    success = compare_datasets()
    sys.exit(0 if success else 1)
