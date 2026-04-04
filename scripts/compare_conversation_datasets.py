#!/usr/bin/env python3
"""
Compare DailyDialog and Empathetic Dialogues datasets side by side.
This script demonstrates the differences in structure and content.
"""

import pandas as pd
from pathlib import Path


def compare_datasets(dd_path: str, ed_path: str):
    """
    Compare DailyDialog and Empathetic Dialogues datasets.

    Args:
        dd_path: Path to DailyDialog parquet file
        ed_path: Path to Empathetic Dialogues parquet file
    """
    print("=" * 80)
    print("DATASET COMPARISON: DailyDialog vs Empathetic Dialogues")
    print("=" * 80)

    # Load datasets
    print("\nLoading datasets...")
    dd = pd.read_parquet(dd_path)
    ed = pd.read_parquet(ed_path)

    print(f"✓ DailyDialog: {len(dd)} examples")
    print(f"✓ Empathetic Dialogues: {len(ed)} examples")

    # Basic statistics
    print("\n" + "=" * 80)
    print("BASIC STATISTICS")
    print("=" * 80)

    print("\nDataFrame Shape:")
    print(f"  DailyDialog:           {dd.shape}")
    print(f"  Empathetic Dialogues:  {ed.shape}")

    print("\nData Source:")
    print(f"  DailyDialog:           {dd['data_source'].unique()}")
    print(f"  Empathetic Dialogues:  {ed['data_source'].unique()}")

    print("\nAbility:")
    print(f"  DailyDialog:           {dd['ability'].unique()}")
    print(f"  Empathetic Dialogues:  {ed['ability'].unique()}")

    # Response statistics
    print("\n" + "=" * 80)
    print("RESPONSE STATISTICS")
    print("=" * 80)

    print("\nResponse Word Count:")
    print(f"  DailyDialog:")
    print(f"    - Mean:  {dd['response_words'].mean():.1f}")
    print(f"    - Median: {dd['response_words'].median():.1f}")
    print(f"    - Min:   {dd['response_words'].min()}")
    print(f"    - Max:   {dd['response_words'].max()}")

    print(f"\n  Empathetic Dialogues:")
    print(f"    - Mean:  {ed['response_words'].mean():.1f}")
    print(f"    - Median: {ed['response_words'].median():.1f}")
    print(f"    - Min:   {ed['response_words'].min()}")
    print(f"    - Max:   {ed['response_words'].max()}")

    # Conversation statistics
    print("\n" + "=" * 80)
    print("CONVERSATION STATISTICS")
    print("=" * 80)

    dd_convs = dd['metadata'].apply(lambda x: x['conv_id']).nunique()
    ed_convs = ed['metadata'].apply(lambda x: x['conv_id']).nunique()

    print("\nUnique Conversations:")
    print(f"  DailyDialog:           {dd_convs}")
    print(f"  Empathetic Dialogues:  {ed_convs}")

    print("\nExamples per Conversation:")
    print(f"  DailyDialog:           {len(dd) / dd_convs:.1f}")
    print(f"  Empathetic Dialogues:  {len(ed) / ed_convs:.1f}")

    dd_turns = dd['metadata'].apply(lambda x: x['total_turns'])
    ed_turns = ed['metadata'].apply(lambda x: x['total_turns'])

    print("\nTurns per Conversation:")
    print(f"  DailyDialog:")
    print(f"    - Mean:  {dd_turns.mean():.1f}")
    print(f"    - Median: {dd_turns.median():.1f}")
    print(f"    - Min:   {dd_turns.min()}")
    print(f"    - Max:   {dd_turns.max()}")

    print(f"\n  Empathetic Dialogues:")
    print(f"    - Mean:  {ed_turns.mean():.1f}")
    print(f"    - Median: {ed_turns.median():.1f}")
    print(f"    - Min:   {ed_turns.min()}")
    print(f"    - Max:   {ed_turns.max()}")

    # Metadata comparison
    print("\n" + "=" * 80)
    print("METADATA DIFFERENCES")
    print("=" * 80)

    dd_meta_keys = set(dd.iloc[0]['metadata'].keys())
    ed_meta_keys = set(ed.iloc[0]['metadata'].keys())

    print("\nMetadata Keys:")
    print(f"  Both datasets:         {dd_meta_keys & ed_meta_keys}")
    print(f"  DailyDialog only:      {dd_meta_keys - ed_meta_keys}")
    print(f"  Empathetic only:       {ed_meta_keys - dd_meta_keys}")

    # Empathetic Dialogues specific: emotion labels
    if 'emotion_label' in ed.iloc[0]['metadata']:
        emotion_counts = ed['metadata'].apply(lambda x: x['emotion_label']).value_counts()
        print("\n" + "=" * 80)
        print("EMOTION LABELS (Empathetic Dialogues)")
        print("=" * 80)
        print(f"\nTotal unique emotions: {len(emotion_counts)}")
        print(f"\nTop 10 emotions:")
        for emotion, count in emotion_counts.head(10).items():
            print(f"  {emotion:20s}: {count:5d} ({count/len(ed)*100:.1f}%)")

    # Sample examples
    print("\n" + "=" * 80)
    print("SAMPLE EXAMPLES")
    print("=" * 80)

    print("\n--- DailyDialog Example ---")
    dd_sample = dd.iloc[0]
    print(f"\nSystem Prompt:\n{dd_sample['raw_system_prompt'][:150]}...")
    print(f"\nUser Prompt:\n{dd_sample['raw_user_prompt'][:200]}...")
    print(f"\nGround Truth:\n{dd_sample['reward_model']['ground_truth'][:150]}...")
    print(f"\nMetadata:")
    for key, value in dd_sample['metadata'].items():
        if key != 'dialogue_history':
            print(f"  {key}: {value}")

    print("\n--- Empathetic Dialogues Example ---")
    ed_sample = ed.iloc[0]
    print(f"\nSystem Prompt:\n{ed_sample['raw_system_prompt'][:150]}...")
    print(f"\nUser Prompt:\n{ed_sample['raw_user_prompt'][:200]}...")
    print(f"\nGround Truth:\n{ed_sample['reward_model']['ground_truth'][:150]}...")
    print(f"\nMetadata:")
    for key, value in ed_sample['metadata'].items():
        if key != 'dialogue_history':
            print(f"  {key}: {value}")

    # Speaker comparison
    print("\n" + "=" * 80)
    print("SPEAKER PATTERNS")
    print("=" * 80)

    dd_speakers = dd['metadata'].apply(lambda x: x['responding_speaker']).value_counts()
    ed_speakers = ed['metadata'].apply(lambda x: x['responding_speaker']).value_counts()

    print("\nDailyDialog Speakers:")
    for speaker, count in dd_speakers.items():
        print(f"  {speaker:20s}: {count:5d} ({count/len(dd)*100:.1f}%)")

    print("\nEmpathetic Dialogues Speakers:")
    for speaker, count in ed_speakers.items():
        print(f"  {speaker:20s}: {count:5d} ({count/len(ed)*100:.1f}%)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("""
Key Differences:

1. Dataset Focus:
   - DailyDialog: General daily conversations, topic-based
   - Empathetic Dialogues: Emotional situations, empathy-focused

2. Metadata:
   - DailyDialog: Generic speaker labels (Speaker_1, Speaker_2)
   - Empathetic Dialogues: Role-based labels (Speaker, Listener) + emotion labels

3. Prompts:
   - DailyDialog: Generic conversation continuation
   - Empathetic Dialogues: Can include emotional context

4. Use Cases:
   - DailyDialog: General dialogue modeling, chitchat
   - Empathetic Dialogues: Emotional understanding, supportive responses

5. Size & Structure:
   - DailyDialog: Fewer conversations, longer on average
   - Empathetic Dialogues: More conversations, emotion-grounded
    """)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare DailyDialog and Empathetic Dialogues datasets"
    )
    parser.add_argument(
        "--dailydialog",
        type=str,
        default="data/DailyDialog.parquet",
        help="Path to DailyDialog parquet file"
    )
    parser.add_argument(
        "--empathetic",
        type=str,
        default="data/EmpatheticDialogues.parquet",
        help="Path to Empathetic Dialogues parquet file"
    )

    args = parser.parse_args()

    # Check if files exist
    dd_path = Path(args.dailydialog)
    ed_path = Path(args.empathetic)

    if not dd_path.exists():
        print(f"Error: DailyDialog file not found: {dd_path}")
        print("\nGenerate it with:")
        print("  python scripts/convert_dailydialog.py --sample-size 1000")
        return 1

    if not ed_path.exists():
        print(f"Error: Empathetic Dialogues file not found: {ed_path}")
        print("\nGenerate it with:")
        print("  python scripts/convert_empathetic_dialogues.py --sample-size 1000")
        return 1

    # Run comparison
    compare_datasets(str(dd_path), str(ed_path))

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
