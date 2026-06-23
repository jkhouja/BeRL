#!/usr/bin/env python3
"""
Example script showing how to use merge_parquet.py programmatically or from command line.
"""

import subprocess
import sys
from pathlib import Path

# Example 1: Merge DailyDialog and EmpatheticDialogues datasets
def example_merge_dialogue_datasets():
    """Merge daily dialogue and empathetic dialogue datasets"""
    print("=" * 60)
    print("Example 1: Merging DailyDialog and EmpatheticDialogues")
    print("=" * 60)

    input_files = [
        "data/DailyDialog_train_limit1000.parquet",
        "data/EmpatheticDialogues_sample.parquet"
    ]
    output_file = "data/merged_dialogue_datasets.parquet"

    cmd = ["python", "merge_parquet.py"] + input_files + ["-o", output_file]
    subprocess.run(cmd)
    print()


# Example 2: Merge all DailyDialog test files
def example_merge_test_files():
    """Merge all test files with wildcard pattern"""
    print("=" * 60)
    print("Example 2: Merging Multiple Test Files")
    print("=" * 60)

    # Note: Wildcard expansion happens in shell, so we use glob in Python
    import glob
    input_files = glob.glob("data/DailyDialog_test_*.parquet")
    output_file = "data/all_dailydialog_tests.parquet"

    if not input_files:
        print("No test files found!")
        return

    cmd = ["python", "merge_parquet.py"] + input_files + ["-o", output_file]
    subprocess.run(cmd)
    print()


# Example 3: Using the merge function directly (requires importing the module)
def example_programmatic_usage():
    """Use the merge function programmatically"""
    print("=" * 60)
    print("Example 3: Programmatic Usage")
    print("=" * 60)

    # Import the function from merge_parquet module
    sys.path.insert(0, str(Path(__file__).parent))
    from merge_parquet import merge_parquet_files

    input_files = [
        "data/DailyDialog_train_limit1000.parquet",
        "data/EmpatheticDialogues_sample.parquet"
    ]
    output_file = "data/programmatic_merge.parquet"

    try:
        merge_parquet_files(input_files, output_file)
    except Exception as e:
        print(f"Error: {e}")
    print()


# Example 4: Verify the merged file
def example_verify_merged_file():
    """Read and verify a merged parquet file"""
    print("=" * 60)
    print("Example 4: Verifying Merged File")
    print("=" * 60)

    import pandas as pd

    merged_file = "data/merged_dialogue_datasets.parquet"

    if not Path(merged_file).exists():
        print(f"File {merged_file} doesn't exist. Run example 1 or 3 first.")
        return

    df = pd.read_parquet(merged_file)

    print(f"✓ Successfully read {merged_file}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    print(f"\n  Column names:")
    for col in df.columns:
        print(f"    - {col}")

    if 'data_source' in df.columns:
        print(f"\n  Data source distribution:")
        print(df['data_source'].value_counts().to_string(prefix="    "))

    print(f"\n  First few rows:")
    print(df.head(2)[['data_source', 'ability', 'response_words']])
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("MERGE PARQUET - EXAMPLES")
    print("=" * 60 + "\n")

    # Run examples
    example_merge_dialogue_datasets()
    example_merge_test_files()
    example_programmatic_usage()
    example_verify_merged_file()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
