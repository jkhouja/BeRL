#!/usr/bin/env python3
"""
Merge multiple parquet files into a single parquet file.

This script takes a list of parquet file paths and merges them into a single
output parquet file. It's useful for combining datasets like DailyDialog and
EmpatheticDialogues.

Usage:
    python merge_parquet.py file1.parquet file2.parquet ... -o output.parquet
    python merge_parquet.py --input file1.parquet file2.parquet --output merged.parquet
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def merge_parquet_files(input_paths: List[str], output_path: str) -> None:
    """
    Merge multiple parquet files into a single parquet file.

    Args:
        input_paths: List of paths to input parquet files
        output_path: Path to the output merged parquet file

    Raises:
        FileNotFoundError: If any input file doesn't exist
        ValueError: If input_paths is empty
    """
    if not input_paths:
        raise ValueError("No input files provided")

    # Validate input files
    for path in input_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    print(f"Merging {len(input_paths)} parquet files...")

    # Read all parquet files as PyArrow tables
    tables = []
    total_rows = 0
    for i, path in enumerate(input_paths, 1):
        print(f"  [{i}/{len(input_paths)}] Reading {path}...")
        table = pq.read_table(path)
        print(f"       Shape: ({table.num_rows}, {table.num_columns}), Columns: {table.num_columns}")
        tables.append(table)
        total_rows += table.num_rows

    # Concatenate all tables using PyArrow
    # This handles schema differences more gracefully
    print("\nConcatenating tables...")
    try:
        # Try to concatenate with promote to handle string/large_string differences
        merged_table = pa.concat_tables(tables, promote_options="permissive")
    except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
        print(f"  PyArrow concatenation failed, using pandas fallback...")
        print(f"  Reason: {str(e)[:100]}...")

        # Fallback to pandas concatenation which is more lenient with mixed types
        dataframes = []
        for table in tables:
            df = table.to_pandas()
            # Convert struct columns to dictionaries to avoid type conflicts
            for col in df.columns:
                if df[col].dtype == object and len(df) > 0:
                    first_val = df[col].iloc[0]
                    # Check if it's a pyarrow-generated struct/dict
                    if first_val is not None and hasattr(first_val, 'items'):
                        # Keep as dict, pandas handles this well
                        pass
            dataframes.append(df)

        merged_df = pd.concat(dataframes, ignore_index=True)

        # Write using pandas to_parquet which handles mixed types better
        # We'll return early from this fallback path
        print(f"Merged dataframe shape: {merged_df.shape}")
        print(f"Total rows: {len(merged_df)}")
        print(f"Total columns: {len(merged_df.columns)}")

        # Check data sources
        if 'data_source' in merged_df.columns:
            print("\nData source distribution:")
            print(merged_df['data_source'].value_counts())

        # Create output directory if needed
        output_dir = Path(output_path).parent
        if output_dir and not output_dir.exists():
            print(f"\nCreating output directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)

        # Write using pandas with object preservation
        print(f"\nWriting merged parquet to {output_path}...")

        # For parquet files, we need to handle nested structures carefully
        # Copy the dataframe to avoid modifying the original
        df_to_write = merged_df.copy()

        # Convert complex nested dict/list columns to strings for parquet compatibility
        import json
        complex_cols = []
        for col in df_to_write.columns:
            if df_to_write[col].dtype == object and len(df_to_write) > 0:
                first_val = df_to_write[col].iloc[0]
                # Check if it contains dicts or complex structures
                if first_val is not None and (isinstance(first_val, (dict, list)) or hasattr(first_val, 'items')):
                    complex_cols.append(col)
                    df_to_write[col] = df_to_write[col].apply(
                        lambda x: json.dumps(dict(x)) if x is not None and hasattr(x, 'items')
                        else json.dumps(x) if x is not None and isinstance(x, list)
                        else x
                    )

        if complex_cols:
            print(f"  Note: Converted {len(complex_cols)} columns with nested structures to JSON strings: {complex_cols}")

        # Now write the parquet file
        df_to_write.to_parquet(output_path, index=False, engine='pyarrow')

        # Verify output file
        output_size = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"✓ Successfully created {output_path} ({output_size:.2f} MB)")
        return

    print(f"Merged table shape: ({merged_table.num_rows}, {merged_table.num_columns})")
    print(f"Total rows: {merged_table.num_rows}")
    print(f"Total columns: {merged_table.num_columns}")

    # Check data sources if the column exists
    if 'data_source' in merged_table.column_names:
        merged_df_temp = merged_table.to_pandas()
        print("\nData source distribution:")
        print(merged_df_temp['data_source'].value_counts())

    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    if output_dir and not output_dir.exists():
        print(f"\nCreating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    # Write merged parquet file
    print(f"\nWriting merged parquet to {output_path}...")
    pq.write_table(merged_table, output_path)

    # Verify output file
    output_size = Path(output_path).stat().st_size / (1024 * 1024)  # Convert to MB
    print(f"✓ Successfully created {output_path} ({output_size:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple parquet files into a single parquet file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge two files
  python merge_parquet.py file1.parquet file2.parquet -o merged.parquet

  # Merge DailyDialog and EmpatheticDialogues
  python merge_parquet.py \\
    data/DailyDialog_train_limit1000.parquet \\
    data/EmpatheticDialogues_sample.parquet \\
    -o data/merged_dialogue_datasets.parquet

  # Using --input flag
  python merge_parquet.py --input file1.parquet file2.parquet --output merged.parquet
        """
    )

    parser.add_argument(
        'input_files',
        nargs='*',
        help='Parquet files to merge (can also use --input)'
    )

    parser.add_argument(
        '-i', '--input',
        nargs='+',
        dest='input_files_flag',
        help='Parquet files to merge'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output path for the merged parquet file'
    )

    args = parser.parse_args()

    # Combine positional and flag-based input files
    input_files = (args.input_files or []) + (args.input_files_flag or [])

    if not input_files:
        parser.error("No input files provided. Use positional arguments or --input flag.")

    try:
        merge_parquet_files(input_files, args.output)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
