# Parquet File Merger

A Python script for merging multiple parquet files into a single parquet file. Designed to handle datasets with varying schemas and complex nested structures.

## Features

- ✅ Merges multiple parquet files with compatible schemas
- ✅ Handles schema mismatches (different column counts, types)
- ✅ Supports complex nested structures (structs, lists, dicts)
- ✅ Automatic fallback for incompatible PyArrow schemas
- ✅ Progress reporting during merge process
- ✅ Validates input files before processing
- ✅ Creates output directories automatically

## Requirements

```bash
pip install pandas pyarrow
```

## Usage

### Basic Usage

```bash
python merge_parquet.py file1.parquet file2.parquet -o merged.parquet
```

### Merge DailyDialog and EmpatheticDialogues

```bash
python merge_parquet.py \
  data/DailyDialog_train_limit1000.parquet \
  data/EmpatheticDialogues_sample.parquet \
  -o data/merged_dialogue_datasets.parquet
```

### Merge Multiple Test Files

```bash
python merge_parquet.py data/DailyDialog_test_*.parquet -o data/all_tests.parquet
```

### Using --input Flag

```bash
python merge_parquet.py --input file1.parquet file2.parquet --output merged.parquet
```

## Command-Line Arguments

- **Positional arguments**: List of input parquet files to merge
- `-i, --input`: Alternative way to specify input files
- `-o, --output`: Output path for the merged parquet file (required)
- `-h, --help`: Show help message

## How It Works

1. **Validation**: Checks that all input files exist
2. **Reading**: Loads parquet files as PyArrow tables
3. **Concatenation**:
   - First attempts PyArrow concatenation with permissive type promotion
   - Falls back to pandas if schemas are incompatible
4. **Type Handling**: Converts complex nested structures (dicts, lists) to JSON strings when needed
5. **Writing**: Saves the merged data as a parquet file

## Schema Handling

The script intelligently handles schema mismatches:

- **String types**: Automatically promotes `string` to `large_string`
- **Missing columns**: Fills with null values
- **Incompatible structs**: Falls back to pandas and converts to JSON strings
- **Different column counts**: Merges all unique columns

## Example Output

```
Merging 2 parquet files...
  [1/2] Reading data/DailyDialog_train_limit1000.parquet...
       Shape: (1000, 12), Columns: 12
  [2/2] Reading data/EmpatheticDialogues_sample.parquet...
       Shape: (1000, 12), Columns: 12

Concatenating tables...
  PyArrow concatenation failed, using pandas fallback...
  Reason: Unable to merge: Field metadata has incompatible types...
Merged dataframe shape: (2000, 12)
Total rows: 2000
Total columns: 12

Data source distribution:
data_source
dailydialog             1000
empathetic_dialogues    1000
Name: count, dtype: int64

Writing merged parquet to data/merged_dialogue_datasets.parquet...
  Note: Converted 2 columns with nested structures to JSON strings: ['reward_model', 'metadata']
✓ Successfully created data/merged_dialogue_datasets.parquet (1.36 MB)
```

## Use Cases

- Combining training and test datasets
- Merging data from different sources with similar schemas
- Consolidating multiple parquet files for easier processing
- Creating unified datasets from multiple conversation datasets

## Notes

- The script preserves all columns from all input files
- When PyArrow schemas are incompatible, complex nested structures are converted to JSON strings
- The `data_source` column (if present) can be used to track which file each row came from
- Output file sizes are optimized using parquet compression

## Troubleshooting

### Error: "Input file not found"
Make sure all file paths are correct and files exist.

### Error: "No input files provided"
You must provide at least one input file using positional arguments or `--input` flag.

### Complex type conversion warnings
This is normal when merging files with incompatible nested structures. Data is preserved as JSON strings.
