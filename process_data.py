import time
import argparse
import os
import pandas as pd
import numpy as np
import concurrent
from concurrent.futures import ProcessPoolExecutor
import mmh3
from typing import List, Tuple
from collections import defaultdict
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constant import (
    DEFAULT_COLUMN_NAMES,
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    DEFAULT_LABEL_NAMES
)

def str_to_bool(value):
    """Convert a string to a boolean."""
    if value.lower() in ('true', '1', 'yes', 'y'):
        return True
    elif value.lower() in ('false', '0', 'no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected. Got {value}")

def process_and_save(df, output_dir):        
    cat_value_columns = []
    cat_length_columns = []
    cont_value_columns = []
    label_columns = []

    for column in df.columns:
        if 'hashed' in column:
            cat_value_columns.append(column)
        elif 'length' in column:
            cat_length_columns.append(column)
        elif column in DEFAULT_INT_NAMES:
            cont_value_columns.append(column)
        elif column in DEFAULT_LABEL_NAMES:
            label_columns.append(column)
            
    cat_values = df[cat_value_columns].to_numpy()
    flattened_cat_values = cat_values.transpose(1, 0).reshape(-1).astype(np.int64)
    cat_values_bytes = flattened_cat_values.tobytes()
    cat_value_save_path = os.path.join(output_dir, 'cat_value.bin')
    with open(cat_value_save_path, 'wb') as f:
        f.write(cat_values_bytes)

    cat_lengths = df[cat_length_columns].to_numpy()
    flattened_cat_lengths = cat_lengths.transpose(1, 0).reshape(-1).astype(np.int32)
    cat_lengths_bytes = flattened_cat_lengths.tobytes()
    cat_lengths_save_path = os.path.join(output_dir, 'cat_length.bin')
    with open(cat_lengths_save_path, 'wb') as f:
        f.write(cat_lengths_bytes)

    flattened_cat_cum_lengths = np.cumsum(flattened_cat_lengths.astype(np.int64))
    cat_cum_lengths_bytes = flattened_cat_cum_lengths.tobytes()
    cat_cum_length_save_path = os.path.join(output_dir, 'cat_cum_length.bin')
    with open(cat_cum_length_save_path, 'wb') as f:
        f.write(cat_cum_lengths_bytes)

    cont_values = df[cont_value_columns].to_numpy()
    flattened_cont_values = cont_values.reshape(-1).astype(np.float32)
    cont_values_bytes = flattened_cont_values.tobytes()
    cont_value_save_path = os.path.join(output_dir, 'numerical.bin')
    with open(cont_value_save_path, 'wb') as f:
        f.write(cont_values_bytes)

    labels = df[label_columns].to_numpy().reshape(-1).astype(np.int32)
    labels_bytes = labels.tobytes()
    labels_save_path = os.path.join(output_dir, 'label.bin')
    with open(labels_save_path, 'wb') as f:
        f.write(labels_bytes)

    return df

from sklearn.model_selection import train_test_split

def process_and_save_split(df, output_dir, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    """
    Split data into train, valid, test sets and save them in respective folders
    """
    # Verify split ratios sum to 1
    assert np.isclose(train_ratio + valid_ratio + test_ratio, 1.0)

    # Create directories for train, valid, test
    train_dir = os.path.join(output_dir, 'train')
    valid_dir = os.path.join(output_dir, 'valid')
    test_dir = os.path.join(output_dir, 'test')
    
    for directory in [train_dir, valid_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)

    # First split: separate train and temp (valid + test)
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_ratio, 
        random_state=42
    )

    # Second split: separate valid and test from temp
    valid_ratio_adjusted = valid_ratio / (valid_ratio + test_ratio)
    valid_df, test_df = train_test_split(
        temp_df, 
        train_size=valid_ratio_adjusted,
        random_state=42
    )

    # Print split sizes
    print(f"Total rows: {len(df)}")
    print(f"Train rows: {len(train_df)} ({len(train_df)/len(df):.2%})")
    print(f"Valid rows: {len(valid_df)} ({len(valid_df)/len(df):.2%})")
    print(f"Test rows: {len(test_df)} ({len(test_df)/len(df):.2%})")

    # Process and save each split
    process_and_save(train_df, train_dir)
    process_and_save(valid_df, valid_dir)
    process_and_save(test_df, test_dir)


def hash_row(args: Tuple[List[str], List[str], int, int]) -> Tuple[int, List[int], List[str]]:
    """
    Hash an entire row of data.
    
    Args:
        args: Tuple containing (row, column_names, num_embeddings, row_index)
    
    Returns:
        Tuple of (row_index, hashed_values, original_combined_values)
    """
    row, num_embeddings, row_index = args
    hashed_row = [row_index]
    
    for value, col_name in zip(row, DEFAULT_COLUMN_NAMES):
        if col_name in DEFAULT_CAT_NAMES:
            combined = f"{col_name}_{value}"
            hashed_row.append(combined)
            hashed = abs(mmh3.hash64(combined)[0]) % num_embeddings
            hashed_row.append(hashed)
            hashed_row.append(1)  # length of the value, used for list
        elif col_name in DEFAULT_INT_NAMES:
            hashed_row.append(np.float32(value))
        elif col_name in DEFAULT_LABEL_NAMES:
            value = 1 if int(value) >= 1 else 0
            hashed_row.append(np.int32(value))
    
    return hashed_row

def calculate_conflicts(df) -> dict:
    """
    Calculate hash conflicts for each column and overall totals.
    
    Returns:
        Dictionary containing conflict statistics for each column and totals
    """
    conflicts = {}
    
    # Initialize totals
    total_hash_to_orig = defaultdict(set)
    total_original_values = set()
    
    for col_name in DEFAULT_CAT_NAMES:
        # Create mapping of hash value to original values
        hash_to_orig = defaultdict(set)
        original_values = df[f'{col_name}_combined'].tolist()
        hashed_values = df[f'{col_name}_hashed'].tolist()

        for row_idx in range(len(original_values)):
            orig_val = original_values[row_idx]
            hash_val = hashed_values[row_idx]
            hash_to_orig[hash_val].add(orig_val)
            
            # Add to totals with column name to distinguish same values from different columns
            total_hash_to_orig[hash_val].add(orig_val)
            total_original_values.add(orig_val)
        
        # Count conflicts for this column
        total_values = len(original_values)
        unique_original = len(set(row for row in original_values))
        unique_hashed = len(hash_to_orig)
        conflict_count = sum(1 for orig_set in hash_to_orig.values() if len(orig_set) > 1)
        
        conflicts[col_name] = {
            'total_values': total_values,
            'unique_original': unique_original,
            'unique_hashed': unique_hashed,
            'conflict_count': conflict_count,
            'conflict_rate': conflict_count / unique_original if unique_original > 0 else 0,
            'collisions': {
                hash_val: list(orig_vals) 
                for hash_val, orig_vals in hash_to_orig.items() 
                if len(orig_vals) > 1
            }
        }
    
    # Calculate total conflicts across all columns
    total_values = len(original_values) * len(DEFAULT_CAT_NAMES)
    unique_original_total = len(total_original_values)
    unique_hashed_total = len(total_hash_to_orig)
    conflict_count_total = sum(1 for orig_set in total_hash_to_orig.values() if len(orig_set) > 1)
    
    conflicts['total'] = {
        'total_values': total_values,
        'unique_original': unique_original_total,
        'unique_hashed': unique_hashed_total,
        'conflict_count': conflict_count_total,
        'conflict_rate': conflict_count_total / unique_original_total if unique_original_total > 0 else 0,
        'collisions': {
            hash_val: [orig_val.split('_', 1)[1] + f" (from {orig_val.split('_', 1)[0]})" 
                      for orig_val in orig_vals]
            for hash_val, orig_vals in total_hash_to_orig.items() 
            if len(orig_vals) > 1
        }
    }
    
    return conflicts

def parallel_hash_data(data: List[List[str]], 
                      num_embeddings: int,
                      max_workers: int = 4) -> Tuple[List[List[int]], dict]:
    """
    Parallel process the data list by hashing each row and calculate conflicts.
    
    Returns:
        Tuple of (hashed_data, conflicts_info)
    """
    # Prepare arguments for parallel processing
    args_list = [(row, num_embeddings, i) for i, row in enumerate(data)]
    total_rows = len(data)
    
    # Process in parallel with progress bar
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create future objects for all tasks
        futures = [executor.submit(hash_row, args) for args in args_list]
        
        # Use tqdm to show progress
        with tqdm(total=total_rows, desc="Hashing rows") as pbar:
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                pbar.update(1)
    
    # Sort results by row index to maintain original order
    results.sort(key=lambda x: x[0])
    # Separate hashed data and original combined values

    column_names_expand = ['id']
    for col_name in DEFAULT_COLUMN_NAMES:
        if col_name in DEFAULT_CAT_NAMES:
            column_names_expand.extend([f"{col_name}_combined", f"{col_name}_hashed", f'{col_name}_length'])
        else:
            column_names_expand.append(col_name)
    
    df = pd.DataFrame(results, columns = column_names_expand)
    print(f"df shape: {df.shape}")
    print(df.head(10))

    # Calculate conflicts
    conflicts = calculate_conflicts(df)
    
    return df, conflicts

def print_conflict_summary(conflicts: dict, output_file: str = "conflict.txt"):
    """
    Print a summary of hash conflicts to both console and file.
    
    Args:
        conflicts: Dictionary containing conflict statistics
        output_file: Path to output file (default: "conflict.txt")
    """
    # Create a function to generate the output lines
    def generate_summary():
        lines = []
        lines.append("\nHash Conflict Summary:")
        lines.append("-" * 80)
        
        for col_name, stats in conflicts.items():
            lines.append(f"\nColumn: {col_name}")
            lines.append(f"Total values: {stats['total_values']}")
            lines.append(f"Unique original values: {stats['unique_original']}")
            lines.append(f"Unique hash values: {stats['unique_hashed']}")
            lines.append(f"Number of conflicts: {stats['conflict_count']}")
            lines.append(f"Conflict rate: {stats['conflict_rate']:.4%}")
            
            if stats['conflict_count'] > 0:
                lines.append("Sample conflicts:")
                # Print up to 3 examples of conflicts
                for hash_val, orig_vals in list(stats['collisions'].items())[:3]:
                    lines.append(f"  Hash value {hash_val}: {orig_vals}")
        
        return lines

    # Generate the summary lines
    summary_lines = generate_summary()
    
    # Print to console
    for line in summary_lines:
        print(line)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in summary_lines:
            f.write(line + '\n')
    
    print(f"\nConflict summary has been saved to {output_file}")
    
def main(args):

    start_time = time.time()

    with open(args.data_path, 'r') as f:
        data = [line.strip().split('\t') for line in f if line.strip()]

    print(f"len of data: {len(data)}, data: {data[:5]}")

    # Process the data and get conflicts
    df, conflicts = parallel_hash_data(data, args.num_embeddings, max_workers=args.num_workers)

    # Usage in main code:
    os.makedirs(args.output_dir, exist_ok=True)
    process_and_save_split(df, args.output_dir)

    conflict_output_file = os.path.join(args.output_dir, "conflict.txt")
    print_conflict_summary(conflicts, conflict_output_file)

    duration_total = time.time() - start_time
    print(f"Total time: {duration_total:.2} s.")
    
def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--column_name_path', type=str, default='', required=True)
    parser.add_argument('--data_path', type=str, default='', required=True)
    parser.add_argument('--output_dir', type=str, default='', required=True)     
    parser.add_argument('--num_files', type=int, default=1)
    parser.add_argument('--num_embeddings', type=int, default=10000000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--shuffle', type=str_to_bool, default=False,
                    help="Whether to shuffle the dataset. Use 'true' or 'false'.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = parse_args()
    main(args)