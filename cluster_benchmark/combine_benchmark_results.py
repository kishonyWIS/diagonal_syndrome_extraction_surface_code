#!/usr/bin/env python3
"""
Combine individual benchmark result CSV files into a single combined file.

When multiple results exist for the same configuration, combines them by:
- Summing shots and errors
- Computing logical_error_rate = total_errors / total_shots
- Computing error_bar from combined statistics

Usage:
    python combine_benchmark_results.py
    python combine_benchmark_results.py --input-dir benchmark_output --output benchmark_data/combined_results.csv
"""

import argparse
import csv
import glob
import math
import os
import sys


def combine_csv_files(input_dir: str, output_file: str, verbose: bool = True) -> int:
    """
    Combine all CSV files in input_dir into a single output file.
    
    When duplicates exist for the same (decoder, k, physical_error_rate, flag_config, direction),
    combines them by summing shots and errors, then recomputing logical_error_rate.
    
    Args:
        input_dir: Directory containing individual result CSV files
        output_file: Path to combined output CSV file
        verbose: Whether to print progress
        
    Returns:
        Number of results combined
    """
    # Find all CSV files
    pattern = os.path.join(input_dir, "*.csv")
    csv_files = sorted(glob.glob(pattern))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return 0
    
    if verbose:
        print(f"Found {len(csv_files)} CSV files in {input_dir}")
    
    # Expected fieldnames
    fieldnames = [
        'direction', 'flag_config', 'k', 'distance', 'physical_error_rate',
        'decoder', 'logical_error_rate', 'errors', 'shots', 'error_bar', 'decode_time',
    ]
    
    # Collect all results, combining duplicates
    # key -> {'shots': total_shots, 'errors': total_errors, 'decode_time': total_time, 'row': first_row}
    results_dict = {}
    files_with_errors = []
    duplicates_combined = 0
    
    def make_key(row):
        """Create unique key from (decoder, k, physical_error_rate, flag_config, direction)"""
        return (
            row.get('decoder', ''),
            row.get('k', ''),
            row.get('physical_error_rate', ''),
            row.get('flag_config', ''),
            row.get('direction', ''),
        )
    
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = make_key(row)
                    
                    # Parse numeric values
                    try:
                        shots = int(row.get('shots', 0))
                        errors = int(row.get('errors', 0))
                        decode_time = float(row.get('decode_time', 0))
                    except (ValueError, TypeError):
                        shots = 0
                        errors = 0
                        decode_time = 0
                    
                    if key not in results_dict:
                        results_dict[key] = {
                            'shots': shots,
                            'errors': errors,
                            'decode_time': decode_time,
                            'row': row,  # Keep first row for non-aggregated fields
                        }
                    else:
                        # Combine: sum shots, errors, and decode_time
                        results_dict[key]['shots'] += shots
                        results_dict[key]['errors'] += errors
                        results_dict[key]['decode_time'] += decode_time
                        duplicates_combined += 1
        except Exception as e:
            files_with_errors.append((csv_file, str(e)))
            if verbose:
                print(f"  Warning: Error reading {csv_file}: {e}")
    
    if verbose and duplicates_combined > 0:
        print(f"  Combined {duplicates_combined} duplicate entries")
    
    # Build final results with recomputed statistics
    all_results = []
    for key, data in results_dict.items():
        row = data['row'].copy()
        total_shots = data['shots']
        total_errors = data['errors']
        
        # Recompute logical_error_rate and error_bar
        if total_shots > 0:
            logical_error_rate = total_errors / total_shots
            # Error bar: standard error assuming binomial distribution
            # stderr = sqrt(p * (1-p) / n) ≈ sqrt(p / n) for small p
            if logical_error_rate > 0 and logical_error_rate < 1:
                error_bar = math.sqrt(logical_error_rate * (1 - logical_error_rate) / total_shots)
            else:
                error_bar = 0
        else:
            logical_error_rate = 0
            error_bar = 0
        
        row['shots'] = total_shots
        row['errors'] = total_errors
        row['logical_error_rate'] = logical_error_rate
        row['error_bar'] = error_bar
        row['decode_time'] = data['decode_time']
        
        all_results.append(row)
    
    if verbose:
        print(f"Loaded {len(all_results)} unique configurations from {len(csv_files) - len(files_with_errors)} files")
        if files_with_errors:
            print(f"  {len(files_with_errors)} files had errors")
    
    # Sort results for consistent ordering
    # Sort by: decoder, k, physical_error_rate, flag_config, direction
    def sort_key(row):
        try:
            return (
                row.get('decoder', ''),
                int(row.get('k', 0)),
                float(row.get('physical_error_rate', 0)),
                row.get('flag_config', ''),
                row.get('direction', ''),
            )
        except (ValueError, TypeError):
            return ('', 0, 0, '', '')
    
    all_results.sort(key=sort_key)
    
    # Write combined file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            # Ensure all fields exist
            clean_row = {field: row.get(field, '') for field in fieldnames}
            writer.writerow(clean_row)
    
    if verbose:
        print(f"Wrote {len(all_results)} results to {output_file}")
    
    return len(all_results)


def check_completeness(input_dir: str, verbose: bool = True) -> dict:
    """
    Check how many jobs have completed vs expected.
    
    Returns dict with counts.
    """
    import numpy as np
    
    # Expected configurations (match submit_benchmark_jobs.sh)
    decoders = ['tesseract', 'pymatching', 'correlated_pymatching']
    k_values_by_decoder = {
        'tesseract': [1, 2, 3, 4],
        'pymatching': [1, 2, 3, 4, 5, 6],
        'correlated_pymatching': [1, 2, 3, 4],
    }
    # Use strings to preserve exact filename format (floats lose precision)
    noise_values = ['0.01', '0.005623413251903491', '0.0031622776601683794', 
                   '0.0017782794100389228', '0.001', '0.0005623413251903491', '0.00031622776601683794',
                   '0.00017782794100389228', '0.0001']
    flag_configs = ['none', 'partial', 'all']
    directions = ['y']
    
    expected_total = sum(
        len(k_values_by_decoder[d]) * len(noise_values) * len(flag_configs) * len(directions)
        for d in decoders
    )
    
    # Count completed
    pattern = os.path.join(input_dir, "*.csv")
    csv_files = glob.glob(pattern)
    completed = len(csv_files)
    
    # Check for expected combinations
    expected_files = set()
    for decoder in decoders:
        for k in k_values_by_decoder[decoder]:
            for noise in noise_values:
                for flag_config in flag_configs:
                    for direction in directions:
                        filename = f"result_{decoder}_k{k}_p{noise}_{flag_config}_{direction}.csv"
                        expected_files.add(filename)
    
    actual_files = {os.path.basename(f) for f in csv_files}
    missing = expected_files - actual_files
    extra = actual_files - expected_files
    
    result = {
        'expected': expected_total,
        'completed': completed,
        'missing': len(missing),
        'extra': len(extra),
        'missing_files': sorted(missing),
        'extra_files': sorted(extra),
    }
    
    if verbose:
        print(f"\nCompleteness check:")
        print(f"  Expected: {expected_total}")
        print(f"  Completed: {completed}")
        print(f"  Missing: {len(missing)}")
        if missing and len(missing) <= 10:
            for f in sorted(missing):
                print(f"    - {f}")
        elif missing:
            print(f"    (showing first 10)")
            for f in sorted(missing)[:10]:
                print(f"    - {f}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Combine benchmark CSV results")
    parser.add_argument('--input-dir', type=str, default='cluster_benchmark/output',
                        help='Directory containing individual result CSV files')
    parser.add_argument('--output', type=str, 
                        default='cluster_benchmark/combined_results.csv',
                        help='Output combined CSV file')
    parser.add_argument('--check-only', action='store_true',
                        help='Only check completeness, do not combine')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    print("=" * 60)
    print("Benchmark Results Combiner")
    print("=" * 60)
    
    # Check completeness
    status = check_completeness(args.input_dir, verbose=verbose)
    
    if args.check_only:
        if status['missing'] > 0:
            sys.exit(1)
        sys.exit(0)
    
    # Combine results
    print()
    count = combine_csv_files(args.input_dir, args.output, verbose=verbose)
    
    if count == 0:
        print("\nNo results to combine!")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print(f"Combined {count} results into {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
