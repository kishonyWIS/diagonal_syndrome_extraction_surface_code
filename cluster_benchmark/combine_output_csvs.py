#!/usr/bin/env python3
"""
Combine all CSV files in each subdirectory of cluster_benchmark/output/
into a single CSV file per directory with "from_cluster" in the filename.
"""

import os
import csv
from pathlib import Path
from typing import List


def combine_csvs_in_directory(directory: Path) -> None:
    """Combine all CSV files in a directory into a single CSV file."""
    # Find all CSV files in the directory
    csv_files = sorted(directory.glob("*.csv"))
    
    if not csv_files:
        print(f"  No CSV files found in {directory}")
        return
    
    # Determine output filename based on directory name
    dir_name = directory.name
    output_filename = f"result_{dir_name}_from_cluster.csv"
    output_path = directory / output_filename
    
    print(f"  Combining {len(csv_files)} CSV files in {directory.name}/")
    
    # Read and combine CSV files
    combined_rows = []
    header = None
    
    for csv_file in csv_files:
        # Skip the output file if it already exists (to avoid including it in itself)
        if csv_file.name == output_filename:
            continue
            
        try:
            with open(csv_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                
                # Store header from first file
                if header is None:
                    header = reader.fieldnames
                
                # Read all rows
                for row in reader:
                    combined_rows.append(row)
                    
        except Exception as e:
            print(f"    Warning: Error reading {csv_file.name}: {e}")
            continue
    
    if not combined_rows:
        print(f"  No data rows found in {directory}")
        return
    
    # Write combined CSV
    if header is None:
        print(f"  Warning: No header found, skipping {directory}")
        return
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(combined_rows)
    
    print(f"  Created: {output_path.name} ({len(combined_rows)} rows)")


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        return 1
    
    print(f"Combining CSV files in subdirectories of: {output_dir}")
    print("=" * 70)
    
    # Find all subdirectories
    subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
    
    if not subdirs:
        print("No subdirectories found in output directory")
        return 1
    
    # Process each subdirectory
    for subdir in sorted(subdirs):
        print(f"\nProcessing: {subdir.name}/")
        combine_csvs_in_directory(subdir)
    
    print("\n" + "=" * 70)
    print("Done!")
    
    return 0


if __name__ == "__main__":
    exit(main())
