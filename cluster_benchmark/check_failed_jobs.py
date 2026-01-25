#!/usr/bin/env python3
"""
Check for failed jobs by comparing expected CSV files vs actual files,
and by checking LSF job status.

Can operate in two modes:
1. CSV-based: Check which expected CSV files are missing
2. LSF-based: Query LSF to find jobs that failed (exit code != 0)
"""

import argparse
import os
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


# Expected configurations for each experiment type
EXPERIMENT_CONFIGS = {
    'memory': {
        'schedules': ['N/Z', 'diagonal'],
        'k_values': [1, 2, 3, 4],
        'noise_values': list(np.logspace(-4, -2, 9)[::-1]),
        'output_dir': 'cluster_benchmark/output/memory',
    },
    'x_junction': {
        'schedules': ['N/Z', 'diagonal'],
        'k_values': [1, 2, 3, 4],
        'noise_values': list(np.logspace(-4, -2, 9)[::-1]),
        'output_dir': 'cluster_benchmark/output/x_junction',
    },
    'patch_rotation': {
        'k_values': [1, 2, 3, 4],
        'basis': 'z',
        'noise_values': list(np.logspace(-4, -2, 9)[::-1]),
        'output_dir': 'cluster_benchmark/output/patch_rotation',
    },
    'spatial_hadamard_interface': {
        'k_values': [1, 2, 3, 4],
        'flag_configs': ['none', 'partial', 'all'],
        'direction': 'y',
        'noise_mode': 'interface_only',
        'noise_values': list(np.logspace(-4, -2, 9)[::-1]),
        'output_dir': 'cluster_benchmark/output/spatial_hadamard_interface',
    },
    'spatial_hadamard_full': {
        'k_values': [1, 2, 3, 4],
        'flag_configs': ['none', 'partial', 'all'],
        'direction': 'y',
        'noise_mode': 'full',
        'noise_values': list(np.logspace(-4, -2, 9)[::-1]),
        'output_dir': 'cluster_benchmark/output/spatial_hadamard_full',
    },
}


def generate_expected_csv_filename(experiment_type: str, **params) -> str:
    """Generate expected CSV filename based on experiment type and parameters."""
    output_dir = EXPERIMENT_CONFIGS[experiment_type]['output_dir']
    
    if experiment_type == 'memory':
        schedule_suffix = params['schedule'].replace('/', '_').lower()
        return f"{output_dir}/result_memory_{schedule_suffix}_k{params['k']}_p{params['noise']}.csv"
    elif experiment_type == 'x_junction':
        schedule_suffix = params['schedule'].replace('/', '_').lower()
        return f"{output_dir}/result_x_junction_{schedule_suffix}_k{params['k']}_p{params['noise']}.csv"
    elif experiment_type == 'patch_rotation':
        return f"{output_dir}/result_patch_rotation_k{params['k']}_p{params['noise']}.csv"
    elif experiment_type == 'spatial_hadamard_interface':
        return f"{output_dir}/result_spatial_hadamard_interface_{params['flag_config']}_k{params['k']}_p{params['noise']}.csv"
    elif experiment_type == 'spatial_hadamard_full':
        return f"{output_dir}/result_spatial_hadamard_full_{params['flag_config']}_k{params['k']}_p{params['noise']}.csv"
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")


def check_missing_csvs(experiment_type: Optional[str] = None) -> List[Dict]:
    """Check for missing CSV files based on expected configurations."""
    failed_jobs = []
    
    experiment_types = [experiment_type] if experiment_type else EXPERIMENT_CONFIGS.keys()
    
    for exp_type in experiment_types:
        if exp_type not in EXPERIMENT_CONFIGS:
            continue
        
        config = EXPERIMENT_CONFIGS[exp_type]
        output_dir = Path(config['output_dir'])
        
        if exp_type in ['memory', 'x_junction']:
            for schedule in config['schedules']:
                for k in config['k_values']:
                    for noise in config['noise_values']:
                        expected_file = generate_expected_csv_filename(
                            exp_type, schedule=schedule, k=k, noise=noise
                        )
                        if not Path(expected_file).exists():
                            failed_jobs.append({
                                'experiment_type': exp_type,
                                'schedule': schedule,
                                'k': k,
                                'noise': noise,
                                'missing_file': expected_file,
                            })
        
        elif exp_type == 'patch_rotation':
            for k in config['k_values']:
                for noise in config['noise_values']:
                    expected_file = generate_expected_csv_filename(
                        exp_type, k=k, noise=noise
                    )
                    if not Path(expected_file).exists():
                        failed_jobs.append({
                            'experiment_type': exp_type,
                            'k': k,
                            'noise': noise,
                            'missing_file': expected_file,
                        })
        
        elif exp_type in ['spatial_hadamard_interface', 'spatial_hadamard_full']:
            for k in config['k_values']:
                for flag_config in config['flag_configs']:
                    for noise in config['noise_values']:
                        expected_file = generate_expected_csv_filename(
                            exp_type, flag_config=flag_config, k=k, noise=noise
                        )
                        if not Path(expected_file).exists():
                            failed_jobs.append({
                                'experiment_type': exp_type,
                                'flag_config': flag_config,
                                'k': k,
                                'noise': noise,
                                'missing_file': expected_file,
                            })
    
    return failed_jobs


def check_lsf_failed_jobs(job_id_file: Optional[str] = None) -> List[Dict]:
    """Check LSF for failed jobs by reading job ID files and querying bhist."""
    failed_jobs = []
    
    if job_id_file:
        job_id_files = [job_id_file]
    else:
        # Find all job_id files
        project_dir = Path(__file__).parent.parent
        job_id_files = list(project_dir.glob('cluster_benchmark/job_ids_*.txt'))
    
    for job_id_file_path in job_id_files:
        if not job_id_file_path.exists():
            continue
        
        experiment_type = job_id_file_path.stem.replace('job_ids_', '')
        
        # Read job IDs from file
        with open(job_id_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                job_id = parts[0]
                
                # Query bhist for this job
                try:
                    result = subprocess.run(
                        ['bhist', '-l', job_id],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    # Skip if bhist command itself failed (e.g., job not found)
                    if result.returncode != 0:
                        continue
                    
                    # Check if job failed (exit code != 0)
                    if 'Exited with exit code' in result.stdout:
                        # Extract exit code
                        exit_code_line = [l for l in result.stdout.split('\n') if 'Exited with exit code' in l]
                        if exit_code_line:
                            exit_code = exit_code_line[0].split('Exited with exit code')[-1].strip()
                            try:
                                exit_code = int(exit_code)
                                if exit_code != 0:
                                    # Parse job configuration from the line
                                    if experiment_type == 'memory' and len(parts) >= 5:
                                        failed_jobs.append({
                                            'job_id': job_id,
                                            'experiment_type': experiment_type,
                                            'schedule': parts[2],
                                            'k': int(parts[3]),
                                            'noise': float(parts[4]),
                                            'exit_code': exit_code,
                                        })
                                    elif experiment_type == 'x_junction' and len(parts) >= 5:
                                        failed_jobs.append({
                                            'job_id': job_id,
                                            'experiment_type': experiment_type,
                                            'schedule': parts[2],
                                            'k': int(parts[3]),
                                            'noise': float(parts[4]),
                                            'exit_code': exit_code,
                                        })
                                    elif experiment_type == 'patch_rotation' and len(parts) >= 4:
                                        failed_jobs.append({
                                            'job_id': job_id,
                                            'experiment_type': experiment_type,
                                            'k': int(parts[2]),
                                            'noise': float(parts[3]),
                                            'exit_code': exit_code,
                                        })
                                    elif experiment_type in ['spatial_hadamard_interface', 'spatial_hadamard_full'] and len(parts) >= 5:
                                        failed_jobs.append({
                                            'job_id': job_id,
                                            'experiment_type': experiment_type,
                                            'flag_config': parts[2],
                                            'k': int(parts[3]),
                                            'noise': float(parts[4]),
                                            'exit_code': exit_code,
                                        })
                            except ValueError:
                                pass
                
                except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                    # Skip if bhist fails
                    continue
    
    return failed_jobs


def main():
    parser = argparse.ArgumentParser(
        description='Check for failed jobs by CSV presence or LSF status'
    )
    parser.add_argument('--mode', type=str, default='csv',
                        choices=['csv', 'lsf', 'both'],
                        help='Detection mode: csv (check missing files), lsf (check job status), both')
    parser.add_argument('--experiment', type=str, default=None,
                        choices=['memory', 'x_junction', 'patch_rotation', 
                                'spatial_hadamard_interface', 'spatial_hadamard_full'],
                        help='Filter by experiment type')
    parser.add_argument('--job-id-file', type=str, default=None,
                        help='Specific job ID file to check (for LSF mode)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for failure report (default: print to stdout)')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary statistics')
    
    args = parser.parse_args()
    
    failed_jobs = []
    
    if args.mode in ['csv', 'both']:
        csv_failed = check_missing_csvs(args.experiment)
        failed_jobs.extend(csv_failed)
        if args.mode == 'both':
            print(f"CSV-based detection: {len(csv_failed)} failed jobs")
    
    if args.mode in ['lsf', 'both']:
        lsf_failed = check_lsf_failed_jobs(args.job_id_file)
        failed_jobs.extend(lsf_failed)
        if args.mode == 'both':
            print(f"LSF-based detection: {len(lsf_failed)} failed jobs")
    
    # Remove duplicates (same configuration)
    # Create a unique key based on experiment parameters (excluding job_id and exit_code)
    unique_failed = {}
    for job in failed_jobs:
        # Create a key from the configuration parameters
        if job.get('experiment_type') == 'memory' or job.get('experiment_type') == 'x_junction':
            key = (job.get('experiment_type'), job.get('schedule'), job.get('k'), job.get('noise'))
        elif job.get('experiment_type') == 'patch_rotation':
            key = (job.get('experiment_type'), job.get('k'), job.get('noise'))
        elif job.get('experiment_type') in ['spatial_hadamard_interface', 'spatial_hadamard_full']:
            key = (job.get('experiment_type'), job.get('flag_config'), job.get('k'), job.get('noise'))
        else:
            # Fallback: use all items except job_id and exit_code
            key = tuple(sorted((k, v) for k, v in job.items() if k not in ['job_id', 'exit_code', 'missing_file']))
        
        if key not in unique_failed:
            unique_failed[key] = job
    
    failed_jobs = list(unique_failed.values())
    
    if args.summary:
        print("\n" + "=" * 70)
        print("Failure Summary")
        print("=" * 70)
        print(f"Total failed jobs: {len(failed_jobs)}")
        
        by_experiment = {}
        for job in failed_jobs:
            exp_type = job.get('experiment_type', 'unknown')
            by_experiment[exp_type] = by_experiment.get(exp_type, 0) + 1
        
        print("\nBy experiment type:")
        for exp_type, count in sorted(by_experiment.items()):
            print(f"  {exp_type}: {count}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(failed_jobs, f, indent=2)
        print(f"\nFailure report saved to: {args.output}")
    else:
        print("\n" + "=" * 70)
        print("Failed Jobs")
        print("=" * 70)
        if failed_jobs:
            for job in failed_jobs:
                print(json.dumps(job, indent=2))
        else:
            print("No failed jobs found!")
    
    return 0 if not failed_jobs else 1


if __name__ == "__main__":
    exit(main())
