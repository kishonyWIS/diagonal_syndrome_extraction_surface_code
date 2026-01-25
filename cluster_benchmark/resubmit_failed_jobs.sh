#!/bin/bash
# Resubmit failed jobs based on failure report from check_failed_jobs.py

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Default values
EXPERIMENT="all"
FAILURE_REPORT=""
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        --failure-report)
            FAILURE_REPORT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--experiment EXPERIMENT] [--failure-report FILE] [--dry-run]"
            echo "  --experiment: memory, x_junction, patch_rotation, spatial_hadamard_interface, spatial_hadamard_full, or all (default: all)"
            echo "  --failure-report: Path to JSON failure report (default: auto-detect latest)"
            echo "  --dry-run: Show what would be resubmitted without actually submitting"
            exit 1
            ;;
    esac
done

# Find latest failure report if not specified
if [ -z "$FAILURE_REPORT" ]; then
    # Look for failure reports in cluster_benchmark directory
    LATEST_REPORT=$(ls -t cluster_benchmark/failure_report_*.json 2>/dev/null | head -1)
    if [ -n "$LATEST_REPORT" ]; then
        FAILURE_REPORT="$LATEST_REPORT"
        echo "Using latest failure report: $FAILURE_REPORT"
    else
        echo "Error: No failure report found. Run check_failed_jobs.py first with --output option."
        exit 1
    fi
fi

if [ ! -f "$FAILURE_REPORT" ]; then
    echo "Error: Failure report not found: $FAILURE_REPORT"
    exit 1
fi

# Check if jq is available for JSON parsing
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required for parsing JSON"
    exit 1
fi

echo "=============================================="
echo "Resubmitting Failed Jobs"
echo "=============================================="
echo "Failure report: $FAILURE_REPORT"
echo "Experiment filter: $EXPERIMENT"
echo "Dry run: $DRY_RUN"
echo ""

# Parse JSON and resubmit jobs
python3 << 'PYTHON_SCRIPT'
import json
import sys
import subprocess
import os

failure_report = sys.argv[1]
experiment_filter = sys.argv[2]
dry_run = sys.argv[3] == 'true'

with open(failure_report, 'r') as f:
    failed_jobs = json.load(f)

# Filter by experiment type
if experiment_filter != 'all':
    failed_jobs = [j for j in failed_jobs if j.get('experiment_type') == experiment_filter]

if not failed_jobs:
    print("No failed jobs to resubmit (after filtering).")
    sys.exit(0)

print(f"Found {len(failed_jobs)} failed jobs to resubmit")
print("")

# Group by experiment type for batch submission
jobs_by_experiment = {}
for job in failed_jobs:
    exp_type = job.get('experiment_type', 'unknown')
    if exp_type not in jobs_by_experiment:
        jobs_by_experiment[exp_type] = []
    jobs_by_experiment[exp_type].append(job)

# Resubmit each job
total_resubmitted = 0
for exp_type, jobs in jobs_by_experiment.items():
    print(f"\n{exp_type}: {len(jobs)} jobs")
    
    for job in jobs:
        # Determine parameters based on experiment type
        if exp_type == 'memory':
            schedule = job.get('schedule', '')
            k = job.get('k')
            noise = job.get('noise')
            max_shots = 1000000000
            output_file = f"cluster_benchmark/output/memory/result_memory_{schedule.replace('/', '_').lower()}_k{k}_p{noise}.csv"
            env_vars = f"EXPERIMENT_TYPE=memory SCHEDULE={schedule} K_VAL={k} NOISE={noise} NOISE_MODE= FLAG_CONFIG= DIRECTION= BASIS= MAX_SHOTS={max_shots} MAX_ERRORS=3000 NUM_WORKERS=4 OUTPUT_FILE={output_file}"
            job_name = f"mem_{schedule.replace('/', '_').lower()[:3]}_k{k}"
        
        elif exp_type == 'x_junction':
            schedule = job.get('schedule', '')
            k = job.get('k')
            noise = job.get('noise')
            max_shots = 100000000
            output_file = f"cluster_benchmark/output/x_junction/result_x_junction_{schedule.replace('/', '_').lower()}_k{k}_p{noise}.csv"
            env_vars = f"EXPERIMENT_TYPE=x_junction SCHEDULE={schedule} K_VAL={k} NOISE={noise} NOISE_MODE= FLAG_CONFIG= DIRECTION= BASIS= MAX_SHOTS={max_shots} MAX_ERRORS=3000 NUM_WORKERS=4 OUTPUT_FILE={output_file}"
            job_name = f"xj_{schedule.replace('/', '_').lower()[:3]}_k{k}"
        
        elif exp_type == 'patch_rotation':
            k = job.get('k')
            noise = job.get('noise')
            basis = 'z'
            max_shots = 100000000
            output_file = f"cluster_benchmark/output/patch_rotation/result_patch_rotation_k{k}_p{noise}.csv"
            env_vars = f"EXPERIMENT_TYPE=patch_rotation SCHEDULE= K_VAL={k} NOISE={noise} NOISE_MODE= FLAG_CONFIG= DIRECTION= BASIS={basis} MAX_SHOTS={max_shots} MAX_ERRORS=3000 NUM_WORKERS=4 OUTPUT_FILE={output_file}"
            job_name = f"pr_k{k}"
        
        elif exp_type == 'spatial_hadamard_interface':
            k = job.get('k')
            noise = job.get('noise')
            flag_config = job.get('flag_config', '')
            max_shots = 100000000
            output_file = f"cluster_benchmark/output/spatial_hadamard_interface/result_spatial_hadamard_interface_{flag_config}_k{k}_p{noise}.csv"
            env_vars = f"EXPERIMENT_TYPE=spatial_hadamard SCHEDULE= K_VAL={k} NOISE={noise} NOISE_MODE=interface_only FLAG_CONFIG={flag_config} DIRECTION=y BASIS= MAX_SHOTS={max_shots} MAX_ERRORS=3000 NUM_WORKERS=4 OUTPUT_FILE={output_file}"
            job_name = f"sh_int_{flag_config[:4]}_k{k}"
        
        elif exp_type == 'spatial_hadamard_full':
            k = job.get('k')
            noise = job.get('noise')
            flag_config = job.get('flag_config', '')
            max_shots = 100000000
            output_file = f"cluster_benchmark/output/spatial_hadamard_full/result_spatial_hadamard_full_{flag_config}_k{k}_p{noise}.csv"
            env_vars = f"EXPERIMENT_TYPE=spatial_hadamard SCHEDULE= K_VAL={k} NOISE={noise} NOISE_MODE=full FLAG_CONFIG={flag_config} DIRECTION=y BASIS= MAX_SHOTS={max_shots} MAX_ERRORS=3000 NUM_WORKERS=4 OUTPUT_FILE={output_file}"
            job_name = f"sh_full_{flag_config[:4]}_k{k}"
        
        else:
            print(f"  Skipping unknown experiment type: {exp_type}")
            continue
        
        if dry_run:
            print(f"  [DRY RUN] Would resubmit: {exp_type} k={k} noise={noise}")
        else:
            # Submit job
            cmd = f"bsub -J {job_name} -env '{env_vars}' < cluster_benchmark/job_correlated_pymatching.sh"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd())
            
            # Extract job ID
            job_id = None
            for line in result.stdout.split('\n'):
                if '<' in line and '>' in line:
                    # Extract number between < and >
                    import re
                    match = re.search(r'<(\d+)>', line)
                    if match:
                        job_id = match.group(1)
                        break
            
            if job_id:
                print(f"  Resubmitted: {exp_type} k={k} noise={noise} -> Job ID: {job_id}")
                total_resubmitted += 1
                
                # Update job ID tracking file
                job_id_file = f"cluster_benchmark/job_ids_{exp_type}.txt"
                with open(job_id_file, 'a') as f:
                    if exp_type in ['memory', 'x_junction']:
                        f.write(f"{job_id} {exp_type} {schedule} {k} {noise}\n")
                    elif exp_type == 'patch_rotation':
                        f.write(f"{job_id} {exp_type} {k} {noise}\n")
                    elif exp_type in ['spatial_hadamard_interface', 'spatial_hadamard_full']:
                        f.write(f"{job_id} {exp_type} {flag_config} {k} {noise}\n")
            else:
                print(f"  ERROR resubmitting: {exp_type} k={k} noise={noise}")
                print(f"    Command output: {result.stdout}")
                print(f"    Error output: {result.stderr}")

if not dry_run:
    print(f"\nTotal resubmitted: {total_resubmitted}")
else:
    print(f"\n[DRY RUN] Would resubmit {len(failed_jobs)} jobs")
PYTHON_SCRIPT
"$FAILURE_REPORT" "$EXPERIMENT" "$DRY_RUN"

if [ $? -eq 0 ]; then
    echo ""
    echo "Resubmission complete!"
else
    echo ""
    echo "Error during resubmission"
    exit 1
fi
