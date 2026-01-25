#!/bin/bash
#BSUB -q berg
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model!=AMD_EPYC]"
#BSUB -R "rusage[mem=500]"
#BSUB -M 500
#BSUB -N
#BSUB -u /dev/null
#BSUB -W 48:00
#BSUB -J correlated_pymatching
#BSUB -o cluster_benchmark/log/benchmark-%J.out
#BSUB -e cluster_benchmark/error/benchmark-%J.err

# Correlated PyMatching Benchmark Job
# Minimal RAM specification: 100MB reserved, 100MB hard limit
# -N flag disables email notifications

PROJECT_DIR="/home/labs/berg/gil-adk/diagonal_syndrome_extraction_surface_code"
cd "$PROJECT_DIR"

# Initialize module system (Lmod)
export MODULEPATH="/apps/easybd/easybuild/amd/modules/all:/apps/easybd/easybuild/modules/all:/apps/easybd/modules:/apps/RHEL9/modules/modulefiles:/apps/easybd/modules/hpcx:/apps/easybd/modules/nvidia_hpc_sdk"
source /apps/RHEL9/modules/lmod/lmod/init/bash

# Load required modules
module load Python/3.11.3-GCCcore-12.3.0

# Activate virtual environment
source venv/bin/activate

# Update stim to 1.16.dev version that includes missing_detectors()
pip install --upgrade --pre stim --quiet

# Print configuration
echo "=============================================="
echo "Correlated PyMatching Benchmark Job"
echo "=============================================="
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo ""
echo "Environment variables:"
echo "  EXPERIMENT_TYPE=[$EXPERIMENT_TYPE]"
echo "  SCHEDULE=[$SCHEDULE]"
echo "  K_VAL=[$K_VAL]"
echo "  NOISE=[$NOISE]"
echo "  OUTPUT_FILE=[$OUTPUT_FILE]"
echo ""

mkdir -p "$(dirname "$OUTPUT_FILE")"

# Run benchmark
python cluster_benchmark/run_correlated_pymatching_benchmark.py \
    --experiment-type "$EXPERIMENT_TYPE" \
    --schedule "$SCHEDULE" \
    --k "$K_VAL" \
    --noise "$NOISE" \
    --noise-mode "$NOISE_MODE" \
    --flag-config "$FLAG_CONFIG" \
    --direction "$DIRECTION" \
    --basis "$BASIS" \
    --max-shots "$MAX_SHOTS" \
    --max-errors "$MAX_ERRORS" \
    --num-workers "$NUM_WORKERS" \
    --output "$OUTPUT_FILE"

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "ERROR: Benchmark failed with exit code $exit_code"
    exit $exit_code
fi

echo ""
echo "Benchmark completed successfully"
