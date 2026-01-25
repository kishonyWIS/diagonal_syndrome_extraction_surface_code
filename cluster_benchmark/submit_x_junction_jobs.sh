#!/bin/bash
# Submit benchmark jobs for X junction experiment (both N/Z and diagonal schedules)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

mkdir -p cluster_benchmark/log cluster_benchmark/error cluster_benchmark/output/x_junction

SCHEDULES=("N/Z" "diagonal")
K_VALUES=(4)
NOISE_VALUES=(0.01 0.005623413251903491 0.0031622776601683794 0.0017782794100389228 0.001 0.0005623413251903491 0.00031622776601683794 0.00017782794100389228 0.0001)
MAX_SHOTS=100000000
MAX_ERRORS=3000
NUM_WORKERS=4

TOTAL_JOBS=$((${#SCHEDULES[@]} * ${#K_VALUES[@]} * ${#NOISE_VALUES[@]}))

echo "Submitting $TOTAL_JOBS jobs for X junction experiment..."
echo "  Schedules: ${SCHEDULES[*]}"
echo "  k values: ${K_VALUES[*]}"
echo "  Noise levels: ${#NOISE_VALUES[@]} values"
echo "  Max shots: $MAX_SHOTS"
echo "  Max errors: $MAX_ERRORS"
echo ""

# Job ID tracking file
JOB_ID_FILE="cluster_benchmark/job_ids_x_junction.txt"
touch "$JOB_ID_FILE"

JOB_NUM=0
for schedule in "${SCHEDULES[@]}"; do
    for k in "${K_VALUES[@]}"; do
        for noise in "${NOISE_VALUES[@]}"; do
            JOB_NUM=$((JOB_NUM + 1))
            
            # Create output filename
            schedule_suffix=$(echo "$schedule" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
            OUTPUT_FILE="cluster_benchmark/output/x_junction/result_x_junction_${schedule_suffix}_k${k}_p${noise}.csv"
            JOB_NAME="xj_${schedule_suffix:0:3}_k${k}"
            
            echo "[$JOB_NUM/$TOTAL_JOBS] Submitting: schedule=$schedule k=$k noise=$noise"
            
            # Submit job and capture job ID
            # Create a temporary wrapper script that exports environment variables and includes BSUB directives
            # LSF -env doesn't work reliably with input redirection, so we use a wrapper
            TEMP_SCRIPT=$(mktemp /tmp/job_wrapper_XXXXXX.sh)
            PROJECT_DIR_ABS="$(cd "$PROJECT_DIR" && pwd)"
            cat > "$TEMP_SCRIPT" << WRAPPER_EOF
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
#BSUB -o cluster_benchmark/log/benchmark-%J.out
#BSUB -e cluster_benchmark/error/benchmark-%J.err

cd "$PROJECT_DIR_ABS"
export EXPERIMENT_TYPE=x_junction
export SCHEDULE="$schedule"
export K_VAL=$k
export NOISE=$noise
export NOISE_MODE=
export FLAG_CONFIG=
export DIRECTION=
export BASIS=
export MAX_SHOTS=$MAX_SHOTS
export MAX_ERRORS=$MAX_ERRORS
export NUM_WORKERS=$NUM_WORKERS
export OUTPUT_FILE="$OUTPUT_FILE"

# Source the job script body (skip the #BSUB lines and shebang)
tail -n +18 "$PROJECT_DIR_ABS/cluster_benchmark/job_correlated_pymatching.sh" | bash
WRAPPER_EOF
            chmod +x "$TEMP_SCRIPT"
            JOB_ID=$(bsub -J "$JOB_NAME" \
                < "$TEMP_SCRIPT" 2>&1 | grep -oP '<\K[0-9]+')
            rm -f "$TEMP_SCRIPT"
            
            # Track job ID
            if [ -n "$JOB_ID" ]; then
                echo "$JOB_ID x_junction $schedule $k $noise" >> "$JOB_ID_FILE"
            fi
            
            sleep 0.1
        done
    done
done

echo ""
echo "All $TOTAL_JOBS jobs submitted!"
echo "Job IDs tracked in: $JOB_ID_FILE"
echo "Monitor with: bjobs"
