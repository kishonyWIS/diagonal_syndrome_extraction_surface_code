#!/bin/bash
# Submit benchmark jobs for rectangular memory experiment (diagonal schedule) using the Tesseract decoder.
#
# This uses cluster_benchmark/run_single_benchmark.py (--experiment rectangular_memory --schedule diagonal --decoder tesseract).
#
# Output goes to:
#   cluster_benchmark/output/rectangular_memory_tesseract/
# with filenames:
#   result_tesseract_diagonal_k<k>_p<p>.csv
#
# Job IDs are tracked in:
#   cluster_benchmark/job_ids_rectangular_memory_tesseract.txt

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

mkdir -p \
  cluster_benchmark/log \
  cluster_benchmark/error \
  cluster_benchmark/output/rectangular_memory_tesseract

# Sweep configuration
DECODER="tesseract"
SCHEDULE="diagonal"
K_VALUES=(1 2 3 4)
NOISE_VALUES=(0.01 0.005623413251903491 0.0031622776601683794 0.0017782794100389228 0.001 0.0005623413251903491 0.00031622776601683794 0.00017782794100389228 0.0001)

# Sinter controls (match spatial hadamard full tesseract: inverse-linear max_shots ramp)
MAX_SHOTS_AT_P_HIGH=1000000
P_HIGH=0.01
P_LOW=0.0001
LOW_P_MULT=20
MAX_ERRORS=3000
NUM_WORKERS=4

TOTAL_JOBS=$((${#K_VALUES[@]} * ${#NOISE_VALUES[@]}))

echo "Submitting $TOTAL_JOBS jobs for rectangular memory (diagonal schedule) with Tesseract decoder..."
echo "  Decoder: $DECODER"
echo "  Schedule: $SCHEDULE"
echo "  k values: ${K_VALUES[*]}"
echo "  Noise levels: ${#NOISE_VALUES[@]} values"
echo "  Max shots: varies with p"
echo "    at p=$P_HIGH  -> $MAX_SHOTS_AT_P_HIGH"
echo "    at p=$P_LOW   -> $((MAX_SHOTS_AT_P_HIGH * LOW_P_MULT))"
echo "  Max errors: $MAX_ERRORS"
echo ""

# Job ID tracking file
JOB_ID_FILE="cluster_benchmark/job_ids_rectangular_memory_tesseract.txt"
touch "$JOB_ID_FILE"

JOB_NUM=0
for k in "${K_VALUES[@]}"; do
  for noise in "${NOISE_VALUES[@]}"; do
    JOB_NUM=$((JOB_NUM + 1))

    # Compute per-noise max shots using a linear function of (1/p),
    # such that multiplier is 1 at p=P_HIGH and LOW_P_MULT at p=P_LOW.
    MAX_SHOTS_THIS=$(awk -v p="$noise" \
                         -v max_hi="$MAX_SHOTS_AT_P_HIGH" \
                         -v p_hi="$P_HIGH" \
                         -v p_lo="$P_LOW" \
                         -v mult_lo="$LOW_P_MULT" \
                         'BEGIN{
                            xh=1/p_hi; xl=1/p_lo;
                            A=(mult_lo-1)/(xl-xh);
                            B=1 - A*xh;
                            m=A*(1/p)+B;
                            if(m<1) m=1;
                            if(m>mult_lo) m=mult_lo;
                            printf "%.0f", max_hi*m;
                          }')

    OUTPUT_FILE="cluster_benchmark/output/rectangular_memory_tesseract/result_tesseract_diagonal_k${k}_p${noise}.csv"
    JOB_NAME="rect_tess_k${k}"

    echo "[$JOB_NUM/$TOTAL_JOBS] Submitting: k=$k noise=$noise max_shots=$MAX_SHOTS_THIS"

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

set -euo pipefail
cd "$PROJECT_DIR_ABS"

module load Python/3.11.3-GCCcore-12.3.0
source venv/bin/activate

rm -f "$OUTPUT_FILE"

python3 cluster_benchmark/run_single_benchmark.py \\
  --experiment rectangular_memory \\
  --schedule "$SCHEDULE" \\
  --decoder "$DECODER" \\
  --k $k \\
  --noise $noise \\
  --max-shots $MAX_SHOTS_THIS \\
  --max-errors $MAX_ERRORS \\
  --num-workers $NUM_WORKERS \\
  --output "$OUTPUT_FILE"
WRAPPER_EOF

    chmod +x "$TEMP_SCRIPT"

    JOB_ID=$(bsub -J "$JOB_NAME" < "$TEMP_SCRIPT" 2>&1 | grep -oP '<\K[0-9]+')
    rm -f "$TEMP_SCRIPT"

    if [ -n "${JOB_ID:-}" ]; then
      echo "$JOB_ID rectangular_memory_tesseract diagonal $k $noise" >> "$JOB_ID_FILE"
    fi

    sleep 0.1
  done
done

echo ""
echo "All $TOTAL_JOBS jobs submitted!"
echo "Job IDs tracked in: $JOB_ID_FILE"
echo "Monitor with: bjobs"
