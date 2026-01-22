#!/bin/bash
# Submit pymatching jobs for k=5,6 only

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

mkdir -p cluster_benchmark/log cluster_benchmark/error cluster_benchmark/output

DECODERS=("pymatching")
K_VALUES=(5 6)
NOISE_VALUES=(0.01 0.005623413251903491 0.0031622776601683794 0.0017782794100389228 0.001 0.0005623413251903491 0.00031622776601683794 0.00017782794100389228 0.0001)
FLAG_CONFIGS=("none" "partial" "all")
DIRECTIONS=("y")

TOTAL_JOBS=$((${#DECODERS[@]} * ${#K_VALUES[@]} * ${#NOISE_VALUES[@]} * ${#FLAG_CONFIGS[@]} * ${#DIRECTIONS[@]}))

echo "Submitting $TOTAL_JOBS jobs for pymatching k=5,6..."
echo "  k values: ${K_VALUES[*]}"
echo "  noise values: ${#NOISE_VALUES[@]} values"
echo ""

JOB_NUM=0
for decoder in "${DECODERS[@]}"; do
    for k in "${K_VALUES[@]}"; do
        for noise in "${NOISE_VALUES[@]}"; do
            for flag_config in "${FLAG_CONFIGS[@]}"; do
                for direction in "${DIRECTIONS[@]}"; do
                    JOB_NUM=$((JOB_NUM + 1))
                    OUTPUT_FILE="cluster_benchmark/output/result_${decoder}_k${k}_p${noise}_${flag_config}_${direction}.csv"
                    JOB_NAME="sh_${decoder:0:4}_k${k}_${flag_config:0:4}"
                    
                    echo "[$JOB_NUM/$TOTAL_JOBS] Submitting: k=$k noise=$noise flag=$flag_config"
                    
                    bsub -J "$JOB_NAME" \
                         -env "DECODER=$decoder,K_VAL=$k,NOISE=$noise,FLAG_CONFIG=$flag_config,DIRECTION=$direction,OUTPUT_FILE=$OUTPUT_FILE" \
                         < cluster_benchmark/job_benchmark.sh
                    
                    sleep 0.1
                done
            done
        done
    done
done

echo ""
echo "All $TOTAL_JOBS jobs submitted!"
echo "Monitor with: bjobs"
