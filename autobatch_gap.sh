#!/bin/bash
#SBATCH --job-name=autobench
#SBATCH --output=benchmarks/logs/autobench_%j.log
#SBATCH --time=24:00:00  # Adjust as needed

# List of bgo values
BGOS=(bc bfs cc pr sssp tc)
# List of thread counts
THREADS=(1 2 4 8 16 32)

for BGO in "${BGOS[@]}"; do
    OUTPUT_DIR="benchmarks/gap_results/${BGO}/${BGO}_gap/gnm"
    mkdir -p "$OUTPUT_DIR"
    
    for N in "${THREADS[@]}"; do
        sbatch --dependency=singleton --job-name="${BGO}_bench" --wrap="srun python3 autobench/run_bench.py bgo/${BGO}/${BGO}_gap/ \
            --runs 10 \
            --data G=/var/scratch/dbart/data/random/generated/gnm/* \
            --output ${OUTPUT_DIR}/output_${N}_threads.csv \
            --threads ${N}"
    done
    wait  # Ensure all thread jobs for the current BGO finish before moving to the next BGO
done