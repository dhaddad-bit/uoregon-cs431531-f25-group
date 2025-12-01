#!/bin/bash
module load cuda

# 1. Convert the binary report (.ncu-rep) to a readable CSV
# We extract specific metrics: Duration, Total Bytes, and Memory Efficiency (SOL DRAM)
echo "Extracting data from profile_report.ncu-rep..."

ncu --import profile_report.ncu-rep \
    --csv \
    --page raw \
    --metrics gpu__time_duration.sum,dram__bytes.sum,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed \
    > raw_metrics.csv

# 2. Print a clean summary table
echo ""
echo "========================================================="
echo "  SPMV KERNEL PERFORMANCE SUMMARY"
echo "========================================================="

# We use awk to find the rows matching our metrics and print the value (Column 9)
# Note: $9 refers to the 9th column in the CSV (Metric Value)

grep "gpu__time_duration.sum" raw_metrics.csv | head -n 1 | \
awk -F',' '{printf "  Kernel Duration:       %s ns\n", $9}'

grep "dram__bytes.sum" raw_metrics.csv | head -n 1 | \
awk -F',' '{printf "  Total Data Moved:      %s Bytes\n", $9}'

grep "gpu__compute_memory_throughput" raw_metrics.csv | head -n 1 | \
awk -F',' '{printf "  Memory Saturation:     %s %% (of Hardware Peak)\n", $9}'

echo "========================================================="
echo "Full data saved to: raw_metrics.csv"