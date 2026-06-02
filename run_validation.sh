#!/bin/sh

LOG_FILE="validation_results.log"
DATETIME=$(date '+%Y-%m-%d %H:%M:%S')

echo "====================================================================" > "$LOG_FILE"
echo "   CPU vs GPU NUMERICAL VALIDATION - $DATETIME" >> "$LOG_FILE"
echo "====================================================================" >> "$LOG_FILE"

log_echo() { echo "$1" | tee -a "$LOG_FILE"; }

make validate > /dev/null

log_echo "===================================================================="
log_echo "  TEST 1: Baseline (50 iters)"
log_echo "===================================================================="
./flip_cuda/validate --lockstep 2>&1 | tee -a "$LOG_FILE"

log_echo "===================================================================="
log_echo "  TEST 2: Pressure Solver Converged (500 iters)"
log_echo "===================================================================="
./flip_cuda/validate --lockstep --iters 500 2>&1 | tee -a "$LOG_FILE"

log_echo "Results saved to $LOG_FILE"
