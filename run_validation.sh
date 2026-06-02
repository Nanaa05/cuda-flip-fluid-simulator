#!/bin/sh

LOG_FILE="validation_results.log"
DATETIME=$(date '+%Y-%m-%d %H:%M:%S')

echo "====================================================================" > "$LOG_FILE"
echo "   CPU vs GPU NUMERICAL VALIDATION - $DATETIME" >> "$LOG_FILE"
echo "====================================================================" >> "$LOG_FILE"

log_echo() { echo "$1" | tee -a "$LOG_FILE"; }

make validate > /dev/null

./flip_cuda/validate --lockstep 2>&1 | tee -a "$LOG_FILE"

log_echo "Results saved to $LOG_FILE"
