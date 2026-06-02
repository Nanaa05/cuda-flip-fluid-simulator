#!/bin/sh

LOG_FILE="validation_results.log"
DATETIME=$(date '+%Y-%m-%d %H:%M:%S')

echo "====================================================================" > "$LOG_FILE"
echo "   CPU vs GPU NUMERICAL VALIDATION - $DATETIME" >> "$LOG_FILE"
echo "====================================================================" >> "$LOG_FILE"

make validate > /dev/null

echo "===================================================================="
echo "  LOCKSTEP - Baseline"
echo "===================================================================="
./flip_cuda/validate --lockstep 2>&1 | tee -a "$LOG_FILE"

echo "===================================================================="
echo "  LOCKSTEP - Pressure Solver Converged (500 iters)"
echo "===================================================================="
./flip_cuda/validate --lockstep --iters 500

echo "===================================================================="
echo "  LOCKSTEP - No Gravity"
echo "===================================================================="
./flip_cuda/validate --lockstep --no-gravity

echo "===================================================================="
echo "  LOCKSTEP - No Obstacle"
echo "===================================================================="
./flip_cuda/validate --lockstep --no-obstacle

echo "===================================================================="
echo "  LOCKSTEP - No Push-Apart"
echo "===================================================================="
./flip_cuda/validate --lockstep --no-separate

echo ""
echo "Baseline result saved to $LOG_FILE"
