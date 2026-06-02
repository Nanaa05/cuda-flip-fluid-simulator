#!/bin/sh

mkdir -p output
LOG_FILE="output/validation_results.log"
DATETIME=$(date '+%Y-%m-%d %H:%M:%S')

echo "====================================================================" > "$LOG_FILE"
echo "   CPU vs GPU NUMERICAL VALIDATION - $DATETIME" >> "$LOG_FILE"
echo "====================================================================" >> "$LOG_FILE"

make validate > /dev/null

echo "====================================================================" >> "$LOG_FILE"
echo "  LOCKSTEP - Baseline" >> "$LOG_FILE"
echo "====================================================================" >> "$LOG_FILE"
./flip_cuda/validate --lockstep >> "$LOG_FILE" 2>&1

echo "====================================================================" >> "$LOG_FILE"
echo "  LOCKSTEP - Pressure Solver Converged (500 iters)" >> "$LOG_FILE"
echo "====================================================================" >> "$LOG_FILE"
./flip_cuda/validate --lockstep --iters 500 >> "$LOG_FILE" 2>&1

echo "====================================================================" >> "$LOG_FILE"
echo "  LOCKSTEP - No Gravity" >> "$LOG_FILE"
echo "====================================================================" >> "$LOG_FILE"
./flip_cuda/validate --lockstep --no-gravity >> "$LOG_FILE" 2>&1

echo "====================================================================" >> "$LOG_FILE"
echo "  LOCKSTEP - No Obstacle" >> "$LOG_FILE"
echo "====================================================================" >> "$LOG_FILE"
./flip_cuda/validate --lockstep --no-obstacle >> "$LOG_FILE" 2>&1

echo "====================================================================" >> "$LOG_FILE"
echo "  LOCKSTEP - No Push-Apart" >> "$LOG_FILE"
echo "====================================================================" >> "$LOG_FILE"
./flip_cuda/validate --lockstep --no-separate >> "$LOG_FILE" 2>&1
