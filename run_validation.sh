#!/bin/sh

LOG_FILE="validation_results.log"

DATETIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "====================================================================" > "$LOG_FILE"
echo "   LOG VALIDASI NUMERIK CPU vs GPU FLIP SIMULATOR - $DATETIME" >> "$LOG_FILE"
echo "====================================================================" >> "$LOG_FILE"

log_echo() {
    echo "$1" | tee -a "$LOG_FILE"
}

log_echo "===================================================================="
log_echo "                   MENGUMPULKAN DATA SPESIFIKASI                    "
log_echo "===================================================================="
lscpu | tee -a "$LOG_FILE"
log_echo ""

nvidia-smi | tee -a "$LOG_FILE"
log_echo ""

nvcc --version | tee -a "$LOG_FILE"
log_echo "--------------------------------------------------------------------"

log_echo "Membangun ulang validator..."
make validate > /dev/null

log_echo "===================================================================="
log_echo "             TEST 1: Lockstep - Baseline (50 iters)                 "
log_echo "===================================================================="
./flip_cuda/validate --lockstep 2>&1 | tee -a "$LOG_FILE"

log_echo "===================================================================="
log_echo "         TEST 2: Lockstep - Pressure Solver Dikonvergensikan        "
log_echo "===================================================================="
./flip_cuda/validate --lockstep --iters 500 2>&1 | tee -a "$LOG_FILE"


log_echo "===================================================================="
log_echo "                  SELURUH PENGUJIAN SELESAI                         "
log_echo "===================================================================="
log_echo "[INFO] Seluruh hasil di atas telah disimpan di file: $LOG_FILE"
