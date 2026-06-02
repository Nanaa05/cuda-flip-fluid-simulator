#!/bin/sh

LOG_FILE="benchmark_results.log"

DATETIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "====================================================================" > "$LOG_FILE"
echo "   LOG EKSPERIMEN BENCHMARK FLIP SIMULATOR - $DATETIME" >> "$LOG_FILE"
echo "====================================================================" >> "$LOG_FILE"

log_echo() {
    echo "$1" | tee -a "$LOG_FILE"
}

log_echo "===================================================================="
log_echo "                   MENGUMPULKAN DATA SPESIFIKASI                    "
log_echo "===================================================================="
lscpu | tee -a "$LOG_FILE"
log_echo ""

free -v -h | tee -a "$LOG_FILE"
log_echo ""

cat /etc/os-release | tee -a "$LOG_FILE"
log_echo ""

nvidia-smi | tee -a "$LOG_FILE"
log_echo ""

nvidia-smi --query-gpu=compute_cap --format=csv | tee -a "$LOG_FILE"
log_echo ""

nvcc --version | tee -a "$LOG_FILE"
log_echo "--------------------------------------------------------------------"

export __GL_SYNC_TO_VBLANK=0

log_echo "Membangun ulang biner simulasi..."
make cpu > /dev/null
make cuda > /dev/null

log_echo "===================================================================="
log_echo "                MEMULAI PENGUJIAN OTOMATIS (50-200)                 "
log_echo "===================================================================="

for resolution in 50 100 150 200; do
    log_echo "[PROSES] Menguji Versi CPU - Resolusi Grid $resolution..."
    ./flip_cpu/flip --no-vsync --benchmark $resolution 2>&1 | tee -a "$LOG_FILE"
    
    log_echo "[PROSES] Menguji Versi CUDA - Resolusi Grid $resolution..."
    ./flip_cuda/flip --benchmark $resolution 2>&1 | tee -a "$LOG_FILE"
    log_echo "--------------------------------------------------------------------"
done

log_echo "===================================================================="
log_echo "               SELURUH MATRIKS PENGUJIAN SELESAI                    "
log_echo "===================================================================="
log_echo "[INFO] Seluruh hasil di atas telah disimpan dengan aman di file: $LOG_FILE"
