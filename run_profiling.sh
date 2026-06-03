#!/bin/sh

LOG_FILE="output/profiling_results.log"
DATETIME=$(date '+%Y-%m-%d %H:%M:%S')

rm -f fluid_timeline_res200.sqlite fluid_timeline_res200.nsys-rep "$LOG_FILE" kernel_bound_analysis.png

echo "====================================================================" > "$LOG_FILE"
echo "   LOG EKSPERIMEN PROFILING NSIGHT - $DATETIME" >> "$LOG_FILE"
echo "====================================================================" >> "$LOG_FILE"

export __GL_SYNC_TO_VBLANK=0

echo "[PROSES] Menjalankan nsys (Resolusi 200)..."
nsys profile --stats=true --force-overwrite=true -o fluid_timeline_res200 ./flip_cuda/flip --benchmark 200 >> "$LOG_FILE" 2>&1

echo "[PROSES] Menjalankan ncu (Resolusi 100)..."
sudo ncu --launch-skip 100 --launch-count 10 \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warps_active.avg.pct_of_peak_sustained_active \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    ./flip_cuda/flip --benchmark 100 >> "$LOG_FILE" 2>&1

echo "[PROSES] Pembuatan grafik visualisasi..."
python3 plot.py

echo "[SUKSES] Grafik berhasil disimpan di: kernel_bound_analysis.png"
