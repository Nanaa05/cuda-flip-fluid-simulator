import re
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

os.makedirs("output", exist_ok=True)
LOG_FILE = "output/benchmark_results.log"

def parse_log(path):
    cpu, cuda, cuda_ni = {}, {}, {}
    pattern = re.compile(
        r'\[(BENCHMARK_(?:CPU|CUDA|CUDA_NOINTEROP)_RESULT)\]'
        r' res=(\d+)'
        r' T1=([\d.]+)ms T2=([\d.]+)ms T3=([\d.]+)ms T4=([\d.]+)ms'
        r' T5=([\d.]+)ms T6=([\d.]+)ms T7=([\d.]+)ms T8=([\d.]+)ms'
        r' T9=([\d.]+)ms T10=([\d.]+)ms T_total=([\d.]+)ms'
    )
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            label = m.group(1)
            res = int(m.group(2))
            vals = [float(m.group(i)) for i in range(3, 14)]
            entry = dict(zip(['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T_total'], vals))
            if 'CPU' in label and 'CUDA' not in label:
                cpu[res] = entry
            elif 'NOINTEROP' in label:
                cuda_ni[res] = entry
            else:
                cuda[res] = entry
    return cpu, cuda, cuda_ni

cpu, cuda, cuda_ni = parse_log(LOG_FILE)
resolutions = sorted(cpu.keys())
stages = ['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10']
stage_labels = ['T1\nIntegrate','T2\nPush Apart','T3\nCollision','T4\nP2G','T5\nDensity','T6\nPressure','T7\nG2P','T8\nColors','T9\nRender','T10\nH2D/D2H']

fig, axes = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle('CPU vs CUDA FLIP Benchmark Analysis', fontsize=14, fontweight='bold')

ax1 = axes[0]
cpu_totals     = [cpu[r]['T_total'] for r in resolutions]
cuda_totals    = [cuda[r]['T_total'] for r in resolutions]
cuda_ni_totals = [cuda_ni[r]['T_total'] for r in resolutions]

ax1.plot(resolutions, cpu_totals,     'o-', color='#e74c3c', linewidth=2, markersize=7, label='CPU')
ax1.plot(resolutions, cuda_totals,    's-', color='#2ecc71', linewidth=2, markersize=7, label='CUDA + interop')
ax1.plot(resolutions, cuda_ni_totals, 'D-', color='#3498db', linewidth=2, markersize=7, label='CUDA no-interop')

ax1.set_yscale('log')
ax1.set_xlabel('Resolution', fontsize=11)
ax1.set_ylabel('T_total (ms, log scale)', fontsize=11)
ax1.set_title('T_total vs Resolution', fontsize=12)
ax1.set_xticks(resolutions)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1f}'))
ax1.legend()
ax1.grid(True, which='both', linestyle='--', alpha=0.5)

for r, yc, yd in zip(resolutions, cpu_totals, cuda_totals):
    speedup = yc / yd
    ax1.annotate(f'{speedup:.1f}x', xy=(r, yd), xytext=(4, 6),
                 textcoords='offset points', fontsize=8, color='#27ae60')

ax2 = axes[1]
r = 200
speedups = [cpu[r][s] / cuda[r][s] if cuda[r][s] > 0 else 0 for s in stages]
colors = ['#27ae60' if s >= 1.0 else '#e74c3c' for s in speedups]

bars = ax2.bar(stage_labels, speedups, color=colors, edgecolor='white', linewidth=0.8)
ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Breakeven (1x)')
ax2.set_xlabel('Pipeline Stage', fontsize=11)
ax2.set_ylabel('Speedup (CPU time / CUDA time)', fontsize=11)
ax2.set_title(f'Speedup per Stage (res={r})', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

for bar, val in zip(bars, speedups):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f'{val:.1f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
out = "output/benchmark_analysis.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")

print("\nSpeedup Table (CPU / CUDA) per Stage")
print(f"{'Res':>5} | " + " | ".join(f"{s:>6}" for s in stages) + " | T_total")
print("-" * 110)
for r in resolutions:
    row = []
    for s in stages:
        sp = cpu[r][s] / cuda[r][s] if cuda[r][s] > 0 else 0
        row.append(f"{sp:>6.1f}x")
    total_sp = cpu[r]['T_total'] / cuda[r]['T_total']
    print(f"{r:>5} | " + " | ".join(row) + f" | {total_sp:.1f}x")
