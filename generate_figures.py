#!/usr/bin/env python3
"""Generate 4 IEEE-style figures for paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

OUT = 'figures'

# ============================================================
# Figure A: fig_asr_by_trigger.png
# ============================================================
fig, ax = plt.subplots(figsize=(3.5, 2.5))
triggers = ['Word', 'Phrase', 'Long']
raw_asr = [0.9250, 0.9000, 0.9250]
def_asr = [0.1417, 0.0750, 0.1750]
reductions = [84.7, 91.7, 81.1]
x = np.arange(len(triggers))
width = 0.32
ax.bar(x - width/2, raw_asr, width, color='#d62728', alpha=0.85, label='Raw (Backdoor)')
bars_def = ax.bar(x + width/2, def_asr, width, color='#2ca02c', alpha=0.85, label='Defended')
for i, (bar, red) in enumerate(zip(bars_def, reductions)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f'-{red}%', ha='center', va='bottom', fontsize=6.5, fontweight='bold', color='#1a7a1a')
ax.set_ylabel('Triggered ASR  (lower is better)')
ax.set_xticks(x)
ax.set_xticklabels(triggers)
ax.set_ylim(0, 1.05)
ax.legend(framealpha=0.9, edgecolor='gray', fontsize=7)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_title('Triggered ASR Before and After Defense')
fig.tight_layout()
fig.savefig(f'{OUT}/fig_asr_by_trigger.png')
plt.close(fig)
print("Figure A saved")

# ============================================================
# Figure B: fig_pruning_budget_ablation.png
# ============================================================
fig, ax1 = plt.subplots(figsize=(3.5, 2.5))
budgets = [32, 64, 128, 256, 512]
asr_vals = [0.865, 0.855, 0.835, 0.810, 0.820]
ppl_vals = [7.496, 7.496, 7.505, 7.399, 7.345]
color_asr = '#d62728'
color_ppl = '#1f77b4'
ax1.plot(budgets, asr_vals, 'o-', color=color_asr, linewidth=1.5, markersize=5, label='ASR')
ax1.set_xlabel('Pruning Budget (units)')
ax1.set_ylabel('Triggered ASR  (lower is better)', color=color_asr)
ax1.tick_params(axis='y', labelcolor=color_asr)
ax1.set_ylim(0.78, 0.90)
best_idx = 3
ax1.annotate(f'ASR={asr_vals[best_idx]:.3f}',
             xy=(budgets[best_idx], asr_vals[best_idx]),
             xytext=(budgets[best_idx]+80, asr_vals[best_idx]+0.01),
             arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
             fontsize=7, color=color_asr)
ax2 = ax1.twinx()
ax2.plot(budgets, ppl_vals, 's--', color=color_ppl, linewidth=1.5, markersize=5, label='PPL')
ax2.set_ylabel('Perplexity  (lower is better)', color=color_ppl)
ax2.tick_params(axis='y', labelcolor=color_ppl)
ax2.set_ylim(7.30, 7.55)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, framealpha=0.9, edgecolor='gray', fontsize=7)
ax1.grid(alpha=0.3, linestyle='--')
ax1.set_title('Effect of Pruning Budget')
fig.tight_layout()
fig.savefig(f'{OUT}/fig_pruning_budget_ablation.png')
plt.close(fig)
print("Figure B saved")

# ============================================================
# Figure C: fig_lambda_align_phase.png
# ============================================================
fig, ax = plt.subplots(figsize=(3.5, 2.5))
la_vals = [1.0, 1.5, 2.0]
asr_la = [0.1750, 0.2583, 0.1417]
hr_la  = [0.8250, 0.7583, 0.8667]
bfr_la = [0.2600, 0.3400, 0.3900]
ax.plot(la_vals, asr_la, 'o-', color='#d62728', linewidth=1.5, markersize=6, label='ASR')
ax.plot(la_vals, hr_la, 'o-', color='#2ca02c', linewidth=1.5, markersize=6, label='HarmRef')
ax.plot(la_vals, bfr_la, 'o-', color='#ff7f0e', linewidth=1.5, markersize=6, label='BFR')
ax.annotate('Balanced\npoint', xy=(2.0, 0.1417),
            xytext=(1.55, 0.10), fontsize=7, fontweight='bold', color='#2ca02c',
            arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.2))
ax.annotate('Unstable\nregion', xy=(1.5, 0.2583),
            xytext=(1.15, 0.32), fontsize=6.5, color='#d62728',
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=0.8))
ax.set_xlabel(r'$\lambda_{align}$')
ax.set_ylabel('Ratio  (ASR' + chr(8595) + ' / HarmRef' + chr(8593) + ' / BFR' + chr(8595) + ')')
ax.set_xticks(la_vals)
ax.legend(framealpha=0.9, edgecolor='gray', fontsize=6.5)
ax.grid(alpha=0.3, linestyle='--')
ax.set_ylim(0, 0.45)
ax.set_title(r'Recovery Alignment Strength ($\lambda_{align}$)')
fig.tight_layout()
fig.savefig(f'{OUT}/fig_lambda_align_phase.png')
plt.close(fig)
print("Figure C saved")

# ============================================================
# Figure D: fig_schedule_tradeoff.png
# ============================================================
fig, ax = plt.subplots(figsize=(3.5, 2.8))
schedules = {
    'Simul. $\lambda$=1.0':       {'BFR': 0.2600, 'ASR': 0.1750, 'HR': 0.8250},
    'Simul. $\lambda$=2.0':       {'BFR': 0.3900, 'ASR': 0.1417, 'HR': 0.8667},
    'Alternating':         {'BFR': 0.3200, 'ASR': 0.3750, 'HR': 0.6500},
    'Alternating-soft':    {'BFR': 0.1800, 'ASR': 0.3750, 'HR': 0.6667},
    'Alternating-2c1s':    {'BFR': 0.2700, 'ASR': 0.3500, 'HR': 0.6583},
    'Alt.-then-simul.':    {'BFR': 0.2200, 'ASR': 0.4000, 'HR': 0.6500},
}
colors = ['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e', '#9467bd', '#8c564b']
markers = ['*', 'o', 's', '^', 'D', 'v']

for i, (name, d) in enumerate(schedules.items()):
    size = 60 + d['HR'] * 80
    lbl = None if name == 'Simul. $\\lambda$=2.0' else name
    ax.scatter(d['BFR'], d['ASR'], s=size, c=colors[i], marker=markers[i],
               alpha=0.85, edgecolors='black', linewidths=0.5, label=lbl)
    if '2.0' in name:
        ax.annotate('Selected\nbalanced', xy=(d['BFR'], d['ASR']),
                    xytext=(d['BFR']+0.06, d['ASR']-0.04),
                    fontsize=7, fontweight='bold', color='#1f77b4',
                    arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.2))
        # Still add a legend entry for the selected point
        ax.scatter([], [], s=100, c='#1f77b4', marker='o', label='Simul. $\\lambda$=2.0 (selected)',
                   edgecolors='black', linewidths=0.5)

ax.legend(framealpha=0.9, edgecolor='gray', fontsize=5.5, ncol=2, loc='upper left')
ax.set_xlabel('Benign False Refusal (BFR)  (lower is better)')
ax.set_ylabel('Triggered ASR  (lower is better)')
ax.set_xlim(0.12, 0.48)
ax.set_ylim(0.08, 0.45)
ax.grid(alpha=0.3, linestyle='--')
ax.set_title('Objective Schedule Trade-off')
fig.tight_layout()
fig.savefig(f'{OUT}/fig_schedule_tradeoff.png')
plt.close(fig)
print("Figure D saved")

print("\nAll 4 figures generated successfully.")
