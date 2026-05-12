"""
SNS-THEORY-001 Figure 2: Z/9Z Gap Staircase K=8 (Theorems 2-13)
================================================================
Step chart of M3(8,R) vs naive prediction R+1 across gaps R in {1..9}.
Highlights the two plateau regions where naive prediction fails.

Output: SNS_FIG2_Z9Z_staircase.png / .pdf

Run on: USLMAX002
Requires: matplotlib, numpy
Author: Ahmed M. Mansour, Steward and Sync LLC
Document: SNS-THEORY-001, Table 2
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

gaps   = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
actual = np.array([2, 3, 4, 5, 5, 6, 7, 8, 8])
naive  = gaps + 1

fig, ax = plt.subplots(figsize=(10, 6))

# Naive prediction line
ax.step(gaps, naive, where='post', color='#999999',
        linewidth=1.5, linestyle='--', label='Naive prediction R+1')

# Actual staircase
ax.step(gaps, actual, where='post', color='#2166ac',
        linewidth=2.5, label=r'$M_3(8,R)$ — exhaustive enumeration')

# Shade plateau regions
ax.axvspan(4, 5, alpha=0.12, color='#d73027', label='Plateau 1: R∈{4,5}, ceiling=5')
ax.axvspan(8, 9, alpha=0.12, color='#fc8d59', label='Plateau 2: R∈{8,9}, ceiling=8')

# Mark actual data points
ax.scatter(gaps, actual, color='#2166ac', s=60, zorder=5)
ax.scatter(gaps, naive,  color='#999999', s=40, zorder=4, marker='x')

# Annotate deficit where naive fails
for g, a, n in zip(gaps, actual, naive):
    if a != n:
        ax.annotate(f'deficit {n-a}',
                    xy=(g, a), xytext=(g + 0.15, a - 0.45),
                    fontsize=8, color='#d73027',
                    arrowprops=dict(arrowstyle='->', color='#d73027', lw=0.8))

ax.set_xticks(gaps)
ax.set_xticklabels([f'R={g}' for g in gaps], fontsize=11)
ax.set_yticks(range(2, 11))
ax.set_ylim(1, 11)
ax.set_xlabel('Gap parameter R = n − k', fontsize=12, labelpad=8)
ax.set_ylabel(r'Maximum min $\Delta_3$ distance', fontsize=12, labelpad=8)
ax.set_title(
    r'Gap staircase $M_3(8,R)$ over $\mathbb{Z}/9\mathbb{Z}$, $K=8$'
    '\n(Theorems 2–13, SNS-THEORY-001)',
    fontsize=13, pad=12
)
ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
ax.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
for ext in ['png', 'pdf']:
    path = Path(f'SNS_FIG2_Z9Z_staircase.{ext}')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')

plt.close()
print('Figure 2 complete.')
