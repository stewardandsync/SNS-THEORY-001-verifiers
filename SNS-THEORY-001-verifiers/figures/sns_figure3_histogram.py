"""
SNS-THEORY-001 Figure 3: Distance Histogram Z/9Z (K=7,R=7) (Theorem 24)
========================================================================
Bar chart of the full seed-space distance distribution from Table 5.
Shows the extreme rarity of max-attaining seeds (7,014 out of 4,782,969).

Output: SNS_FIG3_Z9Z_histogram.png / .pdf

Run on: USLMAX002
Requires: matplotlib, numpy
Author: Ahmed M. Mansour, Steward and Sync LLC
Document: SNS-THEORY-001, Table 5, Theorem 24
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Data from SNS-THEORY-001 Table 5
distances = [1,       2,      3,    4,      5,       6,         7,         8    ]
counts    = [1,    2234,    882, 45192, 112560, 1031268,   3583818,      7014 ]
total     = 4_782_969

colors = ['#d73027'] * 7 + ['#d4a017']  # last bar gold for max-attaining

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6),
                                gridspec_kw={'width_ratios': [2, 1]})

# --- Left: log scale bar chart ---
bars = ax1.bar(distances, counts, color=colors, edgecolor='black', linewidth=0.7)
ax1.set_yscale('log')
ax1.set_xticks(distances)
ax1.set_xticklabels([f'd={d}' for d in distances], fontsize=11)
ax1.set_xlabel(r'Minimum $\Delta_3$ distance', fontsize=12, labelpad=8)
ax1.set_ylabel('Seed count (log scale)', fontsize=12, labelpad=8)
ax1.set_title(
    r'Distance histogram over $(\mathbb{Z}/9\mathbb{Z})^7$, $R=7$'
    '\n(Theorem 24, SNS-THEORY-001)',
    fontsize=12, pad=10
)

# Annotate each bar
for d, c in zip(distances, counts):
    ax1.text(d, c * 1.8, f'{c:,}', ha='center', va='bottom',
             fontsize=8, rotation=45)

ax1.axhline(7014, color='#d4a017', linestyle='--', linewidth=1.2, alpha=0.7)
ax1.text(1.2, 7014 * 1.5, '7,014 max-attaining', fontsize=9,
         color='#d4a017', va='bottom')

ax1.grid(axis='y', linestyle=':', alpha=0.4)

# --- Right: pie chart showing rarity of max-attaining seeds ---
non_max = total - 7014
pie_sizes  = [7014, non_max]
pie_colors = ['#d4a017', '#c6dbef']
pie_labels = [f'Max-attaining\n7,014 seeds\n(0.147%)', f'Other seeds\n4,775,955\n(99.853%)']
ax2.pie(pie_sizes, colors=pie_colors, labels=pie_labels,
        startangle=90, wedgeprops={'edgecolor': 'black', 'linewidth': 0.8},
        textprops={'fontsize': 10})
ax2.set_title('Max-attaining seed\nrarity', fontsize=12, pad=10)

# Add 501 orbits annotation
fig.text(0.5, 0.01,
         '7,014 max-attaining seeds  |  501 free orbits of size 14  |  '
         'Witnesses: (6,3,1,1,4,0,0) and (0,0,1,4,1,3,6)',
         ha='center', fontsize=9, color='#444444',
         bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

plt.tight_layout(rect=[0, 0.04, 1, 1])
for ext in ['png', 'pdf']:
    path = Path(f'SNS_FIG3_Z9Z_histogram.{ext}')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')

plt.close()
print('Figure 3 complete.')
