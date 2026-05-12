"""
SNS-THEORY-001 Figure 1: Z/4Z Phase Diagram Heat Map
=====================================================
Generates a color-coded 7x7 grid of M2(K,R) values over Z/4Z.
Highlights the headline cell (K=7,R=7) and the K-axis non-monotonicity.

Output: SNS_FIG1_Z4Z_phase_diagram.png / .pdf

Run on: USLMAX002
Requires: matplotlib, numpy
Install:  pip install matplotlib numpy --break-system-packages

Author: Ahmed M. Mansour, Steward and Sync LLC
Document: SNS-THEORY-001, Theorem 19
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# Phase diagram data from SNS-THEORY-001 Table 3 (Theorem 19)
# Rows = K in {2,3,4,5,6,7,8}, Cols = R in {1,2,3,4,5,6,7}
phase_data = np.array([
    [2, 4, 4, 5, 5, 6, 6],  # K=2
    [2, 3, 4, 4, 5, 6, 6],  # K=3
    [2, 3, 4, 4, 4, 5, 5],  # K=4
    [2, 2, 3, 4, 6, 6, 6],  # K=5
    [2, 2, 4, 4, 5, 6, 6],  # K=6
    [2, 2, 3, 4, 5, 6, 8],  # K=7  <- headline row
    [2, 2, 3, 4, 4, 5, 6],  # K=8
])

K_labels = [2, 3, 4, 5, 6, 7, 8]
R_labels = [1, 2, 3, 4, 5, 6, 7]

# Custom colormap: light to dark blue, with headline cell in gold
cmap = LinearSegmentedColormap.from_list(
    'sns', ['#e8f4f8', '#2166ac'], N=256
)

fig, ax = plt.subplots(figsize=(9, 7))

im = ax.imshow(phase_data, cmap=cmap, vmin=2, vmax=8, aspect='auto')

# Annotate each cell
for i in range(7):
    for j in range(7):
        val = phase_data[i, j]
        # Headline cell (K=7, R=7) = row 5, col 6
        if i == 5 and j == 6:
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                fill=True, facecolor='#d4a017', edgecolor='black', linewidth=2, zorder=2))
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=14, fontweight='bold', color='black', zorder=3)
        # Contrast cell (K=8, R=7) = row 6, col 6
        elif i == 6 and j == 6:
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                fill=True, facecolor='#f4a582', edgecolor='black', linewidth=1.5, zorder=2))
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=12, fontweight='bold', color='black', zorder=3)
        else:
            color = 'white' if val <= 4 else 'black'
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=12, color=color, zorder=3)

# Naive prediction diagonal annotation
for i in range(7):
    for j in range(7):
        k, r = K_labels[i], R_labels[j]
        naive = r + 1
        if phase_data[i, j] != naive:
            ax.text(j + 0.35, i - 0.35, '≠', ha='center', va='center',
                    fontsize=7, color='#cc0000', alpha=0.8, zorder=4)

ax.set_xticks(range(7))
ax.set_xticklabels([f'R={r}' for r in R_labels], fontsize=11)
ax.set_yticks(range(7))
ax.set_yticklabels([f'K={k}' for k in K_labels], fontsize=11)
ax.set_xlabel('Redundancy parameter R', fontsize=12, labelpad=8)
ax.set_ylabel('Dimension parameter K', fontsize=12, labelpad=8)
ax.set_title(
    r'Phase diagram $M_2(K,R)$ over $\mathbb{Z}/4\mathbb{Z}$ under $\Delta_2$ metric'
    '\n(Theorem 19, SNS-THEORY-001)',
    fontsize=13, pad=12
)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label(r'Max min $\Delta_2$ distance', fontsize=11)

gold_patch = mpatches.Patch(facecolor='#d4a017', edgecolor='black',
                             label='Headline cell (K=7,R=7): d=8 (Thm 20)')
orange_patch = mpatches.Patch(facecolor='#f4a582', edgecolor='black',
                               label='Contrast cell (K=8,R=7): d=6 (Thm 21)')
red_text = mpatches.Patch(facecolor='white', edgecolor='grey',
                           label='≠ marks where naive R+1 fails')
ax.legend(handles=[gold_patch, orange_patch, red_text],
          loc='upper left', fontsize=9, framealpha=0.9)

plt.tight_layout()
for ext in ['png', 'pdf']:
    path = Path(f'SNS_FIG1_Z4Z_phase_diagram.{ext}')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')

plt.close()
print('Figure 1 complete.')
