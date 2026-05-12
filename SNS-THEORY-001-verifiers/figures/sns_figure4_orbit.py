"""
SNS-THEORY-001 Figure 4: Orbit Structure of Max-Attaining Seeds (Theorem 24)
=============================================================================
Circular diagram showing the symmetry group action <sigma, -id> on a
representative max-attaining seed orbit over Z/9Z^7.
Illustrates the 14-element free orbit structure (7 cyclic shifts + negations).

Output: SNS_FIG4_orbit_structure.png / .pdf

Run on: USLMAX002
Requires: matplotlib, numpy
Author: Ahmed M. Mansour, Steward and Sync LLC
Document: SNS-THEORY-001, Theorem 24
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

MOD = 9
K   = 7

# Representative witness seed and its orbit
seed = (6, 3, 1, 1, 4, 0, 0)

def cyclic_shift(s):
    return (s[-1],) + s[:-1]

def negate(s):
    return tuple((-x) % MOD for x in s)

def build_orbit(s):
    orbit_positive = []
    current = s
    for _ in range(K):
        orbit_positive.append(current)
        current = cyclic_shift(current)
    orbit_negative = [negate(x) for x in orbit_positive]
    return orbit_positive, orbit_negative

pos_orbit, neg_orbit = build_orbit(seed)
assert len(set(pos_orbit + neg_orbit)) == 14, "Orbit size should be 14"

fig, ax = plt.subplots(figsize=(11, 11))
ax.set_aspect('equal')
ax.axis('off')

radius_pos = 3.5
radius_neg = 5.5
node_size  = 0.38

# Draw nodes
def draw_ring(orbit, radius, color, label_prefix):
    n = len(orbit)
    angles = [2 * np.pi * i / n - np.pi/2 for i in range(n)]
    positions = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
    for i, (pos, s) in enumerate(zip(positions, orbit)):
        circle = plt.Circle(pos, node_size, color=color,
                            ec='black', linewidth=1.2, zorder=3)
        ax.add_patch(circle)
        label = f'σ^{i}(s)' if label_prefix == 'pos' else f'σ^{i}(-s)'
        ax.text(pos[0], pos[1] + 0.01,
                label, ha='center', va='center',
                fontsize=7, fontweight='bold', zorder=4)
        seed_str = str(list(s))
        ax.text(pos[0], pos[1] - 0.55,
                seed_str, ha='center', va='top',
                fontsize=5.5, color='#333333', zorder=4)
    return positions

pos_positions = draw_ring(pos_orbit, radius_pos, '#aec7e8', 'pos')
neg_positions = draw_ring(neg_orbit, radius_neg, '#ffbb78', 'neg')

# Draw cyclic shift arrows (inner ring)
n = len(pos_orbit)
for i in range(n):
    p1 = np.array(pos_positions[i])
    p2 = np.array(pos_positions[(i + 1) % n])
    mid = (p1 + p2) / 2
    direction = p2 - p1
    norm = np.linalg.norm(direction)
    direction = direction / norm
    start = p1 + direction * node_size
    end   = p2 - direction * node_size
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color='#2166ac',
                                lw=1.0, connectionstyle='arc3,rad=0.1'))

# Draw cyclic shift arrows (outer ring)
for i in range(n):
    p1 = np.array(neg_positions[i])
    p2 = np.array(neg_positions[(i + 1) % n])
    direction = p2 - p1
    norm = np.linalg.norm(direction)
    direction = direction / norm
    start = p1 + direction * node_size
    end   = p2 - direction * node_size
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color='#d62728',
                                lw=1.0, connectionstyle='arc3,rad=0.1'))

# Draw negation spokes (connecting pos to neg)
for i in range(n):
    p1 = np.array(pos_positions[i])
    p2 = np.array(neg_positions[i])
    direction = p2 - p1
    norm = np.linalg.norm(direction)
    direction = direction / norm
    start = p1 + direction * node_size
    end   = p2 - direction * node_size
    ax.plot([start[0], end[0]], [start[1], end[1]],
            color='#666666', linewidth=0.8, linestyle=':', zorder=2)

# Legend and labels
ax.text(0, 0, f'1 free orbit\nof size 14\nunder ⟨σ, −id⟩',
        ha='center', va='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#aaaaaa', alpha=0.9))

ax.text(0, radius_pos - 0.1, 'Positive orbit\n(cyclic shifts of s)',
        ha='center', va='center', fontsize=9, color='#2166ac')
ax.text(0, -(radius_neg + 0.7), 'Negative orbit\n(cyclic shifts of −s)',
        ha='center', va='center', fontsize=9, color='#d62728')

blue_patch  = mpatches.Patch(color='#aec7e8', label='σ^i(s): cyclic shifts')
orange_patch = mpatches.Patch(color='#ffbb78', label='σ^i(−s): negated shifts')
ax.legend(handles=[blue_patch, orange_patch], loc='lower right',
          fontsize=10, framealpha=0.9)

ax.set_title(
    'Orbit structure of a representative max-attaining seed\n'
    r'$\mathbf{s} = (6,3,1,1,4,0,0)$ over $\mathbb{Z}/9\mathbb{Z}^7$, '
    r'$K=7, R=7$'
    '\n(Theorem 24, SNS-THEORY-001 — 501 such orbits exist)',
    fontsize=12, pad=14
)

ax.set_xlim(-7.5, 7.5)
ax.set_ylim(-7.5, 7.5)

plt.tight_layout()
for ext in ['png', 'pdf']:
    path = Path(f'SNS_FIG4_orbit_structure.{ext}')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')

plt.close()
print('Figure 4 complete.')
