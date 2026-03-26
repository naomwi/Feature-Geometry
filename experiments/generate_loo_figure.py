"""Generate LOO stability bar chart (Fig 4) — fixed overlap."""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

names = ['DINO','MoCo-v3','MAE','CLIP','SigLIP','BEiT','BEiTv2','iBOT','EVA-02','OpenCLIP','MetaCLIP']
r_loo = [0.9252, 0.8785, 0.8837, 0.8842, 0.8639, 0.8743, 0.8573, 0.8858, 0.8591, 0.8879, 0.8791]
r_full = 0.8793

# Sort ascending
idx = np.argsort(r_loo)
names_s = [names[i] for i in idx]
r_s = [r_loo[i] for i in idx]

fig, ax = plt.subplots(figsize=(6, 3.5))
colors = ['#2ecc71' if r > r_full else '#e74c3c' for r in r_s]
bars = ax.barh(range(len(names_s)), r_s, color=colors, edgecolor='white', height=0.7, alpha=0.85)

ax.axvline(r_full, color='#2c3e50', linestyle='--', linewidth=1.5, label=f'Full set r = {r_full:.3f}')

# Add value labels at end of each bar
for i, (val, bar) in enumerate(zip(r_s, bars)):
    ax.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=7, color='#333333')

ax.set_yticks(range(len(names_s)))
ax.set_yticklabels(names_s, fontsize=9)
ax.set_xlabel('Pearson r (PSA vs SC, N=10)', fontsize=10)
ax.set_title('Leave-One-Out Stability', fontsize=11, fontweight='bold')
ax.set_xlim(0.84, 0.945)
ax.legend(fontsize=8, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('paper/figures/loo_stability.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/loo_stability.png', dpi=300, bbox_inches='tight')
print('Saved: paper/figures/loo_stability.pdf + .png')
