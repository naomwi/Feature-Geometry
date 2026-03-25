"""Generate Figure 3: PSA vs SC scatter plot with 3 marker groups."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── Verified data (all values from experiments) ──
SSL_CORE = {
    "DINO":    (0.611, 0.558), "MoCo-v3": (0.851, 0.555),
    "MAE":     (0.714, 0.527), "CLIP":    (0.715, 0.500),
    "SigLIP":  (0.449, 0.452), "BEiT":    (0.515, 0.448),
    "BEiTv2":  (0.449, 0.390), "iBOT":    (0.713, 0.571),
}
SUPERVISED = {
    "SAM": (0.702, 0.534), "DeiT": (0.570, 0.498), "DeiT-III": (0.694, 0.484),
}
OUTLIERS = {
    "Data2Vec": (0.715, 0.363), "BEiT3": (0.625, 0.329),
}

# Trendline on SSL core only
ssl_psa = np.array([v[0] for v in SSL_CORE.values()])
ssl_sc = np.array([v[1] for v in SSL_CORE.values()])
slope, intercept, r_val, p_val, _ = stats.linregress(ssl_psa, ssl_sc)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.set_facecolor('#FAFAFA')
fig.patch.set_facecolor('white')

# Trendline + confidence band
x_line = np.linspace(0.36, 0.92, 200)
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, color="#2563EB", linewidth=1.8, linestyle="--", alpha=0.45, zorder=1)
y_pred = slope * ssl_psa + intercept
se = np.sqrt(np.sum((ssl_sc - y_pred)**2) / (len(ssl_sc) - 2))
ax.fill_between(x_line, y_line - 1.5*se, y_line + 1.5*se, color="#2563EB", alpha=0.06, zorder=0)

# SSL Core (blue circles)
ssl_offsets = {
    'DINO': (-40, -1), 'MoCo-v3': (8, -1), 'MAE': (8, -1), 'CLIP': (8, -1),
    'SigLIP': (-46, -1), 'BEiT': (8, -1), 'BEiTv2': (8, -1), 'iBOT': (10, -1),
}
for name, (psa, sc) in SSL_CORE.items():
    ax.scatter(psa, sc, color="#2563EB", s=75, zorder=4, edgecolors="white", linewidths=0.7)
    ox, oy = ssl_offsets[name]
    fw = 'bold' if name == 'iBOT' else 'normal'
    ax.annotate(name, (psa, sc), xytext=(ox, oy), textcoords='offset points',
                fontsize=8.5, color="#1e3a5f", fontweight=fw)
ax.scatter(0.713, 0.571, s=140, zorder=3, facecolors="none", edgecolors="#2563EB", linewidths=2.0)

# Supervised (gray triangles)
sup_offsets = {'SAM': (-34, -1), 'DeiT': (-38, -1), 'DeiT-III': (8, -1)}
for name, (psa, sc) in SUPERVISED.items():
    ax.scatter(psa, sc, marker="^", s=85, zorder=4, facecolors="none",
              edgecolors="#6B7280", linewidths=1.6)
    ox, oy = sup_offsets[name]
    ax.annotate(name, (psa, sc), xytext=(ox, oy), textcoords='offset points',
                fontsize=8.5, color="#6B7280")

# Outliers (red crosses)
out_offsets = {'Data2Vec': (8, -1), 'BEiT3': (-48, -1)}
for name, (psa, sc) in OUTLIERS.items():
    ax.scatter(psa, sc, marker="X", s=100, zorder=5, color="#DC2626",
              edgecolors="white", linewidths=0.7)
    ox, oy = out_offsets[name]
    ax.annotate(name, (psa, sc), xytext=(ox, oy), textcoords='offset points',
                fontsize=8.5, color="#DC2626", fontweight="bold")

# Stats box
ax.text(0.82, 0.32, f"$r = {r_val:.3f}$\n$p = {p_val:.3f}$\n$N\\!=\\!8$ SSL",
        fontsize=9.5, color="#2563EB", ha='center',
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#2563EB", alpha=0.92, linewidth=1.2))

# Legend
legend_handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2563EB",
               markeredgecolor="white", markersize=9, linewidth=0, label="SSL Core ($N\\!=\\!8$, trendline)"),
    plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="none",
               markeredgecolor="#6B7280", markersize=9, linewidth=0,
               markeredgewidth=1.5, label="Supervised / Specialized"),
    plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="#DC2626",
               markeredgecolor="white", markersize=9, linewidth=0,
               label="Outliers (Ctx. / Multimodal)"),
]
ax.legend(handles=legend_handles, fontsize=8.5, loc="upper left",
          framealpha=0.95, edgecolor="#d1d5db", fancybox=True)

ax.set_xlabel("Patch Spatial Autocorrelation (PSA)", fontsize=11.5, labelpad=8)
ax.set_ylabel("Segmentation Covering (SC)", fontsize=11.5, labelpad=8)
ax.set_xlim(0.36, 0.92); ax.set_ylim(0.28, 0.61)
ax.grid(True, alpha=0.15, linewidth=0.5, color='#888')
ax.spines[["top", "right"]].set_visible(False)
ax.spines["left"].set_color('#ccc'); ax.spines["bottom"].set_color('#ccc')
ax.tick_params(colors='#444', labelsize=9.5)

plt.tight_layout()
import os
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'paper', 'figures')
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, 'fig_psa_sc_v2.png'), dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(out_dir, 'fig_psa_sc_v2.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved to {out_dir}/fig_psa_sc_v2.png + .pdf")
plt.close()
