"""Fig 2: PCA Feature Visualization — Input | PCA Features
3 backbones: iBOT (best), OpenCLIP (mid), BEiTv2 (worst).
"""
import os, sys, torch, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import load_ibot, load_beitv2, load_openclip, process_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Image 268002 (cormorant)
IMG_PATH = os.path.join(ROOT, 'data', 'BSDS500', 'images', 'train', '268002.jpg')

def pca_colorize(features, side):
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(features)
    for c in range(3):
        p2, p98 = np.percentile(rgb[:, c], [2, 98])
        rgb[:, c] = np.clip((rgb[:, c] - p2) / max(p98 - p2, 1e-8), 0, 1)
    return rgb.reshape(side, side, 3)

def process_backbone(name, loader_fn):
    print(f"  Loading {name}...")
    model, tf, mt = loader_fn()
    dev = next(model.parameters()).device
    raw, side = process_image(IMG_PATH, model, tf, mt, dev)
    pca_img = pca_colorize(raw, side)
    del model; torch.cuda.empty_cache()
    return pca_img, side

orig = Image.open(IMG_PATH).convert('RGB').resize((224, 224))

backbones = [
    ('iBOT  (PSA = 0.71, SC = 0.57)',     load_ibot),
    ('OpenCLIP  (PSA = 0.60, SC = 0.51)',  load_openclip),
    ('BEiTv2  (PSA = 0.45, SC = 0.34)',    load_beitv2),
]

results = []
for name, fn in backbones:
    pca_img, side = process_backbone(name, fn)
    results.append((name, pca_img))

# --- Plot: 3 rows x 2 cols ---
fig, axes = plt.subplots(3, 2, figsize=(6, 7.5))

axes[0, 0].set_title('Input Image', fontsize=11, fontweight='bold', pad=10)
axes[0, 1].set_title('PCA Features (3 PC → RGB)', fontsize=11, fontweight='bold', pad=10)

for i, (name, pca_img) in enumerate(results):
    axes[i, 0].imshow(orig)
    axes[i, 0].set_ylabel(name, fontsize=9, fontweight='bold',
                          rotation=0, labelpad=120, va='center', ha='right')
    axes[i, 1].imshow(pca_img, interpolation='nearest')

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(0.5)

# Arrow from top to bottom
fig.text(0.5, 0.005,
         'Higher PSA → spatially smoother features → better segmentation quality',
         ha='center', fontsize=9, style='italic', color='#444444')

plt.subplots_adjust(left=0.28, right=0.98, top=0.94, bottom=0.04, hspace=0.06, wspace=0.05)
plt.savefig(os.path.join(ROOT, 'paper', 'figures', 'feature_pca_viz.pdf'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(ROOT, 'paper', 'figures', 'feature_pca_viz.png'),
            dpi=300, bbox_inches='tight')
print('Saved: paper/figures/feature_pca_viz.pdf + .png')
