"""Within-backbone spectral analysis (Sect 4.7 / Simpson's paradox).
Per-image n80 vs SC correlation WITHIN each backbone.
"""
import os, sys, glob, random, numpy as np, torch
from tqdm import tqdm
from scipy.stats import pearsonr

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import process_image, TF_IMAGENET
from src.metrics import compute_sc_bsds, cluster_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BSDS_IMG = os.path.join(ROOT, 'data', 'BSDS500', 'images', 'test')
BSDS_GT = os.path.join(ROOT, 'data', 'BSDS500', 'ground_truth', 'test')
IMGS = sorted(glob.glob(os.path.join(BSDS_IMG, '*.jpg')))  # All 200 test images

# Mixed-architecture set for within-backbone analysis
BACKBONES = {
    'MAE':     lambda: __import__('src.pipeline', fromlist=['load_mae']).load_mae(),
    'DINO-S8': lambda: (torch.hub.load('facebookresearch/dino:main', 'dino_vits8', trust_repo=True).cuda().eval(), TF_IMAGENET, 'dino'),
    'iBOT':    lambda: __import__('src.pipeline', fromlist=['load_ibot']).load_ibot(),
    'DINOv2':  lambda: (__import__('timm').create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, dynamic_img_size=True).cuda().eval(), TF_IMAGENET, 'timm'),
    'CLIP':    lambda: __import__('src.pipeline', fromlist=['load_clip']).load_clip(),
    'MoCo-v3': lambda: __import__('src.pipeline', fromlist=['load_mocov3']).load_mocov3(),
}

print(f"Within-backbone spectral analysis: {len(IMGS)} BSDS500 images")


def compute_per_image_n80(features):
    """Per-image n80 from SVD."""
    U, S, Vt = np.linalg.svd(features, full_matrices=False)
    S_norm = S / S.sum()
    return int(np.searchsorted(np.cumsum(S_norm), 0.80) + 1)


for name, loader_fn in BACKBONES.items():
    print(f"\n--- {name} ---")
    model, transform, model_type = loader_fn()
    device = next(model.parameters()).device

    n80_list, sc_list = [], []
    for ip in tqdm(IMGS, desc=name, leave=False):
        try:
            raw, side = process_image(ip, model, transform, model_type, device)
            n80 = compute_per_image_n80(raw)
            bn = os.path.splitext(os.path.basename(ip))[0]
            gp = os.path.join(BSDS_GT, bn + '.mat')
            if os.path.exists(gp):
                labels = cluster_features(raw, k=4).reshape(side, side)
                sc = compute_sc_bsds(labels, gp, side, side)
                n80_list.append(n80)
                sc_list.append(sc)
        except:
            continue

    del model; torch.cuda.empty_cache()

    if len(n80_list) >= 10:
        r, p = pearsonr(n80_list, sc_list)
        print(f"  n80 vs SC: r={r:.3f}, p={p:.4f}, N={len(n80_list)}")
        sig = "YES" if p < 0.01 else "no"
        print(f"  Significant at p<0.01? {sig}")
    else:
        print(f"  Too few valid images: {len(n80_list)}")
