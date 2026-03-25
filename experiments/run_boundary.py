"""Boundary analysis: test 5 additional backbones to map where PSA fails."""
import os, sys, glob, numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import SSL_CORE, BOUNDARY, ALL_BACKBONES, process_image
from src.metrics import compute_psa, compute_sc_bsds, cluster_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, 'data', 'BSDS500', 'images', 'test')
GT_DIR = os.path.join(ROOT, 'data', 'BSDS500', 'ground_truth', 'test')


def run_one(name, loader_fn, imgs):
    print(f"\n--- {name} ---")
    model, transform, model_type = loader_fn()
    device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cuda'
    psa_list, sc_list = [], []
    for ip in tqdm(imgs, desc=name, leave=False):
        try:
            raw, side = process_image(ip, model, transform, model_type, device)
            psa_list.append(compute_psa(raw, side, side))
            bn = os.path.splitext(os.path.basename(ip))[0]
            gt_path = os.path.join(GT_DIR, bn + '.mat')
            if os.path.exists(gt_path):
                labels = cluster_features(raw, k=4).reshape(side, side)
                sc_list.append(compute_sc_bsds(labels, gt_path, side, side))
        except:
            continue
    import torch; del model; torch.cuda.empty_cache()
    if psa_list and sc_list:
        return np.mean(psa_list), np.mean(sc_list)
    return None


def main():
    imgs = sorted(glob.glob(os.path.join(IMG_DIR, '*.jpg')))[:50]
    results = {}

    # Run all 13 backbones
    for name, loader_fn in ALL_BACKBONES.items():
        r = run_one(name, loader_fn, imgs)
        if r:
            results[name] = r
            print(f"  PSA={r[0]:.3f}  SC={r[1]:.3f}")

    # Summary
    print(f"\n\n{'='*60}")
    print(f"BOUNDARY ANALYSIS — ALL {len(results)} BACKBONES")
    print(f"{'='*60}")
    for n, (psa, sc) in results.items():
        group = 'SSL' if n in SSL_CORE else 'EXT'
        print(f"  [{group}] {n:12s}: PSA={psa:.3f}  SC={sc:.3f}")

    # SSL-only correlation
    ssl_names = [n for n in results if n in SSL_CORE]
    r8, p8 = pearsonr([results[n][0] for n in ssl_names],
                       [results[n][1] for n in ssl_names])
    print(f"\nSSL Core (N={len(ssl_names)}): r={r8:.3f}, p={p8:.4f}")

    # All correlation
    all_names = list(results.keys())
    r_all, p_all = pearsonr([results[n][0] for n in all_names],
                             [results[n][1] for n in all_names])
    print(f"All (N={len(all_names)}): r={r_all:.3f}, p={p_all:.4f}")


if __name__ == '__main__':
    main()
