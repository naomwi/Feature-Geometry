"""Table 2: All metrics + SC on BSDS500 for 8 SSL core ViT-B/16 backbones."""
import os, sys, glob, numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import SSL_CORE, process_image
from src.metrics import compute_psa, compute_geometry_metrics, compute_sc_bsds, cluster_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, 'data', 'BSDS500', 'images', 'test')
GT_DIR = os.path.join(ROOT, 'data', 'BSDS500', 'ground_truth', 'test')


def run_backbone(name, loader_fn, n_images=50):
    """Run all metrics for one backbone on BSDS500."""
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    model, transform, model_type = loader_fn()
    device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cuda'

    imgs = sorted(glob.glob(os.path.join(IMG_DIR, '*.jpg')))[:n_images]
    psa_list, sc_list = [], []
    geo_lists = {'n80': [], 'lid': [], 'nesum': [], 'stable_rank': []}

    for ip in tqdm(imgs, desc=name, leave=False):
        try:
            raw, side = process_image(ip, model, transform, model_type, device)
            psa_list.append(compute_psa(raw, side, side))

            # Geometry metrics (from un-normalized features — use raw before L2 norm)
            geo = compute_geometry_metrics(raw)
            for k in geo_lists:
                geo_lists[k].append(geo[k])

            # SC
            bn = os.path.splitext(os.path.basename(ip))[0]
            gt_path = os.path.join(GT_DIR, bn + '.mat')
            if os.path.exists(gt_path):
                labels = cluster_features(raw, k=4).reshape(side, side)
                sc_list.append(compute_sc_bsds(labels, gt_path, side, side))
        except Exception as e:
            continue

    import torch; del model; torch.cuda.empty_cache()

    result = {
        'PSA': np.mean(psa_list),
        'SC': np.mean(sc_list),
        'n80': np.mean(geo_lists['n80']),
        'LID': np.mean(geo_lists['lid']),
        'NESum': np.mean(geo_lists['nesum']),
        'SR': np.mean(geo_lists['stable_rank']),
        'N': len(sc_list),
    }
    print(f"  PSA={result['PSA']:.3f}  SC={result['SC']:.3f}  n80={result['n80']:.1f}  "
          f"LID={result['LID']:.2f}  NESum={result['NESum']:.1f}  SR={result['SR']:.2f}")
    return result


def main():
    results = {}
    for name, loader_fn in SSL_CORE.items():
        results[name] = run_backbone(name, loader_fn)

    # Print Table 2
    print(f"\n\n{'='*70}")
    print("TABLE 2: All metrics for 8 SSL ViT-B/16 backbones (BSDS500)")
    print(f"{'='*70}")
    print(f"{'Backbone':12s} {'SC':>6s} {'n80':>6s} {'LID':>6s} {'NESum':>6s} {'SR':>6s} {'PSA':>6s}")
    print("-" * 56)
    for n, r in sorted(results.items(), key=lambda x: -x[1]['SC']):
        print(f"{n:12s} {r['SC']:.3f} {r['n80']:6.1f} {r['LID']:6.2f} {r['NESum']:6.1f} {r['SR']:6.2f} {r['PSA']:.3f}")

    # PSA correlation
    names = list(results.keys())
    psa = [results[n]['PSA'] for n in names]
    sc = [results[n]['SC'] for n in names]
    r, p = pearsonr(psa, sc)
    print(f"\nPSA vs SC: r={r:.3f}, p={p:.4f}, N={len(names)}")

    # Geometry correlations
    for metric in ['n80', 'LID', 'NESum', 'SR']:
        vals = [results[n][metric] for n in names]
        r, p = pearsonr(vals, sc)
        print(f"{metric} vs SC: r={r:.3f}, p={p:.4f}")


if __name__ == '__main__':
    main()
