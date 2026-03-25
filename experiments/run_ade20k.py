"""Cross-dataset: PSA vs SC on ADE20K for 8 SSL core ViT-B/16 backbones."""
import os, sys, glob, numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import SSL_CORE, process_image
from src.metrics import compute_psa, compute_sc_ade, cluster_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADE_IMG = os.path.join(ROOT, 'data', 'ADE20K', 'images', 'validation')
ADE_ANN = os.path.join(ROOT, 'data', 'ADE20K', 'annotations', 'validation')


def main():
    imgs = sorted(glob.glob(os.path.join(ADE_IMG, '*.jpg')))[:50]
    print(f"ADE20K: {len(imgs)} images")

    results = {}
    for name, loader_fn in SSL_CORE.items():
        print(f"\n--- {name} ---")
        model, transform, model_type = loader_fn()
        device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cuda'

        psa_list, sc_list = [], []
        for ip in tqdm(imgs, desc=name, leave=False):
            try:
                raw, side = process_image(ip, model, transform, model_type, device)
                psa_list.append(compute_psa(raw, side, side))

                bn = os.path.splitext(os.path.basename(ip))[0]
                ann_path = os.path.join(ADE_ANN, bn + '.png')
                if os.path.exists(ann_path):
                    labels = cluster_features(raw, k=4).reshape(side, side)
                    sc = compute_sc_ade(labels, ann_path, side, side)
                    if not np.isnan(sc):
                        sc_list.append(sc)
            except:
                continue

        import torch; del model; torch.cuda.empty_cache()
        if psa_list and sc_list:
            results[name] = (np.mean(psa_list), np.mean(sc_list))
            print(f"  PSA={results[name][0]:.3f}  SC_ADE={results[name][1]:.3f}")

    names = list(results.keys())
    psa = [results[n][0] for n in names]
    sc = [results[n][1] for n in names]
    r, p = pearsonr(psa, sc)
    print(f"\nPSA vs SC_ADE20K: r={r:.3f}, p={p:.4f}, N={len(names)}")


if __name__ == '__main__':
    main()
