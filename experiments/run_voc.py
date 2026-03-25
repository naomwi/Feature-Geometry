"""Cross-dataset: PSA vs SC on PASCAL VOC for 8 SSL core ViT-B/16 backbones."""
import os, sys, glob, numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import SSL_CORE, process_image
from src.metrics import compute_psa, compute_sc_voc, cluster_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOC_IMG = os.path.join(ROOT, 'data', 'VOC2012', 'JPEGImages')
VOC_GT = os.path.join(ROOT, 'data', 'VOC2012', 'SegmentationClass')
VOC_LIST = os.path.join(ROOT, 'data', 'VOC2012', 'ImageSets', 'Segmentation', 'val.txt')


def main():
    with open(VOC_LIST) as f:
        val_ids = [l.strip() for l in f if l.strip()][:30]

    results = {}
    for name, loader_fn in SSL_CORE.items():
        print(f"\n--- {name} ---")
        model, transform, model_type = loader_fn()
        device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cuda'

        psa_list, sc_list = [], []
        for sid in tqdm(val_ids, desc=name, leave=False):
            try:
                ip = os.path.join(VOC_IMG, sid + '.jpg')
                gp = os.path.join(VOC_GT, sid + '.png')
                if not (os.path.exists(ip) and os.path.exists(gp)):
                    continue
                raw, side = process_image(ip, model, transform, model_type, device)
                psa_list.append(compute_psa(raw, side, side))
                labels = cluster_features(raw, k=4).reshape(side, side)
                sc = compute_sc_voc(labels, gp, side, side)
                if not np.isnan(sc):
                    sc_list.append(sc)
            except:
                continue

        import torch; del model; torch.cuda.empty_cache()
        if psa_list and sc_list:
            results[name] = (np.mean(psa_list), np.mean(sc_list))
            print(f"  PSA={results[name][0]:.3f}  SC_VOC={results[name][1]:.3f}")

    # Correlation
    names = list(results.keys())
    psa = [results[n][0] for n in names]
    sc = [results[n][1] for n in names]
    r, p = pearsonr(psa, sc)
    print(f"\nPSA vs SC_VOC: r={r:.3f}, p={p:.4f}, N={len(names)}")


if __name__ == '__main__':
    main()
