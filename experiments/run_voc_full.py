"""VOC re-run: 150 input images -> ~50 valid multi-class, all 11 SSL."""
import os, sys, random, numpy as np, torch
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import SSL_CORE, process_image
from src.metrics import compute_sc_voc, cluster_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOC_IMG = os.path.join(ROOT, 'data', 'VOC2012', 'JPEGImages')
VOC_GT = os.path.join(ROOT, 'data', 'VOC2012', 'SegmentationClass')
with open(os.path.join(ROOT, 'data', 'VOC2012', 'ImageSets', 'Segmentation', 'val.txt')) as f:
    ids = [l.strip() for l in f if l.strip()][:150]

print(f"VOC: {len(ids)} input images, seed={SEED}")

PSA = {'iBOT': 0.713, 'MoCo-v3': 0.861, 'DINO': 0.625, 'MAE': 0.731,
       'OpenCLIP': 0.596, 'MetaCLIP': 0.672, 'CLIP': 0.702,
       'SigLIP': 0.449, 'BEiT': 0.515, 'EVA-02': 0.479, 'BEiTv2': 0.449}

results = {}
for name, fn in SSL_CORE.items():
    m, tf, mt = fn()
    dev = next(m.parameters()).device
    sc = []
    for sid in tqdm(ids, desc=name, leave=False):
        try:
            ip = os.path.join(VOC_IMG, sid + '.jpg')
            gp = os.path.join(VOC_GT, sid + '.png')
            raw, side = process_image(ip, m, tf, mt, dev)
            labels = cluster_features(raw, k=4).reshape(side, side)
            s = compute_sc_voc(labels, gp, side, side)
            if not np.isnan(s):
                sc.append(s)
        except:
            continue
    del m; torch.cuda.empty_cache()
    mean_sc = np.mean(sc) if sc else float('nan')
    results[name] = {'SC': mean_sc, 'N': len(sc)}
    print(f"  {name}: SC={mean_sc:.3f} N={len(sc)}")

print(f"\n{'='*50}")
for n in sorted(results, key=lambda x: -results[x]['SC']):
    print(f"  {n:10s} SC={results[n]['SC']:.3f} N={results[n]['N']} PSA={PSA[n]:.3f}")

pv = [PSA[n] for n in results]
sv = [results[n]['SC'] for n in results]
r, p = pearsonr(pv, sv)
rho, sp = spearmanr(pv, sv)
print(f"\n  r={r:.4f} p={p:.6f} rho={rho:.4f} p_rho={sp:.6f} N=11")
