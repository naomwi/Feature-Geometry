"""SC on ADE20K for all 11 SSL ViT-B/16 — fixed seed=42, 50 images."""
import os, sys, glob, random, numpy as np, torch
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import SSL_CORE, process_image
from src.metrics import compute_psa, compute_sc_ade, cluster_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADE_IMG = os.path.join(ROOT, 'data', 'ADE20K', 'images', 'validation')
ADE_ANN = os.path.join(ROOT, 'data', 'ADE20K', 'annotations', 'validation')
N_ADE = 50

IMGS = sorted(glob.glob(os.path.join(ADE_IMG, '*.jpg')))[:N_ADE]
print(f"ADE20K: {len(IMGS)} images, seed={SEED}")

# PSA from unified run (BSDS500)
PSA_BSDS = {
    'iBOT': 0.713, 'MoCo-v3': 0.861, 'DINO': 0.625, 'MAE': 0.731,
    'OpenCLIP': 0.596, 'MetaCLIP': 0.672, 'CLIP': 0.702,
    'SigLIP': 0.449, 'BEiT': 0.515, 'EVA-02': 0.479, 'BEiTv2': 0.449,
}

results = {}
for name, loader_fn in SSL_CORE.items():
    print(f"\n--- {name} ---")
    model, transform, model_type = loader_fn()
    device = next(model.parameters()).device
    sc_list = []
    for ip in tqdm(IMGS, desc=name, leave=False):
        try:
            raw, side = process_image(ip, model, transform, model_type, device)
            bn = os.path.splitext(os.path.basename(ip))[0]
            ann = os.path.join(ADE_ANN, bn + '.png')
            if os.path.exists(ann):
                labels = cluster_features(raw, k=4).reshape(side, side)
                sc = compute_sc_ade(labels, ann, side, side)
                if not np.isnan(sc):
                    sc_list.append(sc)
        except:
            continue
    del model; torch.cuda.empty_cache()
    sc_ade = np.mean(sc_list) if sc_list else float('nan')
    results[name] = sc_ade
    print(f"  SC(ADE)={sc_ade:.3f}  N={len(sc_list)}")

# Summary
print(f"\n{'='*50}")
print(f"SC(ADE20K) for all 11 SSL backbones")
print(f"{'='*50}")
for n in sorted(results, key=lambda x: -results[x]):
    psa = PSA_BSDS.get(n, 0)
    print(f"  {n:10s}: SC(ADE)={results[n]:.3f}  PSA(BSDS)={psa:.3f}")

# Correlation
names = [n for n in results if not np.isnan(results[n])]
psa = [PSA_BSDS[n] for n in names]
sc = [results[n] for n in names]
r, p = pearsonr(psa, sc)
rho, sp = spearmanr(psa, sc)
print(f"\nPSA(BSDS) vs SC(ADE20K):")
print(f"  Pearson r={r:.4f}, p={p:.6f}")
print(f"  Spearman rho={rho:.4f}, p={sp:.6f}")
print(f"  N={len(names)}")
