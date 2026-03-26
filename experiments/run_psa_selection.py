"""PSA-Guided Backbone Selection (Sect 4.6).
Rank 11 SSL backbones by PSA on BSDS500, verify SC on ADE20K.
"""
import os, sys, random, numpy as np, torch, glob
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import SSL_CORE, process_image
from src.metrics import compute_psa, compute_sc_ade, cluster_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BSDS_IMG = os.path.join(ROOT, 'data', 'BSDS500', 'images', 'test')
ADE_IMG = os.path.join(ROOT, 'data', 'ADE20K', 'images', 'validation')
ADE_ANN = os.path.join(ROOT, 'data', 'ADE20K', 'annotations', 'validation')
N_BSDS = 50; N_ADE = 50

BSDS_IMGS = sorted(glob.glob(os.path.join(BSDS_IMG, '*.jpg')))[:N_BSDS]
ADE_IMGS = sorted(glob.glob(os.path.join(ADE_IMG, '*.jpg')))[:N_ADE]

print(f"PSA-Guided Selection: {N_BSDS} BSDS images (PSA), {N_ADE} ADE images (SC)")

results = {}
for name, fn in SSL_CORE.items():
    print(f"\n--- {name} ---")
    model, tf, mt = fn(); dev = next(model.parameters()).device

    # PSA on BSDS500 (no labels)
    psa_list = []
    for ip in tqdm(BSDS_IMGS, desc=f'{name}/PSA', leave=False):
        try:
            raw, side = process_image(ip, model, tf, mt, dev)
            psa_list.append(compute_psa(raw, side, side))
        except: continue

    # SC on ADE20K (with labels — for verification only)
    sc_list = []
    for ip in tqdm(ADE_IMGS, desc=f'{name}/ADE', leave=False):
        try:
            raw, side = process_image(ip, model, tf, mt, dev)
            bn = os.path.splitext(os.path.basename(ip))[0]
            ann = os.path.join(ADE_ANN, bn + '.png')
            if os.path.exists(ann):
                labels = cluster_features(raw, k=4).reshape(side, side)
                sc = compute_sc_ade(labels, ann, side, side)
                if not np.isnan(sc): sc_list.append(sc)
        except: continue

    del model; torch.cuda.empty_cache()
    results[name] = {'PSA': np.mean(psa_list), 'SC_ADE': np.mean(sc_list)}
    print(f"  PSA(BSDS)={results[name]['PSA']:.3f}  SC(ADE)={results[name]['SC_ADE']:.3f}")

# Rank by PSA (label-free), compare to actual SC rank
names = list(results.keys())
psa_rank = sorted(names, key=lambda n: -results[n]['PSA'])
sc_rank = sorted(names, key=lambda n: -results[n]['SC_ADE'])

print(f"\n{'='*60}")
print("PSA-GUIDED SELECTION")
print(f"{'='*60}")
print(f"{'Rank':>4s}  {'PSA Ranking':20s}({'PSA':>6s})  {'Actual SC Ranking':20s}({'SC':>6s})")
for i in range(len(names)):
    pn = psa_rank[i]; sn = sc_rank[i]
    print(f"  {i+1:2d}.  {pn:20s}({results[pn]['PSA']:6.3f})  {sn:20s}({results[sn]['SC_ADE']:6.3f})")

psa_v = [results[n]['PSA'] for n in names]
sc_v = [results[n]['SC_ADE'] for n in names]
r, p = pearsonr(psa_v, sc_v)
rho, sp = spearmanr(psa_v, sc_v)
print(f"\nPearson r={r:.4f}, p={p:.6f}")
print(f"Spearman rho={rho:.4f}, p={sp:.6f}")
print(f"Top-3 PSA: {psa_rank[:3]}")
print(f"Top-3 SC:  {sc_rank[:3]}")
