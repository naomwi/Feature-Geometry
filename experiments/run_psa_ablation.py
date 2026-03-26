"""PSA Variants Ablation — 4-conn, 8-conn, L2, weighted — all 11 SSL, seed=42, 50 BSDS."""
import os, sys, random, numpy as np, torch, glob
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import SSL_CORE, process_image
from src.metrics import compute_psa, cluster_features, compute_sc_bsds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BSDS_IMG = os.path.join(ROOT, 'data', 'BSDS500', 'images', 'test')
BSDS_GT = os.path.join(ROOT, 'data', 'BSDS500', 'ground_truth', 'test')
IMGS = sorted(glob.glob(os.path.join(BSDS_IMG, '*.jpg')))[:50]

print(f"PSA Variants Ablation: {len(IMGS)} images, seed={SEED}")


def psa_8conn(features, h, w):
    grid = features.reshape(h, w, -1)
    sims = []
    for i in range(h):
        for j in range(w):
            patch = grid[i, j].reshape(1, -1)
            nb = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0: continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        nb.append(grid[ni, nj])
            if nb:
                sims.append(cosine_similarity(patch, np.stack(nb)).mean())
    return float(np.mean(sims))


def psa_l2(features, h, w):
    grid = features.reshape(h, w, -1)
    dists = []
    for i in range(h):
        for j in range(w):
            patch = grid[i, j]
            nb = []
            if i > 0:     nb.append(grid[i-1, j])
            if i < h - 1: nb.append(grid[i+1, j])
            if j > 0:     nb.append(grid[i, j-1])
            if j < w - 1: nb.append(grid[i, j+1])
            if nb:
                dists.append(np.mean([np.linalg.norm(patch - n) for n in nb]))
    return float(-np.mean(dists))


def psa_weighted(features, h, w):
    pca = PCA(n_components=min(32, features.shape[1]))
    reduced = pca.fit_transform(features)
    reduced = normalize(reduced, norm='l2')
    weights = pca.explained_variance_ / pca.explained_variance_.sum()
    grid = reduced.reshape(h, w, -1)
    sims = []
    for i in range(h):
        for j in range(w):
            wp = grid[i, j] * weights
            nb = []
            if i > 0:     nb.append(grid[i-1, j] * weights)
            if i < h - 1: nb.append(grid[i+1, j] * weights)
            if j > 0:     nb.append(grid[i, j-1] * weights)
            if j < w - 1: nb.append(grid[i, j+1] * weights)
            if nb:
                wsims = []
                for n in nb:
                    cos = np.dot(wp, n) / (np.linalg.norm(wp) * np.linalg.norm(n) + 1e-10)
                    wsims.append(cos)
                sims.append(np.mean(wsims))
    return float(np.mean(sims))


results = {}
for name, loader_fn in SSL_CORE.items():
    print(f"\n--- {name} ---")
    model, transform, model_type = loader_fn()
    device = next(model.parameters()).device

    v4, v8, vl2, vw, sc_list = [], [], [], [], []
    for ip in tqdm(IMGS, desc=name, leave=False):
        try:
            raw, side = process_image(ip, model, transform, model_type, device)
            v4.append(compute_psa(raw, side, side))
            v8.append(psa_8conn(raw, side, side))
            vl2.append(psa_l2(raw, side, side))
            vw.append(psa_weighted(raw, side, side))

            bn = os.path.splitext(os.path.basename(ip))[0]
            gp = os.path.join(BSDS_GT, bn + '.mat')
            if os.path.exists(gp):
                labels = cluster_features(raw, k=4).reshape(side, side)
                sc_list.append(compute_sc_bsds(labels, gp, side, side))
        except:
            continue

    del model; torch.cuda.empty_cache()
    results[name] = {
        'PSA_4': np.mean(v4), 'PSA_8': np.mean(v8),
        'PSA_L2': np.mean(vl2), 'PSA_W': np.mean(vw),
        'SC': np.mean(sc_list)
    }
    r = results[name]
    print(f"  PSA4={r['PSA_4']:.3f} PSA8={r['PSA_8']:.3f} L2={r['PSA_L2']:.3f} W={r['PSA_W']:.3f} SC={r['SC']:.3f}")

# Correlations
print(f"\n\n{'='*60}")
print("PSA VARIANTS ABLATION (N=11)")
print(f"{'='*60}")
names = list(results.keys())
sc = [results[n]['SC'] for n in names]
for var, label in [('PSA_4', 'PSA (4-conn, cosine)'), ('PSA_8', 'PSA (8-conn, cosine)'),
                    ('PSA_L2', 'PSA (4-conn, neg-L2)'), ('PSA_W', 'PSA-weighted (eigenval)')]:
    vals = [results[n][var] for n in names]
    r, p = pearsonr(vals, sc)
    print(f"  {label:30s}: r={r:.4f}, p={p:.6f}")

# Per-backbone table
print(f"\n{'Backbone':10s} {'PSA_4':>7s} {'PSA_8':>7s} {'PSA_L2':>7s} {'PSA_W':>7s} {'SC':>7s}")
print("-" * 50)
for n in sorted(results, key=lambda x: -results[x]['SC']):
    r = results[n]
    print(f"{n:10s} {r['PSA_4']:7.3f} {r['PSA_8']:7.3f} {r['PSA_L2']:7.3f} {r['PSA_W']:7.3f} {r['SC']:7.3f}")
