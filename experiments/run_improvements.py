"""All 3 paper improvements in one script:
1. Expand SSL set with new ViT-B/16 backbones (EVA-02, OpenCLIP, MetaCLIP)
2. PSA variants ablation (4-connected vs 8-connected, cosine vs L2, weighted)
3. PSA-guided backbone selection (rank by PSA on BSDS, verify on ADE20K)
"""
import os, sys, glob, math, numpy as np, torch
from PIL import Image
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import timm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import SSL_CORE, process_image, extract_patches, TF_IMAGENET, TF_CLIP
from src.metrics import compute_psa, compute_sc_bsds, compute_sc_ade, cluster_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device('cuda')

BSDS_IMG = os.path.join(ROOT, 'data', 'BSDS500', 'images', 'test')
BSDS_GT = os.path.join(ROOT, 'data', 'BSDS500', 'ground_truth', 'test')
ADE_IMG = os.path.join(ROOT, 'data', 'ADE20K', 'images', 'validation')
ADE_ANN = os.path.join(ROOT, 'data', 'ADE20K', 'annotations', 'validation')


# ═══════════════════════════════════════════════════════
# PSA VARIANTS
# ═══════════════════════════════════════════════════════

def compute_psa_8connected(features, h, w):
    """PSA with 8-connected neighborhood (includes diagonals)."""
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


def compute_psa_l2(features, h, w):
    """PSA using negative L2 distance (higher = more similar)."""
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
                d = np.mean([np.linalg.norm(patch - n) for n in nb])
                dists.append(d)
    # Return negative mean distance (so higher = more similar, like cosine PSA)
    return float(-np.mean(dists))


def compute_psa_weighted(features_raw, features_pca, eigenvalues, h, w):
    """PSA weighted by PCA eigenvalue explained variance."""
    grid = features_pca.reshape(h, w, -1)
    weights = eigenvalues / eigenvalues.sum()  # normalize
    sims = []
    for i in range(h):
        for j in range(w):
            patch = grid[i, j]
            nb = []
            if i > 0:     nb.append(grid[i-1, j])
            if i < h - 1: nb.append(grid[i+1, j])
            if j > 0:     nb.append(grid[i, j-1])
            if j < w - 1: nb.append(grid[i, j+1])
            if nb:
                # Weighted cosine: weight each dimension by eigenvalue
                wp = patch * weights
                wsims = []
                for n in nb:
                    wn = n * weights
                    cos = np.dot(wp, wn) / (np.linalg.norm(wp) * np.linalg.norm(wn) + 1e-10)
                    wsims.append(cos)
                sims.append(np.mean(wsims))
    return float(np.mean(sims))


# ═══════════════════════════════════════════════════════
# NEW BACKBONE LOADERS
# ═══════════════════════════════════════════════════════

def load_eva02(device='cuda'):
    model = timm.create_model('eva02_base_patch16_clip_224.merged2b', pretrained=True, dynamic_img_size=True)
    return model.to(device).eval(), TF_IMAGENET, 'timm'

def load_openclip_laion(device='cuda'):
    model = timm.create_model('vit_base_patch16_clip_224.laion2b', pretrained=True, dynamic_img_size=True)
    return model.to(device).eval(), TF_IMAGENET, 'timm'

def load_metaclip(device='cuda'):
    model = timm.create_model('vit_base_patch16_clip_224.metaclip_2pt5b', pretrained=True, dynamic_img_size=True)
    return model.to(device).eval(), TF_IMAGENET, 'timm'

NEW_BACKBONES = {
    'EVA-02':    load_eva02,
    'OpenCLIP':  load_openclip_laion,
    'MetaCLIP':  load_metaclip,
}


# ═══════════════════════════════════════════════════════
# HELPER: run one backbone on BSDS500
# ═══════════════════════════════════════════════════════

def run_bsds(name, loader_fn, n_images=50, compute_variants=False):
    """Run PSA + SC + optional variants on BSDS500."""
    model, transform, model_type = loader_fn()
    dev = next(model.parameters()).device if hasattr(model, 'parameters') else device
    imgs = sorted(glob.glob(os.path.join(BSDS_IMG, '*.jpg')))[:n_images]

    psa_list, sc_list = [], []
    psa8_list, psal2_list, psaw_list = [], [], []

    for ip in tqdm(imgs, desc=name, leave=False):
        try:
            raw, side = process_image(ip, model, transform, model_type, dev)
            psa_list.append(compute_psa(raw, side, side))

            # SC
            bn = os.path.splitext(os.path.basename(ip))[0]
            gp = os.path.join(BSDS_GT, bn + '.mat')
            if os.path.exists(gp):
                labels = cluster_features(raw, k=4).reshape(side, side)
                sc_list.append(compute_sc_bsds(labels, gp, side, side))

            # Variants
            if compute_variants:
                psa8_list.append(compute_psa_8connected(raw, side, side))
                psal2_list.append(compute_psa_l2(raw, side, side))
                pca = PCA(n_components=min(32, raw.shape[1]))
                reduced = pca.fit_transform(raw)
                reduced_n = normalize(reduced, norm='l2')
                psaw_list.append(compute_psa_weighted(raw, reduced_n, pca.explained_variance_, side, side))
        except:
            continue

    del model; torch.cuda.empty_cache()

    result = {'PSA': np.mean(psa_list), 'SC': np.mean(sc_list), 'N': len(sc_list)}
    if compute_variants:
        result['PSA_8conn'] = np.mean(psa8_list)
        result['PSA_L2'] = np.mean(psal2_list)
        result['PSA_weighted'] = np.mean(psaw_list)
    print(f"  {name}: PSA={result['PSA']:.3f}  SC={result['SC']:.3f}")
    return result


def run_ade(name, loader_fn, n_images=50):
    """Run PSA + SC on ADE20K."""
    model, transform, model_type = loader_fn()
    dev = next(model.parameters()).device if hasattr(model, 'parameters') else device
    imgs = sorted(glob.glob(os.path.join(ADE_IMG, '*.jpg')))[:n_images]
    psa_list, sc_list = [], []
    for ip in tqdm(imgs, desc=name, leave=False):
        try:
            raw, side = process_image(ip, model, transform, model_type, dev)
            psa_list.append(compute_psa(raw, side, side))
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
    return {'PSA': np.mean(psa_list), 'SC': np.mean(sc_list)}


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    ALL = {**SSL_CORE, **NEW_BACKBONES}

    # ── PART 1: Run all backbones on BSDS500 (with variants for original 8) ──
    print("\n" + "=" * 60)
    print("PART 1: ALL BACKBONES ON BSDS500")
    print("=" * 60)

    bsds_results = {}
    for name, loader_fn in ALL.items():
        is_original = name in SSL_CORE
        bsds_results[name] = run_bsds(name, loader_fn, compute_variants=is_original)

    # ── PART 2: PSA variants ablation ──
    print("\n" + "=" * 60)
    print("PART 2: PSA VARIANTS ABLATION")
    print("=" * 60)

    original_names = [n for n in bsds_results if n in SSL_CORE]
    sc_vals = [bsds_results[n]['SC'] for n in original_names]

    for variant in ['PSA', 'PSA_8conn', 'PSA_L2', 'PSA_weighted']:
        vals = [bsds_results[n].get(variant, 0) for n in original_names]
        if all(v != 0 for v in vals):
            r, p = pearsonr(vals, sc_vals)
            print(f"  {variant:15s}: r={r:.3f}, p={p:.4f}")

    # ── PART 3: Expanded correlation ──
    print("\n" + "=" * 60)
    print(f"PART 3: EXPANDED CORRELATION (N={len(ALL)})")
    print("=" * 60)

    all_names = list(bsds_results.keys())
    all_psa = [bsds_results[n]['PSA'] for n in all_names]
    all_sc = [bsds_results[n]['SC'] for n in all_names]
    r, p = pearsonr(all_psa, all_sc)
    print(f"  Pearson r={r:.3f}, p={p:.4f}, N={len(all_names)}")
    rho, sp = spearmanr(all_psa, all_sc)
    print(f"  Spearman rho={rho:.3f}, p={sp:.4f}")

    for n in sorted(all_names, key=lambda x: -bsds_results[x]['SC']):
        tag = '' if n in SSL_CORE else ' [NEW]'
        print(f"    {n:12s}: PSA={bsds_results[n]['PSA']:.3f}  SC={bsds_results[n]['SC']:.3f}{tag}")

    # ── PART 4: PSA-guided backbone selection ──
    print("\n" + "=" * 60)
    print("PART 4: PSA-GUIDED BACKBONE SELECTION")
    print("=" * 60)
    print("Ranking backbones by PSA (BSDS500), verifying on ADE20K...")

    ade_results = {}
    for name, loader_fn in ALL.items():
        ade_results[name] = run_ade(name, loader_fn)

    # PSA ranking (from BSDS500) vs actual SC ranking (from ADE20K)
    names = list(ALL.keys())
    psa_ranks = np.argsort(np.argsort([-bsds_results[n]['PSA'] for n in names]))
    sc_ade_ranks = np.argsort(np.argsort([-ade_results[n]['SC'] for n in names]))

    rho, p_rho = spearmanr(
        [bsds_results[n]['PSA'] for n in names],
        [ade_results[n]['SC'] for n in names]
    )
    r_pear, p_pear = pearsonr(
        [bsds_results[n]['PSA'] for n in names],
        [ade_results[n]['SC'] for n in names]
    )

    print(f"\n  PSA (BSDS500) → SC (ADE20K):")
    print(f"  Spearman rho={rho:.3f}, p={p_rho:.4f}")
    print(f"  Pearson r={r_pear:.3f}, p={p_pear:.4f}")
    print(f"\n  {'Backbone':12s} {'PSA(BSDS)':>10s} {'rank':>5s}  {'SC(ADE)':>10s} {'rank':>5s}")
    print("  " + "-" * 50)
    for n in sorted(names, key=lambda x: -bsds_results[x]['PSA']):
        pi = list(names).index(n)
        print(f"  {n:12s} {bsds_results[n]['PSA']:10.3f} {psa_ranks[pi]+1:5d}  {ade_results[n]['SC']:10.3f} {sc_ade_ranks[pi]+1:5d}")


if __name__ == '__main__':
    main()
