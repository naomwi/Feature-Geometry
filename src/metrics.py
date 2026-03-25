"""Metrics: PSA, SC, and feature geometry metrics (n80, LID, NESum, Stable Rank)."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from PIL import Image


# ═══════════════════════════════════════════════════════
# Class B: Spatial Arrangement
# ═══════════════════════════════════════════════════════

def compute_psa(features, h, w):
    """Patch Spatial Autocorrelation — mean cosine similarity to 4-connected neighbors.

    Args:
        features: [N, D] L2-normalized features (N = h*w)
        h, w: spatial grid dimensions

    Returns:
        float: mean PSA value in [0, 1]
    """
    grid = features.reshape(h, w, -1)
    sims = []
    for i in range(h):
        for j in range(w):
            patch = grid[i, j].reshape(1, -1)
            neighbors = []
            if i > 0:     neighbors.append(grid[i-1, j])
            if i < h - 1: neighbors.append(grid[i+1, j])
            if j > 0:     neighbors.append(grid[i, j-1])
            if j < w - 1: neighbors.append(grid[i, j+1])
            if neighbors:
                sims.append(cosine_similarity(patch, np.stack(neighbors)).mean())
    return float(np.mean(sims))


# ═══════════════════════════════════════════════════════
# Class A: Feature Geometry
# ═══════════════════════════════════════════════════════

def compute_geometry_metrics(features):
    """Compute all Class A geometry metrics from raw features.

    Args:
        features: [N, D] raw (un-normalized) features

    Returns:
        dict with keys: n80, lid, nesum, stable_rank
    """
    U, S, Vt = np.linalg.svd(features, full_matrices=False)

    # n80: minimum components for 80% variance
    S_norm = S / S.sum()
    cumsum = np.cumsum(S_norm)
    n80 = int(np.searchsorted(cumsum, 0.80) + 1)

    # NESum: normalized eigenvalue sum
    nesum = float(np.sum(S) / S[0]) if S[0] > 0 else 0.0

    # Stable Rank: ||F||_F^2 / ||F||_2^2
    stable_rank = float(np.sum(S**2) / (S[0]**2)) if S[0] > 0 else 0.0

    # LID: Local Intrinsic Dimensionality (MLE)
    from sklearn.neighbors import NearestNeighbors
    feat_n = normalize(features, norm='l2', axis=1)
    k = min(20, len(feat_n) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(feat_n)
    dists, _ = nn.kneighbors(feat_n)
    dists = np.maximum(dists[:, 1:], 1e-10)  # exclude self
    rk = dists[:, -1:]
    log_ratios = np.log(dists / rk)
    lid_per_point = -k / np.sum(log_ratios, axis=1)
    lid = float(np.mean(lid_per_point))

    return {'n80': n80, 'lid': lid, 'nesum': nesum, 'stable_rank': stable_rank}


# ═══════════════════════════════════════════════════════
# Segmentation Covering (SC)
# ═══════════════════════════════════════════════════════

def compute_sc_bsds(pred_labels, gt_path, h, w):
    """Segmentation Covering for BSDS500 (.mat ground truth).

    Args:
        pred_labels: [h, w] integer cluster labels
        gt_path: path to .mat ground truth file
        h, w: grid dimensions

    Returns:
        float: best SC across multiple ground truth segmentations
    """
    from scipy.io import loadmat
    gt_data = loadmat(gt_path)
    gt_segs = gt_data['groundTruth'][0]
    best_sc = 0.0
    for seg_idx in range(len(gt_segs)):
        gt = gt_segs[seg_idx][0][0][0]
        gt_r = np.array(Image.fromarray(gt.astype(np.uint8)).resize((w, h), Image.NEAREST))
        segs = np.unique(gt_r)
        if len(segs) < 2:
            continue
        total = gt_r.size
        sc = 0.0
        for r in segs:
            mr = (gt_r == r)
            sr = mr.sum()
            if sr == 0:
                continue
            best_iou = 0.0
            for s in np.unique(pred_labels):
                ms = (pred_labels == s)
                inter = (mr & ms).sum()
                union = (mr | ms).sum()
                if union > 0:
                    best_iou = max(best_iou, inter / union)
            sc += sr * best_iou
        best_sc = max(best_sc, sc / total)
    return best_sc


def compute_sc_voc(pred_labels, gt_path, h, w):
    """Segmentation Covering for PASCAL VOC (PNG semantic masks).

    Args:
        pred_labels: [h, w] integer cluster labels
        gt_path: path to PNG segmentation mask
        h, w: grid dimensions

    Returns:
        float: SC (ignoring background 0 and boundary 255)
    """
    gt = np.array(Image.open(gt_path))
    gt_r = np.array(Image.fromarray(gt.astype(np.uint8)).resize((w, h), Image.NEAREST))
    segs = np.unique(gt_r)
    segs = segs[(segs > 0) & (segs < 255)]  # exclude background and boundary
    if len(segs) < 2:
        return float('nan')
    total = np.sum((gt_r > 0) & (gt_r < 255))
    if total == 0:
        return float('nan')
    sc = 0.0
    for r in segs:
        mr = (gt_r == r)
        sr = mr.sum()
        if sr == 0:
            continue
        best_iou = 0.0
        for s in np.unique(pred_labels):
            ms = (pred_labels == s)
            inter = (mr & ms).sum()
            union = (mr | ms).sum()
            if union > 0:
                best_iou = max(best_iou, inter / union)
        sc += sr * best_iou
    return sc / total


def compute_sc_ade(pred_labels, gt_path, h, w):
    """Segmentation Covering for ADE20K (PNG class-ID masks).

    Args:
        pred_labels: [h, w] integer cluster labels
        gt_path: path to PNG annotation mask (pixel value = class ID, 0 = bg)
        h, w: grid dimensions

    Returns:
        float: SC (ignoring background class 0)
    """
    gt = np.array(Image.open(gt_path))
    gt_r = np.array(Image.fromarray(gt.astype(np.uint8)).resize((w, h), Image.NEAREST))
    segs = np.unique(gt_r)
    segs = segs[segs > 0]
    if len(segs) < 2:
        return float('nan')
    total = np.sum(gt_r > 0)
    if total == 0:
        return float('nan')
    sc = 0.0
    for r in segs:
        mr = (gt_r == r)
        sr = mr.sum()
        if sr == 0:
            continue
        best_iou = 0.0
        for s in np.unique(pred_labels):
            ms = (pred_labels == s)
            inter = (mr & ms).sum()
            union = (mr | ms).sum()
            if union > 0:
                best_iou = max(best_iou, inter / union)
        sc += sr * best_iou
    return sc / total


# ═══════════════════════════════════════════════════════
# Clustering Helper
# ═══════════════════════════════════════════════════════

def cluster_features(features, k=4, pca_dim=32):
    """PCA → L2 norm → KMeans clustering.

    Args:
        features: [N, D] L2-normalized features
        k: number of clusters
        pca_dim: PCA dimensionality

    Returns:
        labels: [N] integer cluster assignments
    """
    pca = PCA(n_components=min(pca_dim, features.shape[1]))
    reduced = normalize(pca.fit_transform(features), norm='l2')
    return KMeans(k, n_init=3, max_iter=100, random_state=42).fit_predict(reduced)
