"""Table 1: Clustering-invariant backbone rankings (KMeans, GMM, Spectral).
Reproduces Sect 4.1 — mixed-architecture set (6 backbones).
"""
import os, sys, glob, random, numpy as np, torch
from tqdm import tqdm
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import process_image, TF_IMAGENET
from src.metrics import compute_sc_bsds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BSDS_IMG = os.path.join(ROOT, 'data', 'BSDS500', 'images', 'test')
BSDS_GT = os.path.join(ROOT, 'data', 'BSDS500', 'ground_truth', 'test')
IMGS = sorted(glob.glob(os.path.join(BSDS_IMG, '*.jpg')))[:50]

# Mixed-architecture set (Table 1 in paper)
BACKBONES = {
    'DINO ViT-S/8':     lambda: __import__('torch').hub.load('facebookresearch/dino:main', 'dino_vits8', trust_repo=True),
    'DINOv2 ViT-S/14':  lambda: __import__('timm').create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, dynamic_img_size=True),
    'MoCo-v3 ViT-B/16': 'mocov3',
    'iBOT ViT-S/16':    'ibot',
    'MAE ViT-B/16':     lambda: __import__('timm').create_model('vit_base_patch16_224.mae', pretrained=True, dynamic_img_size=True),
    'CLIP ViT-B/16':    'clip',
}

K = 4


def cluster_kmeans(features, k=K, pca_dim=32):
    pca = PCA(n_components=min(pca_dim, features.shape[1]))
    reduced = normalize(pca.fit_transform(features), norm='l2')
    return KMeans(k, n_init=3, max_iter=100, random_state=SEED).fit_predict(reduced)


def cluster_gmm(features, k=K, pca_dim=32):
    pca = PCA(n_components=min(pca_dim, features.shape[1]))
    reduced = pca.fit_transform(features)
    return GaussianMixture(k, covariance_type='full', random_state=SEED, max_iter=100).fit_predict(reduced)


def cluster_spectral(features, k=K, pca_dim=32, max_pixels=5000):
    pca = PCA(n_components=min(pca_dim, features.shape[1]))
    reduced = normalize(pca.fit_transform(features), norm='l2')
    if len(reduced) > max_pixels:
        idx = np.random.choice(len(reduced), max_pixels, replace=False)
        sub = reduced[idx]
    else:
        sub = reduced
        idx = np.arange(len(reduced))
    try:
        labels_sub = SpectralClustering(k, affinity='rbf', random_state=SEED, n_init=3).fit_predict(sub)
        # Assign remaining points via nearest centroid
        if len(reduced) > max_pixels:
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(n_neighbors=1).fit(sub, labels_sub)
            return knn.predict(reduced)
        return labels_sub
    except Exception:
        return None


def compute_bf1(pred_labels, gt_path, h, w):
    """Boundary F1-score (simplified: boundary pixel overlap)."""
    from scipy.io import loadmat
    from scipy.ndimage import binary_dilation

    gt_data = loadmat(gt_path)
    gt_segs = gt_data['groundTruth'][0]
    best_f1 = 0.0
    pred = pred_labels.reshape(h, w)

    # Pred boundaries
    pred_bnd = np.zeros_like(pred, dtype=bool)
    pred_bnd[:-1, :] |= (pred[:-1, :] != pred[1:, :])
    pred_bnd[:, :-1] |= (pred[:, :-1] != pred[:, 1:])

    for seg_idx in range(len(gt_segs)):
        gt = gt_segs[seg_idx][0][0][0]
        from PIL import Image
        gt_r = np.array(Image.fromarray(gt.astype(np.uint8)).resize((w, h), Image.NEAREST))
        gt_bnd = np.zeros_like(gt_r, dtype=bool)
        gt_bnd[:-1, :] |= (gt_r[:-1, :] != gt_r[1:, :])
        gt_bnd[:, :-1] |= (gt_r[:, :-1] != gt_r[:, 1:])

        # Tolerance: dilate GT boundary by 1 pixel
        gt_bnd_d = binary_dilation(gt_bnd, iterations=1)
        pred_bnd_d = binary_dilation(pred_bnd, iterations=1)

        tp = np.sum(pred_bnd & gt_bnd_d)
        prec = tp / max(np.sum(pred_bnd), 1)
        rec = tp / max(np.sum(gt_bnd), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        best_f1 = max(best_f1, f1)
    return best_f1


def load_backbone(name):
    from src.pipeline import load_dino, load_mocov3, load_mae, load_clip, load_ibot
    import timm

    if name == 'DINO ViT-S/8':
        m = torch.hub.load('facebookresearch/dino:main', 'dino_vits8', trust_repo=True)
        return m.cuda().eval(), TF_IMAGENET, 'dino'
    elif name == 'DINOv2 ViT-S/14':
        m = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, dynamic_img_size=True)
        return m.cuda().eval(), TF_IMAGENET, 'timm'
    elif name == 'MoCo-v3 ViT-B/16':
        return load_mocov3()
    elif name == 'iBOT ViT-S/16':
        return load_ibot()
    elif name == 'MAE ViT-B/16':
        return load_mae()
    elif name == 'CLIP ViT-B/16':
        return load_clip()


def main():
    print(f"Clustering Invariance: {len(IMGS)} BSDS500 images, K={K}, seed={SEED}")
    results = {}

    for name in BACKBONES:
        print(f"\n--- {name} ---")
        model, transform, model_type = load_backbone(name)
        device = next(model.parameters()).device

        sc_km, sc_gmm, sc_sp, bf1_list = [], [], [], []
        for ip in tqdm(IMGS, desc=name, leave=False):
            try:
                raw, side = process_image(ip, model, transform, model_type, device)
                bn = os.path.splitext(os.path.basename(ip))[0]
                gp = os.path.join(BSDS_GT, bn + '.mat')
                if not os.path.exists(gp):
                    continue

                # KMeans
                labels_km = cluster_kmeans(raw).reshape(side, side)
                sc_km.append(compute_sc_bsds(labels_km, gp, side, side))
                bf1_list.append(compute_bf1(labels_km, gp, side, side))

                # GMM
                labels_gmm = cluster_gmm(raw).reshape(side, side)
                sc_gmm.append(compute_sc_bsds(labels_gmm, gp, side, side))

                # Spectral
                labels_sp = cluster_spectral(raw)
                if labels_sp is not None:
                    labels_sp = labels_sp.reshape(side, side)
                    sc_sp.append(compute_sc_bsds(labels_sp, gp, side, side))
            except Exception as e:
                continue

        del model; torch.cuda.empty_cache()
        results[name] = {
            'KMeans': np.mean(sc_km) if sc_km else float('nan'),
            'GMM': np.mean(sc_gmm) if sc_gmm else float('nan'),
            'Spectral': np.mean(sc_sp) if sc_sp else '---',
            'BF1': np.mean(bf1_list) if bf1_list else float('nan'),
        }
        r = results[name]
        sp_str = f"{r['Spectral']:.3f}" if isinstance(r['Spectral'], float) else r['Spectral']
        print(f"  KMeans={r['KMeans']:.3f} GMM={r['GMM']:.3f} Spectral={sp_str} BF1={r['BF1']:.3f}")

    # LaTeX table
    print(f"\n{'='*60}")
    print("TABLE 1 — Clustering Invariance")
    print(f"{'='*60}")
    for name in BACKBONES:
        r = results[name]
        sp = f"{r['Spectral']:.3f}" if isinstance(r['Spectral'], float) else '---'
        print(f"  {name:20s} & {r['KMeans']:.3f} & {r['GMM']:.3f} & {sp} & {r['BF1']:.3f} \\\\")


if __name__ == '__main__':
    main()
