"""K ablation: DINO ViT-S/8 on BSDS500, SC + BF1 for K=2,4,6,10."""
import os, sys, glob, random, numpy as np, torch
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_dilation

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import process_image, TF_IMAGENET
from src.metrics import compute_sc_bsds, cluster_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BSDS_IMG = os.path.join(ROOT, 'data', 'BSDS500', 'images', 'test')
BSDS_GT = os.path.join(ROOT, 'data', 'BSDS500', 'ground_truth', 'test')
IMGS = sorted(glob.glob(os.path.join(BSDS_IMG, '*.jpg')))[:50]


def compute_bf1(pred_labels, gt_path, h, w):
    from scipy.io import loadmat
    gt_data = loadmat(gt_path)
    gt_segs = gt_data['groundTruth'][0]
    best_f1 = 0.0
    pred = pred_labels.reshape(h, w)
    pred_bnd = np.zeros_like(pred, dtype=bool)
    pred_bnd[:-1, :] |= (pred[:-1, :] != pred[1:, :])
    pred_bnd[:, :-1] |= (pred[:, :-1] != pred[:, 1:])
    for seg_idx in range(len(gt_segs)):
        gt = gt_segs[seg_idx][0][0][0]
        gt_r = np.array(Image.fromarray(gt.astype(np.uint8)).resize((w, h), Image.NEAREST))
        gt_bnd = np.zeros_like(gt_r, dtype=bool)
        gt_bnd[:-1, :] |= (gt_r[:-1, :] != gt_r[1:, :])
        gt_bnd[:, :-1] |= (gt_r[:, :-1] != gt_r[:, 1:])
        gt_bnd_d = binary_dilation(gt_bnd, iterations=1)
        tp = np.sum(pred_bnd & gt_bnd_d)
        prec = tp / max(np.sum(pred_bnd), 1)
        rec = tp / max(np.sum(gt_bnd), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        best_f1 = max(best_f1, f1)
    return best_f1


# Load DINO ViT-S/8
print("Loading DINO ViT-S/8...")
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8', trust_repo=True).cuda().eval()
transform = TF_IMAGENET
model_type = 'dino'

# Extract features once
print(f"Extracting features from {len(IMGS)} BSDS500 images...")
all_feats = []
for ip in tqdm(IMGS, desc='Extract'):
    raw, side = process_image(ip, model, transform, model_type, 'cuda')
    bn = os.path.splitext(os.path.basename(ip))[0]
    gp = os.path.join(BSDS_GT, bn + '.mat')
    all_feats.append((raw, side, gp))

del model; torch.cuda.empty_cache()

# Run K ablation
print(f"\nK Ablation (DINO ViT-S/8, BSDS500, seed={SEED})")
print(f"{'K':>4s}  {'SC':>6s}  {'BF1':>6s}")
print("-" * 20)

for K in [2, 4, 6, 10]:
    sc_list, bf1_list = [], []
    for raw, side, gp in all_feats:
        if not os.path.exists(gp):
            continue
        labels = cluster_features(raw, k=K).reshape(side, side)
        sc_list.append(compute_sc_bsds(labels, gp, side, side))
        bf1_list.append(compute_bf1(labels, gp, side, side))
    sc = np.mean(sc_list)
    bf1 = np.mean(bf1_list)
    print(f"  {K:2d}   {sc:.3f}   {bf1:.3f}")
