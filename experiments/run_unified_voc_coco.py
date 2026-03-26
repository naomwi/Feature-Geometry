"""SC on VOC + COCO for all 11 SSL ViT-B/16 — fixed seed=42."""
import os, sys, glob, random, json, numpy as np, torch
from PIL import Image
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import SSL_CORE, process_image
from src.metrics import compute_psa, compute_sc_voc, cluster_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# VOC paths
VOC_IMG = os.path.join(ROOT, 'data', 'VOC2012', 'JPEGImages')
VOC_GT  = os.path.join(ROOT, 'data', 'VOC2012', 'SegmentationClass')
VOC_LIST = os.path.join(ROOT, 'data', 'VOC2012', 'ImageSets', 'Segmentation', 'val.txt')

# COCO paths
COCO_IMG = os.path.join(ROOT, 'data', 'COCO', 'val2017')
COCO_ANN = os.path.join(ROOT, 'data', 'COCO', 'annotations', 'instances_val2017.json')

N_VOC = 50
N_COCO = 50

# Load VOC IDs
with open(VOC_LIST) as f:
    VOC_IDS = [l.strip() for l in f if l.strip()][:N_VOC]

# Load COCO annotation
print("Loading COCO annotations...")
from pycocotools.coco import COCO
coco = COCO(COCO_ANN)
coco_img_ids = sorted(coco.getImgIds())[:N_COCO]
print(f"VOC: {len(VOC_IDS)} images | COCO: {len(coco_img_ids)} images | seed={SEED}")

# PSA from unified BSDS run
PSA_BSDS = {
    'iBOT': 0.713, 'MoCo-v3': 0.861, 'DINO': 0.625, 'MAE': 0.731,
    'OpenCLIP': 0.596, 'MetaCLIP': 0.672, 'CLIP': 0.702,
    'SigLIP': 0.449, 'BEiT': 0.515, 'EVA-02': 0.479, 'BEiTv2': 0.449,
}


def compute_sc_coco(pred_labels, img_id, h, w):
    """Segmentation Covering using COCO instance masks."""
    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    if len(anns) < 2:
        return float('nan')

    img_info = coco.loadImgs(img_id)[0]
    oh, ow = img_info['height'], img_info['width']

    # Build GT label map from instance masks
    gt = np.zeros((oh, ow), dtype=np.int32)
    for i, ann in enumerate(anns):
        mask = coco.annToMask(ann)
        gt[mask > 0] = i + 1

    # Resize GT to patch grid
    gt_resized = np.array(Image.fromarray(gt.astype(np.uint8)).resize(
        (w, h), Image.NEAREST))

    gt_flat = gt_resized.flatten()
    pred_flat = pred_labels.flatten()

    # Segmentation covering
    unique_gt = [s for s in np.unique(gt_flat) if s > 0]
    if len(unique_gt) < 2:
        return float('nan')

    unique_pred = np.unique(pred_flat)
    total_pixels = np.sum(gt_flat > 0)
    if total_pixels == 0:
        return float('nan')

    sc = 0.0
    for s in unique_gt:
        gt_mask = (gt_flat == s)
        best_iou = 0.0
        for r in unique_pred:
            pred_mask = (pred_flat == r)
            intersection = np.sum(gt_mask & pred_mask)
            union = np.sum(gt_mask | pred_mask)
            if union > 0:
                best_iou = max(best_iou, intersection / union)
        sc += np.sum(gt_mask) * best_iou
    return sc / total_pixels


def run_backbone(name, loader_fn):
    print(f"\n--- {name} ---")
    model, transform, model_type = loader_fn()
    device = next(model.parameters()).device

    # VOC
    sc_voc = []
    for sid in tqdm(VOC_IDS, desc=f'{name}/VOC', leave=False):
        try:
            ip = os.path.join(VOC_IMG, sid + '.jpg')
            gp = os.path.join(VOC_GT, sid + '.png')
            if not (os.path.exists(ip) and os.path.exists(gp)):
                continue
            raw, side = process_image(ip, model, transform, model_type, device)
            labels = cluster_features(raw, k=4).reshape(side, side)
            sc = compute_sc_voc(labels, gp, side, side)
            if not np.isnan(sc):
                sc_voc.append(sc)
        except:
            continue

    # COCO
    sc_coco = []
    for img_id in tqdm(coco_img_ids, desc=f'{name}/COCO', leave=False):
        try:
            img_info = coco.loadImgs(img_id)[0]
            ip = os.path.join(COCO_IMG, img_info['file_name'])
            if not os.path.exists(ip):
                continue
            raw, side = process_image(ip, model, transform, model_type, device)
            labels = cluster_features(raw, k=4).reshape(side, side)
            sc = compute_sc_coco(labels, img_id, side, side)
            if not np.isnan(sc):
                sc_coco.append(sc)
        except:
            continue

    del model; torch.cuda.empty_cache()

    voc_mean = np.mean(sc_voc) if sc_voc else float('nan')
    coco_mean = np.mean(sc_coco) if sc_coco else float('nan')
    print(f"  SC(VOC)={voc_mean:.3f} (N={len(sc_voc)})  SC(COCO)={coco_mean:.3f} (N={len(sc_coco)})")
    return {'SC_VOC': voc_mean, 'SC_COCO': coco_mean}


results = {}
for name, loader_fn in SSL_CORE.items():
    results[name] = run_backbone(name, loader_fn)

# Summary
print(f"\n\n{'='*70}")
print("SC(VOC) and SC(COCO) for all 11 SSL backbones")
print(f"{'='*70}")
print(f"{'Backbone':10s} {'SC(VOC)':>8s} {'SC(COCO)':>9s} {'PSA(BSDS)':>10s}")
print("-" * 40)
for n in sorted(results, key=lambda x: -results[x].get('SC_VOC', 0)):
    r = results[n]
    print(f"{n:10s} {r['SC_VOC']:8.3f} {r['SC_COCO']:9.3f} {PSA_BSDS[n]:10.3f}")

# Correlations
names = list(results.keys())
psa = [PSA_BSDS[n] for n in names]

for ds in ['SC_VOC', 'SC_COCO']:
    vals = [results[n][ds] for n in names]
    valid = [(p, v) for p, v in zip(psa, vals) if not np.isnan(v)]
    if len(valid) >= 5:
        pp, vv = zip(*valid)
        r, p = pearsonr(pp, vv)
        rho, sp = spearmanr(pp, vv)
        print(f"\nPSA(BSDS) vs {ds}: r={r:.4f} p={p:.6f} | rho={rho:.4f} p={sp:.6f} | N={len(valid)}")
