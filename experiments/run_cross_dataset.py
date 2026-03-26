"""Cross-dataset SC table for mixed-architecture set (Table 4).
6 backbones across BSDS500, ADE20K, COCO.
"""
import os, sys, glob, random, numpy as np, torch
from tqdm import tqdm

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import process_image, TF_IMAGENET
from src.metrics import compute_sc_bsds, compute_sc_ade, compute_sc_coco, cluster_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BSDS_IMG = os.path.join(ROOT, 'data', 'BSDS500', 'images', 'test')
BSDS_GT = os.path.join(ROOT, 'data', 'BSDS500', 'ground_truth', 'test')
ADE_IMG = os.path.join(ROOT, 'data', 'ADE20K', 'images', 'validation')
ADE_ANN = os.path.join(ROOT, 'data', 'ADE20K', 'annotations', 'validation')
COCO_IMG = os.path.join(ROOT, 'data', 'COCO', 'val2017')
COCO_ANN = os.path.join(ROOT, 'data', 'COCO', 'annotations', 'instances_val2017.json')

N = 50  # images per dataset

BSDS_IMGS = sorted(glob.glob(os.path.join(BSDS_IMG, '*.jpg')))[:N]
ADE_IMGS = sorted(glob.glob(os.path.join(ADE_IMG, '*.jpg')))[:N]

# Load COCO
from pycocotools.coco import COCO
coco = COCO(COCO_ANN)
COCO_IDS = sorted(coco.getImgIds())[:N]

BACKBONES = {
    'DINO ViT-S/8':     lambda: (torch.hub.load('facebookresearch/dino:main', 'dino_vits8', trust_repo=True).cuda().eval(), TF_IMAGENET, 'dino'),
    'MoCo-v3 ViT-B/16': lambda: __import__('src.pipeline', fromlist=['load_mocov3']).load_mocov3(),
    'DINOv2 ViT-S/14':  lambda: (__import__('timm').create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, dynamic_img_size=True).cuda().eval(), TF_IMAGENET, 'timm'),
    'iBOT ViT-S/16':    lambda: __import__('src.pipeline', fromlist=['load_ibot']).load_ibot(),
    'MAE ViT-B/16':     lambda: __import__('src.pipeline', fromlist=['load_mae']).load_mae(),
    'CLIP ViT-B/16':    lambda: __import__('src.pipeline', fromlist=['load_clip']).load_clip(),
}

print(f"Cross-dataset table: {N} images each, seed={SEED}")

results = {}
for name, loader_fn in BACKBONES.items():
    print(f"\n--- {name} ---")
    model, transform, model_type = loader_fn()
    device = next(model.parameters()).device

    # BSDS500
    sc_bsds = []
    for ip in tqdm(BSDS_IMGS, desc=f'{name}/BSDS', leave=False):
        try:
            raw, side = process_image(ip, model, transform, model_type, device)
            bn = os.path.splitext(os.path.basename(ip))[0]
            gp = os.path.join(BSDS_GT, bn + '.mat')
            if os.path.exists(gp):
                labels = cluster_features(raw, k=4).reshape(side, side)
                sc_bsds.append(compute_sc_bsds(labels, gp, side, side))
        except: continue

    # ADE20K
    sc_ade = []
    for ip in tqdm(ADE_IMGS, desc=f'{name}/ADE', leave=False):
        try:
            raw, side = process_image(ip, model, transform, model_type, device)
            bn = os.path.splitext(os.path.basename(ip))[0]
            ann = os.path.join(ADE_ANN, bn + '.png')
            if os.path.exists(ann):
                labels = cluster_features(raw, k=4).reshape(side, side)
                sc = compute_sc_ade(labels, ann, side, side)
                if not np.isnan(sc): sc_ade.append(sc)
        except: continue

    # COCO
    sc_coco = []
    for img_id in tqdm(COCO_IDS, desc=f'{name}/COCO', leave=False):
        try:
            info = coco.loadImgs(img_id)[0]
            ip = os.path.join(COCO_IMG, info['file_name'])
            if not os.path.exists(ip): continue
            raw, side = process_image(ip, model, transform, model_type, device)
            labels = cluster_features(raw, k=4).reshape(side, side)
            sc = compute_sc_coco(labels, coco, img_id, side, side)
            if not np.isnan(sc): sc_coco.append(sc)
        except: continue

    del model; torch.cuda.empty_cache()
    results[name] = {
        'BSDS': np.mean(sc_bsds), 'ADE': np.mean(sc_ade), 'COCO': np.mean(sc_coco)
    }
    print(f"  BSDS={results[name]['BSDS']:.3f} ADE={results[name]['ADE']:.3f} COCO={results[name]['COCO']:.3f}")

# Table
print(f"\n{'='*60}")
print("TABLE 4 — Cross-Dataset SC (Mixed Architecture)")
print(f"{'='*60}")
print(f"{'Backbone':20s} {'BSDS500':>8s} {'ADE20K':>8s} {'COCO':>8s}")
print("-" * 48)
for name in sorted(results, key=lambda x: -results[x]['BSDS']):
    r = results[name]
    print(f"{name:20s} {r['BSDS']:8.3f} {r['ADE']:8.3f} {r['COCO']:8.3f}")
