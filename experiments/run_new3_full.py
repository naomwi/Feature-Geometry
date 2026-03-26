"""Compute ALL metrics for EVA-02, OpenCLIP, MetaCLIP on BSDS500 + VOC."""
import os, sys, glob, math, numpy as np, torch
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import load_eva02, load_openclip, load_metaclip, process_image, extract_patches, TF_IMAGENET
from src.metrics import (compute_psa, compute_sc_bsds, compute_sc_voc,
                         cluster_features, compute_geometry_metrics)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BSDS_IMG = os.path.join(ROOT, 'data', 'BSDS500', 'images', 'test')
BSDS_GT = os.path.join(ROOT, 'data', 'BSDS500', 'ground_truth', 'test')
VOC_IMG = os.path.join(ROOT, 'data', 'VOC2012', 'JPEGImages')
VOC_GT = os.path.join(ROOT, 'data', 'VOC2012', 'SegmentationClass')
VOC_LIST = os.path.join(ROOT, 'data', 'VOC2012', 'ImageSets', 'Segmentation', 'val.txt')


def compute_attn_metrics(model, tensor, model_type):
    """Compute AttnGap and AttnEntropy from last-layer attention."""
    with torch.no_grad():
        if model_type == 'timm':
            # Get attention from last block
            x = model.patch_embed(tensor)
            if hasattr(model, 'cls_token') and model.cls_token is not None:
                x = torch.cat([model.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
            x = x + model.pos_embed
            x = model.pos_drop(x) if hasattr(model, 'pos_drop') else x
            if hasattr(model, 'patch_drop'):
                x = model.patch_drop(x)
            if hasattr(model, 'norm_pre'):
                x = model.norm_pre(x)

            # Run through all blocks except last
            for blk in model.blocks[:-1]:
                x = blk(x)

            # Last block: get attention
            last_blk = model.blocks[-1]
            B, N, C = x.shape
            qkv = last_blk.attn.qkv(last_blk.norm1(x))
            num_heads = last_blk.attn.num_heads
            qkv = qkv.reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            scale = (C // num_heads) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)  # [B, heads, N, N]

            # Average over heads, remove CLS row/col if present
            attn_avg = attn.mean(dim=1).squeeze(0)  # [N, N]
            if attn_avg.shape[0] > 196:  # has CLS
                attn_avg = attn_avg[1:, 1:]  # remove CLS

            # AttnGap: spectral gap of attention matrix
            try:
                eigenvalues = torch.linalg.eigvalsh(attn_avg)
                sorted_eigs = eigenvalues.sort(descending=True).values
                attn_gap = (sorted_eigs[0] - sorted_eigs[1]).item()
            except:
                attn_gap = 0.0

            # AttnEntropy: mean row-wise entropy
            attn_probs = attn_avg / (attn_avg.sum(dim=-1, keepdim=True) + 1e-10)
            row_entropy = -(attn_probs * (attn_probs + 1e-10).log()).sum(dim=-1)
            attn_entropy = row_entropy.mean().item()

            return attn_gap, attn_entropy
    return 0.0, 0.0


def run_backbone(name, loader_fn):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    model, transform, model_type = loader_fn()
    device = next(model.parameters()).device

    # ── BSDS500 ──
    imgs = sorted(glob.glob(os.path.join(BSDS_IMG, '*.jpg')))[:50]
    psa_list, sc_list = [], []
    geo = {'n80': [], 'lid': [], 'nesum': [], 'stable_rank': []}
    attn_gap_list, attn_ent_list = [], []

    for ip in tqdm(imgs, desc=f'{name}/BSDS', leave=False):
        try:
            raw, side = process_image(ip, model, transform, model_type, device)
            psa_list.append(compute_psa(raw, side, side))

            g = compute_geometry_metrics(raw)
            for k in geo:
                geo[k].append(g[k])

            bn = os.path.splitext(os.path.basename(ip))[0]
            gp = os.path.join(BSDS_GT, bn + '.mat')
            if os.path.exists(gp):
                labels = cluster_features(raw, k=4).reshape(side, side)
                sc_list.append(compute_sc_bsds(labels, gp, side, side))

            # Attention metrics
            img = Image.open(ip).convert('RGB').resize((224, 224), Image.Resampling.LANCZOS)
            tensor = transform(img).unsqueeze(0).to(device)
            ag, ae = compute_attn_metrics(model, tensor, model_type)
            attn_gap_list.append(ag)
            attn_ent_list.append(ae)
        except Exception as e:
            continue

    # ── VOC ──
    with open(VOC_LIST) as f:
        val_ids = [l.strip() for l in f if l.strip()][:30]
    sc_voc_list = []
    for sid in tqdm(val_ids, desc=f'{name}/VOC', leave=False):
        try:
            ip = os.path.join(VOC_IMG, sid + '.jpg')
            gp = os.path.join(VOC_GT, sid + '.png')
            if not (os.path.exists(ip) and os.path.exists(gp)):
                continue
            raw, side = process_image(ip, model, transform, model_type, device)
            labels = cluster_features(raw, k=4).reshape(side, side)
            sc = compute_sc_voc(labels, gp, side, side)
            if not np.isnan(sc):
                sc_voc_list.append(sc)
        except:
            continue

    del model; torch.cuda.empty_cache()

    result = {
        'SC_BSDS': np.mean(sc_list),
        'SC_VOC': np.mean(sc_voc_list) if sc_voc_list else float('nan'),
        'n80': np.mean(geo['n80']),
        'LID': np.mean(geo['lid']),
        'NESum': np.mean(geo['nesum']),
        'SR': np.mean(geo['stable_rank']),
        'PSA': np.mean(psa_list),
        'AttnGap': np.mean(attn_gap_list),
        'AttnEnt': np.mean(attn_ent_list),
    }

    print(f"  SC(BSDS)={result['SC_BSDS']:.3f}  SC(VOC)={result['SC_VOC']:.3f}")
    print(f"  n80={result['n80']:.1f}  LID={result['LID']:.2f}  NESum={result['NESum']:.1f}  SR={result['SR']:.2f}")
    print(f"  PSA={result['PSA']:.3f}  AttnGap={result['AttnGap']:.3f}  AttnEnt={result['AttnEnt']:.2f}")
    return result


def main():
    backbones = {
        'EVA-02':   load_eva02,
        'OpenCLIP': load_openclip,
        'MetaCLIP': load_metaclip,
    }

    results = {}
    for name, loader_fn in backbones.items():
        results[name] = run_backbone(name, loader_fn)

    # Final summary table
    print(f"\n\n{'='*80}")
    print("FINAL TABLE DATA FOR 3 NEW BACKBONES")
    print(f"{'='*80}")
    print(f"{'Backbone':10s} {'SC_BSDS':>8s} {'SC_VOC':>8s} {'n80':>6s} {'LID':>6s} {'NESum':>6s} {'SR':>6s} {'PSA':>6s} {'AttnGap':>8s} {'AttnEnt':>8s}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:10s} {r['SC_BSDS']:8.3f} {r['SC_VOC']:8.3f} {r['n80']:6.1f} {r['LID']:6.2f} {r['NESum']:6.1f} {r['SR']:6.2f} {r['PSA']:6.3f} {r['AttnGap']:8.3f} {r['AttnEnt']:8.2f}")


if __name__ == '__main__':
    main()
