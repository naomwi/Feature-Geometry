"""
UNIFIED TABLE 2 BENCHMARK — All 11 SSL ViT-B/16 backbones
Fixed seed=42, fixed 50 BSDS500 images, fixed 30 VOC images.
Computes: SC(BSDS), SC(VOC), n80, LID, NESum, SR, PSA, AttnGap, AttnEntropy.
Final Pearson/Spearman correlations at the end.
"""
import os, sys, glob, math, random, numpy as np, torch
from PIL import Image
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

# ── Fixed seed ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import SSL_CORE, process_image, TF_IMAGENET
from src.metrics import compute_psa, compute_sc_bsds, compute_sc_voc, cluster_features, compute_geometry_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BSDS_IMG = os.path.join(ROOT, 'data', 'BSDS500', 'images', 'test')
BSDS_GT = os.path.join(ROOT, 'data', 'BSDS500', 'ground_truth', 'test')
VOC_IMG = os.path.join(ROOT, 'data', 'VOC2012', 'JPEGImages')
VOC_GT = os.path.join(ROOT, 'data', 'VOC2012', 'SegmentationClass')
VOC_LIST = os.path.join(ROOT, 'data', 'VOC2012', 'ImageSets', 'Segmentation', 'val.txt')

N_BSDS = 50
N_VOC = 30

# ── Fixed image lists ──
BSDS_IMGS = sorted(glob.glob(os.path.join(BSDS_IMG, '*.jpg')))[:N_BSDS]
with open(VOC_LIST) as f:
    VOC_IDS = [l.strip() for l in f if l.strip()][:N_VOC]

import torchvision.transforms as T
tf = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])


def compute_attn_hook(model, tensor):
    """Extract AttnGap + AttnEntropy via forward hook on last block."""
    cap = {}
    last_attn = model.blocks[-1].attn
    nh = last_attn.num_heads
    handles = []

    if last_attn.qkv is not None:
        def hook_qkv(mod, inp, out):
            cap['qkv'] = out.detach()
        handles.append(last_attn.qkv.register_forward_hook(hook_qkv))
    elif hasattr(last_attn, 'q_proj') and last_attn.q_proj is not None:
        def hook_q(mod, inp, out):
            cap['q'] = out.detach()
        def hook_k(mod, inp, out):
            cap['k'] = out.detach()
        handles.append(last_attn.q_proj.register_forward_hook(hook_q))
        handles.append(last_attn.k_proj.register_forward_hook(hook_k))
    else:
        return float('nan'), float('nan')

    with torch.no_grad():
        try:
            model.forward_features(tensor)
        except:
            for h in handles:
                h.remove()
            return float('nan'), float('nan')

    for h in handles:
        h.remove()

    if 'qkv' in cap:
        qkv = cap['qkv']
        B, N, C3 = qkv.shape
        hd = C3 // 3 // nh
        qkv = qkv.reshape(B, N, 3, nh, hd).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
    elif 'q' in cap and 'k' in cap:
        B, N, C = cap['q'].shape
        hd = C // nh
        q = cap['q'].reshape(B, N, nh, hd).permute(0, 2, 1, 3)
        k = cap['k'].reshape(B, N, nh, hd).permute(0, 2, 1, 3)
    else:
        return float('nan'), float('nan')

    attn = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    attn = attn.softmax(dim=-1).mean(dim=1).squeeze(0)
    if attn.shape[0] > 196:
        attn = attn[1:, 1:]

    eigs = torch.linalg.eigvalsh(attn).sort(descending=True).values
    gap = (eigs[0] - eigs[1]).item()

    probs = attn / (attn.sum(-1, keepdim=True) + 1e-10)
    ent = -(probs * (probs + 1e-10).log()).sum(-1).mean().item()
    return gap, ent


def run_backbone(name, loader_fn):
    """Run ALL metrics for one backbone."""
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    model, transform, model_type = loader_fn()
    device = next(model.parameters()).device

    psa_list, sc_bsds_list = [], []
    geo = {'n80': [], 'lid': [], 'nesum': [], 'stable_rank': []}
    attn_gap_list, attn_ent_list = [], []

    # ── BSDS500 ──
    for ip in tqdm(BSDS_IMGS, desc=f'{name}/BSDS', leave=False):
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
                sc_bsds_list.append(compute_sc_bsds(labels, gp, side, side))

            # Attention
            img = Image.open(ip).convert('RGB').resize((224, 224))
            tensor = tf(img).unsqueeze(0).to(device)
            if model_type in ('timm', 'timm_no_cls'):
                ag, ae = compute_attn_hook(model, tensor)
                if not np.isnan(ag):
                    attn_gap_list.append(ag)
                    attn_ent_list.append(ae)
        except Exception as e:
            continue

    # ── VOC ──
    sc_voc_list = []
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
                sc_voc_list.append(sc)
        except:
            continue

    del model; torch.cuda.empty_cache()

    result = {
        'SC_BSDS': np.mean(sc_bsds_list) if sc_bsds_list else float('nan'),
        'SC_VOC': np.mean(sc_voc_list) if sc_voc_list else float('nan'),
        'n80': np.mean(geo['n80']) if geo['n80'] else float('nan'),
        'LID': np.mean(geo['lid']) if geo['lid'] else float('nan'),
        'NESum': np.mean(geo['nesum']) if geo['nesum'] else float('nan'),
        'SR': np.mean(geo['stable_rank']) if geo['stable_rank'] else float('nan'),
        'PSA': np.mean(psa_list) if psa_list else float('nan'),
        'AttnGap': np.mean(attn_gap_list) if attn_gap_list else float('nan'),
        'AttnEnt': np.mean(attn_ent_list) if attn_ent_list else float('nan'),
    }
    print(f"  SC(B)={result['SC_BSDS']:.3f} SC(V)={result['SC_VOC']:.3f} "
          f"n80={result['n80']:.1f} LID={result['LID']:.2f} NE={result['NESum']:.1f} SR={result['SR']:.2f} "
          f"PSA={result['PSA']:.3f} AG={result['AttnGap']:.3f} AE={result['AttnEnt']:.2f}")
    return result


def main():
    print(f"SEED={SEED}  BSDS={N_BSDS} images  VOC={N_VOC} images")
    print(f"BSDS path: {BSDS_IMG} ({len(BSDS_IMGS)} found)")
    print(f"VOC path: {VOC_LIST} ({len(VOC_IDS)} IDs)")

    results = {}
    for name, loader_fn in SSL_CORE.items():
        results[name] = run_backbone(name, loader_fn)

    # ── Final Table ──
    print(f"\n\n{'='*100}")
    print(f"TABLE 2 — UNIFIED (N={len(results)}, seed={SEED}, {N_BSDS} BSDS imgs, {N_VOC} VOC imgs)")
    print(f"{'='*100}")
    header = f"{'Backbone':10s} {'SC_BSDS':>8s} {'SC_VOC':>8s} {'n80':>6s} {'LID':>6s} {'NESum':>6s} {'SR':>6s} {'PSA':>6s} {'AttnGap':>8s} {'AttnEnt':>8s}"
    print(header)
    print("-" * len(header))
    for n in sorted(results, key=lambda x: -results[x]['SC_BSDS']):
        r = results[n]
        ag = f"{r['AttnGap']:.3f}" if not np.isnan(r['AttnGap']) else '  ---'
        ae = f"{r['AttnEnt']:.2f}" if not np.isnan(r['AttnEnt']) else '  ---'
        print(f"{n:10s} {r['SC_BSDS']:8.3f} {r['SC_VOC']:8.3f} {r['n80']:6.1f} {r['LID']:6.2f} {r['NESum']:6.1f} {r['SR']:6.2f} {r['PSA']:6.3f} {ag:>8s} {ae:>8s}")

    # ── Correlations ──
    print(f"\n{'='*60}")
    print("CORRELATIONS (PSA vs SC_BSDS)")
    print(f"{'='*60}")
    names = list(results.keys())
    psa = [results[n]['PSA'] for n in names]
    sc = [results[n]['SC_BSDS'] for n in names]
    r_p, p_p = pearsonr(psa, sc)
    r_s, p_s = spearmanr(psa, sc)
    print(f"  Pearson  r={r_p:.4f}, p={p_p:.6f}")
    print(f"  Spearman ρ={r_s:.4f}, p={p_s:.6f}")

    # Geometry correlations
    for metric in ['n80', 'LID', 'NESum', 'SR', 'AttnGap', 'AttnEnt']:
        vals = [results[n][metric] for n in names]
        valid = [(v, s) for v, s in zip(vals, sc) if not np.isnan(v)]
        if len(valid) >= 5:
            vv, ss = zip(*valid)
            r_g, p_g = pearsonr(vv, ss)
            print(f"  {metric:8s} vs SC: r={r_g:.3f}, p={p_g:.4f}, N={len(valid)}")

    # PSA vs SC_VOC
    sc_voc = [results[n]['SC_VOC'] for n in names]
    valid_voc = [(p, s) for p, s in zip(psa, sc_voc) if not np.isnan(s)]
    if valid_voc:
        pv, sv = zip(*valid_voc)
        r_v, p_v = pearsonr(pv, sv)
        print(f"\n  PSA vs SC_VOC: r={r_v:.4f}, p={p_v:.6f}, N={len(valid_voc)}")

    # LaTeX rows
    print(f"\n{'='*60}")
    print("LATEX TABLE ROWS")
    print(f"{'='*60}")
    for n in sorted(results, key=lambda x: -results[x]['SC_BSDS']):
        r = results[n]
        ag = f"{r['AttnGap']:.3f}" if not np.isnan(r['AttnGap']) else '---'
        ae = f"{r['AttnEnt']:.2f}" if not np.isnan(r['AttnEnt']) else '---'
        print(f"{n:10s} & {r['SC_BSDS']:.3f} & {r['SC_VOC']:.3f} & {r['n80']:.1f} & {r['LID']:.2f} & {r['NESum']:.1f} & {r['SR']:.2f} && {r['PSA']:.3f} & {ag} & {ae} \\\\")


if __name__ == '__main__':
    main()
