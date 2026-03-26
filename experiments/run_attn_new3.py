"""Extract AttnGap/AttnEntropy by hooking q_proj or qkv in last attention block."""
import torch, timm, numpy as np, glob, os
from PIL import Image
import torchvision.transforms as T

tf = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
imgs = sorted(glob.glob(os.path.join(ROOT, 'data', 'BSDS500', 'images', 'test', '*.jpg')))[:20]

KNOWN = {
    'EVA-02':   {'SC_BSDS': 0.376, 'SC_VOC': 0.405, 'n80': 85.3, 'LID': 6.80, 'NESum': 12.9, 'SR': 2.86, 'PSA': 0.479},
    'OpenCLIP': {'SC_BSDS': 0.499, 'SC_VOC': 0.397, 'n80': 98.9, 'LID': 9.50, 'NESum': 12.3, 'SR': 2.40, 'PSA': 0.596},
    'MetaCLIP': {'SC_BSDS': 0.497, 'SC_VOC': 0.425, 'n80': 102.6, 'LID': 10.65, 'NESum': 10.2, 'SR': 1.84, 'PSA': 0.672},
}


def compute_attn_from_hook(model, tensor):
    """Hook q_proj + k_proj outputs (or qkv) to compute attention weights."""
    cap = {}
    last_attn = model.blocks[-1].attn
    nh = last_attn.num_heads
    handles = []

    if last_attn.qkv is not None:
        # Standard timm ViT: single qkv linear
        def hook_qkv(mod, inp, out):
            cap['qkv'] = out.detach()
        handles.append(last_attn.qkv.register_forward_hook(hook_qkv))
    else:
        # EVA-02 style: separate q_proj, k_proj
        def hook_q(mod, inp, out):
            cap['q'] = out.detach()
        def hook_k(mod, inp, out):
            cap['k'] = out.detach()
        handles.append(last_attn.q_proj.register_forward_hook(hook_q))
        handles.append(last_attn.k_proj.register_forward_hook(hook_k))

    with torch.no_grad():
        model.forward_features(tensor)

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


MODELS = [
    ('EVA-02',   'eva02_base_patch16_clip_224.merged2b'),
    ('OpenCLIP', 'vit_base_patch16_clip_224.laion2b'),
    ('MetaCLIP', 'vit_base_patch16_clip_224.metaclip_2pt5b'),
]

for name, mn in MODELS:
    print(f"\n--- {name} ---")
    model = timm.create_model(mn, pretrained=True, dynamic_img_size=True).cuda().eval()

    gaps, ents = [], []
    for ip in imgs:
        img = Image.open(ip).convert('RGB').resize((224, 224))
        tensor = tf(img).unsqueeze(0).cuda()
        g, e = compute_attn_from_hook(model, tensor)
        if not np.isnan(g):
            gaps.append(g)
            ents.append(e)

    k = KNOWN[name]
    ag = np.mean(gaps) if gaps else float('nan')
    ae = np.mean(ents) if ents else float('nan')
    print(f"  SC(BSDS)={k['SC_BSDS']:.3f}  SC(VOC)={k['SC_VOC']:.3f}")
    print(f"  n80={k['n80']:.1f}  LID={k['LID']:.2f}  NESum={k['NESum']:.1f}  SR={k['SR']:.2f}")
    print(f"  PSA={k['PSA']:.3f}  AttnGap={ag:.3f}  AttnEnt={ae:.2f}")

    del model; torch.cuda.empty_cache()
