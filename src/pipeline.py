"""Core pipeline: model loading and feature extraction for all 13 ViT-B/16 backbones."""
import os
import math
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, 'checkpoints')

# ── Transforms ──
TF_IMAGENET = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
TF_CLIP = T.Compose([
    T.ToTensor(),
    T.Normalize([0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711]),
])


# ═══════════════════════════════════════════════════════
# Model Loaders — one function per backbone
# ═══════════════════════════════════════════════════════

def load_dino(device='cuda'):
    """DINO ViT-B/16 — self-distillation (official facebookresearch/dino)."""
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', trust_repo=True)
    return model.to(device).eval(), TF_IMAGENET, 'dino'


def load_mocov3(device='cuda'):
    """MoCo-v3 ViT-B/16 — contrastive (local checkpoint)."""
    import timm
    ckpt_path = os.path.join(CKPT_DIR, 'mocov3_vitb16_300ep.pth.tar')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"MoCo-v3 checkpoint not found at {ckpt_path}. "
                                "Run: python scripts/download_checkpoints.py")
    model = timm.create_model('vit_base_patch16_224', pretrained=False, dynamic_img_size=True)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    cleaned = {}
    for k, v in sd.items():
        if 'momentum' in k:
            continue
        nk = k
        for prefix in ['module.base_encoder.', 'module.', 'base_encoder.']:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
                break
        if nk.startswith('head.') or nk.startswith('predictor.'):
            continue
        cleaned[nk] = v
    model.load_state_dict(cleaned, strict=False)
    return model.to(device).eval(), TF_IMAGENET, 'timm'


def load_mae(device='cuda'):
    """MAE ViT-B/16 — masked autoencoder (timm)."""
    import timm
    model = timm.create_model('vit_base_patch16_224.mae', pretrained=True, dynamic_img_size=True)
    return model.to(device).eval(), TF_IMAGENET, 'timm'


def load_clip(device='cuda'):
    """CLIP ViT-B/16 — language-image contrastive (OpenAI)."""
    import clip
    m, _ = clip.load("ViT-B/16", device=device)
    model = m.visual.float().eval()
    return model, TF_CLIP, 'clip'


def load_siglip(device='cuda'):
    """SigLIP ViT-B/16 — sigmoid contrastive (timm)."""
    import timm
    model = timm.create_model('vit_base_patch16_siglip_224.webli', pretrained=True, dynamic_img_size=True)
    return model.to(device).eval(), TF_IMAGENET, 'timm_no_cls'


def load_beit(device='cuda'):
    """BEiT ViT-B/16 — masked image tokenizer (timm)."""
    import timm
    model = timm.create_model('beit_base_patch16_224.in22k_ft_in22k_in1k', pretrained=True)
    return model.to(device).eval(), TF_IMAGENET, 'timm'


def load_beitv2(device='cuda'):
    """BEiTv2 ViT-B/16 — masked VQ-KD (timm)."""
    import timm
    model = timm.create_model('beitv2_base_patch16_224.in1k_ft_in22k_in1k', pretrained=True)
    return model.to(device).eval(), TF_IMAGENET, 'timm'


def load_ibot(device='cuda'):
    """iBOT ViT-B/16 — masked + self-distillation (local checkpoint)."""
    import timm
    ckpt_path = os.path.join(CKPT_DIR, 'ibot_vitb16_teacher.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"iBOT checkpoint not found at {ckpt_path}. "
                                "Run: python scripts/download_checkpoints.py")
    model = timm.create_model('vit_base_patch16_224', pretrained=False, dynamic_img_size=True)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    cleaned = {k.replace('module.', '').replace('backbone.', ''): v
               for k, v in sd.items()
               if 'head' not in k and 'predictor' not in k}
    model.load_state_dict(cleaned, strict=False)
    return model.to(device).eval(), TF_IMAGENET, 'timm'


# ── Additional backbones (boundary analysis) ──

def load_data2vec(device='cuda'):
    """Data2Vec ViT-B/16 — contextualized reconstruction (HuggingFace)."""
    from transformers import Data2VecVisionModel

    class D2VWrapper(nn.Module):
        def __init__(self, hf_model):
            super().__init__()
            self.model = hf_model

        def forward_features(self, x):
            return self.model(x, return_dict=True).last_hidden_state

    hf_model = Data2VecVisionModel.from_pretrained('facebook/data2vec-vision-base-ft1k')
    return D2VWrapper(hf_model).to(device).eval(), TF_IMAGENET, 'timm'


def load_deit(device='cuda'):
    """DeiT ViT-B/16 — supervised distillation (timm)."""
    import timm
    model = timm.create_model('deit_base_patch16_224.fb_in1k', pretrained=True, dynamic_img_size=True)
    return model.to(device).eval(), TF_IMAGENET, 'timm'


def load_deit3(device='cuda'):
    """DeiT-III ViT-B/16 — supervised strong augmentation (timm)."""
    import timm
    model = timm.create_model('deit3_base_patch16_224.fb_in22k_ft_in1k', pretrained=True, dynamic_img_size=True)
    return model.to(device).eval(), TF_IMAGENET, 'timm'


def load_beit3(device='cuda'):
    """BEiT3 ViT-B/16 — multimodal masked language model (timm)."""
    import timm
    model = timm.create_model('beit3_base_patch16_224.in22k_ft_in1k', pretrained=True, dynamic_img_size=True)
    return model.to(device).eval(), TF_IMAGENET, 'timm'


def load_sam(device='cuda'):
    """SAM ViT-B/16 — segment anything (timm)."""
    import timm
    model = timm.create_model('vit_base_patch16_224.sam_in1k', pretrained=True, dynamic_img_size=True)
    return model.to(device).eval(), TF_IMAGENET, 'timm'


# ═══════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════

SSL_CORE = {
    'DINO':    load_dino,
    'MoCo-v3': load_mocov3,
    'MAE':     load_mae,
    'CLIP':    load_clip,
    'SigLIP':  load_siglip,
    'BEiT':    load_beit,
    'BEiTv2':  load_beitv2,
    'iBOT':    load_ibot,
}

BOUNDARY = {
    'Data2Vec': load_data2vec,
    'DeiT':     load_deit,
    'DeiT-III': load_deit3,
    'BEiT3':    load_beit3,
    'SAM':      load_sam,
}

ALL_BACKBONES = {**SSL_CORE, **BOUNDARY}


# ═══════════════════════════════════════════════════════
# Feature Extraction
# ═══════════════════════════════════════════════════════

def extract_patches(model, tensor, model_type='timm'):
    """Extract patch tokens (excluding CLS) from a ViT forward pass.

    Args:
        model: loaded ViT model
        tensor: preprocessed image tensor [1, 3, 224, 224]
        model_type: 'dino' | 'timm' | 'timm_no_cls' | 'clip'

    Returns:
        patches: [1, N, D] patch features (N = 14*14 = 196 for patch16)
    """
    with torch.no_grad():
        if model_type == 'dino':
            feat = model.get_intermediate_layers(tensor, n=1)[0]
            return feat[:, 1:, :]  # remove CLS

        elif model_type == 'clip':
            x = model.conv1(tensor)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls_emb = model.class_embedding.to(x.dtype)
            x = torch.cat([cls_emb + torch.zeros(x.shape[0], 1, x.shape[-1],
                           dtype=x.dtype, device=x.device), x], dim=1)
            x = x + model.positional_embedding.to(x.dtype)
            x = model.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = model.transformer(x)
            x = x.permute(1, 0, 2)
            return x[:, 1:, :]

        elif model_type == 'timm_no_cls':
            # SigLIP: no CLS token
            return model.forward_features(tensor)

        else:  # 'timm'
            feat = model.forward_features(tensor)
            expected = 196  # 14*14
            if feat.shape[1] > expected:
                return feat[:, 1:, :]  # remove CLS
            return feat


def process_image(image_path, model, transform, model_type, device='cuda'):
    """Load image, extract normalized patch features.

    Returns:
        raw: [N, D] L2-normalized features
        side: spatial grid side length (14 for ViT-B/16 at 224px)
    """
    img = Image.open(image_path).convert('RGB').resize((224, 224), Image.Resampling.LANCZOS)
    tensor = transform(img).unsqueeze(0).to(device)
    patches = extract_patches(model, tensor, model_type)
    n = patches.shape[1]
    side = int(math.sqrt(n))
    if side * side != n:
        raise ValueError(f"Non-square patch grid: {n} patches")
    raw = normalize(patches.squeeze(0).float().cpu().numpy(), norm='l2', axis=1)
    return raw, side
