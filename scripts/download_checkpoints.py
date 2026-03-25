"""Download model checkpoints that are not available via timm/hub."""
import os
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)

CHECKPOINTS = {
    'ibot_vitb16_teacher.pth': {
        'url': 'https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_rand_mask/checkpoint_teacher.pth',
        'size_mb': 327,
        'description': 'iBOT ViT-B/16 teacher (bytedance, rand_mask)',
    },
    'mocov3_vitb16_300ep.pth.tar': {
        'url': 'https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar',
        'size_mb': 823,
        'description': 'MoCo-v3 ViT-B/16 (300 epochs)',
    },
}


def main():
    for filename, info in CHECKPOINTS.items():
        path = os.path.join(CKPT_DIR, filename)
        if os.path.exists(path):
            print(f"  ✓ {filename} already exists ({info['size_mb']} MB)")
        else:
            print(f"  ↓ Downloading {filename} ({info['size_mb']} MB)...")
            print(f"    {info['description']}")
            try:
                torch.hub.download_url_to_file(info['url'], path)
                print(f"  ✓ Done: {path}")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                print(f"    Manual download: {info['url']}")
                print(f"    Save to: {path}")


if __name__ == '__main__':
    main()
