# Feature Geometry Does Not Predict Segmentation Quality

> **Spatial Autocorrelation as a Label-Free Diagnostic for Frozen ViTs**
>
> Thai Quang Tran, Nguyen Tan Khoi Nguyen, Phuoc Tan Phan, Khai Trinh Xuan — FPT University

## Key Finding

Eigenspectral metrics (RankMe, n₈₀, LID) **fail** to predict unsupervised segmentation quality. Only **Patch Spatial Autocorrelation (PSA)** — the mean cosine similarity between neighboring patch features — significantly predicts Segmentation Covering across BSDS500, VOC, and ADE20K.

| Dataset | N | r | p |
|---------|---|-------|---------|
| BSDS500 | 8 | 0.814 | 0.014 ✅ |
| VOC | 8 | 0.943 | 0.0005 ✅ |
| ADE20K | 8 | 0.841 | 0.009 ✅ |

## Repository Structure

```
├── paper/                    # LaTeX source + figures
│   ├── result.tex
│   └── figures/
├── src/                      # Core library
│   ├── pipeline.py           # Model loading & feature extraction
│   └── metrics.py            # PSA, SC, geometry metrics
├── experiments/              # Reproduction scripts
│   ├── run_ssl_core.py       # N=8 SSL backbones on BSDS500
│   ├── run_voc.py            # VOC correlation
│   ├── run_ade20k.py         # ADE20K correlation
│   ├── run_boundary.py       # 13-model boundary analysis
│   └── generate_figures.py   # Scatter plot generation
├── scripts/                  # Setup helpers
│   ├── download_checkpoints.py
│   └── download_data.py
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets
python scripts/download_data.py

# 3. Download model checkpoints (MoCo-v3, iBOT)
python scripts/download_checkpoints.py

# 4. Run core experiment (Table 2)
python experiments/run_ssl_core.py

# 5. Run cross-dataset generalization
python experiments/run_voc.py
python experiments/run_ade20k.py

# 6. Run boundary analysis (Fig. 3)
python experiments/run_boundary.py

# 7. Generate figures
python experiments/generate_figures.py
```

## Models (8 SSL Core ViT-B/16)

| Backbone | Source | Paradigm |
|----------|--------|----------|
| DINO | `torch.hub: facebookresearch/dino` | Self-distillation |
| MoCo-v3 | Local checkpoint (see download script) | Contrastive |
| MAE | `timm: vit_base_patch16_224.mae` | Masked autoencoder |
| CLIP | `openai/clip: ViT-B/16` | Language-Image |
| SigLIP | `timm: vit_base_patch16_siglip_224.webli` | Sigmoid contrastive |
| BEiT | `timm: beit_base_patch16_224.in22k_ft_in22k_in1k` | Masked image tokenizer |
| BEiTv2 | `timm: beitv2_base_patch16_224.in1k_ft_in22k_in1k` | Masked VQ-KD |
| iBOT | Local checkpoint (see download script) | Masked + self-distillation |

## Citation

```bibtex
@inproceedings{tran2025featuregeometry,
  title={Feature Geometry Does Not Predict Segmentation Quality: Spatial Autocorrelation as a Label-Free Diagnostic for Frozen ViTs},
  author={Tran, Thai Quang and Nguyen, Nguyen Tan Khoi and Phan, Phuoc Tan and Xuan, Khai Trinh},
  year={2025}
}
```

## License

MIT
