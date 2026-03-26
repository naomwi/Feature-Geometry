# Feature Geometry Does Not Predict Segmentation Quality

> **Spatial Autocorrelation as a Label-Free Diagnostic for Frozen ViTs**
>
> Thai Quang Tran, Nguyen Tan Khoi Nguyen, Phuoc Tan Phan, Khai Tien Trinh — FPT University

## Key Finding

Eigenspectral metrics (RankMe, n₈₀, LID) **fail** to predict unsupervised segmentation quality. Only **Patch Spatial Autocorrelation (PSA)** — the mean cosine similarity between neighboring patch features — significantly predicts Segmentation Covering across three benchmarks.

| Dataset | N | r | p |
|---------|---|-------|---------|
| BSDS500 | 11 | 0.862 | 0.0007 ✅ |
| COCO | 11 | 0.876 | 0.0004 ✅ |
| ADE20K | 11 | 0.833 | 0.0015 ✅ |

**PSA-weighted (eigenvalue-weighted)**: r=0.964, p<0.00001 on BSDS500.

## Repository Structure

```
├── paper/                    # LaTeX source + figures
│   ├── result.tex
│   └── figures/
├── src/                      # Core library
│   ├── pipeline.py           # Model loading & feature extraction (16 backbones)
│   └── metrics.py            # PSA, SC (BSDS/COCO/ADE20K/VOC), geometry metrics
├── experiments/              # Reproduction scripts
│   ├── run_unified_table2.py # N=11 SSL backbones, all metrics (Table 2)
│   ├── run_unified_ade20k.py # ADE20K cross-dataset (seed=42)
│   ├── run_unified_voc_coco.py # VOC + COCO cross-dataset (seed=42)
│   ├── run_psa_ablation.py   # PSA variants (4-conn, 8-conn, L2, weighted)
│   ├── run_ssl_core.py       # Original N=8 SSL core
│   ├── run_boundary.py       # 16-model boundary analysis
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

# 4. Run unified Table 2 (all 11 backbones, all metrics)
python experiments/run_unified_table2.py

# 5. Run cross-dataset generalization
python experiments/run_unified_ade20k.py
python experiments/run_unified_voc_coco.py

# 6. Run PSA variants ablation
python experiments/run_psa_ablation.py

# 7. Run boundary analysis (Fig. 3, 16 models)
python experiments/run_boundary.py

# 8. Generate figures
python experiments/generate_figures.py
```

## Models (11 SSL Core ViT-B/16)

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
| EVA-02 | `timm: eva02_base_patch16_clip_224.merged2b` | CLIP + MIM |
| OpenCLIP | `timm: vit_base_patch16_clip_224.laion2b` | Contrastive (LAION-2B) |
| MetaCLIP | `timm: vit_base_patch16_clip_224.metaclip_2pt5b` | Contrastive (2.5B) |

## Citation

```bibtex
@inproceedings{tran2025featuregeometry,
  title={Feature Geometry Does Not Predict Segmentation Quality: Spatial Autocorrelation as a Label-Free Diagnostic for Frozen ViTs},
  author={Tran, Thai Quang and Nguyen, Nguyen Tan Khoi and Phan, Phuoc Tan and Trinh, Khai Tien},
  year={2025}
}
```

## License

MIT
