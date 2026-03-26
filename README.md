# Feature Geometry Does Not Predict Segmentation Quality

> **Spatial Autocorrelation as a Label-Free Diagnostic for Frozen ViTs**
>
> Thai Quang Tran, Nguyen Tan Khoi Nguyen, Phuoc Tan Phan, Khai Tien Trinh — FPT University

## Key Finding

Eigenspectral metrics (RankMe, n₈₀, LID) **fail** to predict unsupervised segmentation quality. Only **Patch Spatial Autocorrelation (PSA)** — the mean cosine similarity between neighboring patch features — significantly predicts Segmentation Covering across three benchmarks.

| Dataset | N | Pearson r | p-value | Significant? |
|---------|---|-----------|---------|:---:|
| BSDS500 | 11 | 0.862 | 0.0007 | ✅ |
| COCO    | 11 | 0.876 | 0.0004 | ✅ |
| ADE20K  | 11 | 0.833 | 0.0015 | ✅ |

Rankings are highly consistent across datasets (Spearman ρ = 0.909–0.945, all p < 0.001).

## Repository Structure

```
Feature-Geometry/
├── reproduce_all.py              # ⭐ One-click reproduction (see below)
├── requirements.txt              # Python dependencies
│
├── src/                          # Core library
│   ├── pipeline.py               # Model loading (16 backbones) & feature extraction
│   └── metrics.py                # PSA, SC (BSDS/COCO/ADE20K), geometry metrics
│
├── experiments/                  # Experiment scripts (one per paper result)
│   ├── run_unified_table2.py     # → Table 2: N=11, all metrics, 3 datasets
│   ├── run_clustering_invariance.py → Table 1: K-Means/GMM/Spectral
│   ├── run_unified_ade20k.py     # → Sect 4.2: ADE20K cross-dataset
│   ├── run_unified_voc_coco.py   # → Sect 4.2: COCO cross-dataset
│   ├── run_cross_dataset.py      # → Mixed-arch cross-dataset
│   ├── run_boundary.py           # → Sect 4.3: 16-model boundary analysis
│   ├── run_psa_ablation.py       # → Sect 4.4: PSA variants
│   ├── run_psa_selection.py      # → Sect 4.5: PSA-guided selection
│   ├── run_within_backbone.py    # → Sect 4.6: per-image n80 vs SC
│   ├── generate_pca_figure.py    # → Fig 2: PCA feature visualization
│   ├── generate_figures.py       # → Fig 3: PSA vs SC scatter
│   └── generate_loo_figure.py    # → Fig 4: LOO stability chart
│
├── scripts/                      # Setup helpers
│   ├── download_data.py          # Download BSDS500, COCO, ADE20K
│   └── download_checkpoints.py   # Download MoCo-v3, iBOT checkpoints
│
├── paper/                        # LaTeX source & generated figures
│   └── figures/
│       ├── feature_pca_viz.pdf
│       ├── loo_stability.pdf
│       └── fig_psa_sc_v2.png
│
└── data/                         # Dataset root (not tracked)
    ├── BSDS500/images/{train,val,test}/
    ├── COCO/{val2017/, annotations/instances_val2017.json}
    └── ADE20K/{images/validation/, annotations/validation/}
```

## Quick Start

### 1. Setup

```bash
git clone https://github.com/<your-repo>/Feature-Geometry.git
cd Feature-Geometry
pip install -r requirements.txt
```

### 2. Download Data & Checkpoints

```bash
python scripts/download_data.py          # BSDS500, COCO, ADE20K
python scripts/download_checkpoints.py   # MoCo-v3, iBOT (others auto-download)
```

### 3. Reproduce All Results

```bash
# Run everything (Tables 1-2, Figures 2-4, all sections)
python reproduce_all.py

# Or run specific items:
python reproduce_all.py --table 2         # Table 2 only
python reproduce_all.py --figure 2        # PCA visualization only
python reproduce_all.py --section 4.2     # Cross-dataset section
python reproduce_all.py --list            # Show all available experiments
```

## Paper ↔ Code Mapping

| Paper Result | Script | Key Output |
|---|---|---|
| **Table 1**: Clustering invariance | `run_clustering_invariance.py` | SC per backbone × {K-Means, GMM, Spectral} |
| **Table 2**: All metrics + SC (3 datasets) | `run_unified_table2.py` | SC(BSDS/COCO/ADE) + 7 metrics × 11 backbones |
| **Fig 2**: PCA feature visualization | `generate_pca_figure.py` | `paper/figures/feature_pca_viz.pdf` |
| **Fig 3**: PSA vs SC scatter plot | `generate_figures.py` | `paper/figures/fig_psa_sc_v2.png` |
| **Fig 4**: LOO stability | `generate_loo_figure.py` | `paper/figures/loo_stability.pdf` |
| **Sect 4.2**: Cross-dataset | `run_unified_ade20k.py`, `run_unified_voc_coco.py` | PSA→SC correlations across datasets |
| **Sect 4.3**: Boundary conditions | `run_boundary.py` | 16-model PSA/SC with outlier detection |
| **Sect 4.4**: PSA variants | `run_psa_ablation.py` | 4-conn vs 8-conn, cosine vs L2, weighted |
| **Sect 4.5**: PSA-guided selection | `run_psa_selection.py` | Rank backbones by PSA → predict SC |
| **Sect 4.6**: Within-backbone | `run_within_backbone.py` | Per-image n80 vs SC (6 backbones) |

## Experimental Configuration

All experiments use:
- **Seed**: 42 (fixed for reproducibility)
- **N images**: 50 per metric computation (200 for within-backbone)
- **Clustering**: K-Means, K=4, on 32-d PCA features (ℓ₂-normalized)
- **Evaluation**: Segmentation Covering (SC)

## Models (11 SSL Core ViT-B/16)

| Backbone | Source | Paradigm | PSA | SC (BSDS) |
|----------|--------|----------|-----|-----------|
| **iBOT** | Local checkpoint | Masked + self-distill | .713 | **.570** |
| MoCo-v3 | Local checkpoint | Contrastive | .861 | .555 |
| DINO | `torch.hub: facebookresearch/dino` | Self-distillation | .625 | .546 |
| MAE | `timm: vit_base_patch16_224.mae` | Masked autoencoder | .731 | .509 |
| OpenCLIP | `timm: vit_base_patch16_clip_224.laion2b` | Contrastive (LAION-2B) | .596 | .506 |
| MetaCLIP | `timm: vit_base_patch16_clip_224.metaclip_2pt5b` | Contrastive (2.5B) | .672 | .493 |
| CLIP | `openai/clip: ViT-B/16` | Language-Image | .702 | .486 |
| SigLIP | `timm: vit_base_patch16_siglip_224.webli` | Sigmoid contrastive | .449 | .405 |
| BEiT | `timm: beit_base_patch16_224.in22k_ft_in22k_in1k` | Masked tokenizer | .515 | .399 |
| EVA-02 | `timm: eva02_base_patch16_clip_224.merged2b` | CLIP + MIM | .479 | .376 |
| BEiTv2 | `timm: beitv2_base_patch16_224.in1k_ft_in22k_in1k` | Masked VQ-KD | .449 | .343 |

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
