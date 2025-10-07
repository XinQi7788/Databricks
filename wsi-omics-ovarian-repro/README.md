
# WSI–Omics (Ovary/Endometrium) – Reproducible Pipeline Scaffold

This scaffold sets up a public testbed to reproduce the core workflow described in the AACR abstract (B040) for ovarian cancer:

> tiling -> foundation model feature extraction (CHIEF / Prov-GigaPath) -> Weibull survival head -> ABMIL classifier for RNA-defined TME subtypes.

It uses TCGA ovarian diagnostic FFPE slides as exemplar data and provides placeholders to integrate bulk RNA-seq for TME subtyping (CAF/T-cell/endothelial signatures -> k-means (k=4)).

References:
- Abstract B040 and methods overview: AACR/Cancer Research 85(18_Supp):B040 (2025).
- CHIEF (Yu Lab, HMS DBMI): https://github.com/hms-dbmi/CHIEF
- Prov-GigaPath (Microsoft/Providence): https://github.com/prov-gigapath/prov-gigapath and the HF model card.
- Virchow (Paige AI, optional alternative FM): model on HF + arXiv.
- ABMIL reference (Ilse et al. 2018): PyTorch code.
- CLAM (attention MIL framework with heatmaps; optional): Mahmood Lab.
- TCGA WSIs: GDC Portal, ISB-CGC TCGA images documentation.

---

## 0) Environment

Create a fresh conda env:

```bash
conda update -n base -c defaults conda
conda env create -f environment.yml
conda activate wsi-omics-ovarian
```

GPU is recommended. For Prov-GigaPath tile encoder from HuggingFace, set a read token (required by their gated terms):

```bash
export HF_TOKEN=YOUR_READ_ONLY_TOKEN
```
OR
```Windows 
set HF_TOKEN=YOUR_READ_ONLY_TOKEN
```

---

## 1) Obtain TCGA ovarian diagnostic FFPE slides

Follow `scripts/01_download_tcga_wsi.md` for illustrated steps with the GDC Data Portal and gdc-client, filtering to Diagnostic Slide data (filenames ending in DX#). This will populate `data/WSI/TCGA_OV/` (by default).

---

## 2) Tile WSIs at 20x

```bash
python scripts/02_tile_wsi.py   --wsi_dir data/WSI/TCGA_OV   --out_dir data/tiles/20x_256   --tile_size 256 --step 256 --target_mpp 0.5
```

- `target_mpp` ~0.5 um/px is a common proxy for ~20x; adjust based on scanner metadata.

---

## 3) Extract tile features (choose CHIEF or Prov-GigaPath)

Option A – Prov-GigaPath (HF tile encoder):
```bash
python scripts/03_extract_features.py   --tiles_dir data/tiles/20x_256   --out_dir data/features/gigapath_20x256   --model gigapath   --batch_size 256
```

Option B – CHIEF (Yu Lab):
```bash
# one-time: clone CHIEF and set env var
export CHIEF_REPO=$PWD/externals/CHIEF

python scripts/03_extract_features.py   --tiles_dir data/tiles/20x_256   --out_dir data/features/chief_20x256   --model chief   --batch_size 256
```

Outputs are .npy files per tile plus an index mapping tiles to slides.

---

## 4) Aggregate tile embeddings -> slide-level embeddings

For survival you can start with mean pooling or attention pooling:
```bash
python scripts/04_aggregate_slide_features.py   --feat_dir data/features/gigapath_20x256   --out_csv data/slide_features/gigapath_20x256_mean.csv   --pool mean
```

(Optional) Attention pooling (learned) requires a quick lightweight pre-fit:
```bash
python scripts/04_aggregate_slide_features.py   --feat_dir data/features/gigapath_20x256   --out_csv data/slide_features/gigapath_20x256_attn.csv   --pool attention --epochs 1
```

---

## 5) Weibull survival model (deep parametric)

Prepare a CSV with [slide_id, time, event] and a path to the slide features from step 4:
```bash
python scripts/05_train_survival.py   --features_csv data/slide_features/gigapath_20x256_mean.csv   --surv_csv data/clinical/tcga_ov_survival.csv   --out_dir outputs/survival_gigapath_wbl   --max_epochs 50
```

This trains an MLP that outputs Weibull scale/shape; optimization uses the right-censored Weibull NLL. Reports c-index on a validation split and saves checkpoints.

---

## 6) RNA-defined TME subtypes and ABMIL classifier

1) Compute TME labels from bulk RNA-seq: provide a gene x sample matrix and CAF/T-cell/endothelial marker lists (edit `configs/tme_genes.yaml`).
```bash
python scripts/06_compute_tme_clusters.py   --expr_csv data/rna/tcga_ov_rnaseq_fpkm.csv   --genes_yaml configs/tme_genes.yaml   --out_csv data/labels/tme_k4_labels.csv
```
This performs z-scored signature scoring and k-means (k=4) mapping to the 4 labels: {Immune-Enriched, Immune-Enriched+Fibrotic, Fibrotic, Depleted}.

2) Train ABMIL on cores/tiles (or slide tiles) for TME subtype classification:
```bash
python scripts/07_train_abmil_tme.py   --tiles_dir data/tiles/20x_256   --feat_dir  data/features/gigapath_20x256   --tme_csv   data/labels/tme_k4_labels.csv   --out_dir   outputs/tme_abmil_gigapath   --max_epochs 10
```

---

## 7) Configuration
- `configs/paths.yaml` – change dataset/output paths.
- `configs/models.yaml` – FM choice, embedding dims, transforms.
- `configs/survival.yaml` – survival head dims/hparams.
- `configs/mil.yaml` – ABMIL dims/hparams.

---

Notes:
- Access & licensing: Prov-GigaPath and Virchow require agreeing to their terms and HF gating; CHIEF is AGPL-3.0.
- NECC TMA used in the abstract is not public; this scaffold focuses on TCGA as a reproducible public proxy.
- For robust TME labels, consider ssGSEA or GSVA; this scaffold uses a simpler z-score mean per gene set for portability.
