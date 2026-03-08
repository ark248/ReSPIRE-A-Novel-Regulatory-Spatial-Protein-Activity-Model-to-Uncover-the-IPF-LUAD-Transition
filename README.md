# ReSPIRE: Regulatory Spatial Protein Activity Model

**ReSPIRE** predicts the activity of 7,060 regulatory proteins at single-cell resolution directly from H&E histopathology images, using embeddings from the [UNI pathology foundation model](https://www.nature.com/articles/s41591-024-02857-3) paired with a residual MLP regression head. Ground truth protein activity is derived from Visium HD spatial transcriptomics via MetaVIPER.

The model is applied to study the transition from Idiopathic Pulmonary Fibrosis (IPF) to Lung Adenocarcinoma (LUAD), identifying spatially-localized regulatory switches at the fibrotic-tumor boundary directly from routine clinical slides.

> **Author:** Adhiban Arulselvan

---

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Environment Setup](#environment-setup)
- [Data](#data)
- [Running the Pipeline](#running-the-pipeline)
  - [Step 1 — Preprocessing](#step-1--preprocessing-spatial-transcriptomics)
  - [Step 2 — GRN Construction](#step-2--grn-construction-aracne3)
  - [Step 3 — Protein Activity Inference](#step-3--protein-activity-inference-metaviper)
  - [Step 4 — UNI Feature Extraction](#step-4--uni-feature-extraction)
  - [Step 5 — Multiscale Feature Construction](#step-5--multiscale-feature-construction)
  - [Step 6 — Model Training](#step-6--model-training)
  - [Step 7 — Inference on New H&E Images](#step-7--inference-on-new-he-images)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Evaluation](#evaluation)
- [Results](#results)
- [Hardware Requirements](#hardware-requirements)
- [Repository Structure](#repository-structure)

---

## Pipeline Overview

```
scRNA-seq (HTAN MSK)               Visium HD (6 lung tissues)
        │                                     │
        ▼                                     ▼
  Metacell generation              Spatial H5AD preprocessing
        │                                     │
        ▼                                     ▼
  ARACNe3 GRN inference            MetaVIPER NES computation
  (Immune / Epithelial / Stromal)  (7,060 proteins × ~500K spots)
                                              │
                                             ┌┴────────────────────┐
                                             │                      │
                                             ▼                      ▼
                                    UNI feature extraction    NES targets
                                    (224×224 H&E tiles        (h5ad files)
                                     → 1024-dim embeddings)
                                             │
                                             ▼
                                    Multiscale concatenation
                                    (center + 4 cardinal neighbors
                                     → 5×1024 = 5120-dim vectors)
                                             │
                                             ▼
                                    ReSPIRE MLP training
                                    (LOO cross-validation, k=6)
                                             │
                                             ▼
                                    Spatial protein activity maps
```

---

## Environment Setup

Python 3.10 is required. An Anaconda environment is strongly recommended.

```bash
conda create -n respire python=3.10
conda activate respire
```

### Core dependencies

```bash
# Deep learning
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install lightning timm

# Spatial / single-cell
pip install scanpy anndata pyviper

# Image processing
pip install pyvips Pillow h5py

# Data / numerics
pip install numpy pandas scipy pyarrow scikit-learn

# Visualization
pip install matplotlib seaborn
```

### ARACNe3

ARACNe3 is a C++ executable and must be compiled separately.

```bash
git clone https://github.com/califano-lab/ARACNe3
cd ARACNe3
mkdir build && cd build
cmake ..
make -j8
```

The compiled binary will be at `ARACNe3/build/src/app/ARACNe3_app_release`. Update the path in `run_aracne3.py` to point to your binary.

### UNI Model Weights

UNI weights are available through the [Mahmood Lab HuggingFace](https://huggingface.co/MahmoodLab/UNI). You will need to request access. Once downloaded, the weights are loaded via `timm`:

```python
import timm
model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16,
                           init_values=1e-5, num_classes=0, dynamic_img_size=True)
model.load_state_dict(torch.load("path/to/uni_weights.pth"), strict=True)
```

The model's classification head is removed (`num_classes=0`) and weights are **frozen** during all training.

---

## Data

### Visium HD Spatial Transcriptomics

Download the following datasets and organize them under a common root directory:

**LUAD (10x Genomics public datasets):**
- `Visium HD Spatial Gene Expression Libraries, Post-Xenium, Human Lung Cancer (FFPE)` — 2 sections
- `Visium HD Spatial Gene Expression Library, Human Lung Cancer (Fixed Frozen)` — 1 section
- `Visium HD Spatial Gene Expression Library, Human Lung Cancer, IF Stained (FFPE)` — 1 section

**IPF (GEO: [GSE261731](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE261731)):**
- `GSM8721365` — ASS-ILD, pre-transplant
- `GSM8721366` — ASS-ILD, post-bilateral lung transplant

For each tissue, the pipeline expects these files under `binned_outputs/square_008um/`:
```
filtered_feature_bc_matrix/
    barcodes.tsv.gz
    features.tsv.gz
    matrix.mtx.gz
spatial/
    tissue_positions.parquet
    scalefactors_json.json
    tissue_hires_image.png
```

### scRNA-seq (for GRN construction only)

Download from [CZ CELLxGENE](https://cellxgene.cziscience.com/) — HTAN MSK cohort:
- Immune cells, Lung Adenocarcinoma
- Epithelial cells, Lung Adenocarcinoma
- Stromal cells, Lung Adenocarcinoma

### Expected Directory Structure

```
data/
├── lung1/
│   ├── sample1_tumor/binned_outputs/square_008um/
│   └── sample2_tumor/binned_outputs/square_008um/
├── lung2/sample1/binned_outputs/square_008um/
├── lung3/sample1/binned_outputs/square_008um/
├── lung6/
│   ├── sample1/spatial/   ← IPF
│   └── sample2/spatial/   ← IPF
scrnaseq/
├── immune/
├── epithelial/
└── stromal/
```

---

## Running the Pipeline

### Step 1 — Preprocessing (Spatial Transcriptomics)

Converts raw 10x output files into a unified H5AD with normalized expression.

```bash
python preprocess_spatial.py \
  --input_path data/lung1/sample1_tumor/binned_outputs/square_008um/filtered_feature_bc_matrix \
  --sample_id lung1_sample1 \
  --file_type standard_10x \
  --output_h5ad data/lung1/sample1_tumor/processed.h5ad
```

QC applied: min 200 genes/cell, min 3 cells/gene, `normalize_total` + `log1p`, top 2,000 HVGs identified (full gene set retained for MetaVIPER).

---

### Step 2 — GRN Construction (ARACNe3)

Run separately for each scRNA-seq cell type (immune, epithelial, stromal).

**2a. Metacell generation:**
```bash
python generate_metacells.py \
  --input_h5ad scrnaseq/epithelial/epithelial.h5ad \
  --output_dir grns/epithelial/ \
  --n_metacells 500
```

scRNA-seq QC: min 2,000 genes/cell, min 3 cells/gene, <20% mitochondrial reads, doublet removal via Scrublet, cell type validation via SingleR-py (Blueprint_Encode).

**2b. Run ARACNe3:**
```bash
python run_aracne3.py \
  --exp_mat grns/epithelial/metacells.tsv \
  --output_folder grns/epithelial/network/ \
  --regulators regulators/human_tf_cotf_plus_sig_surf.txt \
  --subnets 100 \
  --threads 32
```

Output: `consolidated-net_defaultid.tsv` (columns: TF, Target, MI score). Each GRN should contain ~10 million edges. Runtime: 1–2 hours per network on 32 threads.

---

### Step 3 — Protein Activity Inference (MetaVIPER)

Runs metaVIPER on each spatial H5AD using all 3 GRN interactomes simultaneously.

```bash
python run_metaviper.py \
  --input_h5ad data/lung1/sample1_tumor/processed.h5ad \
  --network_stromal grns/stromal/network/consolidated-net_defaultid.tsv \
  --network_epithelial grns/epithelial/network/consolidated-net_defaultid.tsv \
  --network_immune grns/immune/network/consolidated-net_defaultid.tsv \
  --output_h5ad pa_tables/lung1_sample1_NES.h5ad \
  --njobs 32
```

**Output format:** AnnData where `obs` = spatial barcodes, `var` = protein names, `X` = NES matrix. NES range ~−5 to +5; positive = active, negative = suppressed. Low-variance proteins are filtered, leaving **7,060 proteins** retained for training.

Runtime: ~30–60 minutes per tissue on 32 cores.

---

### Step 4 — UNI Feature Extraction

Extracts 1,024-dim embeddings from H&E tiles for every spatial barcode using the frozen UNI ViT-Large model.

```bash
python extract_uni_features.py \
  --visium_root data/lung1/sample1_tumor/binned_outputs/square_008um \
  --pa_h5ad pa_tables/lung1_sample1_NES.h5ad \
  --save_dir data/lung1/sample1_tumor/features/ \
  --model_name uni \
  --batch_size 256 \
  --num_threads 16
```

`pyvips` is used for streaming tile extraction to avoid loading full-resolution images into RAM. Each tile is resized to 224×224, normalized with ImageNet stats, and passed through the frozen UNI encoder in batches. Output: `uni_features.h5` with datasets `features` (N × 1024) and `barcodes`.

---

### Step 5 — Multiscale Feature Construction

Builds the final training-ready `.pt` file by concatenating each spot's embedding with its 4 cardinal neighbor embeddings, creating a 5,120-dim multiscale vector per spot.

```bash
python build_multiscale_features.py \
  --config configs/sample_paths.yaml \
  --output metaviperfeatures1.pt
```

`sample_paths.yaml` should list `feature_path`, `pos_path`, and `h5ad_path` for each sample. Neighbors are resolved via grid lookup on `array_row`/`array_col` from the spatial positions parquet. Missing neighbors (tissue edges, background) are zero-padded. Barcodes are intersected across all three sources (features, positions, targets) before assembly.

**Output `.pt` structure:**
```python
{
  "data": {
    "Lung1_S1": {"features": Tensor(N, 5120), "targets": Tensor(N, 7060)},
    ...
  },
  "meta": {
    "proteins": [...],                 # list of 7060 protein names
    "barcodes_per_sample": {...},      # for auditability
    "neighbor_offsets": [(-1,0), (1,0), (0,-1), (0,1)]
  }
}
```

---

### Step 6 — Model Training

Leave-one-out cross-validation across all 6 samples. Per-fold z-score normalization is computed from the training set only and applied before training and evaluation.

```bash
python train.py \
  --data_file metaviperfeatures1.pt \
  --output_dir predictions/ \
  --input_dim 5120 \
  --batch_size 4096 \
  --epochs 300 \
  --lr 2e-4 \
  --patience 8
```

Each fold trains on 4–5 samples and evaluates on the held-out sample. Per-protein Pearson R is logged per fold. Checkpoints, CSV logs, and per-fold prediction `.npy` files are saved to `output_dir`.

---

### Step 7 — Inference on New H&E Images

To run ReSPIRE on a new H&E slide without spatial transcriptomics ground truth:

```bash
python infer.py \
  --he_image path/to/slide.png \
  --model_checkpoint predictions/fold1/best_model.ckpt \
  --output_dir results/ \
  --batch_size 256
```

This runs the sliding-window UNI extraction followed by the trained MLP, producing a full spatial protein activity map saved as an H5AD file.

---

## Model Architecture

ReSPIRE uses a **Residual MLP** with 11.9M trainable parameters. The UNI encoder is fully frozen; only the MLP is trained.

```
Input: 5120-dim  (5× UNI 1024-dim embeddings)
  │
  ├─ ResBlock: 5120 → 2048
  │    LayerNorm → GELU → Dropout(0.2) → Linear
  │    LayerNorm → GELU → Dropout(0.2) → Linear
  │    + skip projection (Linear 5120 → 2048)
  │
  ├─ ResBlock: 2048 → 1024
  ├─ ResBlock: 1024 → 1024
  ├─ ResBlock: 1024 → 512
  │
  └─ Head: LayerNorm → GELU → Dropout(0.2) → Linear(512 → 7060)

Output: 7060-dim NES prediction
```

**Why MLP over GNN?** The dataset contains ~500K spots × 7,060 proteins per sample (~3.5B values). GNNs require kNN graph construction with message-passing over 7K+ dimensional node features, causing over-smoothing within 3–4 layers. Ali et al. (2025) show GNNs provide minimal gain (ΔAUPR < 0.05) over simpler models when node features are already rich. The multiscale concatenation strategy captures local spatial context without graph construction overhead, and the UNI embeddings already encode sophisticated morphological context.

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Loss | `MSE − 0.5 × mean Pearson R` |
| Optimizer | AdamW (`weight_decay=1e-4`) |
| Learning rate | `2e-4` → cosine anneal → `1e-6` |
| Warmup | 3 epochs (linear ramp) |
| Batch size | 1,024 (up to 2,048 if VRAM allows) |
| Max epochs | 300 per fold |
| Early stopping | Patience = 8 (monitors `val_loss`) |
| Gradient clipping | 1.0 |
| Weight init | Kaiming (He) Normal |
| Dropout | 0.2 per ResBlock |
| Activation | GELU throughout |
| Input normalization | Z-score per feature (train-set statistics only) |
| Target normalization | Z-score per protein (train-set statistics only) |
| Low-variance filter | Proteins with std < 0.05 across training spots removed |

The custom loss combines MSE with a Pearson correlation term. This penalizes models that achieve low absolute error but fail to capture the relative ranking of protein activity across spots — which is biologically more meaningful than minimizing raw NES distance alone.

**Cross-validation folds:**

| Fold | Test | Train |
|------|------|-------|
| 1 | Lung1_S1 | Lung1_S2, Lung2_S1, Lung6_S1, Lung6_S2 |
| 2 | Lung1_S2 | Lung1_S1, Lung2_S1, Lung6_S1, Lung6_S2 |
| 3 | Lung2_S1 | Lung1_S1, Lung1_S2, Lung6_S1, Lung6_S2 |
| 4 | Lung6_S1 | Lung1_S1, Lung1_S2, Lung2_S1, Lung6_S2 |
| 5 | Lung6_S2 | Lung1_S1, Lung1_S2, Lung2_S1, Lung6_S1 |

---

## Evaluation

Metrics reported on the blind held-out test sample (Lung1_S1, LUAD):

| Metric | Value |
|--------|-------|
| RMSE | 0.0935 |
| MAE | 0.0653 |
| R² | 0.3400 |
| Pearson (PCC) | 0.6083 |
| Spearman (ρ) | 0.5578 |

Per-protein and per-fold Pearson R values are logged to CSV during training. Predicted NES maps are saved per fold and can be overlaid on the H&E tissue image for visual validation.

---

## Results

The model identifies a distinct regulatory switch at the fibrotic-tumor boundary. Key proteins upregulated at the IPF-LUAD interface include **MIF**, **STEAP1**, **S100A10**, **S100A11**, **LGALS3**, and **LRP1**. Proteins actively suppressed include **ITGA6**, **SPTBN1**, **CDH5**, and **ARHGAP29**. Survival analysis (TCGA-LUAD, n=513) and PPI hub analysis (STRING v12.0) further narrow the signature to **STEAP1, S100A10, S100A11, and ITGA6** as novel bridge protein candidates for therapeutic targeting.

---

## Hardware Requirements

| Component | Minimum | Used in This Study |
|-----------|---------|-------------------|
| GPU | 16 GB VRAM | NVIDIA RTX 3090 (24 GB) |
| CPU | 16 threads | 32 threads |
| RAM | 64 GB | 500 GB |
| Storage | 2 TB | Network-attached |
| OS | Linux | Ubuntu |

> **Note:** The MetaVIPER step is CPU-bound and RAM-intensive (~100+ GB for a single Visium HD tissue). UNI feature extraction and model training are GPU-bound. Reduce batch sizes on GPUs with less VRAM.

---

## Repository Structure

```
ReSPIRE/
├── preprocess_spatial.py           # Step 1: 10x MTX → H5AD
├── generate_metacells.py           # Step 2a: Metacell construction
├── run_aracne3.py                  # Step 2b: ARACNe3 wrapper
├── run_metaviper.py                # Step 3: MetaVIPER NES inference
├── extract_uni_features.py         # Step 4: UNI tile embedding
├── build_multiscale_features.py    # Step 5: 5-scale feature assembly
├── train.py                        # Step 6: LOO cross-validation training
├── infer.py                        # Step 7: Inference on new H&E images
├── model.py                        # ResidualBlock + ProteinPredictor
├── dataset.py                      # ProteinDataset, data loading utils
├── configs/
│   └── sample_paths.yaml           # Per-sample data paths
├── regulators/
│   └── human_tf_cotf_plus_sig_surf.txt   # ~7,700 candidate regulators
└── notebooks/
    ├── visualization.ipynb         # NES overlay on H&E tissue
    └── biomarker_analysis.ipynb    # Downstream protein analysis
```
