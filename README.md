# Text-to-Sketch

> **Hand Simulation Pipeline** — converts anime images into Sketchformer-ready
> stroke-5 vector sequences, complete with realistic kinematics and a K-means
> Tok-Dict motion vocabulary.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Stages](#pipeline-stages)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [1. Download Dataset](#1-download-dataset)
  - [2. Extract Sketches](#2-extract-sketches)
  - [3. Run Full Pipeline](#3-run-full-pipeline)
  - [4. Evaluate Ordering](#4-evaluate-ordering)
- [Output Artefacts](#output-artefacts)
- [Environment Variables](#environment-variables)
- [CLI Reference](#cli-reference)

---

## Overview

**Text-to-Sketch** is a data preprocessing pipeline that transforms raw anime
frames into stroke-5 vector sequences suitable for training
[Sketchformer](https://github.com/leosampaio/sketchformer).  It implements the
complete **Hand Simulation Pipeline** (function *H*):

```
Anime image
  └─ 1: Lineart extraction    (ControlNet LineartAnimeDetector)
  └─ 2: Vectorization + RDP   (OpenCV contours → Ramer-Douglas-Peucker)
  └─ 3: Stroke ordering       (Directional / Greedy / TSP)
  └─ 4: Sigma-Lognormal kinematics
  └─ 5: stroke-5 formatting   → Sketchformer input
       + Tok-Dict              → K-means discrete motion vocabulary & encoding
```

---

## Pipeline Stages

| Stage | Module | Description |
|---|---|---|
| **1** | `pipeline/lineart.py` | ControlNet anime lineart extraction |
| **2** | `pipeline/vectorization.py` | RDP-simplified vector stroke extraction, default epsilon `0.5` |
| **3** | `pipeline/ordering.py` | Directional bias / Greedy NN / TSP ordering |
| **4** | `pipeline/kinematics.py` | Sigma-Lognormal velocity model |
| **5** | `pipeline/stroke5.py` | stroke-5 `[Δx, Δy, p1, p2, p3]` formatter |
| **Tok-Dict** | `prep_data/sketch_token/` + `utils/tokenizer.py` | K-means codebook builder + encoder/decoder |

---

## Project Structure

```
text-to-sketch/
│
├── pipeline/                          # Main pipeline logic
│   ├── lineart.py                     # Stage 1 — Lineart extraction helpers
│   ├── vectorization.py               # Stage 2 — Vectorization + RDP
│   ├── ordering.py                    # Stage 3 — Stroke ordering
│   ├── kinematics.py                  # Stage 4 — Kinematics
│   ├── stroke5.py                     # Stage 5 — stroke-5 formatting
│   ├── workflow.py                    # End-to-end Stage 2–5 + Tok-Dict workflow
│   └── run_pipeline.py                # Interactive pipeline CLI
│
├── prep_data/                         # Data preparation commands
│   ├── download_data.py
│   ├── extract_sketches.py
│   ├── filter_sketches.py
│   ├── prepare_sketchformer.py
│   └── sketch_token/
│       └── create_token_dict.py       # K-means sketch token dictionary
│
├── dataloaders/                       # Future Sketchformer-ready data loaders
├── models/                            # Future Sketchformer model definitions
├── builders/                          # Future losses, schedulers, layers, builders
│   └── layers/
├── core/                              # Future training/experiment orchestration
├── experiments/                       # Future fine-tuning and analysis scripts
├── dependencies/                      # Environment, Docker, and dependency notes
│
├── metrics/                           # Metrics, reports, and visualisation
│   ├── compare_rdp_epsilon.py
│   ├── evaluate_encoder.py
│   ├── evaluate_ordering.py
│   └── visualisation.py
│
├── utils/                             # Shared paths, persistence, tokenizer helpers
│   ├── io.py
│   ├── paths.py
│   └── tokenizer.py                   # encode/decode sketch token sequences
├── weights/                           # Fine-tuned and pretrained model weights
│   ├── pretrained/
│   └── finetuned/
│
├── scripts/                           # Thin compatibility wrappers
│   ├── prepare_data/
│   │   ├── download_data.py           # Kaggle dataset download
│   │   ├── extract_sketches.py        # Stage 1 batch lineart extraction
│   │   ├── filter_sketches_by_points.py
│   │   └── prepare_anime_data.py
│   ├── metrics/
│   │   ├── compare_rdp_epsilon.py     # Top-point RDP comparison report
│   │   ├── evaluate_ordering.py       # Ordering visualisation & evaluation
│   │   └── evaluate_encoder.py        # Tok-Dict encoding/decoding evaluation
│   ├── run_pipeline.py                # Unified interactive pipeline runner (2–5)
│
├── data/                              # All data (git-ignored)
│   ├── raw/                           # Raw downloaded datasets
│   └── processed/
│       ├── sketches/                  # Stage 1 output — binary line-art .png
│       ├── sketches_max_20000/        # Filtered sketches used by the pipeline
│       ├── stroke5/                   # Stage 5 output — stroke-5 .npz files
│       ├── sketch_token/              # codebook.npy + metadata.json
│       ├── tokens/                    # encoded token .npz files
│       └── evaluations/               # Ordering evaluation plots
│
├── sketchformer/                      # External Sketchformer checkout (git-ignored)
├── docs/
├── .env                               # Local environment variables (git-ignored)
├── .gitignore
├── pyproject.toml                     # Package metadata + console commands
├── requirements.txt
└── README.md
```

---

## Requirements

- Python **3.10+**
- A valid **Kaggle API** account (for dataset download only)
- GPU recommended for Stage A (CPU supported)

**Core dependencies:**

```
controlnet-aux     # Stage 1 – LineartAnimeDetector
opencv-python      # Stage 2 – contour extraction
python-tsp         # Stage 3 – TSP ordering
scipy              # Stage 4 – Sigma-Lognormal (lognorm.ppf)
scikit-learn       # Tok-Dict – MiniBatchKMeans
numpy
matplotlib
Pillow
tqdm
python-dotenv
kagglehub
```

---

## Installation

### 1. Clone the Repository

```bash
git clone git@github.com:naolselemon/text-to-sketch.git
cd text-to-sketch
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` makes these top-level packages importable from anywhere
inside the virtual environment and installs the `tts-*` console commands.

### 4. Configure Kaggle Credentials

**Option A — Kaggle JSON (recommended):**
```bash
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
```

**Option B — Environment variables:**
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

---

## Configuration

Create a `.env` file in the project root:

```env
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
KAGGLE_DATASET=diraizel/anime-images-dataset

DATA_RAW_DIR=data/raw
INPUT_DIR=data/raw/data/anime_images
OUTPUT_DIR=data/processed/sketches

DETECT_RES=512
IMAGE_RES=512
MAX_PER_FOLDER=15
```

---

## Usage

### 1. Download Dataset

```bash
python scripts/prepare_data/download_data.py
```

Downloads the configured Kaggle dataset into `data/raw/`.

---

### 2. Extract Sketches

```bash
python scripts/prepare_data/extract_sketches.py
```

With custom options:

```bash
python scripts/prepare_data/extract_sketches.py \
  --input-dir data/raw/data/anime_images \
  --output-dir data/processed/sketches \
  --detect-resolution 512 \
  --image-resolution 512 \
  --max-per-folder 15
```

Produces binary line-art `.png` files in `data/processed/sketches/`.

---

### 3. Filter Sketches

```bash
python scripts/prepare_data/filter_sketches_by_points.py
```

Produces filtered sketch `.png` files in `data/processed/sketches_max_20000/`.
The rest of the codebase uses this folder by default.

---

### 4. Run Full Pipeline

**Interactive mode — asks how many sketches to process and which ordering:**

```bash
python scripts/run_pipeline.py
```

```
╔════════════════════════════════════════════════════════════╗
║          Hand Simulation Pipeline — Text-to-Sketch         ║
║ Stages: Vectorize → Order → Kinematics → Stroke5 → Tok-Dict║
╚════════════════════════════════════════════════════════════╝

  Available sketches : 10300

How many sketches to process? [default: 50]:
Stroke-ordering method:
  1) Directional bias [default]  — top-left → bottom-right
  2) Greedy nearest-neighbor     — minimise pen travel locally
  3) TSP approximation           — globally minimise pen travel
Choose [1/2/3, default: 1]:
```

Produces:
- `data/processed/stroke5/<name>.npz` — stroke-5 arrays, shape `(N+1, 5)`
- `data/processed/sketch_token/codebook.npy` — K-means centroids, shape `(K, 2)`
- `data/processed/sketch_token/metadata.json` — K, n_samples, timestamp
- `data/processed/tokens/<name>.npz` — encoded discrete motion token sequences

---

### 5. Evaluate Output

**Evaluate Ordering:**
```bash
python scripts/metrics/evaluate_ordering.py
python scripts/metrics/evaluate_ordering.py --samples 20
```
Saves side-by-side evaluation plots to `data/processed/evaluations/`.
Uses `data/processed/sketches_max_20000/` by default.

**Evaluate Tok-Dict Encoder:**
```bash
python scripts/metrics/evaluate_encoder.py
```
Tests the encoding-decoding cycle and computes quantization loss.

**Compare RDP epsilon output on the densest sketches:**
```bash
python scripts/metrics/compare_rdp_epsilon.py
```

By default this scans `data/processed/sketches_max_20000/`, selects the 20
sketches with the most pre-simplification contour points, simplifies them with
RDP epsilon `0.5`, and saves:

- `data/processed/rdp_epsilon_0_5/reports/top_stroke_point_sketches.csv`
- `data/processed/rdp_epsilon_0_5/visualizations/*.png`
- `data/processed/rdp_epsilon_0_5/stroke5/*.npz`

---

## Output Artefacts

### stroke-5 format (`data/processed/stroke5/*.npz`)

Each `.npz` contains a single array `stroke5` of shape `(N+1, 5)`:

| Column | Meaning |
|---|---|
| `Δx` | Relative X displacement to previous point |
| `Δy` | Relative Y displacement to previous point |
| `p1` | `1` = pen is drawing (mid-stroke) |
| `p2` | `1` = last point of stroke (pen lifts next) |
| `p3` | `1` = end-of-sketch sentinel (final row only) |

Load with:
```python
from utils.io import load_stroke5
s5 = load_stroke5("data/processed/stroke5/my_sketch.npz")
```

### Tok-Dict Codebook (`data/processed/sketch_token/`)

| File | Contents |
|---|---|
| `codebook.npy` | `(K, 2)` float32 array of (Δx, Δy) cluster centroids |
| `metadata.json` | `K`, `n_samples`, `codebook_shape`, `timestamp` |

Encode a sketch to token indices:
```python
from utils.io import load_codebook
from utils.tokenizer import encode_stroke5
from utils.tokenizer import decode_tokens

codebook = load_codebook("data/processed/sketch_token/codebook.npy")
tokens   = encode_stroke5(s5, codebook)   # shape (N+1,), dtype int32
# tokens[i] ∈ [0, K-1]  → motion token
# tokens[i] == K         → pen-lift token
# tokens[i] == K+1       → end-of-sketch token

# Decode back to stroke-5
reconstructed_s5 = decode_tokens(tokens, codebook)
```

**Tokens format (`data/processed/tokens/*.npz`)**:
Each `.npz` contains a discrete sequence array named `tokens` resulting from encoding `stroke-5` with the codebook.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `KAGGLE_USERNAME` | _(none)_ | Kaggle account username |
| `KAGGLE_KEY` | _(none)_ | Kaggle API key |
| `KAGGLE_DATASET` | `diraizel/anime-images-dataset` | Kaggle dataset slug |
| `DATA_RAW_DIR` | `data/raw` | Destination for downloaded dataset |
| `INPUT_DIR` | `data/raw/data/anime_images` | Source for lineart extraction |
| `OUTPUT_DIR` | `data/processed/sketches` | Output for extracted sketches |
| `DETECT_RES` | `512` | Detector input resolution |
| `IMAGE_RES` | `512` | Output image resolution |
| `MAX_PER_FOLDER` | `15` | Max images processed per subdirectory |

---

## CLI Reference

| Script | Purpose |
|---|---|
| `python scripts/prepare_data/download_data.py` or `tts-download-data` | Download Kaggle dataset |
| `python scripts/prepare_data/extract_sketches.py [--input-dir] [--output-dir] [--detect-resolution] [--image-resolution] [--max-per-folder]` or `tts-extract-sketches` | Run Stage A lineart extraction |
| `python scripts/prepare_data/filter_sketches_by_points.py` or `tts-filter-sketches` | Filter dense sketches before vectorization |
| `python scripts/run_pipeline.py` or `tts-run-pipeline` | Interactive Stages B–E + Tok-Dict |
| `python scripts/metrics/evaluate_ordering.py [--samples N]` or `tts-evaluate-ordering` | Visualise ordering quality |
| `python scripts/metrics/evaluate_encoder.py` or `tts-evaluate-encoder` | Evaluate Tok-Dict encoding/decoding |
| `python scripts/metrics/compare_rdp_epsilon.py` or `tts-compare-rdp` | Compare RDP simplification on dense sketches |
