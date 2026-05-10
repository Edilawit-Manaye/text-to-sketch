# Text-to-Sketch

Text-to-Sketch is a preprocessing and training-preparation project for turning
anime images into vector sketch data that can be used with Sketchformer.

The current project focuses on the data pipeline:

```text
anime images
  -> line-art sketches
  -> vector strokes
  -> ordered drawing paths
  -> human-like timing / kinematics
  -> stroke-5 arrays
  -> sketch-token codebook and token sequences
  -> Sketchformer-ready stroke3 chunks
```

The project is also being shaped to follow the original Sketchformer repository
layout. That means folders such as `models/`, `dataloaders/`, `builders/`,
`core/`, `experiments/`, `metrics/`, `dependencies/`, and `weights/` already
exist so future fine-tuning work has a clear home.

## What This Project Is For

This repo is for preparing anime-style image data so it can eventually be used
to fine-tune or reproduce Sketchformer-style models.

At the moment, it can:

- download the anime dataset from Kaggle
- extract binary line-art sketches from anime images
- filter sketches that are too dense
- vectorize line-art into stroke paths
- order strokes using several drawing-order strategies
- add simple hand-motion timing with Sigma-Lognormal kinematics
- convert timed strokes into stroke-5 arrays
- build a K-means sketch-token codebook
- encode stroke-5 arrays into token sequences
- convert stroke-5 arrays into Sketchformer-style stroke3 train/valid/test chunks
- generate basic evaluation plots and reports

The next major direction is:

- add real Sketchformer dataloaders
- add model code under `models/`
- add training orchestration under `core/`
- add fine-tuning experiments under `experiments/`
- manage pretrained and fine-tuned weights under `weights/`

## Project Layout

The layout intentionally follows Sketchformer's style.

```text
text-to-sketch/
├── pipeline/
│   ├── lineart.py
│   ├── vectorization.py
│   ├── ordering.py
│   ├── kinematics.py
│   ├── stroke5.py
│   ├── workflow.py
│   └── run_pipeline.py
│
├── prep_data/
│   ├── download_data.py
│   ├── extract_sketches.py
│   ├── filter_sketches.py
│   ├── prepare_sketchformer.py
│   └── sketch_token/
│       └── create_token_dict.py
│
├── metrics/
│   ├── compare_rdp_epsilon.py
│   ├── evaluate_encoder.py
│   ├── evaluate_ordering.py
│   └── visualisation.py
│
├── utils/
│   ├── io.py
│   ├── paths.py
│   └── tokenizer.py
│
├── builders/
│   └── layers/
├── core/
├── dataloaders/
├── models/
├── experiments/
├── dependencies/
├── weights/
│   ├── pretrained/
│   └── finetuned/
│
├── scripts/
│   ├── prepare_data/
│   ├── metrics/
│   └── run_pipeline.py
│
├── data/
├── sketchformer/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Folder Guide

| Folder | Purpose |
|---|---|
| `pipeline/` | Core image-to-sketch pipeline stages. |
| `prep_data/` | Dataset download, extraction, filtering, and Sketchformer data preparation. |
| `prep_data/sketch_token/` | K-means sketch-token codebook generation, matching Sketchformer's `prep_data/sketch_token` idea. |
| `metrics/` | Evaluation scripts, reports, and visual plots. |
| `utils/` | Shared paths, file IO, and token encode/decode helpers. |
| `scripts/` | Thin command wrappers. These keep command paths simple and stable. |
| `dataloaders/` | Future Sketchformer-style data loaders. |
| `models/` | Future Sketchformer model definitions. |
| `builders/` | Future layers, schedulers, losses, and model-building utilities. |
| `core/` | Future training and experiment orchestration. |
| `experiments/` | Future fine-tuning and analysis experiments. |
| `dependencies/` | Environment notes, Dockerfiles, and training dependency setup. |
| `weights/` | Pretrained and fine-tuned model weights. |
| `data/` | Local generated data. This is git-ignored. |
| `sketchformer/` | External Sketchformer checkout. This is git-ignored. |

## Pipeline Stages

| Step | File | What It Does |
|---|---|---|
| 1 | `pipeline/lineart.py` | Uses ControlNet's `LineartAnimeDetector` to create binary line-art sketches. |
| 2 | `pipeline/vectorization.py` | Uses OpenCV contours and RDP simplification to turn sketches into vector strokes. |
| 3 | `pipeline/ordering.py` | Orders strokes using directional, greedy nearest-neighbor, or TSP-style ordering. |
| 4 | `pipeline/kinematics.py` | Adds simple human-like timing with a Sigma-Lognormal model. |
| 5 | `pipeline/stroke5.py` | Converts timed strokes to stroke-5 format: `[dx, dy, p1, p2, p3]`. |
| 6 | `prep_data/sketch_token/create_token_dict.py` | Builds a K-means codebook for sketch-token encoding. |
| 7 | `utils/tokenizer.py` | Encodes stroke-5 arrays to token sequences and decodes them back. |
| 8 | `prep_data/prepare_sketchformer.py` | Converts stroke-5 data to Sketchformer-style stroke3 chunks. |

## Data Flow

### 1. Raw Images

Downloaded anime images live under:

```text
data/raw/
```

The default image input path is:

```text
data/raw/data/anime_images/
```

### 2. Extracted Sketches

Line-art extraction writes binary sketch images to:

```text
data/processed/sketches/
```

### 3. Filtered Sketches

Dense sketches are filtered before vectorization. The default filtered dataset is:

```text
data/processed/sketches_max_20000/
```

### 4. Stroke-5 Arrays

The main pipeline writes stroke-5 arrays to:

```text
data/processed/stroke5/
```

Each file contains:

```text
stroke5: shape (N + 1, 5)
```

The columns are:

| Column | Meaning |
|---|---|
| `dx` | X movement from previous point |
| `dy` | Y movement from previous point |
| `p1` | pen is drawing |
| `p2` | pen lifts after this point |
| `p3` | end-of-sketch marker |

### 5. Sketch Token Codebook

The K-means sketch-token codebook is written to:

```text
data/processed/sketch_token/codebook.npy
data/processed/sketch_token/metadata.json
```

This mirrors Sketchformer's idea of building a dictionary before using a
dictionary-based tokenizer.

### 6. Token Sequences

Encoded token sequences are written to:

```text
data/processed/tokens/
```

### 7. Sketchformer-Ready Data

The Sketchformer preparation step writes chunked stroke3 data to:

```text
data/processed/sketchformer-ready-data/stroke3/
```

That folder contains files such as:

```text
train_000.npz
train_001.npz
valid.npz
test.npz
meta.npz
```

## Requirements

Use Python 3.10 or newer.

Main dependencies include:

- `controlnet-aux`
- `opencv-python-headless`
- `python-tsp`
- `scipy`
- `scikit-learn`
- `numpy`
- `matplotlib`
- `Pillow`
- `tqdm`
- `python-dotenv`
- `kagglehub`

Install everything with:

```bash
pip install -r requirements.txt
```

Optional but recommended:

```bash
pip install -e .
```

`pip install -e .` uses `pyproject.toml` to make the project packages importable
and to install shortcut commands such as `tts-run-pipeline`.

## What `pyproject.toml` Does Here

`pyproject.toml` is the project's Python packaging file.

In this repo, it is used for three things:

- tells Python how to install this repo in editable mode
- exposes top-level packages like `pipeline`, `prep_data`, `metrics`, and `utils`
- creates command shortcuts such as `tts-download-data` and `tts-run-pipeline`

It does not replace `requirements.txt`.

Use both:

```bash
pip install -r requirements.txt
pip install -e .
```

## Setup

### 1. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Configure Kaggle

Use either Kaggle's JSON credentials:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Or environment variables:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

You can also create a `.env` file:

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

## How To Run The Current Pipeline

You can run commands through `scripts/` wrappers or through the `tts-*` commands
installed by `pip install -e .`.

### Step 1: Download Data

```bash
python scripts/prepare_data/download_data.py
```

Shortcut:

```bash
tts-download-data
```

### Step 2: Extract Line-Art Sketches

```bash
python scripts/prepare_data/extract_sketches.py
```

With explicit paths:

```bash
python scripts/prepare_data/extract_sketches.py \
  --input-dir data/raw/data/anime_images \
  --output-dir data/processed/sketches \
  --detect-resolution 512 \
  --image-resolution 512 \
  --max-per-folder 15
```

Shortcut:

```bash
tts-extract-sketches
```

### Step 3: Filter Dense Sketches

```bash
python scripts/prepare_data/filter_sketches_by_points.py
```

Default behavior:

- reads from `data/processed/sketches/`
- keeps sketches with at most 20,000 foreground points
- writes kept sketches to `data/processed/sketches_max_20000/`
- writes a CSV report to `data/processed/sketch_point_filter_report.csv`

Shortcut:

```bash
tts-filter-sketches
```

### Step 4: Run The Main Pipeline

```bash
python scripts/run_pipeline.py
```

Shortcut:

```bash
tts-run-pipeline
```

This step:

- samples sketches from `data/processed/sketches_max_20000/`
- vectorizes each sketch
- orders strokes
- adds timing
- saves stroke-5 arrays
- builds the sketch-token codebook
- saves token sequences

### Step 5: Prepare Sketchformer Stroke3 Data

```bash
python scripts/prepare_data/prepare_anime_data.py
```

Shortcut:

```bash
tts-prepare-sketchformer
```

This converts stroke-5 arrays into Sketchformer-style stroke3 chunks.

## Evaluation Commands

### Compare Stroke Ordering

```bash
python scripts/metrics/evaluate_ordering.py --samples 20
```

Shortcut:

```bash
tts-evaluate-ordering --samples 20
```

Output:

```text
data/processed/evaluations/
```

### Evaluate Token Encoder / Decoder

```bash
python scripts/metrics/evaluate_encoder.py
```

Shortcut:

```bash
tts-evaluate-encoder
```

This checks how much error is introduced when stroke-5 data is encoded into
tokens and decoded back.

### Compare RDP Epsilon

```bash
python scripts/metrics/compare_rdp_epsilon.py
```

Shortcut:

```bash
tts-compare-rdp
```

Default output:

```text
data/processed/rdp_epsilon_0_5/
```

## Important Formats

### stroke-5

Stroke-5 is the current main vector format used by this preprocessing pipeline.

```text
[dx, dy, p1, p2, p3]
```

Example loading code:

```python
from utils.io import load_stroke5

s5 = load_stroke5("data/processed/stroke5/example.npz")
```

### stroke3

Stroke3 is the Sketchformer-style format prepared for training.

```text
[dx, dy, pen_state]
```

The conversion happens in:

```text
prep_data/prepare_sketchformer.py
```

### Sketch Tokens

Sketch tokens are integer IDs produced by quantizing `[dx, dy]` movements with
a K-means codebook.

The codebook is created in:

```text
prep_data/sketch_token/create_token_dict.py
```

Encoding and decoding are in:

```text
utils/tokenizer.py
```

Example:

```python
from utils.io import load_codebook, load_stroke5
from utils.tokenizer import encode_stroke5, decode_tokens

stroke5 = load_stroke5("data/processed/stroke5/example.npz")
codebook = load_codebook("data/processed/sketch_token/codebook.npy")

tokens = encode_stroke5(stroke5, codebook)
reconstructed = decode_tokens(tokens, codebook)
```

## Command Reference

| Task | Script | Shortcut |
|---|---|---|
| Download dataset | `python scripts/prepare_data/download_data.py` | `tts-download-data` |
| Extract line-art | `python scripts/prepare_data/extract_sketches.py` | `tts-extract-sketches` |
| Filter sketches | `python scripts/prepare_data/filter_sketches_by_points.py` | `tts-filter-sketches` |
| Run full pipeline | `python scripts/run_pipeline.py` | `tts-run-pipeline` |
| Prepare Sketchformer data | `python scripts/prepare_data/prepare_anime_data.py` | `tts-prepare-sketchformer` |
| Evaluate ordering | `python scripts/metrics/evaluate_ordering.py` | `tts-evaluate-ordering` |
| Evaluate token encoder | `python scripts/metrics/evaluate_encoder.py` | `tts-evaluate-encoder` |
| Compare RDP epsilon | `python scripts/metrics/compare_rdp_epsilon.py` | `tts-compare-rdp` |

## Current Status

Implemented now:

- image download
- line-art extraction
- sketch filtering
- vectorization
- stroke ordering
- kinematics
- stroke-5 export
- sketch-token codebook generation
- token encoding / decoding
- stroke5-to-stroke3 preparation
- simple metrics and visual reports

Prepared for future work:

- `dataloaders/`
- `models/`
- `builders/`
- `core/`
- `experiments/`
- `dependencies/`
- `weights/`

These folders are intentionally present because the project is moving toward
Sketchformer fine-tuning and eventual Sketchformer-style replication.

## Notes

- `data/`, `weights/`, `.env`, `.venv`, and `sketchformer/` are local/runtime
  folders and should not be committed with generated artifacts.
- `scripts/` contains wrappers. The real implementation lives in folders such
  as `pipeline/`, `prep_data/`, `metrics/`, and `utils/`.
- The local `sketchformer/` folder is treated as an external reference checkout.
  This project is being organized to become compatible with that style over time.
