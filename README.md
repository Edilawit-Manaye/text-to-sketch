# Text-to-Sketch

Text-to-Sketch is a preprocessing project for turning anime portrait images
into vector sketch data that can be used for Sketchformer-style experiments.

The current pipeline is data preparation only. It downloads anime portrait
images, extracts line-art sketches, filters noisy sketches, vectorizes them,
orders the strokes, exports stroke-5 arrays, builds a sketch-token codebook,
and prepares Sketchformer-style stroke3 chunks.

```text
Danbooru2019 Portraits
  -> line-art sketches
  -> filtered sketches
  -> vector strokes
  -> ordered drawing paths
  -> human-like timing / kinematics
  -> stroke-5 arrays
  -> sketch-token codebook and token sequences
  -> Sketchformer-ready stroke3 chunks
```

## Current Capabilities

- Download a user-selected number of Danbooru2019 Portraits images over `rsync`.
- Skip already-installed images and resume incomplete downloads.
- Extract binary anime line-art sketches with ControlNet's `LineartAnimeDetector`.
- Filter sketches by foreground point count before vectorization.
- Vectorize sketches into OpenCV contour strokes with RDP simplification.
- Order strokes with directional, greedy nearest-neighbor, or TSP-style ordering.
- Add simple Sigma-Lognormal hand-motion timing.
- Export stroke-5 `.npz` files.
- Build a K-means sketch-token codebook and token sequences.
- Convert stroke-5 arrays into Sketchformer-style stroke3 train/valid/test chunks.
- Generate basic evaluation reports and visualizations.

Model training code is not implemented yet. The folders `models/`, `dataloaders/`,
`builders/`, `core/`, `experiments/`, and `weights/` are present so the project
can grow toward training and fine-tuning later.

## Project Layout

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
├── scripts/
│   ├── prepare_data/
│   ├── metrics/
│   └── run_pipeline.py
│
├── utils/
│   ├── io.py
│   ├── paths.py
│   └── tokenizer.py
│
├── data/
├── weights/
├── .env.example
├── pyproject.toml
├── requirements.txt
└── README.md
```

| Path | Purpose |
|---|---|
| `prep_data/download_data.py` | Downloads Danbooru2019 Portraits images. |
| `prep_data/extract_sketches.py` | Converts raw portrait images to binary line-art sketches. |
| `prep_data/filter_sketches.py` | Filters sketches by point count. |
| `pipeline/` | Vectorization, stroke ordering, kinematics, and stroke-5 export. |
| `prep_data/sketch_token/` | K-means sketch-token codebook generation. |
| `prep_data/prepare_sketchformer.py` | Converts stroke-5 files into Sketchformer-style stroke3 chunks. |
| `metrics/` | Evaluation and visualization scripts. |
| `utils/paths.py` | Shared default paths. |
| `data/` | Local generated data. This is git-ignored. |
| `weights/` | Placeholder for pretrained and fine-tuned weights. |

## Requirements

Use Python 3.10 or newer.

Python dependencies are installed from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The dataset downloader also requires the system `rsync` binary. On
Debian/Ubuntu:

```bash
sudo apt install rsync
```

Install the project in editable mode to make the `tts-*` shortcuts available:

```bash
pip install -e .
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Create local environment config:

```bash
cp .env.example .env
```

The default `.env.example` points the pipeline at Danbooru2019 Portraits:

```env
DOWNLOAD_TARGET_DIR=data/raw/portraits
DANBOORU2019_PORTRAITS_RSYNC_URL=rsync://176.9.41.242:873/biggan/portraits/

INPUT_DIR=data/raw/portraits
OUTPUT_DIR=data/processed/sketches

DETECT_RES=512
IMAGE_RES=512
MAX_PER_FOLDER=15
```

## Data Locations

| Stage | Default Path |
|---|---|
| Raw portrait images | `data/raw/portraits/` |
| Download manifest | `data/raw/portraits/.danbooru2019-portraits-files.txt` |
| Extracted sketches | `data/processed/sketches/` |
| Filtered sketches | `data/processed/sketches_max_20000/` |
| Filter report | `data/processed/sketch_point_filter_report.csv` |
| Stroke-5 arrays | `data/processed/stroke5/` |
| Sketch-token codebook | `data/processed/sketch_token/codebook.npy` |
| Token sequences | `data/processed/tokens/` |
| Sketchformer-ready chunks | `data/processed/sketchformer-ready-data/stroke3/` |
| Evaluation outputs | `data/processed/evaluations/` |

## Run The Pipeline

You can use either the script paths or the installed `tts-*` shortcuts.

### 1. Download Portrait Images

Choose how many raw images should exist locally:

```bash
python scripts/prepare_data/download_data.py --num-images 5000
```

Equivalent shortcut:

```bash
tts-download-data --num-images 5000
```

The downloader lists the remote portrait images, checks local files by path and
file size, skips complete images, and downloads only the missing remainder.

Preview without downloading:

```bash
python scripts/prepare_data/download_data.py --num-images 5000 --dry-run
```

Limit bandwidth:

```bash
python scripts/prepare_data/download_data.py --num-images 5000 --bwlimit 5m
```

Override the destination or rsync source:

```bash
python scripts/prepare_data/download_data.py \
  --num-images 5000 \
  --target-dir data/raw/portraits \
  --rsync-url rsync://176.9.41.242:873/biggan/portraits/
```

### 2. Extract Line-Art Sketches

```bash
python scripts/prepare_data/extract_sketches.py
```

Equivalent shortcut:

```bash
tts-extract-sketches
```

With explicit settings:

```bash
python scripts/prepare_data/extract_sketches.py \
  --input-dir data/raw/portraits \
  --output-dir data/processed/sketches \
  --detect-resolution 512 \
  --image-resolution 512 \
  --max-per-folder 15
```

Existing output sketches are skipped, so this step is resumable.

### 3. Filter Noisy Sketches

```bash
python scripts/prepare_data/filter_sketches_by_points.py
```

Equivalent shortcut:

```bash
tts-filter-sketches
```

Default behavior:

- Reads sketches from `data/processed/sketches/`.
- Keeps sketches with at most 20,000 foreground points.
- Copies kept sketches to `data/processed/sketches_max_20000/`.
- Writes `data/processed/sketch_point_filter_report.csv`.

Useful options:

```bash
python scripts/prepare_data/filter_sketches_by_points.py \
  --max-points 20000 \
  --count original \
  --limit 100
```

### 4. Vectorize, Order, Time, And Tokenize

```bash
python scripts/run_pipeline.py
```

Equivalent shortcut:

```bash
tts-run-pipeline
```

This interactive command samples filtered sketches, asks how many to process,
asks which ordering method to use, then writes:

- stroke-5 arrays under `data/processed/stroke5/`
- sketch-token codebook under `data/processed/sketch_token/`
- token sequences under `data/processed/tokens/`

### 5. Prepare Sketchformer-Style Stroke3 Data

```bash
python scripts/prepare_data/prepare_anime_data.py
```

Equivalent shortcut:

```bash
tts-prepare-sketchformer
```

This converts stroke-5 arrays into chunked stroke3 data:

```text
data/processed/sketchformer-ready-data/stroke3/
├── train_000.npz
├── train_001.npz
├── ...
├── valid.npz
├── test.npz
└── meta.npz
```

## Evaluation Commands

Compare stroke ordering strategies:

```bash
python scripts/metrics/evaluate_ordering.py --samples 20
```

Shortcut:

```bash
tts-evaluate-ordering --samples 20
```

Evaluate sketch-token encoding and decoding:

```bash
python scripts/metrics/evaluate_encoder.py
```

Shortcut:

```bash
tts-evaluate-encoder
```

Compare RDP simplification on dense sketches:

```bash
python scripts/metrics/compare_rdp_epsilon.py
```

Shortcut:

```bash
tts-compare-rdp
```

## Formats

### Stroke-5

Stroke-5 is the main vector format produced by this preprocessing pipeline.
Each `.npz` file contains a `stroke5` array with shape `(N + 1, 5)`.

```text
[dx, dy, p1, p2, p3]
```

| Column | Meaning |
|---|---|
| `dx` | X movement from the previous point. |
| `dy` | Y movement from the previous point. |
| `p1` | Pen is drawing. |
| `p2` | Pen lifts after this point. |
| `p3` | End-of-sketch marker. |

Example:

```python
from utils.io import load_stroke5

s5 = load_stroke5("data/processed/stroke5/example.npz")
```

### Stroke3

Stroke3 is the Sketchformer-style training format:

```text
[dx, dy, pen_state]
```

`prep_data/prepare_sketchformer.py` converts stroke-5 files into stroke3
train/valid/test chunks.

### Sketch Tokens

Sketch tokens are integer IDs produced by quantizing `[dx, dy]` movements with
a K-means codebook.

```python
from utils.io import load_codebook, load_stroke5
from utils.tokenizer import decode_tokens, encode_stroke5

stroke5 = load_stroke5("data/processed/stroke5/example.npz")
codebook = load_codebook("data/processed/sketch_token/codebook.npy")

tokens = encode_stroke5(stroke5, codebook)
reconstructed = decode_tokens(tokens, codebook)
```

## Command Reference

| Task | Script | Shortcut |
|---|---|---|
| Download portraits | `python scripts/prepare_data/download_data.py --num-images 5000` | `tts-download-data --num-images 5000` |
| Extract line-art | `python scripts/prepare_data/extract_sketches.py` | `tts-extract-sketches` |
| Filter sketches | `python scripts/prepare_data/filter_sketches_by_points.py` | `tts-filter-sketches` |
| Run main pipeline | `python scripts/run_pipeline.py` | `tts-run-pipeline` |
| Prepare stroke3 data | `python scripts/prepare_data/prepare_anime_data.py` | `tts-prepare-sketchformer` |
| Evaluate ordering | `python scripts/metrics/evaluate_ordering.py --samples 20` | `tts-evaluate-ordering --samples 20` |
| Evaluate encoder | `python scripts/metrics/evaluate_encoder.py` | `tts-evaluate-encoder` |
| Compare RDP epsilon | `python scripts/metrics/compare_rdp_epsilon.py` | `tts-compare-rdp` |

