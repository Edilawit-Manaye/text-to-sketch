# Sketchformer Codebase Fine-Tuning Integration

This folder contains the adapter for fine-tuning with the original
Sketchformer codebase.

Text-to-Sketch remains responsible for generating and preparing stroke data.
After the handoff point, this integration launches the local git-ignored
`sketchformer/` checkout for training, checkpoint loading, loss computation,
and evaluation.

The adapter does not reimplement Sketchformer's model, dataloader, masks,
losses, or checkpoint structure.

## Contents

| Path | Purpose |
|---|---|
| `launcher.py` | Builds Docker commands for original Sketchformer training and evaluation. |
| `docker/` | CPU TensorFlow 2.1 image and GPU image used to run the legacy code. |

## Common Commands

Build the image:

```bash
python scripts/sketchformer_codebase_finetune.py --sudo build-image
```

Prepare legacy-compatible stroke3 chunks:

```bash
python scripts/sketchformer_codebase_finetune.py --sudo prepare-data \
  --source-dir data/processed/stroke5 \
  --target-dir data/processed/sketchformer-ready-data/stroke3 \
  --n-chunks 10 \
  --n-classes 345
```

Evaluate pretrained reconstruction:

```bash
python scripts/sketchformer_codebase_finetune.py --sudo evaluate-reconstruction
```

Fine-tune:

```bash
python scripts/sketchformer_codebase_finetune.py --sudo finetune-continuous
```

Variable-length anime sketches are handled by keeping the saved `x` arrays
ragged and passing a fixed `--max-seq-len` per run. The Sketchformer dataloader
pads shorter sketches and truncates longer sketches before computing masks.
Use separate run IDs for each cap:

```bash
python scripts/sketchformer_codebase_finetune.py --dry-run finetune-continuous \
  --run-id anime-continuous-l100 \
  --resume none \
  --max-seq-len 100

python scripts/sketchformer_codebase_finetune.py --dry-run finetune-continuous \
  --run-id anime-continuous-l500 \
  --resume none \
  --max-seq-len 500 \
  --base-hparams batch_size=2,num_epochs=1,save_every=1.0,safety_save=1.0,log_every=10,notify_every=100000,slack_config=
```

The released continuous checkpoint was trained with `max_seq_len=200`; changing
that cap while resuming it can fail on sequence-length-dependent TensorFlow
weights.
