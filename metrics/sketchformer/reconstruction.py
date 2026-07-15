"""Reconstruction reporting helpers for native Sketchformer evaluation."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from metrics.sketchformer.free_running import decode_token_sequence


@dataclass(frozen=True)
class ReconstructionExample:
    """One target/prediction pair prepared for qualitative inspection."""

    target: np.ndarray
    prediction: np.ndarray
    length: int
    source_file: str
    source_index: int
    label: int | None = None
    sample_id: str | None = None
    prediction_length: int | None = None
    decode_mode: str = "teacher-forced"
    coordinate_mode: str = "target-normalized"
    statistics: Mapping[str, float | int | str] = field(default_factory=dict)


def prediction_to_stroke3(output: Any) -> torch.Tensor:
    """Convert a model output object into predicted ``(dx, dy, pen)`` strokes."""

    if output.reconstruction is None:
        raise ValueError("Model output does not include reconstruction predictions")
    if getattr(output.reconstruction, "token_logits", None) is not None:
        raise ValueError("Token predictions require a codebook; use collect_reconstruction_examples")

    xy = output.reconstruction.xy
    pen = torch.argmax(output.reconstruction.pen_logits, dim=-1).to(dtype=xy.dtype)
    return torch.cat([xy, pen.unsqueeze(-1)], dim=-1)


def _batch_lengths(batch: Mapping[str, Any], fallback_mask: torch.Tensor | None) -> list[int]:
    lengths = batch.get("lengths")
    if torch.is_tensor(lengths):
        return [int(value) for value in lengths.detach().cpu().tolist()]
    if lengths is not None:
        return [int(value) for value in lengths]
    if fallback_mask is not None:
        return [int(value) for value in fallback_mask.detach().cpu().sum(dim=1).tolist()]
    return []


def collect_reconstruction_examples(
    output: Any,
    batch: Mapping[str, Any],
    *,
    max_examples: int,
    codebook: np.ndarray | None = None,
    token_layout: Mapping[str, Any] | None = None,
    coordinate_mode: str | None = None,
) -> list[ReconstructionExample]:
    """Collect a small CPU copy of target and predicted stroke3 sequences."""

    if max_examples <= 0:
        return []

    token_logits = (
        None
        if output.reconstruction is None
        else getattr(output.reconstruction, "token_logits", None)
    )
    if token_logits is not None and codebook is None:
        raise ValueError("A tok-dict codebook is required to plot token reconstructions")

    predictions = (
        torch.argmax(token_logits, dim=-1).detach().cpu().numpy()
        if token_logits is not None
        else prediction_to_stroke3(output).detach().cpu().numpy()
    )
    target_tensor = (
        output.loss_targets
        if getattr(output, "loss_targets", None) is not None
        else batch["targets"]
    )
    targets = target_tensor.detach().cpu().numpy()
    valid_mask = (
        output.loss_valid_mask
        if getattr(output, "loss_valid_mask", None) is not None
        else batch.get("valid_mask")
    )
    lengths = _batch_lengths(batch, valid_mask)
    if not lengths:
        lengths = [targets.shape[1]] * targets.shape[0]

    source_files = batch.get("source_files") or [""] * targets.shape[0]
    source_indices = batch.get("source_indices")
    labels = batch.get("labels")
    sample_ids = batch.get("sample_ids")
    if torch.is_tensor(source_indices):
        source_indices = source_indices.detach().cpu().tolist()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().tolist()

    examples: list[ReconstructionExample] = []
    for row in range(min(max_examples, targets.shape[0])):
        length = min(int(lengths[row]), targets.shape[1])
        label = int(labels[row]) if labels is not None else None
        source_index = int(source_indices[row]) if source_indices is not None else row
        if token_logits is not None:
            assert codebook is not None
            target = decode_token_sequence(
                np.asarray(targets[row, :length], dtype=np.int64),
                codebook,
                token_layout=token_layout,
            )
            prediction = decode_token_sequence(
                np.asarray(predictions[row, :length], dtype=np.int64),
                codebook,
                token_layout=token_layout,
            )
        else:
            target = np.asarray(targets[row, :length], dtype=np.float32)
            prediction = np.asarray(predictions[row, :length], dtype=np.float32)
        examples.append(
            ReconstructionExample(
                target=target,
                prediction=prediction,
                length=length,
                source_file=str(source_files[row]),
                source_index=source_index,
                label=label,
                sample_id=str(sample_ids[row]) if sample_ids is not None else None,
                prediction_length=length,
                decode_mode="teacher-forced",
                coordinate_mode=(
                    coordinate_mode
                    or ("canvas" if _is_v3_layout(token_layout) else "target-normalized")
                ),
            )
        )
    return examples


def collect_generated_reconstruction_examples(
    generation: Any,
    batch: Mapping[str, Any],
    *,
    max_examples: int,
    codebook: np.ndarray,
    token_layout: Mapping[str, Any] | None = None,
    records: list[Mapping[str, float | int | str]] | None = None,
    coordinate_mode: str | None = None,
) -> list[ReconstructionExample]:
    """Collect target/free-running prediction pairs for qualitative plots."""

    if max_examples <= 0:
        return []
    predictions = generation.tokens.detach().cpu().numpy()
    prediction_lengths = generation.lengths.detach().cpu().tolist()
    targets = batch["targets"].detach().cpu().numpy()
    lengths = _batch_lengths(batch, batch.get("valid_mask"))
    source_files = batch.get("source_files") or [""] * targets.shape[0]
    source_indices = batch.get("source_indices")
    labels = batch.get("labels")
    sample_ids = batch.get("sample_ids")
    if torch.is_tensor(source_indices):
        source_indices = source_indices.detach().cpu().tolist()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().tolist()

    examples: list[ReconstructionExample] = []
    for row in range(min(max_examples, targets.shape[0])):
        target_length = min(int(lengths[row]), targets.shape[1])
        prediction_length = min(int(prediction_lengths[row]), predictions.shape[1])
        target = decode_token_sequence(
            np.asarray(targets[row, :target_length], dtype=np.int64),
            codebook,
            token_layout=token_layout,
        )
        prediction = decode_token_sequence(
            np.asarray(predictions[row, :prediction_length], dtype=np.int64),
            codebook,
            token_layout=token_layout,
        )
        examples.append(
            ReconstructionExample(
                target=target,
                prediction=prediction,
                length=target_length,
                source_file=str(source_files[row]),
                source_index=(
                    int(source_indices[row]) if source_indices is not None else row
                ),
                label=int(labels[row]) if labels is not None else None,
                sample_id=str(sample_ids[row]) if sample_ids is not None else None,
                prediction_length=prediction_length,
                decode_mode="free-running",
                coordinate_mode=(
                    coordinate_mode
                    or ("canvas" if _is_v3_layout(token_layout) else "target-normalized")
                ),
                statistics=(dict(records[row]) if records is not None else {}),
            )
        )
    return examples


def tensor_logs_to_floats(logs: Mapping[str, torch.Tensor | float | int]) -> dict[str, float]:
    """Convert scalar tensor logs into JSON-safe floats."""

    result: dict[str, float] = {}
    for key, value in logs.items():
        if torch.is_tensor(value):
            result[key] = float(value.detach().cpu())
        else:
            result[key] = float(value)
    return result


def write_metrics_report(
    output_path: str | Path,
    logs: Mapping[str, torch.Tensor | float | int],
    *,
    metadata: Mapping[str, Any] | None = None,
    records: list[Mapping[str, Any]] | None = None,
) -> Path:
    """Write a JSON report with aggregates and optional per-sample diagnostics."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": tensor_logs_to_floats(logs),
        "metadata": dict(metadata or {}),
    }
    if records is not None:
        report["records"] = [_json_safe_mapping(record) for record in records]
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return path


def _json_safe_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values.items():
        if torch.is_tensor(value):
            result[str(key)] = value.detach().cpu().item()
        elif isinstance(value, np.generic):
            result[str(key)] = value.item()
        else:
            result[str(key)] = value
    return result


def _is_v3_layout(token_layout: Mapping[str, Any] | None) -> bool:
    layout = token_layout or {}
    return str(layout.get("version", layout.get("token_layout_version", ""))).lower() in {
        "3",
        "v3",
        "anchored_v3",
    }
