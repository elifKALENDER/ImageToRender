#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageToRender_HybridFuzzyReconstruction_v1.py
=============================================

Hybrid restoration prototype:
    OWL-ViT/SAM/DINOv2 outputs
        +
    expert-validated architectural inventory
        +
    architectural relations / bay sequence / courses
        +
    local CNN patch model
        +
    DINOv2 similarity
        +
    fuzzy evidence fusion
        +
    expert feedback loop
        ->
    reconstruction hypotheses

This is intentionally a single orchestration file.
It does not replace the existing detection / inventory stages.
It consumes their outputs and moves the actual reconstruction decision
into one learnable + fuzzy decision layer.

Core principle:
    "Do not ask a generative model to invent a missing piece first.
     First estimate WHAT SHOULD EXIST and WHY."

Outputs:
    hybrid_reconstruction_hypotheses.json
    hybrid_feature_rows.json
    hybrid_expert_feedback_queue.json
    hybrid_report.txt
    hybrid_overlay.png
    hybrid_masks/*.png

Optional capabilities:
    - DINOv2 embeddings if torch + transformers are installed and a local
      model can be loaded.
    - CNN patch classifier if torch is available.
    - Otherwise, deterministic fallback features remain available.

Expert feedback:
    The generated queue can be reviewed later and appended as:
        feedback: 1.0 / 0.75 / 0.5 / 0.25 / 0.0
    or:
        expert_label: "yes" / "mostly" / "uncertain" / "no"
    and fed back into future training.

IMPORTANT:
    This prototype does not auto-edit the observed inventory.
    It only proposes reconstruction candidates.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Optional ML/DL dependencies
# ---------------------------------------------------------------------------

TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = None

try:
    from transformers import AutoImageProcessor, AutoModel
    TRANSFORMERS_AVAILABLE = True
except Exception:
    AutoImageProcessor = None
    AutoModel = None

try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    output_root: str = "outputs"
    parent_run_dir: str = ""
    train_root: str = "outputs"

    # Candidate generation
    min_damage_ratio: float = 0.20
    min_gap_width_px: int = 15

    # Fuzzy thresholds
    fuzzy_accept: float = 0.72
    fuzzy_review: float = 0.48
    fuzzy_reject: float = 0.30

    # Expert feedback
    expert_feedback_file: str = ""
    use_expert_feedback: bool = True

    # DINOv2
    use_dino: bool = True
    dino_model_id: str = "facebook/dinov2-small"
    dino_device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

    # CNN
    use_cnn: bool = True
    cnn_patch_size: int = 128
    cnn_epochs: int = 4
    cnn_batch_size: int = 8
    cnn_lr: float = 1e-3
    cnn_min_train_rows: int = 16

    # CNN uses synthetic masking from intact local crops.
    # A positive training example is an intact architecture patch,
    # a negative one is a visually unrelated/local background patch.
    cnn_negative_margin: int = 12

    # DINO similarity
    dino_top_k: int = 5

    # Learned geometry
    use_geometry_regressor: bool = True
    geometry_min_rows: int = 8
    random_state: int = 42
    rf_estimators: int = 250
    rf_max_depth: int = 7
    rf_min_samples_leaf: int = 2

    # Safety
    only_inside_damage_mask: bool = True
    never_modify_observed_inventory: bool = True

    # Output
    save_models: bool = True
    save_overlay: bool = True


CFG = Config()

REQUIRED_FILES = (
    "input.png",
    "completion_mask.png",
    "source_expert_validated_inventory.json",
    "stage3c_b_facade_sectors.json",
    "stage3c_b_support_axis_sequence.json",
    "stage3c_b_ordered_bay_sequence.json",
    "stage3c_b_arch_role_subfamilies.json",
    "stage3c_b_pilaster_role_subfamilies.json",
)

COURSE_FILE_VARIANTS = (
    "source_stage3c_horizontal_courses.json",
    "stage3c_horizontal_courses.json",
    "source_stage3c_a_horizontal_courses.json",
)

CLASS_COLORS = {
    "arch": (34, 197, 94),
    "pilaster": (14, 165, 233),
    "cornice": (236, 72, 153),
    "base": (245, 158, 11),
    "opening": (168, 85, 247),
    "other": (148, 163, 184),
}


# =============================================================================
# UTILITIES
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def create_run_dir(root: Path) -> tuple[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    while True:
        run_id = datetime.now().strftime("restore_%Y%m%d_%H%M%S")
        run_dir = root / run_id
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
            return run_id, run_dir
        except FileExistsError:
            time.sleep(1.05)


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"hybrid_reconstruction_{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(formatter)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def clip_box(box: Sequence[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [x1, y1, x2, y2]


def box_width(box: Sequence[float]) -> float:
    return max(1.0, float(box[2]) - float(box[0]))


def box_height(box: Sequence[float]) -> float:
    return max(1.0, float(box[3]) - float(box[1]))


def box_center(box: Sequence[float]) -> tuple[float, float]:
    return (
        (float(box[0]) + float(box[2])) / 2.0,
        (float(box[1]) + float(box[3])) / 2.0,
    )


def safe_div(a: float, b: float, fallback: float = 0.0) -> float:
    if abs(float(b)) < 1e-9:
        return fallback
    return float(a) / float(b)


def median_or(values: Iterable[float], fallback: float) -> float:
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(median(vals)) if vals else float(fallback)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    paths = (
        [
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        if bold else
        [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for path in paths:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    image = Image.open(path).convert("L")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.NEAREST)
    return np.where(np.asarray(image) >= 128, 255, 0).astype(np.uint8)


def mask_ratio(mask: np.ndarray, box: Sequence[float]) -> float:
    x1, y1, x2, y2 = clip_box(box, mask.shape[1], mask.shape[0])
    crop = mask[y1:y2, x1:x2]
    return float(np.mean(crop > 0)) if crop.size else 0.0


def intersection_over_box(mask: np.ndarray, box: Sequence[float]) -> float:
    return mask_ratio(mask, box)


# =============================================================================
# RUN DISCOVERY / INPUT
# =============================================================================

def course_file(run_dir: Path) -> Path | None:
    for name in COURSE_FILE_VARIANTS:
        p = run_dir / name
        if p.exists():
            return p
    return None


def is_compatible_run(run_dir: Path) -> bool:
    return (
        run_dir.is_dir()
        and all((run_dir / name).exists() for name in REQUIRED_FILES)
        and course_file(run_dir) is not None
    )


def find_parent(cfg: Config) -> Path:
    if cfg.parent_run_dir:
        parent = Path(cfg.parent_run_dir)
        if not is_compatible_run(parent):
            missing = [n for n in REQUIRED_FILES if not (parent / n).exists()]
            if course_file(parent) is None:
                missing.append("horizontal_courses")
            raise FileNotFoundError(
                "Parent run hibrit sistemle uyumlu değil: " + ", ".join(missing)
            )
        return parent

    candidates = sorted(
        [p for p in Path(cfg.output_root).glob("restore_*") if p.is_dir()],
        reverse=True,
    )
    for p in candidates:
        if is_compatible_run(p):
            return p

    raise FileNotFoundError("Uygun restore_* parent run bulunamadı.")


def find_train_runs(cfg: Config) -> list[Path]:
    runs = [
        p for p in Path(cfg.train_root).glob("restore_*")
        if p.is_dir() and is_compatible_run(p)
    ]
    return sorted(runs)


def load_run(run_dir: Path) -> dict[str, Any]:
    image = Image.open(run_dir / "input.png").convert("RGB")
    size = image.size

    course_path = course_file(run_dir)

    return {
        "run_dir": run_dir,
        "image": image,
        "width": size[0],
        "height": size[1],
        "damage": load_mask(run_dir / "completion_mask.png", size),
        "inventory": read_json(run_dir / "source_expert_validated_inventory.json"),
        "sectors": read_json(run_dir / "stage3c_b_facade_sectors.json"),
        "supports": read_json(run_dir / "stage3c_b_support_axis_sequence.json"),
        "bays": read_json(run_dir / "stage3c_b_ordered_bay_sequence.json"),
        "arch_roles": read_json(run_dir / "stage3c_b_arch_role_subfamilies.json"),
        "pilaster_roles": read_json(run_dir / "stage3c_b_pilaster_role_subfamilies.json"),
        "courses": read_json(course_path) if course_path else {},
    }


# =============================================================================
# NORMALIZATION
# =============================================================================

def observed_elements(doc: dict[str, Any]) -> list[dict[str, Any]]:
    elements = []
    for idx, item in enumerate(doc["inventory"].get("elements", []), start=1):
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "")).strip().lower()
        if status not in {"observed_intact", "observed_fragmented"}:
            continue

        box = clip_box(
            item.get("bbox_xyxy", [0, 0, 1, 1]),
            doc["width"],
            doc["height"],
        )
        cx, cy = box_center(box)
        elements.append({
            **item,
            "id": str(item.get("id") or f"element_{idx:03d}"),
            "class": str(item.get("class", "other")).lower(),
            "status": status,
            "bbox_xyxy": box,
            "center": [cx, cy],
            "width": box_width(box),
            "height": box_height(box),
        })
    return elements


def element_lookup(elements: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {e["id"]: e for e in elements}


def support_sequence(doc: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for item in doc["supports"].get("supports", []):
        try:
            out.append({
                **item,
                "sequence_index": int(item["sequence_index"]),
                "x": float(item["x"]),
                "primary_element_id": str(item["primary_element_id"]),
            })
        except Exception:
            continue
    return sorted(out, key=lambda s: s["sequence_index"])


def bay_intervals(doc: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for item in doc["bays"].get("adjacent_intervals", []):
        try:
            out.append({
                **item,
                "sequence_index": int(item["sequence_index"]),
                "left_support_index": int(item["left_support_index"]),
                "right_support_index": int(item["right_support_index"]),
                "left_x": float(item["left_x"]),
                "right_x": float(item["right_x"]),
                "gap_width": max(1.0, float(item["right_x"]) - float(item["left_x"])),
                "direct_arch_ids": [str(v) for v in item.get("direct_arch_ids", []) if v],
                "damage_overlap_ratio": float(item.get("damage_overlap_ratio", 0.0)),
            })
        except Exception:
            continue
    return sorted(out, key=lambda x: x["sequence_index"])


# =============================================================================
# STRUCTURAL FEATURES
# =============================================================================

STRUCTURAL_FEATURES = [
    "gap_width_ratio",
    "prev_gap_ratio",
    "next_gap_ratio",
    "damage_overlap",
    "prev_arch",
    "next_arch",
    "neighbor_arch_count",
    "left_support_intact",
    "right_support_intact",
    "left_support_width_ratio",
    "right_support_width_ratio",
    "position_ratio",
]


def build_structural_features(
    interval: dict[str, Any],
    intervals: list[dict[str, Any]],
    supports: list[dict[str, Any]],
    elements: list[dict[str, Any]],
    doc: dict[str, Any],
) -> np.ndarray:
    s_lookup = {s["sequence_index"]: s for s in supports}
    e_lookup = element_lookup(elements)

    seq = interval["sequence_index"]
    prev_i = next((i for i in intervals if i["sequence_index"] == seq - 1), None)
    next_i = next((i for i in intervals if i["sequence_index"] == seq + 1), None)

    left_s = s_lookup.get(interval["left_support_index"])
    right_s = s_lookup.get(interval["right_support_index"])

    left_e = e_lookup.get(left_s["primary_element_id"]) if left_s else None
    right_e = e_lookup.get(right_s["primary_element_id"]) if right_s else None

    median_gap = median_or([i["gap_width"] for i in intervals], interval["gap_width"])
    median_support_w = median_or(
        [e["width"] for e in elements if e["class"] in {"pilaster", "column"}],
        30.0,
    )

    total_span = sum(i["gap_width"] for i in intervals) or 1.0
    position = sum(
        i["gap_width"] for i in intervals
        if i["sequence_index"] < seq
    ) / total_span

    prev_arch = int(bool(prev_i and prev_i.get("direct_arch_ids")))
    next_arch = int(bool(next_i and next_i.get("direct_arch_ids")))

    vector = np.array([
        safe_div(interval["gap_width"], median_gap, 1.0),
        safe_div(prev_i["gap_width"], interval["gap_width"], 1.0) if prev_i else 1.0,
        safe_div(next_i["gap_width"], interval["gap_width"], 1.0) if next_i else 1.0,
        float(interval.get("damage_overlap_ratio", 0.0)),
        prev_arch,
        next_arch,
        prev_arch + next_arch,
        int(left_e is not None and left_e["status"] == "observed_intact"),
        int(right_e is not None and right_e["status"] == "observed_intact"),
        safe_div(left_e["width"], median_support_w, 1.0) if left_e else 1.0,
        safe_div(right_e["width"], median_support_w, 1.0) if right_e else 1.0,
        position,
    ], dtype=np.float32)

    return np.nan_to_num(vector, nan=0.0, posinf=10.0, neginf=-10.0)


# =============================================================================
# DINOv2
# =============================================================================

class DINOFeatureExtractor:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.enabled = False
        self.logger = logger
        self.processor = None
        self.model = None
        self.device = None

        if not cfg.use_dino:
            return
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            logger.warning("[DINO] torch/transformers unavailable; DINO disabled.")
            return

        try:
            self.device = torch.device(cfg.dino_device)
            self.processor = AutoImageProcessor.from_pretrained(cfg.dino_model_id)
            self.model = AutoModel.from_pretrained(cfg.dino_model_id)
            self.model.to(self.device)
            self.model.eval()
            self.enabled = True
            logger.info("[DINO] loaded: %s on %s", cfg.dino_model_id, self.device)
        except Exception as exc:
            logger.warning("[DINO] could not load: %s", exc)

    @torch.no_grad() if TORCH_AVAILABLE else (lambda f: f)
    def encode(self, image: Image.Image) -> np.ndarray | None:
        if not self.enabled:
            return None

        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output = self.model(**inputs)

        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            vec = output.pooler_output[0]
        else:
            vec = output.last_hidden_state[:, 0, :][0]

        vec = F.normalize(vec.float(), dim=0)
        return vec.cpu().numpy().astype(np.float32)

    @staticmethod
    def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            return float("nan")
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-9:
            return 0.0
        return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


# =============================================================================
# CNN
# =============================================================================

if TORCH_AVAILABLE:
    class PatchDataset(Dataset):
        def __init__(self, images: list[np.ndarray], labels: list[int], patch_size: int):
            self.images = images
            self.labels = labels
            self.patch_size = patch_size

        def __len__(self) -> int:
            return len(self.images)

        def __getitem__(self, idx: int):
            arr = self.images[idx]
            if arr.shape[0] != self.patch_size or arr.shape[1] != self.patch_size:
                pil = Image.fromarray(arr)
                pil = pil.resize(
                    (self.patch_size, self.patch_size),
                    Image.Resampling.BILINEAR,
                )
                arr = np.asarray(pil)

            x = arr.astype(np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))
            return torch.tensor(x), torch.tensor(self.labels[idx], dtype=torch.long)


    class SmallPatchCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 24, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(24, 48, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(48, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.head(self.features(x)).squeeze(1)


def crop_patch(image: Image.Image, box: Sequence[float], patch_size: int) -> np.ndarray:
    width, height = image.size
    x1, y1, x2, y2 = clip_box(box, width, height)
    crop = image.crop((x1, y1, x2, y2)).convert("RGB")
    crop = crop.resize((patch_size, patch_size), Image.Resampling.BILINEAR)
    return np.asarray(crop)


def build_cnn_training_set(
    train_docs: list[dict[str, Any]],
    cfg: Config,
) -> tuple[list[np.ndarray], list[int]]:
    images = []
    labels = []

    for doc in train_docs:
        elems = observed_elements(doc)
        for e in elems:
            # Positives: observed intact architecture pieces.
            if e["class"] not in {"arch", "pilaster", "cornice", "base"}:
                continue
            if e["status"] != "observed_intact":
                continue

            images.append(crop_patch(doc["image"], e["bbox_xyxy"], cfg.cnn_patch_size))
            labels.append(1)

        # Negatives: random regions outside known observed element boxes.
        # This is intentionally weak/self-supervised-ish and should not be
        # mistaken for a high-quality semantic negative dataset.
        observed_boxes = [e["bbox_xyxy"] for e in elems]
        w, h = doc["width"], doc["height"]

        attempts = 0
        while attempts < max(5, len(images) // 8) and len(labels) < len(images) * 2:
            attempts += 1
            pw = max(24, int(np.median([
                e["width"] for e in elems
            ])) if elems else 60)
            ph = max(24, int(np.median([
                e["height"] for e in elems
            ])) if elems else 80)

            x1 = random.randint(0, max(0, w - pw - 1))
            y1 = random.randint(0, max(0, h - ph - 1))
            box = [x1, y1, x1 + pw, y1 + ph]

            overlap = False
            for ob in observed_boxes:
                ix1 = max(box[0], ob[0])
                iy1 = max(box[1], ob[1])
                ix2 = min(box[2], ob[2])
                iy2 = min(box[3], ob[3])
                if max(0, ix2 - ix1) * max(0, iy2 - iy1) > 0.25 * pw * ph:
                    overlap = True
                    break

            if overlap:
                continue

            images.append(crop_patch(doc["image"], box, cfg.cnn_patch_size))
            labels.append(0)

    # Balance lightly.
    pos = [x for x, y in zip(images, labels) if y == 1]
    neg = [x for x, y in zip(images, labels) if y == 0]
    n = min(len(pos), len(neg))
    if n == 0:
        return [], []

    pos = pos[:n]
    neg = neg[:n]
    return pos + neg, [1] * n + [0] * n


class CNNPatchScorer:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.model = None
        self.device = None
        self.enabled = False

        if not cfg.use_cnn:
            return
        if not TORCH_AVAILABLE:
            logger.warning("[CNN] torch unavailable; CNN disabled.")
            return

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def train(
        self,
        train_docs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        info = {
            "enabled": False,
            "rows": 0,
            "positive": 0,
            "negative": 0,
            "epochs": 0,
        }

        if not TORCH_AVAILABLE:
            return info

        images, labels = build_cnn_training_set(train_docs, self.cfg)

        info["rows"] = len(labels)
        info["positive"] = int(sum(labels))
        info["negative"] = int(len(labels) - sum(labels))

        if len(labels) < self.cfg.cnn_min_train_rows:
            self.logger.warning(
                "[CNN] insufficient rows: %d < %d",
                len(labels), self.cfg.cnn_min_train_rows
            )
            return info

        self.model = SmallPatchCNN().to(self.device)
        self.model.train()

        dataset = PatchDataset(images, labels, self.cfg.cnn_patch_size)
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.cnn_batch_size,
            shuffle=True,
        )

        optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.cnn_lr)
        loss_fn = nn.BCEWithLogitsLoss()

        for epoch in range(self.cfg.cnn_epochs):
            running = 0.0
            count = 0

            for x, y in loader:
                x = x.to(self.device)
                y = y.float().to(self.device)

                optim.zero_grad()
                logits = self.model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optim.step()

                running += float(loss.item()) * len(y)
                count += len(y)

            self.logger.info(
                "[CNN] epoch=%d/%d loss=%.4f",
                epoch + 1,
                self.cfg.cnn_epochs,
                safe_div(running, count, 0.0)
            )

        self.model.eval()
        self.enabled = True
        info["enabled"] = True
        info["epochs"] = self.cfg.cnn_epochs
        return info

    @torch.no_grad() if TORCH_AVAILABLE else (lambda f: f)
    def score_patch(self, image: Image.Image, box: Sequence[float]) -> float:
        if not self.enabled or self.model is None:
            return float("nan")

        arr = crop_patch(image, box, self.cfg.cnn_patch_size).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)
        logit = self.model(x)
        return float(torch.sigmoid(logit)[0].cpu().item())


# =============================================================================
# DINO REFERENCE DATABASE
# =============================================================================

def build_dino_reference_db(
    train_docs: list[dict[str, Any]],
    extractor: DINOFeatureExtractor,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    if not extractor.enabled:
        return []

    refs = []

    for doc in train_docs:
        elems = observed_elements(doc)

        for e in elems:
            if e["class"] not in {"arch", "pilaster", "cornice", "base"}:
                continue
            if e["status"] != "observed_intact":
                continue

            patch = doc["image"].crop(tuple(e["bbox_xyxy"]))
            vec = extractor.encode(patch)
            if vec is None:
                continue

            refs.append({
                "run": doc["run_dir"].name,
                "id": e["id"],
                "class": e["class"],
                "bbox": e["bbox_xyxy"],
                "embedding": vec.tolist(),
            })

    logger.info("[DINO] reference elements: %d", len(refs))
    return refs


def dino_class_similarity(
    extractor: DINOFeatureExtractor,
    image: Image.Image,
    box: Sequence[int],
    refs: list[dict[str, Any]],
    class_name: str,
    top_k: int,
) -> tuple[float, list[str]]:
    if not extractor.enabled or not refs:
        return float("nan"), []

    target_patch = image.crop(tuple(box))
    target = extractor.encode(target_patch)
    if target is None:
        return float("nan"), []

    scored = []
    for ref in refs:
        if ref["class"] != class_name:
            continue
        vec = np.asarray(ref["embedding"], dtype=np.float32)
        scored.append(
            (
                extractor.cosine(target, vec),
                f"{ref['run']}:{ref['id']}",
            )
        )

    if not scored:
        return float("nan"), []

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:top_k]

    # Convert cosine [-1, 1] -> fuzzy-ish [0, 1]
    score = float(np.mean([0.5 * (s + 1.0) for s, _ in top]))
    return score, [ref_id for _, ref_id in top]


# =============================================================================
# FUZZY LOGIC
# =============================================================================

def ramp_up(x: float, low: float, high: float) -> float:
    if x <= low:
        return 0.0
    if x >= high:
        return 1.0
    return (x - low) / max(1e-9, high - low)


def ramp_down(x: float, high: float, low: float) -> float:
    if x <= low:
        return 1.0
    if x >= high:
        return 0.0
    return (high - x) / max(1e-9, high - low)


def fuzzy_membership(value: float, kind: str) -> dict[str, float]:
    """
    Small, transparent fuzzy layer.

    Each input is converted to:
        low
        medium
        high

    This intentionally stays simple and interpretable.
    """
    x = float(np.clip(value, 0.0, 1.0))

    low = ramp_down(x, 0.60, 0.20)
    high = ramp_up(x, 0.40, 0.80)
    medium = max(0.0, 1.0 - abs(x - 0.50) / 0.30)

    return {
        "low": float(np.clip(low, 0.0, 1.0)),
        "medium": float(np.clip(medium, 0.0, 1.0)),
        "high": float(np.clip(high, 0.0, 1.0)),
    }


def fuzzify_evidence(
    structural: float,
    dino: float,
    cnn: float,
    geometry: float,
    damage: float,
    expert_prior: float,
) -> dict[str, Any]:
    evidence = {
        "structural": structural,
        "dino": dino,
        "cnn": cnn,
        "geometry": geometry,
        "damage": damage,
        "expert_prior": expert_prior,
    }

    memberships = {
        key: fuzzy_membership(value, key)
        for key, value in evidence.items()
        if np.isfinite(value)
    }

    return memberships


def fuzzy_inference(
    structural: float,
    dino: float,
    cnn: float,
    geometry: float,
    damage: float,
    expert_prior: float,
) -> dict[str, Any]:
    """
    Interpretable fuzzy rule layer.

    High confidence:
        strong structural + strong geometry + strong visual evidence
    Medium confidence:
        strong structure + one strong visual source
    Low confidence:
        weak structure even if one visual source is strong

    Missing optional signals are ignored and weights are renormalized.
    """
    terms = {
        "structural": structural,
        "dino": dino,
        "cnn": cnn,
        "geometry": geometry,
        "damage": damage,
        "expert_prior": expert_prior,
    }

    valid = {k: float(v) for k, v in terms.items() if np.isfinite(v)}
    if not valid:
        return {
            "confidence": 0.0,
            "state": "reject",
            "rules": [],
            "memberships": {},
        }

    # Strong individual evidence.
    strong = {
        k: fuzzy_membership(v, k)["high"]
        for k, v in valid.items()
    }

    # Core architectural safety:
    # damage + structure are necessary for an aggressive proposal.
    structural_high = strong.get("structural", 0.0)
    geometry_high = strong.get("geometry", 0.0)
    damage_high = strong.get("damage", 0.0)
    dino_high = strong.get("dino", 0.0)
    cnn_high = strong.get("cnn", 0.0)
    expert_high = strong.get("expert_prior", 0.0)

    rules = []

    r1 = min(structural_high, geometry_high, damage_high)
    rules.append(("strong_structure_and_geometry", r1, 0.86))

    visual_high = max(dino_high, cnn_high)
    r2 = min(structural_high, visual_high, damage_high)
    rules.append(("structure_plus_visual", r2, 0.78))

    r3 = min(structural_high, dino_high, cnn_high)
    rules.append(("two_visual_sources_plus_structure", r3, 0.88))

    r4 = min(structural_high, expert_high, damage_high)
    rules.append(("expert_prior_reinforcement", r4, 0.84))

    # Weak structure is actively penalized.
    weak_structure = fuzzy_membership(valid.get("structural", 0.0), "structural")["low"]
    penalty = 0.20 * weak_structure

    # Aggregate by weighted average of active rules.
    numerator = sum(fire * weight for _, fire, weight in rules)
    denominator = sum(fire for _, fire, _ in rules)

    confidence = safe_div(numerator, denominator, 0.0)
    confidence = float(np.clip(confidence - penalty, 0.0, 1.0))

    if confidence >= CFG.fuzzy_accept:
        state = "accept"
    elif confidence >= CFG.fuzzy_review:
        state = "review"
    else:
        state = "reject"

    return {
        "confidence": confidence,
        "state": state,
        "rules": [
            {"rule": name, "fire": float(fire), "weight": float(weight)}
            for name, fire, weight in rules
        ],
        "memberships": {
            key: fuzzy_membership(value, key)
            for key, value in valid.items()
        },
    }


# =============================================================================
# EXPERT FEEDBACK
# =============================================================================

def load_expert_feedback(path: str) -> dict[str, float]:
    if not path:
        return {}

    p = Path(path)
    if not p.exists():
        return {}

    doc = read_json(p)
    feedback = {}

    if isinstance(doc, dict):
        rows = doc.get("feedback", [])
    elif isinstance(doc, list):
        rows = doc
    else:
        rows = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        hyp_id = str(row.get("hypothesis_id", "")).strip()
        if not hyp_id:
            continue

        if row.get("feedback") is not None:
            try:
                feedback[hyp_id] = float(np.clip(float(row["feedback"]), 0.0, 1.0))
                continue
            except Exception:
                pass

        mapping = {
            "yes": 1.0,
            "mostly": 0.75,
            "uncertain": 0.50,
            "mostly_no": 0.25,
            "no": 0.0,
        }
        label = str(row.get("expert_label", "")).lower().strip()
        if label in mapping:
            feedback[hyp_id] = mapping[label]

    return feedback


def build_expert_queue(hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue = []

    for hyp in hypotheses:
        queue.append({
            "hypothesis_id": hyp["id"],
            "question": (
                f"Bu {hyp['class']} önerisi mevcut yapının mimari dili, "
                f"ritmi ve geometrik devamlılığı ile uyumlu mu?"
            ),
            "suggested_options": [
                "yes",
                "mostly",
                "uncertain",
                "mostly_no",
                "no",
            ],
            "feedback": None,
            "expert_label": None,
            "note": "",
        })

    return queue


# =============================================================================
# GEOMETRY
# =============================================================================

def observed_arch_geometry(
    doc: dict[str, Any],
) -> dict[str, float]:
    elems = [
        e for e in observed_elements(doc)
        if e["class"] == "arch" and e["status"] == "observed_intact"
    ]

    if not elems:
        return {
            "width": 100.0,
            "height": max(40.0, doc["height"] * 0.15),
            "top": doc["height"] * 0.30,
            "bottom": doc["height"] * 0.52,
        }

    return {
        "width": median_or([e["width"] for e in elems], 100.0),
        "height": median_or([e["height"] for e in elems], 100.0),
        "top": median_or([e["bbox_xyxy"][1] for e in elems], doc["height"] * 0.30),
        "bottom": median_or([e["bbox_xyxy"][3] for e in elems], doc["height"] * 0.52),
    }


def predicted_arch_box(
    doc: dict[str, Any],
    interval: dict[str, Any],
    geometry_model: Any = None,
) -> list[int]:
    supports = support_sequence(doc)
    lookup = {s["sequence_index"]: s for s in supports}

    left = lookup.get(interval["left_support_index"])
    right = lookup.get(interval["right_support_index"])

    if left is None or right is None:
        return [0, 0, 1, 1]

    left_x = float(left["x"])
    right_x = float(right["x"])
    span = max(1.0, right_x - left_x)

    geo = observed_arch_geometry(doc)

    if geometry_model is not None:
        try:
            features = build_structural_features(
                interval,
                bay_intervals(doc),
                supports,
                observed_elements(doc),
                doc,
            )
            pred = geometry_model.predict(features.reshape(1, -1))[0]
            rel_left = float(pred[0])
            rel_right = float(pred[1])
            width_ratio = float(np.clip(pred[2], 0.45, 1.20))

            x1 = left_x + rel_left * span
            x2 = right_x + rel_right * span

            center = (x1 + x2) / 2.0
            width = max(20.0, span * width_ratio)
            x1 = center - width / 2.0
            x2 = center + width / 2.0
        except Exception:
            x1 = (left_x + right_x) / 2.0 - geo["width"] / 2.0
            x2 = (left_x + right_x) / 2.0 + geo["width"] / 2.0
    else:
        target_width = min(geo["width"], span * 1.10)
        center = (left_x + right_x) / 2.0
        x1 = center - target_width / 2.0
        x2 = center + target_width / 2.0

    return clip_box(
        [x1, geo["top"], x2, geo["bottom"]],
        doc["width"],
        doc["height"],
    )


# =============================================================================
# TRAINING DATA FOR GEOMETRY
# =============================================================================

def geometry_training_rows(
    docs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []

    for doc in docs:
        intervals = bay_intervals(doc)
        supports = support_sequence(doc)
        elements = observed_elements(doc)
        e_lookup = element_lookup(elements)
        s_lookup = {s["sequence_index"]: s for s in supports}

        for interval in intervals:
            ids = interval.get("direct_arch_ids", [])
            if not ids:
                continue

            arches = [e_lookup[i] for i in ids if i in e_lookup]
            if not arches:
                continue

            arch = max(arches, key=lambda e: e["width"] * e["height"])
            left = s_lookup.get(interval["left_support_index"])
            right = s_lookup.get(interval["right_support_index"])
            if not left or not right:
                continue

            span = max(1.0, right["x"] - left["x"])
            box = arch["bbox_xyxy"]

            X = build_structural_features(
                interval,
                intervals,
                supports,
                elements,
                doc,
            )

            y = np.array([
                safe_div(box[0] - left["x"], span, 0.0),
                safe_div(box[2] - right["x"], span, 0.0),
                safe_div(box_width(box), span, 1.0),
            ], dtype=np.float32)

            rows.append({
                "run": doc["run_dir"].name,
                "features": X.tolist(),
                "target": y.tolist(),
            })

    return rows


def train_geometry_model(
    docs: list[dict[str, Any]],
    cfg: Config,
    logger: logging.Logger,
):
    if not cfg.use_geometry_regressor or not SKLEARN_AVAILABLE:
        return None, {
            "enabled": False,
            "reason": "disabled_or_sklearn_unavailable",
        }

    rows = geometry_training_rows(docs)

    if len(rows) < cfg.geometry_min_rows:
        return None, {
            "enabled": False,
            "reason": f"insufficient_rows_{len(rows)}",
            "rows": len(rows),
        }

    X = np.asarray([r["features"] for r in rows], dtype=np.float32)
    y = np.asarray([r["target"] for r in rows], dtype=np.float32)

    model = RandomForestRegressor(
        n_estimators=cfg.rf_estimators,
        max_depth=cfg.rf_max_depth,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    model.fit(X, y)

    logger.info("[GEOMETRY ML] trained rows=%d", len(rows))

    return model, {
        "enabled": True,
        "model": "RandomForestRegressor",
        "rows": len(rows),
    }


# =============================================================================
# CANDIDATE GENERATION
# =============================================================================

def generate_candidates(
    doc: dict[str, Any],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    intervals = bay_intervals(doc)
    candidates = []

    for interval in intervals:
        if interval.get("direct_arch_ids"):
            continue

        damage = float(interval.get("damage_overlap_ratio", 0.0))
        if damage < CFG.min_damage_ratio:
            continue

        if interval["gap_width"] < CFG.min_gap_width_px:
            continue

        candidates.append({
            "candidate_type": "missing_arch_between_observed_supports",
            "class": "arch",
            "interval": interval,
            "damage": damage,
        })

    logger.info("[CANDIDATES] missing-arch candidates=%d", len(candidates))
    return candidates


# =============================================================================
# HYBRID SCORING
# =============================================================================

def structural_score(interval: dict[str, Any], doc: dict[str, Any]) -> float:
    intervals = bay_intervals(doc)
    supports = support_sequence(doc)
    elements = observed_elements(doc)

    feat = build_structural_features(
        interval,
        intervals,
        supports,
        elements,
        doc,
    )

    # Features:
    # width, prev/next consistency, damage, neighbors, support integrity...
    width_consistency = math.exp(-abs(float(feat[0]) - 1.0))
    neighbor = float(feat[4] + feat[5]) / 2.0
    support = float(feat[7] + feat[8]) / 2.0
    damage = float(feat[3])

    score = (
        0.25 * width_consistency
        + 0.30 * neighbor
        + 0.20 * support
        + 0.25 * damage
    )
    return float(np.clip(score, 0.0, 1.0))


def geometry_consistency_score(
    interval: dict[str, Any],
    doc: dict[str, Any],
) -> float:
    supports = support_sequence(doc)
    support_lookup = {s["sequence_index"]: s for s in supports}
    left = support_lookup.get(interval["left_support_index"])
    right = support_lookup.get(interval["right_support_index"])

    if not left or not right:
        return 0.0

    span = max(1.0, right["x"] - left["x"])
    geo = observed_arch_geometry(doc)

    # Is the expected median width compatible with this bay span?
    width_ratio = safe_div(geo["width"], span, 1.0)
    width_term = 1.0 - min(1.0, abs(width_ratio - 0.85) / 0.85)

    # Height / top / bottom are stable across observed intact arches.
    arches = [
        e for e in observed_elements(doc)
        if e["class"] == "arch" and e["status"] == "observed_intact"
    ]
    if not arches:
        vertical_term = 0.5
    else:
        top_std = np.std([e["bbox_xyxy"][1] for e in arches])
        bottom_std = np.std([e["bbox_xyxy"][3] for e in arches])
        vertical_scale = max(1.0, doc["height"])
        vertical_term = math.exp(-(top_std + bottom_std) / vertical_scale * 5.0)

    return float(np.clip(0.60 * width_term + 0.40 * vertical_term, 0.0, 1.0))


def evaluate_candidate(
    candidate: dict[str, Any],
    doc: dict[str, Any],
    dino_extractor: DINOFeatureExtractor,
    dino_refs: list[dict[str, Any]],
    cnn_scorer: CNNPatchScorer,
    geometry_model: Any,
    expert_feedback: dict[str, float],
) -> dict[str, Any]:
    interval = candidate["interval"]

    structural = structural_score(interval, doc)
    geometry = geometry_consistency_score(interval, doc)

    box = predicted_arch_box(doc, interval, geometry_model)

    dino_score, dino_refs_used = dino_class_similarity(
        dino_extractor,
        doc["image"],
        box,
        dino_refs,
        "arch",
        CFG.dino_top_k,
    )

    cnn_score = cnn_scorer.score_patch(doc["image"], box)

    # If no feedback for current id, neutral prior.
    expert_prior = 0.50

    fuzzy = fuzzy_inference(
        structural=structural,
        dino=dino_score,
        cnn=cnn_score,
        geometry=geometry,
        damage=candidate["damage"],
        expert_prior=expert_prior,
    )

    return {
        "id": f"hybrid_arch_{len(expert_feedback) + random.randint(1000, 9999)}",
        "class": "arch",
        "status": fuzzy["state"],
        "bbox_xyxy": box,
        "confidence": round(float(fuzzy["confidence"]), 6),
        "source": "hybrid_fuzzy_reconstruction",
        "bay_id": interval.get("id"),
        "sequence_index": interval["sequence_index"],
        "damage_overlap_ratio": round(float(candidate["damage"]), 6),
        "evidence": {
            "structural": round(structural, 6),
            "geometry": round(geometry, 6),
            "dino_similarity": None if np.isnan(dino_score) else round(float(dino_score), 6),
            "cnn_probability": None if np.isnan(cnn_score) else round(float(cnn_score), 6),
            "damage": round(float(candidate["damage"]), 6),
            "expert_prior": expert_prior,
        },
        "dino_reference_ids": dino_refs_used,
        "fuzzy": fuzzy,
        "expert_feedback": None,
        "expert_label": None,
        "next_action": (
            "accept_candidate"
            if fuzzy["state"] == "accept"
            else "expert_review"
        ),
    }


# =============================================================================
# OUTPUT MASK
# =============================================================================

def build_arch_mask(size: tuple[int, int], box: Sequence[int]) -> np.ndarray:
    width, height = size
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)

    x1, y1, x2, y2 = clip_box(box, width, height)
    cx = (x1 + x2) / 2.0
    rx = max(1.0, (x2 - x1) / 2.0)
    ry = max(1.0, y2 - y1)
    thick = max(2, int(min(x2 - x1, y2 - y1) * 0.10))

    for t in range(thick):
        b = [
            cx - rx + t,
            y2 - ry + t,
            cx + rx - t,
            y2 + ry * 0.72 - t,
        ]
        draw.arc(b, 180, 360, fill=255, width=1)

    return np.asarray(img)


def build_hypothesis_mask(hyp: dict[str, Any], size: tuple[int, int]) -> np.ndarray:
    if hyp["class"] == "arch":
        return build_arch_mask(size, hyp["bbox_xyxy"])

    width, height = size
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = clip_box(hyp["bbox_xyxy"], width, height)
    draw.rectangle([x1, y1, x2, y2], fill=255)
    return np.asarray(img)


# =============================================================================
# MAIN
# =============================================================================

def run(cfg: Config) -> Path:
    set_seed(cfg.random_state)

    parent = find_parent(cfg)
    run_id, run_dir = create_run_dir(Path(cfg.output_root))
    logger = setup_logger(run_dir)

    logger.info("=" * 90)
    logger.info("[HYBRID FUZZY RECONSTRUCTION] %s", run_id)
    logger.info("[PARENT] %s", parent.resolve())
    logger.info("=" * 90)

    parent_doc = load_run(parent)
    train_runs = find_train_runs(cfg)
    if parent not in train_runs:
        train_runs.append(parent)

    train_runs = sorted(set(train_runs), key=lambda p: str(p))
    train_docs = [load_run(p) for p in train_runs]

    logger.info(
        "[DATA] parent=%s | training structures=%d",
        parent.name,
        len(train_docs),
    )

    # -------------------------------
    # DINO
    # -------------------------------
    dino = DINOFeatureExtractor(cfg, logger)
    dino_refs = build_dino_reference_db(train_docs, dino, logger)

    # -------------------------------
    # CNN
    # -------------------------------
    cnn = CNNPatchScorer(cfg, logger)
    cnn_info = cnn.train(train_docs)

    # -------------------------------
    # Geometry ML
    # -------------------------------
    geometry_model, geometry_info = train_geometry_model(
        train_docs,
        cfg,
        logger,
    )

    # -------------------------------
    # Expert feedback
    # -------------------------------
    feedback = {}
    if cfg.use_expert_feedback:
        feedback = load_expert_feedback(cfg.expert_feedback_file)

    # -------------------------------
    # Candidates + hybrid inference
    # -------------------------------
    candidates = generate_candidates(parent_doc, logger)
    hypotheses = []

    for candidate in candidates:
        hyp = evaluate_candidate(
            candidate,
            parent_doc,
            dino,
            dino_refs,
            cnn,
            geometry_model,
            feedback,
        )

        # Safety: proposal must overlap completion mask.
        if cfg.only_inside_damage_mask:
            overlap = intersection_over_box(
                parent_doc["damage"],
                hyp["bbox_xyxy"],
            )
            hyp["damage_geometry_overlap"] = round(float(overlap), 6)

            if overlap < 0.20:
                hyp["status"] = "reject"
                hyp["next_action"] = "reject_mask_mismatch"

        hypotheses.append(hyp)

    # Make IDs deterministic after scoring.
    for idx, hyp in enumerate(hypotheses, start=1):
        hyp["id"] = f"hybrid_arch_{idx:03d}"

    # Queue all non-rejected candidates.
    queue = build_expert_queue(
        [h for h in hypotheses if h["status"] != "reject"]
    )

    # Copy immutable input references.
    shutil.copy2(parent / "input.png", run_dir / "input.png")
    shutil.copy2(parent / "completion_mask.png", run_dir / "completion_mask.png")

    # Save metadata / rows.
    model_info = {
        "created_at": now_iso(),
        "parent_run": parent.name,
        "training_structures": [p.name for p in train_runs],
        "dino": {
            "enabled": dino.enabled,
            "model_id": cfg.dino_model_id,
            "device": str(dino.device) if dino.enabled else None,
            "references": len(dino_refs),
        },
        "cnn": cnn_info,
        "geometry": geometry_info,
        "fuzzy": {
            "accept": cfg.fuzzy_accept,
            "review": cfg.fuzzy_review,
            "reject": cfg.fuzzy_reject,
        },
        "feature_names": STRUCTURAL_FEATURES,
        "expert_feedback_rows": len(feedback),
    }

    write_json(run_dir / "hybrid_model_info.json", model_info)
    write_json(
        run_dir / "hybrid_reconstruction_hypotheses.json",
        {
            "run_id": run_id,
            "stage": "hybrid_fuzzy_reconstruction",
            "policy": {
                "observed_inventory_immutable": True,
                "expert_review_required": True,
                "only_completion_mask": True,
                "stable_diffusion_not_used": True,
            },
            "hypotheses": hypotheses,
        },
    )

    write_json(
        run_dir / "hybrid_expert_feedback_queue.json",
        {
            "run_id": run_id,
            "instructions": (
                "Uzman her öneri için yes/mostly/uncertain/mostly_no/no seçebilir "
                "veya 0..1 arasında feedback verebilir."
            ),
            "feedback": queue,
        },
    )

    if dino_refs:
        write_json(
            run_dir / "hybrid_dino_reference_index.json",
            [
                {
                    "run": r["run"],
                    "id": r["id"],
                    "class": r["class"],
                    "bbox": r["bbox"],
                }
                for r in dino_refs
            ],
        )

    # -------------------------------
    # Overlay
    # -------------------------------
    if cfg.save_overlay:
        canvas = parent_doc["image"].convert("RGBA")

        damage_layer = Image.new("RGBA", canvas.size, (239, 68, 68, 0))
        damage_alpha = Image.fromarray(
            np.where(parent_doc["damage"] > 0, 45, 0).astype(np.uint8),
            mode="L",
        )
        damage_layer.putalpha(damage_alpha)
        canvas = Image.alpha_composite(canvas, damage_layer)

        draw = ImageDraw.Draw(canvas, "RGBA")
        font = load_font(14, bold=True)

        for hyp in hypotheses:
            if hyp["status"] == "reject":
                color = (239, 68, 68)
            elif hyp["status"] == "accept":
                color = (34, 197, 94)
            else:
                color = (245, 158, 11)

            x1, y1, x2, y2 = hyp["bbox_xyxy"]

            draw.rectangle(
                [x1, y1, x2, y2],
                outline=(*color, 255),
                width=3,
            )
            label = (
                f"{hyp['id']} "
                f"{hyp['status']} "
                f"{hyp['confidence']:.2f}"
            )
            draw.text(
                (x1 + 3, max(0, y1 - 18)),
                label,
                fill=(*color, 255),
                font=font,
            )

        canvas.convert("RGB").save(run_dir / "hybrid_overlay.png")

    # -------------------------------
    # Individual masks
    # -------------------------------
    mask_dir = run_dir / "hybrid_masks"
    mask_dir.mkdir(exist_ok=True)

    for hyp in hypotheses:
        mask = build_hypothesis_mask(
            hyp,
            (parent_doc["width"], parent_doc["height"]),
        )

        if cfg.only_inside_damage_mask:
            mask = np.where(parent_doc["damage"] > 0, mask, 0).astype(np.uint8)

        Image.fromarray(mask, mode="L").save(
            mask_dir / f"{hyp['id']}.png"
        )

    # -------------------------------
    # Human-readable report
    # -------------------------------
    counts = {
        "accept": sum(h["status"] == "accept" for h in hypotheses),
        "review": sum(h["status"] == "review" for h in hypotheses),
        "reject": sum(h["status"] == "reject" for h in hypotheses),
    }

    report = [
        "HYBRID FUZZY ARCHITECTURAL RECONSTRUCTION",
        f"Run: {run_id}",
        f"Parent: {parent.resolve()}",
        "",
        "SYSTEM:",
        "- Existing OWL-ViT/SAM/DINOv2 outputs are treated as upstream evidence.",
        "- Structural architectural relations are treated as a separate evidence channel.",
        "- CNN is used for local visual patch evidence when torch is available.",
        "- DINOv2 is used for similarity to observed intact architectural elements when available.",
        "- Fuzzy inference combines evidence instead of relying on one hard threshold.",
        "- Expert feedback is saved as a reusable learning signal.",
        "",
        "MODELS:",
        f"- DINO enabled: {dino.enabled}",
        f"- CNN enabled: {cnn_info.get('enabled', False)}",
        f"- Geometry ML enabled: {geometry_info.get('enabled', False)}",
        "",
        "HYPOTHESES:",
        f"- Total: {len(hypotheses)}",
        f"- Accept: {counts['accept']}",
        f"- Review: {counts['review']}",
        f"- Reject: {counts['reject']}",
        "",
        "EXPERT LOOP:",
        "- Review accepted/review proposals.",
        "- Fill hybrid_expert_feedback_queue.json.",
        "- Feed that file into --expert-feedback on a later run.",
        "",
    ]

    for h in hypotheses:
        report.append(
            f"- {h['id']} | state={h['status']} | conf={h['confidence']:.3f} | "
            f"struct={h['evidence']['structural']:.3f} | "
            f"geom={h['evidence']['geometry']:.3f} | "
            f"dino={h['evidence']['dino_similarity']} | "
            f"cnn={h['evidence']['cnn_probability']}"
        )

    (run_dir / "hybrid_report.txt").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "[DONE] hypotheses=%d accept=%d review=%d reject=%d",
        len(hypotheses),
        counts["accept"],
        counts["review"],
        counts["reject"],
    )
    logger.info("[OUTPUT] %s", run_dir.resolve())

    return run_dir


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ImageToRender hibrit ML + DINOv2 + CNN + fuzzy reconstruction."
    )
    parser.add_argument("--parent-run", default="")
    parser.add_argument("--train-root", default="outputs")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--expert-feedback", default="")
    parser.add_argument("--no-dino", action="store_true")
    parser.add_argument("--no-cnn", action="store_true")
    parser.add_argument("--fuzzy-accept", type=float, default=CFG.fuzzy_accept)
    parser.add_argument("--fuzzy-review", type=float, default=CFG.fuzzy_review)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cfg = Config(
        parent_run_dir=args.parent_run,
        train_root=args.train_root,
        output_root=args.output_root,
        expert_feedback_file=args.expert_feedback,
        use_dino=not args.no_dino,
        use_cnn=not args.no_cnn,
        fuzzy_accept=args.fuzzy_accept,
        fuzzy_review=args.fuzzy_review,
    )

    try:
        run(cfg)
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
