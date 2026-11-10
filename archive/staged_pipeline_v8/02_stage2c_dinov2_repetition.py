#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageToRender2Restoration_v8_stage2_dinov2_repetition.py
========================================================
V8 / STAGE 2C — DINOv2 ile yapı-içi tekrar ve parçalı motif analizi

ÖNEMLİ GÜVENLİK KURALI
----------------------
Bu sürüm pilaster/sütun EKLEMEZ ve eksik modül üretmez.
Yalnızca fotoğrafta gerçekten bulunan görsel özelliklerden tekrar eden düşey
motif eksenlerini arar. Bir eksen hasar maskesiyle kısmen kesiliyorsa fakat
sağlam görünen parçaları başka eksenlerle güçlü biçimde benzeşiyorsa bunu
"fragmented_repeated_motif" olarak raporlar.

OWL-ViT sonuçları varsa yalnız etiket desteği verir; tek başına eksen/adayı
oluşturamaz.

Girdi (en son outputs/restore_* klasöründe):
    input.png
    completion_mask.png
    elements.json              isteğe bağlı yardımcı etiket desteği

Çıktılar:
    dinov2_facade_crop.png
    dinov2_similarity_heatmap.png
    dinov2_repetition_profile.png
    dinov2_repetition_axes.png
    repetition_axes.json
    stage2_dinov2_report.txt
    stage2_dinov2_meta.json
    stage2_dinov2.log

Bu aşamada Stable Diffusion / ControlNet / restorasyon üretimi YOKTUR.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import platform
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable

# Hugging Face ayarları transformers importundan önce olmalı.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings(
    "ignore",
    message=r".*torch\.utils\._pytree\._register_pytree_node.*",
    category=FutureWarning,
)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    output_root: str = "outputs"
    analysis_run_dir: str = ""  # boşsa en güncel restore_* klasörü

    model_id: str = "facebook/dinov2-small"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16_if_cuda: bool = True

    # Hasar çevresinden cephe çalışma alanı.
    facade_expand_x_ratio: float = 0.18
    facade_expand_y_top_ratio: float = 0.16
    facade_expand_y_bottom_ratio: float = 0.08
    fallback_x1_ratio: float = 0.08
    fallback_x2_ratio: float = 0.92
    fallback_y1_ratio: float = 0.18
    fallback_y2_ratio: float = 0.82

    # DINOv2 giriş çözünürlüğü. Boyutlar patch_size katına yuvarlanır.
    target_long_side: int = 700
    minimum_short_side: int = 336

    # Düşey şerit tanımı (patch sütunu sayısı).
    strip_widths_patches: tuple[int, ...] = (2, 3, 4, 5)
    vertical_bins: int = 8
    min_visible_patches_per_bin: int = 2
    min_shared_bins: int = 3
    min_axis_separation_ratio: float = 0.075

    # Tekrar kabul eşikleri. Kararsız durumda hiçbir eksen üretmemek tercih edilir.
    min_pair_similarity: float = 0.78
    min_similarity_prominence: float = 0.025
    min_repetition_score: float = 0.56
    top_matches_per_axis: int = 2
    candidate_local_radius_patches: int = 2
    axis_nms_distance_ratio: float = 0.045
    max_axes: int = 14

    # Parçalı motif.
    fragmented_damage_fraction: float = 0.10
    heavily_damaged_fraction: float = 0.65

    # Düşey profil/kenar katkısı.
    edge_energy_weight: float = 0.12
    similarity_weight: float = 0.68
    prominence_weight: float = 0.20

    # Stage 1 etiketleri yalnız yardımcı kanıt.
    support_classes: tuple[str, ...] = (
        "pilaster", "column", "base", "capital", "arch"
    )
    support_axis_tolerance_ratio: float = 0.055


# =============================================================================
# UTILITIES
# =============================================================================
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_restore_run(output_root: Path) -> Path:
    candidates = sorted(
        path for path in output_root.glob("restore_*") if path.is_dir()
    )
    if not candidates:
        raise FileNotFoundError(f"restore_* klasörü bulunamadı: {output_root.resolve()}")
    return candidates[-1]


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"stage2_dinov2_{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"
    )
    file_handler = logging.FileHandler(
        run_dir / "stage2_dinov2.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def load_binary_mask(path: Path, size_xy: tuple[int, int]) -> np.ndarray:
    width, height = size_xy
    image = Image.open(path).convert("L").resize(
        (width, height), Image.Resampling.NEAREST
    )
    array = np.asarray(image)
    return np.where(array >= 128, 255, 0).astype(np.uint8)


def load_optional_elements(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("elements", []))


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def box_center(box: Iterable[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def facade_roi_from_damage(
    damage_mask: np.ndarray,
    width: int,
    height: int,
    cfg: Config,
) -> tuple[int, int, int, int]:
    box = bbox_from_mask(damage_mask)
    if box is None:
        return (
            int(round(width * cfg.fallback_x1_ratio)),
            int(round(height * cfg.fallback_y1_ratio)),
            int(round(width * cfg.fallback_x2_ratio)),
            int(round(height * cfg.fallback_y2_ratio)),
        )

    x1, y1, x2, y2 = box
    ex = int(round(width * cfg.facade_expand_x_ratio))
    et = int(round(height * cfg.facade_expand_y_top_ratio))
    eb = int(round(height * cfg.facade_expand_y_bottom_ratio))
    return (
        max(0, x1 - ex),
        max(0, y1 - et),
        min(width, x2 + ex),
        min(height, y2 + eb),
    )


def round_to_patch(value: int, patch_size: int, minimum: int) -> int:
    value = max(minimum, value)
    return max(patch_size, int(round(value / patch_size)) * patch_size)


def resized_shape_for_model(
    width: int,
    height: int,
    patch_size: int,
    cfg: Config,
) -> tuple[int, int]:
    scale = cfg.target_long_side / max(width, height)
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))

    if min(resized_w, resized_h) < cfg.minimum_short_side:
        extra = cfg.minimum_short_side / max(1, min(resized_w, resized_h))
        resized_w = int(round(resized_w * extra))
        resized_h = int(round(resized_h * extra))

    resized_w = round_to_patch(resized_w, patch_size, patch_size * 12)
    resized_h = round_to_patch(resized_h, patch_size, patch_size * 12)
    return resized_w, resized_h


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# DINOv2 FEATURES
# =============================================================================
def infer_grid_shape(
    token_count: int,
    input_height: int,
    input_width: int,
    patch_size: int,
) -> tuple[int, int]:
    expected_h = input_height // patch_size
    expected_w = input_width // patch_size
    if expected_h * expected_w == token_count:
        return expected_h, expected_w

    target_ratio = input_width / max(1.0, input_height)
    best: tuple[float, int, int] | None = None
    for grid_h in range(1, int(math.sqrt(token_count)) + 2):
        if token_count % grid_h != 0:
            continue
        grid_w = token_count // grid_h
        for candidate_h, candidate_w in ((grid_h, grid_w), (grid_w, grid_h)):
            error = abs((candidate_w / max(1.0, candidate_h)) - target_ratio)
            if best is None or error < best[0]:
                best = (error, candidate_h, candidate_w)
    if best is None:
        raise RuntimeError(
            f"DINOv2 patch grid çözülemedi: token_count={token_count}"
        )
    return best[1], best[2]


def extract_dense_features(
    crop_rgb: np.ndarray,
    crop_damage_mask: np.ndarray,
    cfg: Config,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    logger.info(f"[MODEL] DINOv2 yükleniyor: {cfg.model_id}")
    processor = AutoImageProcessor.from_pretrained(cfg.model_id)
    model = Dinov2Model.from_pretrained(cfg.model_id)
    model.to(cfg.device)
    model.eval()

    use_fp16 = bool(cfg.device == "cuda" and cfg.use_fp16_if_cuda)
    if use_fp16:
        model.half()

    patch_size = int(getattr(model.config, "patch_size", 14))
    crop_h, crop_w = crop_rgb.shape[:2]
    resized_w, resized_h = resized_shape_for_model(
        crop_w, crop_h, patch_size, cfg
    )

    resized_rgb = cv2.resize(
        crop_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LANCZOS4
    )
    resized_damage = cv2.resize(
        crop_damage_mask,
        (resized_w, resized_h),
        interpolation=cv2.INTER_NEAREST,
    )

    pil_image = Image.fromarray(resized_rgb, mode="RGB")
    inputs = processor(
        images=pil_image,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    )
    pixel_values = inputs["pixel_values"].to(cfg.device)
    if use_fp16:
        pixel_values = pixel_values.half()

    logger.info(
        f"[DINOv2] giriş={resized_w}x{resized_h}, patch={patch_size}"
    )
    with torch.inference_mode():
        try:
            outputs = model(
                pixel_values=pixel_values,
                interpolate_pos_encoding=True,
            )
        except TypeError:
            outputs = model(pixel_values=pixel_values)

    tokens = outputs.last_hidden_state[:, 1:, :].detach().float().cpu()[0]
    grid_h, grid_w = infer_grid_shape(
        int(tokens.shape[0]), resized_h, resized_w, patch_size
    )
    feature_dim = int(tokens.shape[-1])
    features = tokens.reshape(grid_h, grid_w, feature_dim).numpy()
    features = features / np.maximum(
        np.linalg.norm(features, axis=-1, keepdims=True), 1e-8
    )

    damage_grid = cv2.resize(
        resized_damage.astype(np.float32) / 255.0,
        (grid_w, grid_h),
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    edge_energy = cv2.resize(
        np.abs(sobel_x), (grid_w, grid_h), interpolation=cv2.INTER_AREA
    )
    edge_energy = edge_energy / max(1e-6, float(np.percentile(edge_energy, 95)))
    edge_energy = np.clip(edge_energy, 0.0, 1.0)

    info = {
        "model_id": cfg.model_id,
        "patch_size": patch_size,
        "input_size": [resized_w, resized_h],
        "feature_grid": [grid_w, grid_h],
        "feature_dim": feature_dim,
    }

    del outputs, tokens, pixel_values, inputs, model, processor
    clear_cuda()
    return features, damage_grid, edge_energy, info


# =============================================================================
# STRIP DESCRIPTORS AND SELF-SIMILARITY
# =============================================================================
def build_strip_descriptors(
    features: np.ndarray,
    damage_grid: np.ndarray,
    strip_width: int,
    cfg: Config,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    grid_h, grid_w, feature_dim = features.shape
    descriptors: list[dict[str, Any]] = []
    half = strip_width // 2

    bin_edges = np.linspace(0, grid_h, cfg.vertical_bins + 1).astype(int)
    for center_x in range(grid_w):
        x1 = max(0, center_x - half)
        x2 = min(grid_w, x1 + strip_width)
        x1 = max(0, x2 - strip_width)

        bin_vectors = np.zeros((cfg.vertical_bins, feature_dim), dtype=np.float32)
        valid_bins = np.zeros(cfg.vertical_bins, dtype=bool)
        visible_patch_count = 0
        total_patch_count = 0

        for bin_index in range(cfg.vertical_bins):
            y1 = int(bin_edges[bin_index])
            y2 = int(bin_edges[bin_index + 1])
            if y2 <= y1:
                continue

            patch_features = features[y1:y2, x1:x2, :]
            patch_damage = damage_grid[y1:y2, x1:x2]
            visible = patch_damage < 0.50
            visible_count = int(np.count_nonzero(visible))
            total_count = int(visible.size)
            visible_patch_count += visible_count
            total_patch_count += total_count

            if visible_count < cfg.min_visible_patches_per_bin:
                continue
            vector = patch_features[visible].mean(axis=0)
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-8:
                continue
            bin_vectors[bin_index] = vector / norm
            valid_bins[bin_index] = True

        damage_fraction = 1.0 - (
            visible_patch_count / max(1, total_patch_count)
        )
        descriptors.append(
            {
                "center_x_patch": center_x,
                "x1_patch": x1,
                "x2_patch": x2,
                "vectors": bin_vectors,
                "valid_bins": valid_bins,
                "valid_bin_count": int(np.count_nonzero(valid_bins)),
                "damage_fraction": float(damage_fraction),
            }
        )

    similarity = np.full((grid_w, grid_w), -1.0, dtype=np.float32)
    for i, first in enumerate(descriptors):
        for j in range(i, grid_w):
            second = descriptors[j]
            common = first["valid_bins"] & second["valid_bins"]
            if int(np.count_nonzero(common)) < cfg.min_shared_bins:
                continue
            first_vectors = first["vectors"][common]
            second_vectors = second["vectors"][common]
            score = float(np.mean(np.sum(first_vectors * second_vectors, axis=1)))
            similarity[i, j] = score
            similarity[j, i] = score

    return descriptors, similarity


def normalize_profile(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low = float(np.percentile(finite, 10))
    high = float(np.percentile(finite, 90))
    if high <= low + 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def score_repetition_profile(
    descriptors: list[dict[str, Any]],
    similarity: np.ndarray,
    edge_energy: np.ndarray,
    strip_width: int,
    cfg: Config,
) -> dict[str, Any]:
    grid_w = similarity.shape[0]
    min_sep = max(2, int(round(grid_w * cfg.min_axis_separation_ratio)))

    best_similarity = np.full(grid_w, -1.0, dtype=np.float32)
    mean_top_similarity = np.full(grid_w, -1.0, dtype=np.float32)
    prominence = np.zeros(grid_w, dtype=np.float32)
    match_indices: list[list[int]] = [[] for _ in range(grid_w)]

    for x in range(grid_w):
        far_indices = [
            other for other in range(grid_w)
            if abs(other - x) >= min_sep and similarity[x, other] >= -0.5
        ]
        if not far_indices:
            continue
        ranked = sorted(
            far_indices,
            key=lambda other: float(similarity[x, other]),
            reverse=True,
        )
        selected = ranked[: cfg.top_matches_per_axis]
        scores = [float(similarity[x, other]) for other in selected]
        all_far_scores = np.array(
            [float(similarity[x, other]) for other in far_indices],
            dtype=np.float32,
        )
        best_similarity[x] = max(scores)
        mean_top_similarity[x] = float(np.mean(scores))
        prominence[x] = float(np.mean(scores) - np.median(all_far_scores))
        match_indices[x] = selected

    edge_profile = np.zeros(grid_w, dtype=np.float32)
    half = strip_width // 2
    for x in range(grid_w):
        x1 = max(0, x - half)
        x2 = min(grid_w, x1 + strip_width)
        x1 = max(0, x2 - strip_width)
        visible = edge_energy[:, x1:x2]
        edge_profile[x] = float(np.mean(visible)) if visible.size else 0.0

    similarity_norm = normalize_profile(mean_top_similarity)
    prominence_norm = normalize_profile(prominence)
    edge_norm = normalize_profile(edge_profile)
    final_score = (
        cfg.similarity_weight * similarity_norm
        + cfg.prominence_weight * prominence_norm
        + cfg.edge_energy_weight * edge_norm
    ).astype(np.float32)

    return {
        "strip_width_patches": strip_width,
        "best_similarity": best_similarity,
        "mean_top_similarity": mean_top_similarity,
        "prominence": prominence,
        "edge_profile": edge_profile,
        "final_score": final_score,
        "match_indices": match_indices,
    }


def local_maximum(values: np.ndarray, index: int, radius: int) -> bool:
    left = max(0, index - radius)
    right = min(len(values), index + radius + 1)
    return float(values[index]) >= float(np.max(values[left:right]))


def support_labels_for_axis(
    axis_x_original: float,
    elements: list[dict[str, Any]],
    image_width: int,
    cfg: Config,
) -> list[str]:
    tolerance = image_width * cfg.support_axis_tolerance_ratio
    labels: set[str] = set()
    for element in elements:
        cls = str(element.get("class", "")).lower()
        if cls not in cfg.support_classes:
            continue
        box = element.get("bbox_xyxy") or element.get("bbox")
        if not box or len(box) != 4:
            continue
        center_x, _ = box_center(box)
        if abs(center_x - axis_x_original) <= tolerance:
            labels.add(cls)
    return sorted(labels)


def collect_axis_candidates(
    all_profiles: list[dict[str, Any]],
    all_descriptors: dict[int, list[dict[str, Any]]],
    roi: tuple[int, int, int, int],
    grid_w: int,
    image_width: int,
    elements: list[dict[str, Any]],
    cfg: Config,
) -> list[dict[str, Any]]:
    roi_x1, _, roi_x2, _ = roi
    roi_width = max(1.0, float(roi_x2 - roi_x1))
    candidates: list[dict[str, Any]] = []

    for profile in all_profiles:
        strip_width = int(profile["strip_width_patches"])
        final_score = profile["final_score"]
        best_similarity = profile["best_similarity"]
        prominence = profile["prominence"]
        match_indices = profile["match_indices"]
        descriptors = all_descriptors[strip_width]

        for x in range(grid_w):
            if not local_maximum(
                final_score, x, cfg.candidate_local_radius_patches
            ):
                continue
            if float(best_similarity[x]) < cfg.min_pair_similarity:
                continue
            if float(prominence[x]) < cfg.min_similarity_prominence:
                continue
            if float(final_score[x]) < cfg.min_repetition_score:
                continue
            if descriptors[x]["valid_bin_count"] < cfg.min_shared_bins:
                continue

            axis_x = roi_x1 + ((x + 0.5) / grid_w) * roi_width
            strip_width_original = (strip_width / grid_w) * roi_width
            matches = []
            for match_x in match_indices[x]:
                match_axis_x = roi_x1 + ((match_x + 0.5) / grid_w) * roi_width
                matches.append(
                    {
                        "axis_x": round(match_axis_x, 2),
                        "similarity": round(
                            float(profile["best_similarity"][x])
                            if match_x == match_indices[x][0]
                            else 0.0,
                            6,
                        ),
                        "patch_x": int(match_x),
                    }
                )

            damage_fraction = float(descriptors[x]["damage_fraction"])
            if damage_fraction >= cfg.heavily_damaged_fraction:
                # Yeterli görünür parça kalmamışsa çıkarım yapma.
                continue
            state = (
                "fragmented_repeated_motif"
                if damage_fraction >= cfg.fragmented_damage_fraction
                else "observed_repeated_motif"
            )
            support_labels = support_labels_for_axis(
                axis_x, elements, image_width, cfg
            )

            candidates.append(
                {
                    "axis_x": round(axis_x, 2),
                    "patch_x": int(x),
                    "strip_width_patches": strip_width,
                    "strip_width_original_px": round(strip_width_original, 2),
                    "best_similarity": round(float(best_similarity[x]), 6),
                    "prominence": round(float(prominence[x]), 6),
                    "repetition_score": round(float(final_score[x]), 6),
                    "damage_fraction": round(damage_fraction, 6),
                    "valid_vertical_bins": int(descriptors[x]["valid_bin_count"]),
                    "state": state,
                    "detector_support_labels": support_labels,
                    "matched_axes": matches,
                    "evidence": [
                        "dinov2_yapi_ici_benzerlik",
                        "dikey_bolumlu_descriptor",
                        "hasar_maskesi_disindaki_parcalar",
                    ] + (["stage1_etiket_destegi"] if support_labels else []),
                }
            )

    candidates.sort(
        key=lambda item: float(item["repetition_score"]), reverse=True
    )
    kept: list[dict[str, Any]] = []
    nms_distance = image_width * cfg.axis_nms_distance_ratio
    for candidate in candidates:
        if any(
            abs(float(candidate["axis_x"]) - float(old["axis_x"]))
            <= nms_distance
            for old in kept
        ):
            continue
        kept.append(candidate)
        if len(kept) >= cfg.max_axes:
            break

    kept.sort(key=lambda item: float(item["axis_x"]))
    for index, candidate in enumerate(kept, start=1):
        candidate["id"] = f"repetition_axis_{index:02d}"
    return kept


# =============================================================================
# VISUAL OUTPUTS
# =============================================================================
def similarity_heatmap(similarity: np.ndarray) -> np.ndarray:
    valid = similarity >= -0.5
    display = np.zeros_like(similarity, dtype=np.float32)
    if np.any(valid):
        low = float(np.percentile(similarity[valid], 5))
        high = float(np.percentile(similarity[valid], 98))
        if high > low + 1e-8:
            display[valid] = np.clip(
                (similarity[valid] - low) / (high - low), 0.0, 1.0
            )
    gray = (display * 255.0).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def draw_profile_image(
    crop_rgb: np.ndarray,
    profile: dict[str, Any],
) -> np.ndarray:
    height = 260
    width = max(520, crop_rgb.shape[1])
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    values = np.asarray(profile["final_score"], dtype=np.float32)
    similarities = np.asarray(profile["best_similarity"], dtype=np.float32)
    prominence = np.asarray(profile["prominence"], dtype=np.float32)

    def points(array: np.ndarray, top: int, bottom: int) -> np.ndarray:
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            normalized = np.zeros_like(array)
        else:
            low = float(np.percentile(finite, 5))
            high = float(np.percentile(finite, 95))
            if high <= low + 1e-8:
                normalized = np.zeros_like(array)
            else:
                normalized = np.clip((array - low) / (high - low), 0, 1)
        xs = np.linspace(25, width - 25, len(array))
        ys = bottom - normalized * (bottom - top)
        return np.stack([xs, ys], axis=1).astype(np.int32)

    cv2.putText(
        canvas,
        f"DINOv2 repetition profile — strip={profile['strip_width_patches']} patches",
        (18, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    cv2.polylines(canvas, [points(values, 42, 105)], False, (40, 160, 60), 2)
    cv2.putText(canvas, "final score", (18, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 160, 60), 1)
    cv2.polylines(canvas, [points(similarities, 112, 174)], False, (220, 120, 30), 2)
    cv2.putText(canvas, "best similarity", (18, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 120, 30), 1)
    cv2.polylines(canvas, [points(prominence, 180, 242)], False, (160, 50, 180), 2)
    cv2.putText(canvas, "prominence", (18, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 50, 180), 1)
    return canvas


def draw_axes_overlay(
    original: np.ndarray,
    damage_mask: np.ndarray,
    roi: tuple[int, int, int, int],
    axes: list[dict[str, Any]],
) -> np.ndarray:
    visual = original.astype(np.float32).copy()
    red = np.zeros_like(visual)
    red[..., 0] = 255
    binary = (damage_mask > 0)[..., None].astype(np.float32)
    visual = visual * (1.0 - binary * 0.16) + red * (binary * 0.16)
    visual = np.clip(visual, 0, 255).astype(np.uint8)

    x1, y1, x2, y2 = roi
    cv2.rectangle(visual, (x1, y1), (x2, y2), (255, 215, 40), 2)

    for axis in axes:
        x = int(round(axis["axis_x"]))
        half_width = max(3, int(round(axis["strip_width_original_px"] / 2.0)))
        if axis["state"] == "fragmented_repeated_motif":
            color = (255, 175, 30)
        else:
            color = (50, 220, 80)
        cv2.rectangle(
            visual,
            (max(0, x - half_width), y1),
            (min(visual.shape[1] - 1, x + half_width), y2),
            color,
            2,
        )
        cv2.line(visual, (x, y1), (x, y2), color, 1, cv2.LINE_AA)
        support = ",".join(axis["detector_support_labels"])
        label = (
            f"{axis['id']} {axis['state']} "
            f"R={axis['repetition_score']:.2f}"
        )
        if support:
            label += f" [{support}]"
        cv2.putText(
            visual,
            label,
            (max(2, x - 55), max(16, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
            cv2.LINE_AA,
        )
    return visual


# =============================================================================
# MAIN
# =============================================================================
def analyze(cfg: Config) -> Path:
    run_dir = (
        Path(cfg.analysis_run_dir)
        if cfg.analysis_run_dir
        else latest_restore_run(Path(cfg.output_root))
    )
    logger = setup_logger(run_dir)
    started = time.perf_counter()

    input_path = run_dir / "input.png"
    mask_path = run_dir / "completion_mask.png"
    elements_path = run_dir / "elements.json"
    meta_path = run_dir / "stage2_dinov2_meta.json"

    if not input_path.exists():
        raise FileNotFoundError(f"input.png yok: {input_path.resolve()}")
    if not mask_path.exists():
        raise FileNotFoundError(f"completion_mask.png yok: {mask_path.resolve()}")

    source_path = Path(__file__)
    metadata: dict[str, Any] = {
        "stage": "v8_stage2c_dinov2_self_similarity",
        "status": "running",
        "started_at": now_iso(),
        "analysis_run_dir": str(run_dir.resolve()),
        "source_script": {
            "filename": source_path.name,
            "path": str(source_path.resolve()),
            "sha256": sha256_file(source_path),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": getattr(torch, "__version__", None),
            "transformers": safe_version("transformers"),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "config": asdict(cfg),
        "facade_roi": None,
        "model_info": {},
        "profiles": [],
        "axes": [],
        "summary": {},
    }

    def flush() -> None:
        write_json(meta_path, metadata)

    try:
        logger.info("=" * 72)
        logger.info(f"[RUN] Stage 2C DINOv2 repetition -> {run_dir.name}")
        logger.info("[RULE] Pilaster/sütun eklenmeyecek")
        logger.info("[RULE] Yalnız görüntüdeki tekrar eden motif eksenleri raporlanacak")
        logger.info("=" * 72)

        original = read_rgb(input_path)
        height, width = original.shape[:2]
        damage_mask = load_binary_mask(mask_path, (width, height))
        elements = load_optional_elements(elements_path)

        roi = facade_roi_from_damage(damage_mask, width, height, cfg)
        rx1, ry1, rx2, ry2 = roi
        crop_rgb = original[ry1:ry2, rx1:rx2].copy()
        crop_damage = damage_mask[ry1:ry2, rx1:rx2].copy()
        save_rgb(run_dir / "dinov2_facade_crop.png", crop_rgb)

        features, damage_grid, edge_energy, model_info = extract_dense_features(
            crop_rgb, crop_damage, cfg, logger
        )
        grid_h, grid_w, _ = features.shape
        metadata["facade_roi"] = list(roi)
        metadata["model_info"] = model_info
        flush()

        all_profiles: list[dict[str, Any]] = []
        all_descriptors: dict[int, list[dict[str, Any]]] = {}
        all_similarities: dict[int, np.ndarray] = {}

        for strip_width in cfg.strip_widths_patches:
            if strip_width >= grid_w:
                continue
            logger.info(f"[SIM] strip_width={strip_width} patch")
            descriptors, similarity = build_strip_descriptors(
                features, damage_grid, strip_width, cfg
            )
            profile = score_repetition_profile(
                descriptors, similarity, edge_energy, strip_width, cfg
            )
            all_descriptors[strip_width] = descriptors
            all_similarities[strip_width] = similarity
            all_profiles.append(profile)

        if not all_profiles:
            raise RuntimeError("DINOv2 tekrar profili oluşturulamadı")

        # En güçlü profil yalnız görsel teşhis için kullanılır.
        best_profile = max(
            all_profiles,
            key=lambda item: float(np.nanmax(item["final_score"])),
        )
        best_width = int(best_profile["strip_width_patches"])
        save_rgb(
            run_dir / "dinov2_similarity_heatmap.png",
            similarity_heatmap(all_similarities[best_width]),
        )
        save_rgb(
            run_dir / "dinov2_repetition_profile.png",
            draw_profile_image(crop_rgb, best_profile),
        )

        axes = collect_axis_candidates(
            all_profiles,
            all_descriptors,
            roi,
            grid_w,
            width,
            elements,
            cfg,
        )
        save_rgb(
            run_dir / "dinov2_repetition_axes.png",
            draw_axes_overlay(original, damage_mask, roi, axes),
        )

        serial_profiles = []
        for profile in all_profiles:
            serial_profiles.append(
                {
                    "strip_width_patches": int(profile["strip_width_patches"]),
                    "best_similarity": [
                        round(float(value), 6)
                        for value in profile["best_similarity"]
                    ],
                    "mean_top_similarity": [
                        round(float(value), 6)
                        for value in profile["mean_top_similarity"]
                    ],
                    "prominence": [
                        round(float(value), 6)
                        for value in profile["prominence"]
                    ],
                    "final_score": [
                        round(float(value), 6)
                        for value in profile["final_score"]
                    ],
                }
            )

        axes_document = {
            "run_dir": str(run_dir.resolve()),
            "image_size": [width, height],
            "facade_roi": list(roi),
            "model_info": model_info,
            "safety": {
                "invented_pilasters": 0,
                "invented_modules": 0,
                "missing_inference_enabled": False,
                "note": (
                    "Kayıtlar nesne değil, fotoğrafta tekrar kanıtı bulunan "
                    "görsel motif eksenleridir."
                ),
            },
            "axes": axes,
        }
        write_json(run_dir / "repetition_axes.json", axes_document)

        metadata["profiles"] = serial_profiles
        metadata["axes"] = axes
        metadata["summary"] = {
            "feature_grid": [grid_w, grid_h],
            "tested_strip_widths": [
                int(item["strip_width_patches"]) for item in all_profiles
            ],
            "best_visual_profile_strip_width": best_width,
            "repetition_axis_count": len(axes),
            "fragmented_repeated_motif_count": sum(
                1 for axis in axes
                if axis["state"] == "fragmented_repeated_motif"
            ),
            "observed_repeated_motif_count": sum(
                1 for axis in axes
                if axis["state"] == "observed_repeated_motif"
            ),
            "invented_pilaster_count": 0,
            "invented_module_count": 0,
        }

        report_lines = [
            "V8 STAGE 2C — DINOv2 YAPI-İÇİ TEKRAR RAPORU",
            f"Analiz klasörü: {run_dir.resolve()}",
            "",
            "Bu rapor pilaster/sütun eklemez.",
            "Yalnız fotoğrafta tekrar eden düşey görsel motif eksenlerini kaydeder.",
            "",
            f"DINOv2 model: {cfg.model_id}",
            f"Özellik grid'i: {grid_w} x {grid_h}",
            f"Tekrar ekseni sayısı: {len(axes)}",
            "",
        ]
        for axis in axes:
            report_lines.append(
                f"- {axis['id']}: durum={axis['state']}, "
                f"x={axis['axis_x']:.1f}, tekrar={axis['repetition_score']:.2f}, "
                f"benzerlik={axis['best_similarity']:.2f}, "
                f"hasar={axis['damage_fraction']:.2f}, "
                f"etiket_destegi={','.join(axis['detector_support_labels']) or 'yok'}"
            )
        if not axes:
            report_lines.append(
                "Güvenli eşikleri geçen tekrar ekseni bulunamadı; sistem hiçbir sütun üretmedi."
            )
        (run_dir / "stage2_dinov2_report.txt").write_text(
            "\n".join(report_lines), encoding="utf-8"
        )

        metadata["status"] = "completed" if axes else "completed_no_axes"
        metadata["finished_at"] = now_iso()
        metadata["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        flush()

        logger.info("[OK] Stage 2C tamamlandı")
        logger.info(f"[OUT] {run_dir.resolve()}")
        logger.info(f"[COUNT] tekrar_ekseni={len(axes)}, eklenen_sütun=0")
        logger.info(
            "[FILES] dinov2_similarity_heatmap.png, "
            "dinov2_repetition_axes.png, repetition_axes.json"
        )
        return run_dir

    except Exception as exc:
        metadata["status"] = "failed"
        metadata["finished_at"] = now_iso()
        metadata["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        metadata["error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        flush()
        logger.error(f"[FAIL] {exc}")
        logger.error(traceback.format_exc())
        raise
    finally:
        clear_cuda()


def main() -> None:
    cfg = Config()
    run_dir = analyze(cfg)
    print(f"\nBİTTİ: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
