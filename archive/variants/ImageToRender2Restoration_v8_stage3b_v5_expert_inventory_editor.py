#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageToRender2Restoration_v8_stage3b_v5_expert_inventory_editor.py
==================================================================
V8 / STAGE 3B-v5 — Uzman gözlenen-eleman envanteri editörü

Amaç
----
Stage 3B-v4 tarafından bulunan kabul edilmiş ve inceleme bekleyen mimari
nesneleri, mimar/restoratörün okunaklı bir masaüstü arayüzünde kesin bir
gözlenen-eleman envanterine dönüştürmesini sağlar.

Uzman şunları yapabilir:
- AI kutusunu kabul etme, parçalı kabul etme veya reddetme,
- yanlış sınıfı düzeltme,
- kaymış kutuyu taşıma ve sekiz tutamaçla yeniden boyutlandırma,
- ok tuşlarıyla 1 px, Shift+ok ile 10 px hassas düzeltme,
- kutuyu baştan çizme,
- fotoğrafta gerçekten görünen ama AI'ın kaçırdığı elemanı yeni kutu olarak ekleme,
- aynı elemanı temsil eden iki kutuyu birleştirme,
- kutuyu yakındaki görünür kenarlara yardımcı olarak oturtma,
- geri alma,
- taslak kaydetme veya tüm review adayları çözüldüğünde envanteri kilitleme.

KESİN AYRIM
-----------
Bu editör yalnız fotoğrafta gerçekten görünen kanıtı kaydeder.
Hasarlı boşluğa varsayımsal kemer/pilaster yerleştirmek bu aşamada yasaktır.
Uzman tarafından yeni eklenen kutu:
    source = expert_added_observed_evidence
olarak kaydedilir. Hipotez katmanı daha sonraki aşamada ayrı tutulacaktır.

GİRDİLER (parent run)
---------------------
input.png
completion_mask.png
stage3b_v4_observed_instances.json
stage3b_v4_review_candidates.json

ÇIKTILAR
--------
Her kayıt işleminde yeni bir klasör:
    outputs/restore_YYYYMMDD_HHMMSS

input.png
completion_mask.png
source_stage3b_v4_observed_instances.json
source_stage3b_v4_review_candidates.json
expert_validated_inventory.json
expert_rejected_elements.json
expert_review_actions.json
expert_inventory_overlay.png
expert_intact_elements.png
expert_fragmented_elements.png
expert_rejected_elements.png
expert_inventory_report.txt
run.log
run_meta.json

KULLANIM ÖZETİ
--------------
Sol tık        : seç / sürükleyerek taşı
Köşe-kenar     : kutuyu yeniden boyutlandır
Fare tekeri    : yakınlaştır / uzaklaştır
Orta veya sağ  : görüntüyü kaydır
Ok tuşları     : 1 px taşı
Shift + ok     : 10 px taşı
Ctrl+Z         : geri al
Delete         : uzman eklediyse sil; AI adayıysa reddet
N              : yeni görünen eleman çiz
E              : seçili kutuyu baştan çiz
S              : seçili kutuyu kenarlara yardımcı oturt
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import platform
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Tkinter yüklenemedi. Bu editör masaüstü Tkinter arayüzü gerektirir.") from exc


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    output_root: str = "outputs"
    parent_run_dir: str = ""

    window_width: int = 1480
    window_height: int = 920
    sidebar_width: int = 390
    minimum_window_width: int = 1120
    minimum_window_height: int = 720

    initial_fit_padding: int = 28
    zoom_min: float = 0.20
    zoom_max: float = 8.0
    zoom_step: float = 1.15
    handle_radius_px: int = 6
    hit_tolerance_px: int = 7
    minimum_box_size_px: int = 6
    nudge_small_px: int = 1
    nudge_large_px: int = 10
    snap_search_px: int = 18
    snap_min_gain_ratio: float = 1.12
    history_limit: int = 60

    # Yeni çizilen uzman kutusunun varsayılanı.
    default_new_class: str = "pilaster"
    default_new_status: str = "observed_intact"

    class_options: tuple[str, ...] = (
        "arch",
        "pilaster",
        "column",
        "capital",
        "base",
        "cornice",
        "opening",
        "window",
        "door",
        "wall_fragment",
        "other",
    )
    status_options: tuple[str, ...] = (
        "observed_intact",
        "observed_fragmented",
        "candidate_needs_review",
        "rejected",
    )

    # Görüntü çıktılarında okunaklı ama sade etiketler.
    output_box_thickness: int = 3
    output_label_font_size: int = 14


STATUS_STYLE: dict[str, dict[str, str]] = {
    "observed_intact": {
        "color": "#22c55e",
        "title": "Sağlam gözlenen",
        "description": "Fotoğrafta görünen ve sağlam/prototip niteliğinde kabul edilen eleman.",
    },
    "observed_fragmented": {
        "color": "#f59e0b",
        "title": "Parçalı gözlenen",
        "description": "Elemanın fotoğraftaki kalıntısı görülüyor; tamamı sağlam değil.",
    },
    "candidate_needs_review": {
        "color": "#eab308",
        "title": "İnceleme bekliyor",
        "description": "AI adayı; uzman kararı henüz verilmedi.",
    },
    "rejected": {
        "color": "#94a3b8",
        "title": "Reddedildi",
        "description": "Yanlış sınıf, yanlış sınır, gölge/doku veya yinelenen kutu.",
    },
}

SOURCE_TITLES = {
    "ai_known_prototype": "AI / önceki prototip",
    "ai_recovered_observed": "AI / geri kazanılmış gözlem",
    "ai_review_candidate": "AI / inceleme adayı",
    "expert_added_observed_evidence": "Uzman tarafından eklenen görünen kanıt",
}


# =============================================================================
# BASIC FILE UTILITIES
# =============================================================================
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def create_new_run_dir(output_root: Path) -> tuple[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    while True:
        run_id = datetime.now().strftime("restore_%Y%m%d_%H%M%S")
        run_dir = output_root / run_id
        try:
            run_dir.mkdir(exist_ok=False)
            return run_id, run_dir
        except FileExistsError:
            time.sleep(1.05)


def setup_logger(run_dir: Path, run_id: str) -> logging.Logger:
    logger = logging.getLogger(f"stage3b_v5_{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%H:%M:%S",
    )
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def find_parent_run(cfg: Config) -> Path:
    required = (
        "input.png",
        "completion_mask.png",
        "stage3b_v4_observed_instances.json",
        "stage3b_v4_review_candidates.json",
    )

    if cfg.parent_run_dir:
        parent = Path(cfg.parent_run_dir)
        if not parent.is_dir():
            raise FileNotFoundError(f"Parent run klasörü bulunamadı: {parent.resolve()}")
        missing = [name for name in required if not (parent / name).exists()]
        if missing:
            raise FileNotFoundError(
                "Parent run zorunlu dosyaları eksik: " + ", ".join(missing)
            )
        return parent

    candidates = sorted(
        (path for path in Path(cfg.output_root).glob("restore_*") if path.is_dir()),
        reverse=True,
    )
    for candidate in candidates:
        if all((candidate / name).exists() for name in required):
            return candidate

    raise FileNotFoundError(
        "Stage 3B-v4 observed/review JSON dosyalarını içeren restore_* klasörü bulunamadı."
    )


def load_binary_mask(path: Path, size_xy: tuple[int, int]) -> np.ndarray:
    width, height = size_xy
    image = Image.open(path).convert("L").resize(
        (width, height),
        Image.Resampling.NEAREST,
    )
    return np.where(np.asarray(image) >= 128, 255, 0).astype(np.uint8)


def clip_box(box: Iterable[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [x1, y1, x2, y2]


def box_area(box: list[int]) -> int:
    return max(1, box[2] - box[0]) * max(1, box[3] - box[1])


def box_iou(first: list[int], second: list[int]) -> float:
    ix1 = max(first[0], second[0])
    iy1 = max(first[1], second[1])
    ix2 = min(first[2], second[2])
    iy2 = min(first[3], second[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = box_area(first) + box_area(second) - inter
    return float(inter / max(1, union))


# =============================================================================
# INPUT NORMALIZATION
# =============================================================================
def get_item_id(item: dict[str, Any], fallback: str) -> str:
    for key in ("id", "source_id", "candidate_id"):
        value = item.get(key)
        if value:
            return str(value)
    return fallback


def get_item_class(item: dict[str, Any], fallback: str) -> str:
    for key in ("class", "label", "class_name", "category", "type"):
        value = item.get(key)
        if value:
            return str(value).strip().lower()
    return fallback


def get_item_box(item: dict[str, Any]) -> list[int] | None:
    for key in ("bbox_xyxy", "box_xyxy", "bbox", "box"):
        value = item.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return [int(round(float(v))) for v in value]
    return None


def get_item_score(item: dict[str, Any]) -> float:
    for key in ("final_score", "score", "confidence", "fusion_score"):
        value = item.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    evidence = item.get("evidence", {})
    try:
        return float(evidence.get("fusion_score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def infer_initial_status(item: dict[str, Any], default: str) -> str:
    status = str(item.get("status", "")).strip().lower()
    if status in STATUS_STYLE:
        return status
    if status in {"known_prototype", "validated", "observed"}:
        return "observed_intact"
    if status in {"fragment_supported", "fragmented", "known_fragment"}:
        return "observed_fragmented"
    return default


def make_record(
    item: dict[str, Any],
    fallback_id: str,
    fallback_class: str,
    initial_status: str,
    source: str,
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    box = get_item_box(item)
    if box is None:
        return None
    box = clip_box(box, image_width, image_height)
    record_id = get_item_id(item, fallback_id)
    class_name = get_item_class(item, fallback_class)
    status = infer_initial_status(item, initial_status)
    usable = bool(item.get("usable_as_prototype", status == "observed_intact"))
    if status != "observed_intact":
        usable = False

    return {
        "id": record_id,
        "display_number": 0,
        "class": class_name,
        "status": status,
        "bbox_xyxy": box,
        "original_bbox_xyxy": list(box),
        "source": source,
        "original_candidate_id": record_id if source != "expert_added_observed_evidence" else None,
        "ai_status": str(item.get("status", "")),
        "ai_score": round(get_item_score(item), 6),
        "usable_as_prototype": usable,
        "expert_note": "",
        "expert_action": "unmodified",
        "expert_reviewed": source != "ai_review_candidate",
        "merged_from": [],
        "raw_evidence": item.get("evidence", {}),
        "hidden": False,
    }


def load_editor_records(
    observed_path: Path,
    review_path: Path,
    image_width: int,
    image_height: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observed_doc = read_json(observed_path)
    review_doc = read_json(review_path)
    records: list[dict[str, Any]] = []

    known = observed_doc.get("known_prototypes", {})
    recovered = observed_doc.get("recovered_observed", {})

    groups: list[tuple[list[Any], str, str, str]] = [
        (known.get("arches", []), "arch", "observed_intact", "ai_known_prototype"),
        (known.get("arch_fragments", []), "arch", "observed_fragmented", "ai_known_prototype"),
        (known.get("pilasters", []), "pilaster", "observed_intact", "ai_known_prototype"),
        (known.get("pilaster_fragments", []), "pilaster", "observed_fragmented", "ai_known_prototype"),
        (recovered.get("arches", []), "arch", "observed_intact", "ai_recovered_observed"),
        (recovered.get("pilasters", []), "pilaster", "observed_intact", "ai_recovered_observed"),
        (review_doc.get("arches", []), "arch", "candidate_needs_review", "ai_review_candidate"),
        (review_doc.get("pilasters", []), "pilaster", "candidate_needs_review", "ai_review_candidate"),
    ]

    fallback_index = 1
    seen_ids: set[str] = set()
    for items, fallback_class, initial_status, source in groups:
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            record = make_record(
                item,
                f"candidate_{fallback_index:03d}",
                fallback_class,
                initial_status,
                source,
                image_width,
                image_height,
            )
            fallback_index += 1
            if record is None:
                continue
            base_id = record["id"]
            suffix = 2
            while record["id"] in seen_ids:
                record["id"] = f"{base_id}__{suffix}"
                suffix += 1
            seen_ids.add(record["id"])
            records.append(record)

    # Aynı ID değil ama neredeyse aynı kutu olan review kopyalarını, kabul edilmiş
    # kayıtla birleştirmeyiz; uzman görebilsin. Ancak sıralamada kabul edilen önce gelir.
    source_rank = {
        "ai_known_prototype": 0,
        "ai_recovered_observed": 1,
        "ai_review_candidate": 2,
        "expert_added_observed_evidence": 3,
    }
    records.sort(
        key=lambda record: (
            source_rank.get(record["source"], 9),
            record["class"],
            record["bbox_xyxy"][0],
            record["bbox_xyxy"][1],
        )
    )
    for index, record in enumerate(records, start=1):
        record["display_number"] = index

    context = {
        "observed_run_id": observed_doc.get("run_id"),
        "observed_policy": observed_doc.get("policy", {}),
        "review_note": review_doc.get("note", ""),
    }
    return records, context


# =============================================================================
# EDGE SNAP ASSISTANT
# =============================================================================
def make_edge_energy(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    grad_x = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3))
    return grad_x, grad_y


def best_edge_position(
    energy: np.ndarray,
    fixed_start: int,
    fixed_end: int,
    current: int,
    search: int,
    axis: str,
    image_limit: int,
) -> tuple[int, float, float]:
    low = max(0, current - search)
    high = min(image_limit - 1, current + search)
    fixed_start = max(0, fixed_start)
    if axis == "vertical":
        fixed_end = min(energy.shape[0], fixed_end)
    else:
        fixed_end = min(energy.shape[1], fixed_end)

    scores: list[tuple[int, float]] = []
    for position in range(low, high + 1):
        if axis == "vertical":
            strip = energy[fixed_start:fixed_end, max(0, position - 1):min(energy.shape[1], position + 2)]
        else:
            strip = energy[max(0, position - 1):min(energy.shape[0], position + 2), fixed_start:fixed_end]
        score = float(np.mean(strip)) if strip.size else 0.0
        scores.append((position, score))

    if not scores:
        return current, 0.0, 0.0
    best_position, best_score = max(scores, key=lambda item: item[1])
    current_score = next((score for position, score in scores if position == current), 0.0)
    return best_position, best_score, current_score


def snap_box_to_edges(
    box: list[int],
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    cfg: Config,
    image_width: int,
    image_height: int,
) -> tuple[list[int], dict[str, Any]]:
    x1, y1, x2, y2 = box
    search = cfg.snap_search_px

    nx1, sx1, cx1 = best_edge_position(
        grad_x, y1, y2, x1, search, "vertical", image_width
    )
    nx2, sx2, cx2 = best_edge_position(
        grad_x, y1, y2, x2, search, "vertical", image_width
    )
    ny1, sy1, cy1 = best_edge_position(
        grad_y, x1, x2, y1, search, "horizontal", image_height
    )
    ny2, sy2, cy2 = best_edge_position(
        grad_y, x1, x2, y2, search, "horizontal", image_height
    )

    def accept(candidate: int, best: float, current_score: float, original: int) -> int:
        baseline = max(current_score, 1e-6)
        return candidate if best >= baseline * cfg.snap_min_gain_ratio else original

    nx1 = accept(nx1, sx1, cx1, x1)
    nx2 = accept(nx2, sx2, cx2, x2)
    ny1 = accept(ny1, sy1, cy1, y1)
    ny2 = accept(ny2, sy2, cy2, y2)

    if nx2 - nx1 < cfg.minimum_box_size_px:
        nx1, nx2 = x1, x2
    if ny2 - ny1 < cfg.minimum_box_size_px:
        ny1, ny2 = y1, y2

    snapped = clip_box([nx1, ny1, nx2, ny2], image_width, image_height)
    diagnostics = {
        "before": list(box),
        "after": list(snapped),
        "edge_scores": {
            "left": [round(cx1, 4), round(sx1, 4)],
            "right": [round(cx2, 4), round(sx2, 4)],
            "top": [round(cy1, 4), round(sy1, 4)],
            "bottom": [round(cy2, 4), round(sy2, 4)],
        },
    }
    return snapped, diagnostics


# =============================================================================
# EXPERT DIALOG
# =============================================================================
class ElementDialog(tk.Toplevel):
    def __init__(
        self,
        master: tk.Misc,
        cfg: Config,
        title: str,
        initial_class: str,
        initial_status: str,
        initial_prototype: bool,
        initial_note: str,
    ) -> None:
        super().__init__(master)
        self.cfg = cfg
        self.result: dict[str, Any] | None = None
        self.title(title)
        self.transient(master)
        self.grab_set()
        self.resizable(False, False)
        self.configure(bg="#f8fafc")

        frame = ttk.Frame(self, padding=18)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Sınıf").grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.class_var = tk.StringVar(value=initial_class)
        class_box = ttk.Combobox(
            frame,
            textvariable=self.class_var,
            values=cfg.class_options,
            state="readonly",
            width=28,
        )
        class_box.grid(row=0, column=1, sticky="ew", padx=(12, 0), pady=(0, 10))

        ttk.Label(frame, text="Durum").grid(row=1, column=0, sticky="w", pady=(0, 10))
        self.status_var = tk.StringVar(value=initial_status)
        status_box = ttk.Combobox(
            frame,
            textvariable=self.status_var,
            values=("observed_intact", "observed_fragmented"),
            state="readonly",
            width=28,
        )
        status_box.grid(row=1, column=1, sticky="ew", padx=(12, 0), pady=(0, 10))
        status_box.bind("<<ComboboxSelected>>", self._status_changed)

        ttk.Label(frame, text="Prototipte kullan").grid(row=2, column=0, sticky="w", pady=(0, 10))
        self.prototype_var = tk.BooleanVar(value=initial_prototype)
        self.prototype_check = ttk.Checkbutton(frame, variable=self.prototype_var, text="Evet")
        self.prototype_check.grid(row=2, column=1, sticky="w", padx=(12, 0), pady=(0, 10))

        ttk.Label(frame, text="Uzman notu").grid(row=3, column=0, sticky="nw")
        self.note = tk.Text(frame, width=36, height=5, wrap="word")
        self.note.grid(row=3, column=1, sticky="ew", padx=(12, 0))
        self.note.insert("1.0", initial_note)

        help_text = (
            "Yalnız fotoğrafta gerçekten görünen elemanı kaydedin. "
            "Tamamen yok olmuş hasar boşluğuna hipotez kutusu çizmeyin."
        )
        ttk.Label(frame, text=help_text, wraplength=390, foreground="#475569").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(14, 12)
        )

        buttons = ttk.Frame(frame)
        buttons.grid(row=5, column=0, columnspan=2, sticky="e")
        ttk.Button(buttons, text="İptal", command=self._cancel).pack(side="right", padx=(8, 0))
        ttk.Button(buttons, text="Kaydet", command=self._save).pack(side="right")

        self.bind("<Escape>", lambda _event: self._cancel())
        self.bind("<Return>", lambda _event: self._save())
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self._status_changed()
        self.update_idletasks()
        self.geometry(
            f"+{master.winfo_rootx() + 120}+{master.winfo_rooty() + 100}"
        )

    def _status_changed(self, _event: tk.Event | None = None) -> None:
        if self.status_var.get() != "observed_intact":
            self.prototype_var.set(False)
            self.prototype_check.state(["disabled"])
        else:
            self.prototype_check.state(["!disabled"])

    def _save(self) -> None:
        self.result = {
            "class": self.class_var.get(),
            "status": self.status_var.get(),
            "usable_as_prototype": bool(self.prototype_var.get()),
            "expert_note": self.note.get("1.0", "end").strip(),
        }
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()


# =============================================================================
# MAIN EDITOR UI
# =============================================================================
class ExpertInventoryEditor:
    def __init__(
        self,
        root: tk.Tk,
        cfg: Config,
        parent_run: Path,
        image: Image.Image,
        damage_mask: np.ndarray,
        records: list[dict[str, Any]],
        input_context: dict[str, Any],
    ) -> None:
        self.root = root
        self.cfg = cfg
        self.parent_run = parent_run
        self.image = image.convert("RGB")
        self.image_np = np.asarray(self.image)
        self.image_width, self.image_height = self.image.size
        self.damage_mask = damage_mask
        self.grad_x, self.grad_y = make_edge_energy(self.image_np)
        self.records = records
        self.input_context = input_context

        self.selected_id: str | None = None
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.photo_image: ImageTk.PhotoImage | None = None
        self.drag_mode: str | None = None
        self.drag_handle: str | None = None
        self.drag_start_canvas = (0.0, 0.0)
        self.drag_start_box: list[int] | None = None
        self.pan_start = (0.0, 0.0)
        self.pan_offset_start = (0.0, 0.0)
        self.draw_start_image: tuple[float, float] | None = None
        self.temp_draw_box: list[int] | None = None
        self.mode: str = "select"
        self.merge_base_id: str | None = None
        self.history: list[list[dict[str, Any]]] = []
        self.actions: list[dict[str, Any]] = []
        self.saved_run_dir: Path | None = None
        self.closed_by_save = False

        self.show_status_vars = {
            status: tk.BooleanVar(value=True)
            for status in STATUS_STYLE
        }
        self.show_labels_var = tk.BooleanVar(value=False)
        self.show_damage_var = tk.BooleanVar(value=False)
        self.filter_class_var = tk.StringVar(value="all")

        self.selected_id_var = tk.StringVar(value="—")
        self.selected_source_var = tk.StringVar(value="—")
        self.selected_score_var = tk.StringVar(value="—")
        self.selected_bbox_var = tk.StringVar(value="—")
        self.selected_class_var = tk.StringVar(value=cfg.default_new_class)
        self.selected_status_var = tk.StringVar(value=cfg.default_new_status)
        self.selected_prototype_var = tk.BooleanVar(value=True)
        self.selected_note_var = tk.StringVar(value="")
        self.status_message_var = tk.StringVar(value="Hazır. Bir kutuyu seçin veya N ile yeni eleman çizin.")
        self.summary_var = tk.StringVar(value="")

        self._configure_root()
        self._build_ui()
        self._bind_events()
        self.root.after(80, self.fit_image)
        self.refresh_all()

    # ------------------------------------------------------------------ UI SETUP
    def _configure_root(self) -> None:
        self.root.title("ImageToRender — Stage 3B-v5 Uzman Envanter Editörü")
        self.root.geometry(f"{self.cfg.window_width}x{self.cfg.window_height}")
        self.root.minsize(self.cfg.minimum_window_width, self.cfg.minimum_window_height)
        try:
            self.root.tk.call("tk", "scaling", 1.15)
        except tk.TclError:
            pass

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#f8fafc")
        style.configure("TLabel", background="#f8fafc", foreground="#0f172a", font=("Segoe UI", 10))
        style.configure("Muted.TLabel", foreground="#475569")
        style.configure("Heading.TLabel", font=("Segoe UI Semibold", 12))
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 14))
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 7))
        style.configure("Accent.TButton", font=("Segoe UI Semibold", 10), padding=(10, 8))
        style.configure("TCheckbutton", font=("Segoe UI", 9))
        style.configure("Treeview", font=("Segoe UI", 9), rowheight=25)
        style.configure("Treeview.Heading", font=("Segoe UI Semibold", 9))

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top = ttk.Frame(self.root, padding=(12, 9))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(8, weight=1)

        ttk.Button(top, text="Seç / Taşı", command=lambda: self.set_mode("select")).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(top, text="Yeni Gözlenen Eleman (N)", command=self.start_add_mode).grid(row=0, column=1, padx=6)
        ttk.Button(top, text="Kutuyu Yeniden Çiz (E)", command=self.start_redraw_mode).grid(row=0, column=2, padx=6)
        ttk.Button(top, text="Kenarlara Oturt (S)", command=self.snap_selected).grid(row=0, column=3, padx=6)
        ttk.Button(top, text="Birleştir", command=self.merge_action).grid(row=0, column=4, padx=6)
        ttk.Button(top, text="Geri Al (Ctrl+Z)", command=self.undo).grid(row=0, column=5, padx=6)
        ttk.Button(top, text="Görüntüye Sığdır", command=self.fit_image).grid(row=0, column=6, padx=6)
        ttk.Button(top, text="%100", command=self.zoom_100).grid(row=0, column=7, padx=6)
        ttk.Label(top, textvariable=self.status_message_var, style="Muted.TLabel", anchor="e").grid(
            row=0, column=8, sticky="ew", padx=(12, 0)
        )

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.grid(row=1, column=0, sticky="nsew")

        canvas_frame = tk.Frame(body, bg="#111827")
        sidebar = ttk.Frame(body, width=self.cfg.sidebar_width, padding=(12, 10))
        body.add(canvas_frame, weight=5)
        body.add(sidebar, weight=0)

        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            canvas_frame,
            bg="#111827",
            highlightthickness=0,
            cursor="arrow",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(4, weight=1)

        ttk.Label(sidebar, text="Uzman envanteri", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            sidebar,
            text=(
                "Kutular görselde yalnız numaralı rozetlerle gösterilir. "
                "Sınıf, durum ve açıklamalar bu panelde okunur."
            ),
            wraplength=self.cfg.sidebar_width - 34,
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="ew", pady=(3, 9))

        self._build_legend(sidebar).grid(row=2, column=0, sticky="ew", pady=(0, 9))
        self._build_filters(sidebar).grid(row=3, column=0, sticky="ew", pady=(0, 9))

        list_frame = ttk.LabelFrame(sidebar, text="Eleman listesi", padding=(8, 7))
        list_frame.grid(row=4, column=0, sticky="nsew")
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(
            list_frame,
            columns=("num", "class", "status"),
            show="headings",
            selectmode="browse",
        )
        self.tree.heading("num", text="#")
        self.tree.heading("class", text="Sınıf")
        self.tree.heading("status", text="Durum")
        self.tree.column("num", width=42, anchor="center", stretch=False)
        self.tree.column("class", width=105, anchor="w")
        self.tree.column("status", width=155, anchor="w")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.bind("<<TreeviewSelect>>", self._tree_selected)

        inspector = self._build_inspector(sidebar)
        inspector.grid(row=5, column=0, sticky="ew", pady=(10, 0))

        footer = ttk.Frame(self.root, padding=(12, 9))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.summary_var, style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(footer, text="İptal / Kapat", command=self.cancel_editor).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(footer, text="Taslak Kaydet", command=lambda: self.save_inventory(finalize=False)).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(footer, text="Envanteri Kilitle", style="Accent.TButton", command=lambda: self.save_inventory(finalize=True)).grid(row=0, column=3, padx=(8, 0))

    def _build_legend(self, parent: tk.Misc) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Lejant", padding=(9, 7))
        for row, (status, style_info) in enumerate(STATUS_STYLE.items()):
            swatch = tk.Canvas(frame, width=18, height=18, highlightthickness=0, bg="#f8fafc")
            swatch.create_rectangle(2, 2, 16, 16, fill=style_info["color"], outline=style_info["color"])
            swatch.grid(row=row, column=0, sticky="nw", pady=3)
            text = ttk.Frame(frame)
            text.grid(row=row, column=1, sticky="ew", padx=(7, 0), pady=2)
            ttk.Label(text, text=style_info["title"], font=("Segoe UI Semibold", 9)).pack(anchor="w")
            ttk.Label(
                text,
                text=style_info["description"],
                wraplength=self.cfg.sidebar_width - 100,
                style="Muted.TLabel",
                font=("Segoe UI", 8),
            ).pack(anchor="w")
        frame.columnconfigure(1, weight=1)
        return frame

    def _build_filters(self, parent: tk.Misc) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Görünürlük ve filtre", padding=(9, 7))
        for index, status in enumerate(STATUS_STYLE):
            ttk.Checkbutton(
                frame,
                text=STATUS_STYLE[status]["title"],
                variable=self.show_status_vars[status],
                command=self.refresh_all,
            ).grid(row=index // 2, column=index % 2, sticky="w", padx=(0, 8), pady=2)

        ttk.Checkbutton(
            frame,
            text="Kutu yanında sınıfı da yaz",
            variable=self.show_labels_var,
            command=self.redraw_canvas,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 2))
        ttk.Checkbutton(
            frame,
            text="Hasar maskesini göster",
            variable=self.show_damage_var,
            command=self.redraw_canvas,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Label(frame, text="Sınıf filtresi").grid(row=4, column=0, sticky="w", pady=(6, 0))
        class_filter = ttk.Combobox(
            frame,
            textvariable=self.filter_class_var,
            values=("all",) + self.cfg.class_options,
            state="readonly",
            width=21,
        )
        class_filter.grid(row=4, column=1, sticky="ew", pady=(6, 0))
        class_filter.bind("<<ComboboxSelected>>", lambda _event: self.refresh_all())
        frame.columnconfigure(1, weight=1)
        return frame

    def _build_inspector(self, parent: tk.Misc) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Seçili kutu", padding=(9, 8))
        frame.columnconfigure(1, weight=1)

        labels = [
            ("ID", self.selected_id_var),
            ("Kaynak", self.selected_source_var),
            ("AI puanı", self.selected_score_var),
            ("Kutu", self.selected_bbox_var),
        ]
        for row, (title, variable) in enumerate(labels):
            ttk.Label(frame, text=title, style="Muted.TLabel").grid(row=row, column=0, sticky="nw", pady=2)
            ttk.Label(frame, textvariable=variable, wraplength=245).grid(row=row, column=1, sticky="w", padx=(8, 0), pady=2)

        row = len(labels)
        ttk.Label(frame, text="Sınıf", style="Muted.TLabel").grid(row=row, column=0, sticky="w", pady=(7, 2))
        class_box = ttk.Combobox(
            frame,
            textvariable=self.selected_class_var,
            values=self.cfg.class_options,
            state="readonly",
        )
        class_box.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=(7, 2))

        row += 1
        ttk.Label(frame, text="Durum", style="Muted.TLabel").grid(row=row, column=0, sticky="w", pady=2)
        status_box = ttk.Combobox(
            frame,
            textvariable=self.selected_status_var,
            values=self.cfg.status_options,
            state="readonly",
        )
        status_box.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=2)
        status_box.bind("<<ComboboxSelected>>", self._inspector_status_changed)

        row += 1
        ttk.Label(frame, text="Prototip", style="Muted.TLabel").grid(row=row, column=0, sticky="w", pady=2)
        self.prototype_check = ttk.Checkbutton(
            frame,
            variable=self.selected_prototype_var,
            text="Aile prototipinde kullanılabilir",
        )
        self.prototype_check.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=2)

        row += 1
        ttk.Label(frame, text="Uzman notu", style="Muted.TLabel").grid(row=row, column=0, sticky="nw", pady=(5, 2))
        note_entry = ttk.Entry(frame, textvariable=self.selected_note_var)
        note_entry.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=(5, 2))

        row += 1
        buttons = ttk.Frame(frame)
        buttons.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        ttk.Button(buttons, text="Değişiklikleri Uygula", command=self.apply_inspector).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(buttons, text="Orijinal Kutuya Dön", command=self.reset_selected_box).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        return frame

    def _bind_events(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_press)
        self.canvas.bind("<B2-Motion>", self.on_pan_drag)
        self.canvas.bind("<ButtonPress-3>", self.on_pan_press)
        self.canvas.bind("<B3-Motion>", self.on_pan_drag)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", lambda event: self.on_linux_wheel(event, 1))
        self.canvas.bind("<Button-5>", lambda event: self.on_linux_wheel(event, -1))
        self.canvas.bind("<Configure>", lambda _event: self.redraw_canvas())

        self.root.bind("<Control-z>", lambda _event: self.undo())
        self.root.bind("<Control-Z>", lambda _event: self.undo())
        # Düzenleme kısayolları yalnız görüntü kanvası odaktayken çalışır;
        # böylece sağ panelde not yazarken N/E/S veya ok tuşları yanlışlıkla
        # kutuları değiştirmez.
        self.canvas.bind("<KeyPress-n>", lambda _event: self.start_add_mode())
        self.canvas.bind("<KeyPress-N>", lambda _event: self.start_add_mode())
        self.canvas.bind("<KeyPress-e>", lambda _event: self.start_redraw_mode())
        self.canvas.bind("<KeyPress-E>", lambda _event: self.start_redraw_mode())
        self.canvas.bind("<KeyPress-s>", lambda _event: self.snap_selected())
        self.canvas.bind("<KeyPress-S>", lambda _event: self.snap_selected())
        self.canvas.bind("<Escape>", lambda _event: self.set_mode("select"))
        self.canvas.bind("<Delete>", lambda _event: self.delete_or_reject_selected())
        self.canvas.bind("<Left>", lambda event: self.nudge_selected(-1, 0, event))
        self.canvas.bind("<Right>", lambda event: self.nudge_selected(1, 0, event))
        self.canvas.bind("<Up>", lambda event: self.nudge_selected(0, -1, event))
        self.canvas.bind("<Down>", lambda event: self.nudge_selected(0, 1, event))
        self.root.bind("<Escape>", lambda _event: self.set_mode("select"))
        self.root.protocol("WM_DELETE_WINDOW", self.cancel_editor)

    # ------------------------------------------------------------------ RECORD HELPERS
    def visible_records(self) -> list[dict[str, Any]]:
        class_filter = self.filter_class_var.get()
        result = []
        for record in self.records:
            if record.get("hidden"):
                continue
            status = record["status"]
            if status not in self.show_status_vars or not self.show_status_vars[status].get():
                continue
            if class_filter != "all" and record["class"] != class_filter:
                continue
            result.append(record)
        return result

    def get_record(self, record_id: str | None) -> dict[str, Any] | None:
        if record_id is None:
            return None
        return next(
            (record for record in self.records if record["id"] == record_id and not record.get("hidden")),
            None,
        )

    def push_history(self) -> None:
        self.history.append(copy.deepcopy(self.records))
        if len(self.history) > self.cfg.history_limit:
            self.history = self.history[-self.cfg.history_limit:]

    def log_action(self, action: str, record: dict[str, Any] | None, **details: Any) -> None:
        self.actions.append(
            {
                "timestamp": now_iso(),
                "action": action,
                "record_id": record.get("id") if record else None,
                "details": details,
            }
        )

    def renumber_records(self) -> None:
        number = 1
        for record in self.records:
            if not record.get("hidden"):
                record["display_number"] = number
                number += 1

    # ------------------------------------------------------------------ TRANSFORMS
    def image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y

    def canvas_to_image(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale

    def fit_image(self) -> None:
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        padding = self.cfg.initial_fit_padding
        self.scale = min(
            (canvas_width - 2 * padding) / self.image_width,
            (canvas_height - 2 * padding) / self.image_height,
        )
        self.scale = max(self.cfg.zoom_min, min(self.cfg.zoom_max, self.scale))
        self.offset_x = (canvas_width - self.image_width * self.scale) / 2
        self.offset_y = (canvas_height - self.image_height * self.scale) / 2
        self.redraw_canvas()

    def zoom_100(self) -> None:
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        self.scale = 1.0
        self.offset_x = (canvas_width - self.image_width) / 2
        self.offset_y = (canvas_height - self.image_height) / 2
        self.redraw_canvas()

    def zoom_at(self, canvas_x: float, canvas_y: float, factor: float) -> None:
        image_x, image_y = self.canvas_to_image(canvas_x, canvas_y)
        new_scale = max(self.cfg.zoom_min, min(self.cfg.zoom_max, self.scale * factor))
        if abs(new_scale - self.scale) < 1e-9:
            return
        self.scale = new_scale
        self.offset_x = canvas_x - image_x * self.scale
        self.offset_y = canvas_y - image_y * self.scale
        self.redraw_canvas()

    # ------------------------------------------------------------------ DRAWING
    def redraw_canvas(self) -> None:
        if not self.canvas.winfo_exists():
            return
        self.canvas.delete("all")
        display_width = max(1, int(round(self.image_width * self.scale)))
        display_height = max(1, int(round(self.image_height * self.scale)))
        resized = self.image.resize((display_width, display_height), Image.Resampling.LANCZOS)

        if self.show_damage_var.get():
            mask_pil = Image.fromarray(self.damage_mask, mode="L").resize(
                (display_width, display_height), Image.Resampling.NEAREST
            )
            overlay = Image.new("RGBA", resized.size, (239, 68, 68, 0))
            alpha = mask_pil.point(lambda value: 74 if value > 0 else 0)
            overlay.putalpha(alpha)
            resized = Image.alpha_composite(resized.convert("RGBA"), overlay).convert("RGB")

        self.photo_image = ImageTk.PhotoImage(resized)
        self.canvas.create_image(
            self.offset_x,
            self.offset_y,
            image=self.photo_image,
            anchor="nw",
            tags=("base_image",),
        )

        # Boxes: larger boxes first, so smaller ones remain selectable/visible.
        for record in sorted(self.visible_records(), key=lambda item: box_area(item["bbox_xyxy"]), reverse=True):
            self.draw_record(record)

        selected = self.get_record(self.selected_id)
        if selected and selected in self.visible_records():
            self.draw_handles(selected)

        if self.temp_draw_box is not None:
            x1, y1, x2, y2 = self.temp_draw_box
            cx1, cy1 = self.image_to_canvas(x1, y1)
            cx2, cy2 = self.image_to_canvas(x2, y2)
            self.canvas.create_rectangle(
                cx1, cy1, cx2, cy2,
                outline="#38bdf8",
                width=2,
                dash=(6, 4),
                tags=("temp_box",),
            )

    def draw_record(self, record: dict[str, Any]) -> None:
        style_info = STATUS_STYLE.get(record["status"], STATUS_STYLE["candidate_needs_review"])
        color = style_info["color"]
        x1, y1, x2, y2 = record["bbox_xyxy"]
        cx1, cy1 = self.image_to_canvas(x1, y1)
        cx2, cy2 = self.image_to_canvas(x2, y2)
        selected = record["id"] == self.selected_id
        width = 4 if selected else 2
        dash = (6, 4) if record["status"] == "candidate_needs_review" else None
        if record["status"] == "rejected":
            dash = (3, 4)
        self.canvas.create_rectangle(
            cx1, cy1, cx2, cy2,
            outline="#38bdf8" if selected else color,
            width=width,
            dash=dash,
            tags=("box", f"record:{record['id']}"),
        )

        # Compact numbered badge; long overlapping text is intentionally avoided.
        badge_text = str(record["display_number"])
        if self.show_labels_var.get():
            badge_text += f"  {record['class']}"
        text_id = self.canvas.create_text(
            cx1 + 4,
            cy1 + 3,
            text=badge_text,
            anchor="nw",
            fill="#ffffff",
            font=("Segoe UI Semibold", 9),
            tags=("badge", f"record:{record['id']}"),
        )
        bounds = self.canvas.bbox(text_id)
        if bounds:
            rectangle_id = self.canvas.create_rectangle(
                bounds[0] - 3,
                bounds[1] - 2,
                bounds[2] + 3,
                bounds[3] + 2,
                fill="#0f172a",
                outline=color,
                width=1,
                tags=("badge_bg", f"record:{record['id']}"),
            )
            self.canvas.tag_lower(rectangle_id, text_id)

    def handle_positions(self, box: list[int]) -> dict[str, tuple[float, float]]:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        points = {
            "nw": (x1, y1), "n": (cx, y1), "ne": (x2, y1),
            "e": (x2, cy), "se": (x2, y2), "s": (cx, y2),
            "sw": (x1, y2), "w": (x1, cy),
        }
        return {name: self.image_to_canvas(*point) for name, point in points.items()}

    def draw_handles(self, record: dict[str, Any]) -> None:
        radius = self.cfg.handle_radius_px
        for name, (x, y) in self.handle_positions(record["bbox_xyxy"]).items():
            self.canvas.create_rectangle(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="#ffffff",
                outline="#0284c7",
                width=2,
                tags=("handle", f"handle:{name}"),
            )

    # ------------------------------------------------------------------ HIT TESTING
    def hit_handle(self, canvas_x: float, canvas_y: float) -> str | None:
        record = self.get_record(self.selected_id)
        if record is None:
            return None
        tolerance = self.cfg.handle_radius_px + 3
        for name, (x, y) in self.handle_positions(record["bbox_xyxy"]).items():
            if abs(canvas_x - x) <= tolerance and abs(canvas_y - y) <= tolerance:
                return name
        return None

    def hit_record(self, canvas_x: float, canvas_y: float) -> dict[str, Any] | None:
        image_x, image_y = self.canvas_to_image(canvas_x, canvas_y)
        candidates = []
        tolerance = self.cfg.hit_tolerance_px / max(self.scale, 1e-6)
        for record in self.visible_records():
            x1, y1, x2, y2 = record["bbox_xyxy"]
            if x1 - tolerance <= image_x <= x2 + tolerance and y1 - tolerance <= image_y <= y2 + tolerance:
                candidates.append(record)
        if not candidates:
            return None
        # Prefer the smallest box under the pointer, then closest border.
        return min(candidates, key=lambda item: box_area(item["bbox_xyxy"]))

    # ------------------------------------------------------------------ POINTER EVENTS
    def on_left_press(self, event: tk.Event) -> None:
        self.canvas.focus_set()
        if self.mode in {"add", "redraw"}:
            image_x, image_y = self.canvas_to_image(event.x, event.y)
            image_x = max(0.0, min(self.image_width - 1.0, image_x))
            image_y = max(0.0, min(self.image_height - 1.0, image_y))
            self.draw_start_image = (image_x, image_y)
            self.temp_draw_box = [int(image_x), int(image_y), int(image_x) + 1, int(image_y) + 1]
            self.drag_mode = "draw"
            return

        handle = self.hit_handle(event.x, event.y)
        if handle:
            record = self.get_record(self.selected_id)
            if record:
                self.push_history()
                self.drag_mode = "resize"
                self.drag_handle = handle
                self.drag_start_canvas = (event.x, event.y)
                self.drag_start_box = list(record["bbox_xyxy"])
            return

        record = self.hit_record(event.x, event.y)
        if record is None:
            self.select_record(None)
            return
        self.select_record(record["id"])
        self.push_history()
        self.drag_mode = "move"
        self.drag_start_canvas = (event.x, event.y)
        self.drag_start_box = list(record["bbox_xyxy"])

    def on_left_drag(self, event: tk.Event) -> None:
        if self.drag_mode == "draw" and self.draw_start_image is not None:
            x, y = self.canvas_to_image(event.x, event.y)
            x = max(0.0, min(self.image_width, x))
            y = max(0.0, min(self.image_height, y))
            sx, sy = self.draw_start_image
            self.temp_draw_box = clip_box(
                [min(sx, x), min(sy, y), max(sx, x), max(sy, y)],
                self.image_width,
                self.image_height,
            )
            self.redraw_canvas()
            return

        record = self.get_record(self.selected_id)
        if record is None or self.drag_start_box is None:
            return
        dx = (event.x - self.drag_start_canvas[0]) / self.scale
        dy = (event.y - self.drag_start_canvas[1]) / self.scale

        if self.drag_mode == "move":
            x1, y1, x2, y2 = self.drag_start_box
            width = x2 - x1
            height = y2 - y1
            nx1 = int(round(x1 + dx))
            ny1 = int(round(y1 + dy))
            nx1 = max(0, min(self.image_width - width, nx1))
            ny1 = max(0, min(self.image_height - height, ny1))
            record["bbox_xyxy"] = [nx1, ny1, nx1 + width, ny1 + height]
            self.redraw_canvas()
            self.update_inspector()
            return

        if self.drag_mode == "resize" and self.drag_handle:
            x1, y1, x2, y2 = self.drag_start_box
            handle = self.drag_handle
            if "w" in handle:
                x1 = int(round(x1 + dx))
            if "e" in handle:
                x2 = int(round(x2 + dx))
            if "n" in handle:
                y1 = int(round(y1 + dy))
            if "s" in handle:
                y2 = int(round(y2 + dy))
            if x2 - x1 < self.cfg.minimum_box_size_px:
                if "w" in handle:
                    x1 = x2 - self.cfg.minimum_box_size_px
                else:
                    x2 = x1 + self.cfg.minimum_box_size_px
            if y2 - y1 < self.cfg.minimum_box_size_px:
                if "n" in handle:
                    y1 = y2 - self.cfg.minimum_box_size_px
                else:
                    y2 = y1 + self.cfg.minimum_box_size_px
            record["bbox_xyxy"] = clip_box([x1, y1, x2, y2], self.image_width, self.image_height)
            self.redraw_canvas()
            self.update_inspector()

    def on_left_release(self, _event: tk.Event) -> None:
        if self.drag_mode == "draw":
            box = self.temp_draw_box
            self.temp_draw_box = None
            self.draw_start_image = None
            self.drag_mode = None
            if box is None or box[2] - box[0] < self.cfg.minimum_box_size_px or box[3] - box[1] < self.cfg.minimum_box_size_px:
                self.status_message_var.set("Çizilen kutu çok küçük; işlem iptal edildi.")
                self.redraw_canvas()
                return
            if self.mode == "add":
                self.finish_add_box(box)
            elif self.mode == "redraw":
                self.finish_redraw_box(box)
            return

        record = self.get_record(self.selected_id)
        if record and self.drag_mode in {"move", "resize"} and self.drag_start_box:
            if record["bbox_xyxy"] != self.drag_start_box:
                record["expert_action"] = "bbox_corrected"
                record["expert_reviewed"] = True
                self.log_action(
                    "bbox_corrected",
                    record,
                    before=self.drag_start_box,
                    after=list(record["bbox_xyxy"]),
                )
        self.drag_mode = None
        self.drag_handle = None
        self.drag_start_box = None
        self.refresh_all()

    def on_pan_press(self, event: tk.Event) -> None:
        self.pan_start = (event.x, event.y)
        self.pan_offset_start = (self.offset_x, self.offset_y)

    def on_pan_drag(self, event: tk.Event) -> None:
        self.offset_x = self.pan_offset_start[0] + event.x - self.pan_start[0]
        self.offset_y = self.pan_offset_start[1] + event.y - self.pan_start[1]
        self.redraw_canvas()

    def on_mousewheel(self, event: tk.Event) -> None:
        factor = self.cfg.zoom_step if event.delta > 0 else 1 / self.cfg.zoom_step
        self.zoom_at(event.x, event.y, factor)

    def on_linux_wheel(self, event: tk.Event, direction: int) -> None:
        factor = self.cfg.zoom_step if direction > 0 else 1 / self.cfg.zoom_step
        self.zoom_at(event.x, event.y, factor)

    # ------------------------------------------------------------------ MODE / SELECTION
    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.temp_draw_box = None
        self.draw_start_image = None
        mode_messages = {
            "select": "Seçim modu: kutuyu seçin; içinden sürükleyin veya tutamaçları kullanın.",
            "add": "Yeni eleman modu: fotoğrafta gerçekten görünen elemanın çevresine dikdörtgen çizin.",
            "redraw": "Yeniden çizim modu: seçili elemanın doğru sınırını baştan çizin.",
        }
        self.status_message_var.set(mode_messages.get(mode, "Hazır"))
        self.canvas.configure(cursor="crosshair" if mode in {"add", "redraw"} else "arrow")
        self.redraw_canvas()

    def start_add_mode(self) -> None:
        self.set_mode("add")

    def start_redraw_mode(self) -> None:
        if self.get_record(self.selected_id) is None:
            messagebox.showinfo("Kutu seçilmedi", "Önce yeniden çizilecek kutuyu seçin.", parent=self.root)
            return
        self.set_mode("redraw")

    def select_record(self, record_id: str | None) -> None:
        self.selected_id = record_id
        self.update_inspector()
        self.redraw_canvas()
        if record_id and self.tree.exists(record_id):
            self.tree.selection_set(record_id)
            self.tree.see(record_id)
        elif not record_id:
            self.tree.selection_remove(*self.tree.selection())

    def _tree_selected(self, _event: tk.Event) -> None:
        selection = self.tree.selection()
        if selection:
            self.selected_id = selection[0]
            self.update_inspector()
            self.redraw_canvas()

    # ------------------------------------------------------------------ ADD / REDRAW / EDIT
    def finish_add_box(self, box: list[int]) -> None:
        dialog = ElementDialog(
            self.root,
            self.cfg,
            "Yeni görünen mimari eleman",
            self.cfg.default_new_class,
            self.cfg.default_new_status,
            True,
            "",
        )
        self.root.wait_window(dialog)
        if dialog.result is None:
            self.set_mode("select")
            return

        self.push_history()
        next_number = max((record["display_number"] for record in self.records), default=0) + 1
        class_name = dialog.result["class"]
        base_id = f"expert_{class_name}_{next_number:03d}"
        record_id = base_id
        suffix = 2
        existing_ids = {record["id"] for record in self.records}
        while record_id in existing_ids:
            record_id = f"{base_id}_{suffix}"
            suffix += 1
        record = {
            "id": record_id,
            "display_number": next_number,
            "class": class_name,
            "status": dialog.result["status"],
            "bbox_xyxy": list(box),
            "original_bbox_xyxy": None,
            "source": "expert_added_observed_evidence",
            "original_candidate_id": None,
            "ai_status": "",
            "ai_score": None,
            "usable_as_prototype": bool(dialog.result["usable_as_prototype"]),
            "expert_note": dialog.result["expert_note"],
            "expert_action": "added",
            "expert_reviewed": True,
            "merged_from": [],
            "raw_evidence": {},
            "hidden": False,
        }
        self.records.append(record)
        self.renumber_records()
        self.log_action("expert_added_observed_evidence", record, bbox=list(box))
        self.set_mode("select")
        self.select_record(record["id"])
        self.refresh_all()

    def finish_redraw_box(self, box: list[int]) -> None:
        record = self.get_record(self.selected_id)
        if record is None:
            self.set_mode("select")
            return
        self.push_history()
        before = list(record["bbox_xyxy"])
        record["bbox_xyxy"] = list(box)
        record["expert_action"] = "bbox_redrawn"
        record["expert_reviewed"] = True
        self.log_action("bbox_redrawn", record, before=before, after=list(box))
        self.set_mode("select")
        self.refresh_all()

    def update_inspector(self) -> None:
        record = self.get_record(self.selected_id)
        if record is None:
            self.selected_id_var.set("—")
            self.selected_source_var.set("—")
            self.selected_score_var.set("—")
            self.selected_bbox_var.set("—")
            return
        self.selected_id_var.set(f"#{record['display_number']}  {record['id']}")
        self.selected_source_var.set(SOURCE_TITLES.get(record["source"], record["source"]))
        score = record.get("ai_score")
        self.selected_score_var.set("—" if score is None else f"{float(score):.3f}")
        self.selected_bbox_var.set(str(record["bbox_xyxy"]))
        self.selected_class_var.set(record["class"])
        self.selected_status_var.set(record["status"])
        self.selected_prototype_var.set(bool(record["usable_as_prototype"]))
        self.selected_note_var.set(record.get("expert_note", ""))
        self._inspector_status_changed()

    def _inspector_status_changed(self, _event: tk.Event | None = None) -> None:
        if self.selected_status_var.get() != "observed_intact":
            self.selected_prototype_var.set(False)
            self.prototype_check.state(["disabled"])
        else:
            self.prototype_check.state(["!disabled"])

    def apply_inspector(self) -> None:
        record = self.get_record(self.selected_id)
        if record is None:
            messagebox.showinfo("Kutu seçilmedi", "Önce bir kutu seçin.", parent=self.root)
            return
        before = {
            "class": record["class"],
            "status": record["status"],
            "usable_as_prototype": record["usable_as_prototype"],
            "expert_note": record.get("expert_note", ""),
        }
        after = {
            "class": self.selected_class_var.get(),
            "status": self.selected_status_var.get(),
            "usable_as_prototype": bool(self.selected_prototype_var.get()),
            "expert_note": self.selected_note_var.get().strip(),
        }
        if after["status"] != "observed_intact":
            after["usable_as_prototype"] = False
        if before == after:
            self.status_message_var.set("Seçili kutuda uygulanacak yeni bir değişiklik yok.")
            return
        self.push_history()
        record.update(after)
        record["expert_reviewed"] = True
        if before["class"] != after["class"]:
            record["expert_action"] = "class_corrected"
        elif before["status"] != after["status"]:
            record["expert_action"] = "status_decided"
        else:
            record["expert_action"] = "metadata_updated"
        self.log_action("inspector_updated", record, before=before, after=after)
        self.refresh_all()
        self.status_message_var.set("Seçili kutunun sınıf/durum bilgisi güncellendi.")

    def reset_selected_box(self) -> None:
        record = self.get_record(self.selected_id)
        if record is None:
            return
        original = record.get("original_bbox_xyxy")
        if not original:
            messagebox.showinfo(
                "Orijinal kutu yok",
                "Bu eleman uzman tarafından eklendiği için AI orijinal kutusu bulunmuyor.",
                parent=self.root,
            )
            return
        self.push_history()
        before = list(record["bbox_xyxy"])
        record["bbox_xyxy"] = list(original)
        record["expert_action"] = "bbox_reset_to_original"
        self.log_action("bbox_reset_to_original", record, before=before, after=list(original))
        self.refresh_all()

    def snap_selected(self) -> None:
        record = self.get_record(self.selected_id)
        if record is None:
            messagebox.showinfo("Kutu seçilmedi", "Önce kenara oturtulacak kutuyu seçin.", parent=self.root)
            return
        snapped, diagnostics = snap_box_to_edges(
            record["bbox_xyxy"],
            self.grad_x,
            self.grad_y,
            self.cfg,
            self.image_width,
            self.image_height,
        )
        if snapped == record["bbox_xyxy"]:
            self.status_message_var.set("Yakındaki kenarlar mevcut kutudan daha güçlü bulunmadı; kutu değişmedi.")
            return
        before = list(record["bbox_xyxy"])
        preview = self.image.copy()
        draw = ImageDraw.Draw(preview)
        draw.rectangle(before, outline="#f59e0b", width=3)
        draw.rectangle(snapped, outline="#22c55e", width=3)
        # Keep UX simple: numerical confirmation, not a second image window.
        confirmed = messagebox.askyesno(
            "Kenarlara oturtma önerisi",
            f"Mevcut: {before}\nÖnerilen: {snapped}\n\nÖnerilen kutu uygulansın mı?",
            parent=self.root,
        )
        if not confirmed:
            return
        self.push_history()
        record["bbox_xyxy"] = snapped
        record["expert_action"] = "bbox_edge_snapped"
        record["expert_reviewed"] = True
        self.log_action("bbox_edge_snapped", record, diagnostics=diagnostics)
        self.refresh_all()

    def nudge_selected(self, dx: int, dy: int, event: tk.Event) -> None:
        record = self.get_record(self.selected_id)
        if record is None:
            return
        shift_pressed = bool(event.state & 0x0001)
        step = self.cfg.nudge_large_px if shift_pressed else self.cfg.nudge_small_px
        dx *= step
        dy *= step
        x1, y1, x2, y2 = record["bbox_xyxy"]
        width = x2 - x1
        height = y2 - y1
        nx1 = max(0, min(self.image_width - width, x1 + dx))
        ny1 = max(0, min(self.image_height - height, y1 + dy))
        if nx1 == x1 and ny1 == y1:
            return
        self.push_history()
        before = list(record["bbox_xyxy"])
        record["bbox_xyxy"] = [nx1, ny1, nx1 + width, ny1 + height]
        record["expert_action"] = "bbox_nudged"
        record["expert_reviewed"] = True
        self.log_action("bbox_nudged", record, before=before, after=list(record["bbox_xyxy"]))
        self.refresh_all()

    def delete_or_reject_selected(self) -> None:
        record = self.get_record(self.selected_id)
        if record is None:
            return
        if record["source"] == "expert_added_observed_evidence":
            confirmed = messagebox.askyesno(
                "Uzman kutusunu sil",
                "Uzman tarafından eklenen bu kutu tamamen silinsin mi?",
                parent=self.root,
            )
            if not confirmed:
                return
            self.push_history()
            record["hidden"] = True
            self.log_action("expert_added_record_deleted", record)
            self.selected_id = None
        else:
            self.push_history()
            record["status"] = "rejected"
            record["usable_as_prototype"] = False
            record["expert_reviewed"] = True
            record["expert_action"] = "rejected"
            self.log_action("candidate_rejected", record)
        self.renumber_records()
        self.refresh_all()

    # ------------------------------------------------------------------ MERGE / UNDO
    def merge_action(self) -> None:
        selected = self.get_record(self.selected_id)
        if selected is None:
            messagebox.showinfo("Kutu seçilmedi", "Birleştirme için önce bir kutu seçin.", parent=self.root)
            return
        if self.merge_base_id is None:
            self.merge_base_id = selected["id"]
            self.status_message_var.set(
                f"Birleştirme: ilk kutu #{selected['display_number']} seçildi. İkinci kutuyu seçip Birleştir'e yeniden basın."
            )
            return
        if self.merge_base_id == selected["id"]:
            self.merge_base_id = None
            self.status_message_var.set("Birleştirme iptal edildi.")
            return
        first = self.get_record(self.merge_base_id)
        second = selected
        if first is None:
            self.merge_base_id = None
            return
        if first["class"] != second["class"]:
            proceed = messagebox.askyesno(
                "Sınıflar farklı",
                f"İlk sınıf: {first['class']}\nİkinci sınıf: {second['class']}\n\nYine de birleştirilsin mi? İlk kutunun sınıfı korunur.",
                parent=self.root,
            )
            if not proceed:
                return
        self.push_history()
        merged_box = [
            min(first["bbox_xyxy"][0], second["bbox_xyxy"][0]),
            min(first["bbox_xyxy"][1], second["bbox_xyxy"][1]),
            max(first["bbox_xyxy"][2], second["bbox_xyxy"][2]),
            max(first["bbox_xyxy"][3], second["bbox_xyxy"][3]),
        ]
        first["bbox_xyxy"] = merged_box
        first["merged_from"] = list(dict.fromkeys(
            first.get("merged_from", [])
            + [first.get("original_candidate_id"), second.get("original_candidate_id"), second["id"]]
        ))
        first["merged_from"] = [value for value in first["merged_from"] if value]
        first["expert_action"] = "merged"
        first["expert_reviewed"] = True
        first["usable_as_prototype"] = first["status"] == "observed_intact" and bool(first["usable_as_prototype"])
        second["hidden"] = True
        self.log_action(
            "records_merged",
            first,
            merged_record_id=second["id"],
            merged_bbox=merged_box,
        )
        self.merge_base_id = None
        self.renumber_records()
        self.select_record(first["id"])
        self.refresh_all()
        self.status_message_var.set("İki kutu tek gözlenen eleman kaydı altında birleştirildi.")

    def undo(self) -> None:
        if not self.history:
            self.status_message_var.set("Geri alınacak işlem yok.")
            return
        self.records = self.history.pop()
        if self.get_record(self.selected_id) is None:
            self.selected_id = None
        self.actions.append({"timestamp": now_iso(), "action": "undo", "record_id": None, "details": {}})
        self.refresh_all()
        self.status_message_var.set("Son işlem geri alındı.")

    # ------------------------------------------------------------------ REFRESH / SUMMARY
    def refresh_tree(self) -> None:
        existing = set(self.tree.get_children())
        wanted: set[str] = set()
        for record in self.visible_records():
            record_id = record["id"]
            wanted.add(record_id)
            values = (
                record["display_number"],
                record["class"],
                STATUS_STYLE.get(record["status"], {}).get("title", record["status"]),
            )
            if record_id in existing:
                self.tree.item(record_id, values=values)
            else:
                self.tree.insert("", "end", iid=record_id, values=values)
        for record_id in existing - wanted:
            self.tree.delete(record_id)

    def refresh_summary(self) -> None:
        active = [record for record in self.records if not record.get("hidden")]
        counts = {status: sum(record["status"] == status for record in active) for status in STATUS_STYLE}
        expert_added = sum(record["source"] == "expert_added_observed_evidence" for record in active)
        corrected = sum(record["bbox_xyxy"] != record.get("original_bbox_xyxy") for record in active if record.get("original_bbox_xyxy"))
        self.summary_var.set(
            " | ".join(
                [
                    f"Sağlam: {counts['observed_intact']}",
                    f"Parçalı: {counts['observed_fragmented']}",
                    f"İnceleme: {counts['candidate_needs_review']}",
                    f"Reddedilen: {counts['rejected']}",
                    f"Uzman ekledi: {expert_added}",
                    f"Kutu düzeltildi: {corrected}",
                ]
            )
        )

    def refresh_all(self) -> None:
        self.renumber_records()
        self.refresh_tree()
        self.refresh_summary()
        self.update_inspector()
        self.redraw_canvas()

    # ------------------------------------------------------------------ SAVE OUTPUTS
    def normalized_output_record(self, record: dict[str, Any]) -> dict[str, Any]:
        box = list(record["bbox_xyxy"])
        width = box[2] - box[0]
        height = box[3] - box[1]
        return {
            "id": record["id"],
            "display_number": record["display_number"],
            "class": record["class"],
            "status": record["status"],
            "bbox_xyxy": box,
            "original_bbox_xyxy": record.get("original_bbox_xyxy"),
            "center": [round((box[0] + box[2]) / 2, 3), round((box[1] + box[3]) / 2, 3)],
            "width": width,
            "height": height,
            "source": record["source"],
            "original_candidate_id": record.get("original_candidate_id"),
            "ai_status": record.get("ai_status"),
            "ai_score": record.get("ai_score"),
            "expert_action": record.get("expert_action", "unmodified"),
            "expert_reviewed": bool(record.get("expert_reviewed", False)),
            "expert_note": record.get("expert_note", ""),
            "usable_as_prototype": bool(record.get("usable_as_prototype", False)),
            "merged_from": record.get("merged_from", []),
        }

    def find_font(self, size: int) -> ImageFont.ImageFont:
        candidates = [
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ]
        for path in candidates:
            if path.exists():
                try:
                    return ImageFont.truetype(str(path), size=size)
                except OSError:
                    continue
        return ImageFont.load_default()

    def make_output_visual(self, statuses: set[str], title: str) -> Image.Image:
        output = self.image.copy()
        draw = ImageDraw.Draw(output, "RGBA")
        font = self.find_font(self.cfg.output_label_font_size)
        header_height = 34
        draw.rectangle([0, 0, self.image_width, header_height], fill=(15, 23, 42, 235))
        draw.text((8, 8), title, fill=(255, 255, 255, 255), font=font)

        color_rgb = {
            status: tuple(int(STATUS_STYLE[status]["color"][index:index + 2], 16) for index in (1, 3, 5))
            for status in STATUS_STYLE
        }
        for record in self.records:
            if record.get("hidden") or record["status"] not in statuses:
                continue
            color = color_rgb[record["status"]]
            box = list(record["bbox_xyxy"])
            draw.rectangle(box, outline=color + (255,), width=self.cfg.output_box_thickness)
            label = f"#{record['display_number']} {record['class']}"
            try:
                text_bounds = draw.textbbox((box[0], box[1]), label, font=font)
                text_width = text_bounds[2] - text_bounds[0]
                text_height = text_bounds[3] - text_bounds[1]
            except AttributeError:
                text_width, text_height = draw.textsize(label, font=font)
            label_y = max(header_height + 2, box[1] - text_height - 6)
            draw.rectangle(
                [box[0], label_y, box[0] + text_width + 8, label_y + text_height + 5],
                fill=(15, 23, 42, 220),
                outline=color + (255,),
                width=1,
            )
            draw.text((box[0] + 4, label_y + 2), label, fill=(255, 255, 255, 255), font=font)
        return output

    def save_inventory(self, finalize: bool) -> None:
        active = [record for record in self.records if not record.get("hidden")]
        unresolved = [record for record in active if record["status"] == "candidate_needs_review"]
        if finalize and unresolved:
            messagebox.showwarning(
                "Envanter henüz kilitlenemez",
                f"{len(unresolved)} inceleme adayı hâlâ kararsız.\n\n"
                "Her sarı adayı sağlam, parçalı veya reddedildi durumuna getirin. "
                "Şimdilik Taslak Kaydet kullanılabilir.",
                parent=self.root,
            )
            return

        if not messagebox.askyesno(
            "Envanteri kaydet",
            (
                "Bu envanter KİLİTLENMİŞ uzman çıktısı olarak kaydedilecek."
                if finalize
                else "Bu envanter çözümlenmemiş adayları içeren TASLAK olarak kaydedilecek."
            )
            + "\n\nHerhangi bir önceki klasörün üzerine yazılmayacak. Devam edilsin mi?",
            parent=self.root,
        ):
            return

        run_id, run_dir = create_new_run_dir(Path(self.cfg.output_root))
        logger = setup_logger(run_dir, run_id)
        started = time.perf_counter()
        source_script = Path(__file__)
        try:
            logger.info("=" * 80)
            logger.info(f"[RUN] {run_id}")
            logger.info(f"[PARENT] {self.parent_run.resolve()}")
            logger.info("[STAGE] Stage 3B-v5 — uzman gözlenen-eleman envanteri")
            logger.info("[RULE] Uzman kutusu yalnız fotoğrafta görünen kanıt için kullanılabilir")
            logger.info(f"[MODE] {'FINALIZED' if finalize else 'DRAFT'}")
            logger.info("=" * 80)

            copied: dict[str, dict[str, Any]] = {}
            copy_map = {
                "input.png": "input.png",
                "completion_mask.png": "completion_mask.png",
                "stage3b_v4_observed_instances.json": "source_stage3b_v4_observed_instances.json",
                "stage3b_v4_review_candidates.json": "source_stage3b_v4_review_candidates.json",
            }
            optional_map = {
                "stage3b_v4_all_observed_instances.png": "source_stage3b_v4_all_observed_instances.png",
                "stage3b_v4_review_candidates.png": "source_stage3b_v4_review_candidates.png",
            }
            for source_name, target_name in {**copy_map, **optional_map}.items():
                source = self.parent_run / source_name
                if not source.exists():
                    if source_name in copy_map:
                        raise FileNotFoundError(f"Zorunlu parent girdisi eksik: {source}")
                    continue
                target = run_dir / target_name
                shutil.copy2(source, target)
                copied[source_name] = {
                    "source_path": str(source.resolve()),
                    "source_sha256": sha256_file(source),
                    "run_copy": target_name,
                    "run_copy_sha256": sha256_file(target),
                }

            normalized = [self.normalized_output_record(record) for record in active]
            accepted = [item for item in normalized if item["status"] in {"observed_intact", "observed_fragmented"}]
            rejected = [item for item in normalized if item["status"] == "rejected"]
            review = [item for item in normalized if item["status"] == "candidate_needs_review"]

            counts_by_class: dict[str, dict[str, int]] = {}
            for item in normalized:
                class_counts = counts_by_class.setdefault(
                    item["class"],
                    {status: 0 for status in STATUS_STYLE},
                )
                class_counts[item["status"]] += 1

            inventory_doc = {
                "run_id": run_id,
                "stage": "v8_stage3b_v5_expert_inventory_editor",
                "inventory_state": "finalized" if finalize else "draft",
                "locked": bool(finalize),
                "parent_run": {
                    "run_id": self.parent_run.name,
                    "path": str(self.parent_run.resolve()),
                },
                "policy": {
                    "observed_evidence_only": True,
                    "expert_may_add_visible_elements": True,
                    "expert_may_add_missing_hypothesis": False,
                    "unresolved_review_allowed": not finalize,
                },
                "summary": {
                    "total_active_records": len(normalized),
                    "observed_intact": sum(item["status"] == "observed_intact" for item in normalized),
                    "observed_fragmented": sum(item["status"] == "observed_fragmented" for item in normalized),
                    "rejected": len(rejected),
                    "candidate_needs_review": len(review),
                    "expert_added": sum(item["source"] == "expert_added_observed_evidence" for item in normalized),
                    "bbox_corrected": sum(
                        item.get("original_bbox_xyxy") is not None
                        and item["bbox_xyxy"] != item["original_bbox_xyxy"]
                        for item in normalized
                    ),
                    "counts_by_class": counts_by_class,
                },
                "elements": accepted,
                "unresolved_review": review,
            }
            write_json(run_dir / "expert_validated_inventory.json", inventory_doc)
            write_json(
                run_dir / "expert_rejected_elements.json",
                {"run_id": run_id, "elements": rejected},
            )
            write_json(
                run_dir / "expert_review_actions.json",
                {
                    "run_id": run_id,
                    "action_count": len(self.actions),
                    "actions": self.actions,
                },
            )

            self.make_output_visual(
                {"observed_intact", "observed_fragmented", "rejected", "candidate_needs_review"},
                "EXPERT INVENTORY OVERLAY — OBSERVED EVIDENCE ONLY",
            ).save(run_dir / "expert_inventory_overlay.png")
            self.make_output_visual(
                {"observed_intact"},
                "EXPERT-CONFIRMED INTACT ELEMENTS",
            ).save(run_dir / "expert_intact_elements.png")
            self.make_output_visual(
                {"observed_fragmented"},
                "EXPERT-CONFIRMED FRAGMENTED ELEMENTS",
            ).save(run_dir / "expert_fragmented_elements.png")
            self.make_output_visual(
                {"rejected"},
                "EXPERT-REJECTED CANDIDATES",
            ).save(run_dir / "expert_rejected_elements.png")

            report_lines = [
                "STAGE 3B-v5 — EXPERT OBSERVED-ELEMENT INVENTORY",
                f"Run: {run_id}",
                f"Parent: {self.parent_run.resolve()}",
                f"Inventory state: {'FINALIZED / LOCKED' if finalize else 'DRAFT'}",
                "",
                "POLICY:",
                "- AI candidates may be accepted, corrected, merged or rejected.",
                "- Expert may draw a box only around an element actually visible in the photograph.",
                "- Missing-element hypotheses are forbidden in this stage.",
                "",
                f"Active records: {len(normalized)}",
                f"Observed intact: {inventory_doc['summary']['observed_intact']}",
                f"Observed fragmented: {inventory_doc['summary']['observed_fragmented']}",
                f"Rejected: {inventory_doc['summary']['rejected']}",
                f"Unresolved review: {inventory_doc['summary']['candidate_needs_review']}",
                f"Expert-added observed elements: {inventory_doc['summary']['expert_added']}",
                f"Corrected bounding boxes: {inventory_doc['summary']['bbox_corrected']}",
                "",
                "CLASS COUNTS:",
            ]
            for class_name, class_counts in sorted(counts_by_class.items()):
                report_lines.append(
                    f"- {class_name}: intact={class_counts['observed_intact']}, "
                    f"fragmented={class_counts['observed_fragmented']}, "
                    f"review={class_counts['candidate_needs_review']}, "
                    f"rejected={class_counts['rejected']}"
                )
            (run_dir / "expert_inventory_report.txt").write_text(
                "\n".join(report_lines),
                encoding="utf-8",
            )

            metadata = {
                "run_id": run_id,
                "stage": "v8_stage3b_v5_expert_inventory_editor",
                "status": "completed",
                "inventory_state": "finalized" if finalize else "draft",
                "started_at": now_iso(),
                "finished_at": now_iso(),
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "parent_run": {
                    "run_id": self.parent_run.name,
                    "path": str(self.parent_run.resolve()),
                },
                "source_script": {
                    "filename": source_script.name,
                    "path": str(source_script.resolve()),
                    "sha256": sha256_file(source_script),
                },
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "opencv": cv2.__version__,
                    "numpy": np.__version__,
                    "pillow": getattr(Image, "__version__", "unknown"),
                    "tk": self.root.tk.call("info", "patchlevel"),
                },
                "config": asdict(self.cfg),
                "copied_inputs": copied,
                "summary": inventory_doc["summary"],
                "safety": {
                    "missing_elements_inferred": 0,
                    "hypothesis_boxes_created": 0,
                    "expert_added_records_are_observed_evidence_only": True,
                    "stable_diffusion_used": False,
                },
                "outputs": [
                    "expert_validated_inventory.json",
                    "expert_rejected_elements.json",
                    "expert_review_actions.json",
                    "expert_inventory_overlay.png",
                    "expert_intact_elements.png",
                    "expert_fragmented_elements.png",
                    "expert_rejected_elements.png",
                    "expert_inventory_report.txt",
                    "run.log",
                    "run_meta.json",
                ],
            }
            write_json(run_dir / "run_meta.json", metadata)

            logger.info(
                f"[COUNT] intact={inventory_doc['summary']['observed_intact']}, "
                f"fragmented={inventory_doc['summary']['observed_fragmented']}, "
                f"review={inventory_doc['summary']['candidate_needs_review']}, "
                f"rejected={inventory_doc['summary']['rejected']}, "
                f"expert_added={inventory_doc['summary']['expert_added']}"
            )
            logger.info(f"[OUT] {run_dir.resolve()}")
            logger.info("[OK] Uzman envanteri kaydedildi")

            self.saved_run_dir = run_dir
            self.closed_by_save = True
            messagebox.showinfo(
                "Envanter kaydedildi",
                f"Yeni run klasörü oluşturuldu:\n\n{run_dir.resolve()}",
                parent=self.root,
            )
            self.root.destroy()

        except Exception as exc:
            error_text = traceback.format_exc()
            logger.error(f"[FAIL] {exc}")
            logger.error(error_text)
            try:
                write_json(
                    run_dir / "run_meta.json",
                    {
                        "run_id": run_id,
                        "stage": "v8_stage3b_v5_expert_inventory_editor",
                        "status": "failed",
                        "parent_run": str(self.parent_run.resolve()),
                        "error": {
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                            "traceback": error_text,
                        },
                    },
                )
            except Exception:
                pass
            messagebox.showerror("Kayıt hatası", f"Envanter kaydedilemedi:\n\n{exc}", parent=self.root)

    def cancel_editor(self) -> None:
        if self.closed_by_save:
            self.root.destroy()
            return
        if messagebox.askyesno(
            "Editörü kapat",
            "Kaydedilmemiş uzman düzenlemeleri kaybolacak. Editör kapatılsın mı?",
            parent=self.root,
        ):
            self.root.destroy()


# =============================================================================
# MAIN
# =============================================================================
def run_editor(cfg: Config) -> Path | None:
    parent = find_parent_run(cfg)
    input_path = parent / "input.png"
    image = Image.open(input_path).convert("RGB")
    damage_mask = load_binary_mask(
        parent / "completion_mask.png",
        image.size,
    )
    records, context = load_editor_records(
        parent / "stage3b_v4_observed_instances.json",
        parent / "stage3b_v4_review_candidates.json",
        image.width,
        image.height,
    )
    if not records:
        raise RuntimeError("Editöre yüklenecek observed/review eleman kaydı bulunamadı.")

    root = tk.Tk()
    editor = ExpertInventoryEditor(
        root=root,
        cfg=cfg,
        parent_run=parent,
        image=image,
        damage_mask=damage_mask,
        records=records,
        input_context=context,
    )
    root.mainloop()
    return editor.saved_run_dir


def main() -> None:
    cfg = Config()
    try:
        run_dir = run_editor(cfg)
        if run_dir is not None:
            print(f"\nBİTTİ: {run_dir.resolve()}")
        else:
            print("\nEditör kaydetmeden kapatıldı; yeni run klasörü oluşturulmadı.")
    except Exception as exc:
        print(f"\nHATA: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
