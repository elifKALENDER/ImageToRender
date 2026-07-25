"""
ImageToRender2Restoration_v6.py — YAPININ KENDİSİNDEN OKUYAN ÖZGÜN HÂL ÖNGÖRÜSÜ
====================================================================================
AMAÇ
----
Bu sürüm yalnızca şu soruya odaklanır:

    "Yapının ilk/özgün hâli, yalnızca bu fotoğrafta ayakta kalan kanıtlardan okunarak
     nasıl görünmüş olabilir?"

Venedik Tüzüğü restorasyon modu bu deneyde bilinçli olarak çalıştırılmaz. Önce özgün
hâl öngörüsünün kanıta sadık çalışması hedeflenir.

TEMEL FARK
----------
Önceki inpaint sürümü boşluğu tamamlıyor fakat Stable Diffusion'ın genel "tarihî bina"
ezberinden yabancı kemer, bezeme veya profil ekleyebiliyordu. Bu sürüm maskeli alan için
fotoğrafın KENDİSİNDEN bir "kanıt rehberi" üretir:

  1. Manuel completion_mask.png ile yalnız gerçekten kayıp yapı bölgesi seçilir.
  2. Yapının simetri ekseninin karşı tarafındaki sağlam pikseller maskeli alana taşınarak
     auto_evidence_guide.png hazırlanır.
  3. Lineart ve depth kontrolünün maskeli kısmı bu kanıt rehberinden kurulur.
  4. Inpaint başlangıç görüntüsü de bu rehberle önceden doldurulur.
  5. Maske dışı son görüntü, orijinal fotoğraftan piksel düzeyinde geri alınır.

Bu yöntem kesin restitüsyon üretmez. Aynı yapıdaki tekrar, simetri, profil ve malzeme
izlerinden türetilmiş araştırma öngörüsü üretir. Kanıt zayıfsa uzman rehberi gerekir.

GİRDİLER
--------
  image.png                 Zorunlu fotoğraf.
  completion_mask.png       Beyaz=tamamla, siyah=aynen koru.
                            Dosya yoksa basit maske editörü açılır.
  completion_guide.png      İsteğe bağlı, fotoğrafla aynı kadrajda uzman geometri çizimi.
                            Varsa yalnız maskeli alandaki LINEART rehberini güçlendirir.

ÖNEMLİ AYAR
-----------
  symmetry_axis_x_ratio = 0.50

0.50 görselin tam orta eksenidir. Yapının gerçek simetri ekseni fotoğrafın ortasından
sapıyorsa bu oranı değiştir. Örnek: eksen görsel genişliğinin %47'sindeyse 0.47.

ÇIKTI KLASÖRÜ — DEĞİŞTİRME
---------------------------
  outputs/restore_YYYYMMDD_HHMMSS

Kaynak Python dosyasının adı, tam yolu ve SHA256 değeri run_meta.json içinde tutulur;
çıktı klasörüne source_*.py kopyası atılmaz.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from PIL.PngImagePlugin import PngInfo
from controlnet_aux.lineart import LineartDetector
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetInpaintPipeline,
)
from transformers import AutoImageProcessor, DPTForDepthEstimation

try:
    from diffusers.utils import logging as dlog
    from transformers.utils import logging as tlog

    dlog.set_verbosity_error()
    tlog.set_verbosity_error()
except Exception:
    pass

_RESAMPLING = getattr(Image, "Resampling", Image)
LANCZOS = _RESAMPLING.LANCZOS
NEAREST = _RESAMPLING.NEAREST


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class OriginalStateConfig:
    label: str = "ÖZGÜN HÂL — YAPI İÇİ KANIT ÖNGÖRÜSÜ"

    # Evidence-init bulunduğu için 0.96 yerine daha kontrollü bir değer.
    strength: float = 0.82
    depth_scale: float = 0.46
    lineart_scale: float = 0.72
    guidance_end: tuple[float, float] = (0.68, 0.86)

    # SD1.5 CLIP sınırı nedeniyle en önemli hükümler prompt'un başındadır.
    prompt: str = (
        "exact same building, reconstruct only masked missing masonry from surviving evidence "
        "visible in this photograph, copy and continue adjacent repeated arches, columns, moldings, "
        "cornices, wall thickness, stone courses and supported bilateral symmetry, no new motif, "
        "no foreign ornament, simplest evidence-based continuation, period-authentic stone and lime "
        "mortar, photorealistic, same camera, natural daylight"
    )
    negative: str = (
        "foreign architecture, borrowed historical detail, invented ornament, invented motif, extra arch, "
        "extra window, extra niche, extra column, unrelated capital, fantasy restoration, redesign, "
        "ornate carving not visible in the source, altered proportions, duplicate elements, speculative roof, "
        "modern materials, concrete frame, glass facade, warped geometry, melted masonry, changed camera, "
        "changed background, bright new facade, low quality, blurry, painting, illustration, text, watermark"
    )


@dataclass
class Config:
    # Girdi
    input_photo: str = "image.png"
    completion_mask: str = "completion_mask.png"
    completion_guide: str = "completion_guide.png"
    period_hint: str = ""

    # Maske editörü
    open_mask_editor_if_missing: bool = True
    mask_editor_brush_px: int = 34
    mask_editor_max_width: int = 1200
    mask_editor_max_height: int = 780

    # Yapının kendisinden kanıt rehberi
    # "auto"   : completion_guide varsa manual lineart, yoksa mirror lineart.
    # "manual" : completion_guide zorunlu.
    # "mirror" : yalnız fotoğrafın simetrik karşılığını kullan.
    # "none"   : kanıt rehberi kullanma (karşılaştırma/debug için).
    guide_mode: str = "auto"
    symmetry_axis_x_ratio: float = 0.50
    mirror_fill_radius: int = 4
    save_evidence_debug: bool = True

    # Üretim
    seeds: list[int] = field(default_factory=lambda: [42, 1001, 2024])
    selected_seed_file: str = "selected_seed.json"
    max_side: int = 512
    steps: int = 42
    guidance_scale: float = 5.5

    # Inpaint maskesi ve son birleştirme
    mask_expand_px: int = 4
    mask_feather_px: float = 2.0
    final_feather_px: float = 1.5

    # Modeller
    base_model: str = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
    vae_id: str = "stabilityai/sd-vae-ft-mse"
    depth_controlnet: str = "lllyasviel/control_v11f1p_sd15_depth"
    lineart_controlnet: str = "lllyasviel/control_v11p_sd15_lineart"
    depth_estimator: str = "Intel/dpt-hybrid-midas"
    annotator_repo: str = "lllyasviel/Annotators"
    use_safetensors: bool = False

    # Eski diffusers ortamını bozmamak için kapalı. Destek varsa açılabilir.
    use_ip_adapter: bool = False
    ip_adapter_scale: float = 0.20
    ip_adapter_repo: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "models"
    ip_adapter_weight: str = "ip-adapter_sd15.bin"
    ip_cache_dir: str = r"E:\ImageToRender\models"

    # Bellek / kayıt
    enable_xformers: bool = True
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = True
    enable_cpu_offload: bool = True
    save_raw_debug: bool = False
    save_pip_freeze: bool = False
    output_root: str = "outputs"


MODE = OriginalStateConfig()


# ══════════════════════════════════════════════════════════════════════════════
# RUN / LOG / METADATA
# ══════════════════════════════════════════════════════════════════════════════
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def collect_environment(device: str, dtype: torch.dtype) -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        try:
            gpu = torch.cuda.get_device_name(0)
        except Exception:
            gpu = "CUDA available"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "device": device,
        "dtype": str(dtype),
        "packages": {
            "diffusers": package_version("diffusers"),
            "transformers": package_version("transformers"),
            "controlnet-aux": package_version("controlnet-aux"),
            "Pillow": package_version("Pillow"),
            "numpy": package_version("numpy"),
            "opencv-python": package_version("opencv-python"),
        },
    }


def source_info() -> dict[str, Any]:
    source = Path(__file__).resolve()
    return {
        "filename": source.name,
        "absolute_path": str(source),
        "sha256": sha256_file(source),
    }


def make_run_dir(cfg: Config) -> tuple[str, Path]:
    run_id = datetime.now().strftime("restore_%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def setup_logger(run_dir: Path, run_id: str) -> logging.Logger:
    logger = logging.getLogger(f"original_state.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_pip_freeze(run_dir: Path, logger: logging.Logger) -> str | None:
    out = run_dir / "pip_freeze.txt"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        out.write_text(result.stdout, encoding="utf-8")
        logger.info("[ENV] pip_freeze.txt kaydedildi")
        return out.name
    except Exception as exc:
        logger.warning(f"[ENV] pip freeze alınamadı: {exc}")
        return None


def load_selected_seeds(default: list[int], path: str, logger: logging.Logger) -> list[int]:
    p = Path(path)
    if not p.exists():
        return list(default)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data.get("seeds"), list):
            seeds = [int(value) for value in data["seeds"]]
        elif "seed" in data:
            seeds = [int(data["seed"])]
        else:
            raise ValueError("'seed' veya 'seeds' alanı yok")
        if not seeds:
            raise ValueError("seed listesi boş")
        logger.info(f"[SEED] {p} -> {seeds}")
        return seeds
    except Exception as exc:
        logger.warning(f"[SEED] {p} okunamadı ({exc}); varsayılan {default}")
        return list(default)


# ══════════════════════════════════════════════════════════════════════════════
# GÖRSEL / MASKE
# ══════════════════════════════════════════════════════════════════════════════
def rounded_multiple(value: float, multiple: int = 8, minimum: int = 64) -> int:
    return max(minimum, int(round(value / multiple) * multiple))


def load_photo_preserve_ratio(path: Path, max_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = max_side / max(width, height)
    new_width = rounded_multiple(width * scale)
    new_height = rounded_multiple(height * scale)
    return image.resize((new_width, new_height), LANCZOS)


def load_binary_mask(path: Path, size: tuple[int, int]) -> Image.Image:
    mask = Image.open(path).convert("L").resize(size, NEAREST)
    array = np.array(mask)
    return Image.fromarray(np.where(array >= 128, 255, 0).astype(np.uint8), "L")


def mask_area_ratio(mask: Image.Image) -> float:
    return float((np.array(mask.convert("L")) > 127).mean())


def expand_and_feather_mask(mask: Image.Image, expand_px: int, feather_px: float) -> Image.Image:
    result = mask.copy()
    for _ in range(max(0, int(expand_px))):
        result = result.filter(ImageFilter.MaxFilter(3))
    if feather_px > 0:
        result = result.filter(ImageFilter.GaussianBlur(float(feather_px)))
    return result


def feather_mask(mask: Image.Image, feather_px: float) -> Image.Image:
    if feather_px <= 0:
        return mask.copy()
    return mask.filter(ImageFilter.GaussianBlur(float(feather_px)))


def create_mask_with_editor(
    photo_path: Path,
    output_path: Path,
    brush_px: int = 34,
    max_width: int = 1200,
    max_height: int = 780,
) -> bool:
    """Sol fare ekler, sağ fare siler, tekerlek fırçayı değiştirir."""
    try:
        import tkinter as tk
        from tkinter import messagebox
        from PIL import ImageTk
    except Exception as exc:
        raise RuntimeError(
            "Maske editörü açılamadı. Tkinter ve PIL.ImageTk gerekli. "
            f"Ayrıntı: {exc}"
        ) from exc

    original = Image.open(photo_path).convert("RGB")
    original_width, original_height = original.size
    scale = min(max_width / original_width, max_height / original_height, 1.0)
    display_width = max(1, int(round(original_width * scale)))
    display_height = max(1, int(round(original_height * scale)))
    display_photo = original.resize((display_width, display_height), LANCZOS)
    display_mask = Image.new("L", (display_width, display_height), 0)

    root = tk.Tk()
    root.title("Özgün Hâl — Tamamlama Maskesi")
    root.configure(bg="#202020")

    state = {
        "brush": max(4, int(round(brush_px * scale))),
        "saved": False,
        "painting": False,
        "erasing": False,
    }

    title = tk.Label(
        root,
        text=(
            "Yalnız gerçekten kaybolmuş MİMARİ KÜTLEYİ boya. Kırmızı alan tamamlanır. "
            "Gerçek kapı/kemer açıklıklarını, insanı, zemini ve arka planı boyama."
        ),
        fg="white",
        bg="#202020",
        font=("Segoe UI", 10),
        wraplength=max(520, display_width),
        justify="left",
    )
    title.pack(fill="x", padx=10, pady=(10, 6))

    canvas = tk.Canvas(
        root,
        width=display_width,
        height=display_height,
        highlightthickness=0,
        bg="black",
    )
    canvas.pack(padx=10, pady=4)

    status_var = tk.StringVar()
    status = tk.Label(root, textvariable=status_var, fg="white", bg="#202020")
    status.pack(fill="x", padx=10, pady=(2, 4))
    image_holder: dict[str, Any] = {"image": None}

    def update_status() -> None:
        original_brush = max(1, int(round(state["brush"] / max(scale, 1e-6))))
        status_var.set(
            f"Fırça: yaklaşık {original_brush}px | Sol: ekle | Sağ: sil | Tekerlek: boyut"
        )

    def redraw() -> None:
        preview = display_photo.convert("RGBA")
        red = Image.new("RGBA", (display_width, display_height), (255, 36, 36, 0))
        alpha = display_mask.point(lambda value: 150 if value > 0 else 0)
        red.putalpha(alpha)
        preview = Image.alpha_composite(preview, red).convert("RGB")
        tk_image = ImageTk.PhotoImage(preview)
        image_holder["image"] = tk_image
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=tk_image)

    def paint_at(x: int, y: int, value: int) -> None:
        radius = state["brush"] / 2
        draw = ImageDraw.Draw(display_mask)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=value)
        redraw()

    def left_down(event) -> None:
        state["painting"] = True
        paint_at(event.x, event.y, 255)

    def left_move(event) -> None:
        if state["painting"]:
            paint_at(event.x, event.y, 255)

    def left_up(_event) -> None:
        state["painting"] = False

    def right_down(event) -> None:
        state["erasing"] = True
        paint_at(event.x, event.y, 0)

    def right_move(event) -> None:
        if state["erasing"]:
            paint_at(event.x, event.y, 0)

    def right_up(_event) -> None:
        state["erasing"] = False

    def mouse_wheel(event) -> None:
        delta = 3 if event.delta > 0 else -3
        state["brush"] = max(4, min(180, state["brush"] + delta))
        update_status()

    def clear_mask() -> None:
        display_mask.paste(0, (0, 0, display_width, display_height))
        redraw()

    def save_and_continue() -> None:
        if not np.any(np.array(display_mask) > 0):
            messagebox.showwarning(
                "Maske boş",
                "Tamamlanacak gerçek yapı kaybını sol fareyle işaretle.",
            )
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final = display_mask.resize((original_width, original_height), NEAREST)
        array = np.array(final)
        final = Image.fromarray(np.where(array >= 128, 255, 0).astype(np.uint8), "L")
        final.save(output_path)
        state["saved"] = True
        root.destroy()

    def cancel() -> None:
        root.destroy()

    canvas.bind("<ButtonPress-1>", left_down)
    canvas.bind("<B1-Motion>", left_move)
    canvas.bind("<ButtonRelease-1>", left_up)
    canvas.bind("<ButtonPress-3>", right_down)
    canvas.bind("<B3-Motion>", right_move)
    canvas.bind("<ButtonRelease-3>", right_up)
    canvas.bind("<MouseWheel>", mouse_wheel)

    buttons = tk.Frame(root, bg="#202020")
    buttons.pack(fill="x", padx=10, pady=(4, 10))
    tk.Button(buttons, text="Maskeyi Temizle", command=clear_mask).pack(side="left")
    tk.Button(buttons, text="İptal", command=cancel).pack(side="right", padx=(6, 0))
    tk.Button(
        buttons,
        text="Maskeyi Kaydet ve Devam Et",
        command=save_and_continue,
        font=("Segoe UI", 9, "bold"),
    ).pack(side="right")

    update_status()
    redraw()
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()
    return bool(state["saved"] and output_path.exists())


# ══════════════════════════════════════════════════════════════════════════════
# YAPININ KENDİSİNDEN KANIT REHBERİ
# ══════════════════════════════════════════════════════════════════════════════
def _fill_unresolved_regions(
    image_array: np.ndarray,
    unresolved: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, str]:
    """Aynadan kanıt bulunamayan küçük alanları yalnız komşu görüntüden nötr biçimde doldurur."""
    if not np.any(unresolved):
        return image_array, "none_needed"

    try:
        import cv2

        mask = unresolved.astype(np.uint8) * 255
        bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        filled = cv2.inpaint(bgr, mask, max(1, int(radius)), cv2.INPAINT_TELEA)
        return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB), "opencv_telea"
    except Exception:
        blurred = np.array(
            Image.fromarray(image_array, "RGB").filter(
                ImageFilter.GaussianBlur(max(2.0, float(radius) * 2.0))
            )
        )
        result = image_array.copy()
        result[unresolved] = blurred[unresolved]
        return result, "pil_blur_fallback"


def build_mirror_evidence(
    source: Image.Image,
    mask: Image.Image,
    axis_x_ratio: float,
    fill_radius: int,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """
    Maskeli pikseli, aynı y seviyesinde simetri ekseninin karşısındaki SAĞLAM pikselden alır.
    Donör de maskeliyse veya görüntü dışındaysa nötr komşu doldurma kullanılır.
    """
    source_array = np.array(source.convert("RGB"), dtype=np.uint8)
    mask_array = np.array(mask.resize(source.size, NEAREST).convert("L")) > 127
    height, width = mask_array.shape

    ratio = float(np.clip(axis_x_ratio, 0.0, 1.0))
    axis_x = ratio * (width - 1)
    x = np.arange(width)
    mirrored_x = np.rint(2.0 * axis_x - x).astype(np.int32)
    valid_x = (mirrored_x >= 0) & (mirrored_x < width)
    clipped_x = np.clip(mirrored_x, 0, width - 1)

    donor_mask = mask_array[:, clipped_x]
    donor_valid = np.broadcast_to(valid_x[None, :], (height, width)) & (~donor_mask)
    use_mirror = mask_array & donor_valid

    result = source_array.copy()
    mirrored_source = source_array[:, clipped_x, :]
    result[use_mirror] = mirrored_source[use_mirror]

    unresolved = mask_array & (~use_mirror)
    result, fallback_method = _fill_unresolved_regions(result, unresolved, fill_radius)

    donor_map = Image.fromarray(use_mirror.astype(np.uint8) * 255, "L")
    masked_count = int(mask_array.sum())
    mirror_count = int(use_mirror.sum())
    unresolved_count = int(unresolved.sum())
    stats = {
        "axis_x_ratio": ratio,
        "axis_x_px": round(axis_x, 3),
        "masked_pixels": masked_count,
        "mirror_donor_pixels": mirror_count,
        "unresolved_pixels": unresolved_count,
        "mirror_coverage_of_mask": (mirror_count / masked_count) if masked_count else 0.0,
        "fallback_method": fallback_method,
    }
    return Image.fromarray(result, "RGB"), donor_map, stats


def merge_control_inside_mask(
    original_control: Image.Image,
    evidence_control: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    original = original_control.convert("RGB")
    evidence = evidence_control.convert("RGB").resize(original.size, LANCZOS)
    binary = mask.resize(original.size, NEAREST).convert("L")
    return Image.composite(evidence, original, binary)


def neutralize_control_inside_mask(control: Image.Image, mask: Image.Image) -> Image.Image:
    array = np.array(control.convert("RGB")).copy()
    selected = np.array(mask.resize(control.size, NEAREST).convert("L")) > 127
    outside = array[~selected]
    fill = np.median(outside, axis=0).astype(np.uint8) if outside.size else np.array([127] * 3)
    array[selected] = fill
    return Image.fromarray(array.astype(np.uint8), "RGB")


# ══════════════════════════════════════════════════════════════════════════════
# CONDITION MODELLERİ
# ══════════════════════════════════════════════════════════════════════════════
def prepare_depth(
    photo: Image.Image,
    model_id: str,
    device: str,
    logger: logging.Logger,
) -> Image.Image:
    logger.info(f"[COND] depth: {model_id}")
    model = DPTForDepthEstimation.from_pretrained(model_id).to(device)
    processor = AutoImageProcessor.from_pretrained(model_id)
    inputs = processor(images=photo, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        prediction = model(**inputs).predicted_depth
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=photo.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    array = prediction.squeeze().detach().float().cpu().numpy()
    array -= array.min()
    denominator = float(array.max()) or 1.0
    array = np.clip(array * 255.0 / denominator, 0, 255).astype(np.uint8)
    depth = Image.fromarray(array, "L").convert("RGB")

    del model, processor, inputs, prediction
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return depth


def prepare_lineart(
    source: Image.Image,
    annotator_repo: str,
    logger: logging.Logger,
    label: str,
) -> Image.Image:
    logger.info(f"[COND] lineart ({label}): {annotator_repo}")
    detector = LineartDetector.from_pretrained(annotator_repo)
    result = detector(source)
    if result.mode != "RGB":
        result = result.convert("RGB")
    result = result.resize(source.size, LANCZOS)
    del detector
    gc.collect()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# GÖRSEL KAYIT
# ══════════════════════════════════════════════════════════════════════════════
def find_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                pass
    return ImageFont.load_default()


def stamp(image: Image.Image, text: str) -> Image.Image:
    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    font = find_font(max(13, round(result.width / 42)))
    bbox = draw.textbbox((0, 0), text, font=font)
    bar_height = max(30, bbox[3] - bbox[1] + 12)
    draw.rectangle([0, result.height - bar_height, result.width, result.height], fill=(0, 0, 0))
    draw.text((8, result.height - bar_height + 5), text, fill=(255, 255, 255), font=font)
    return result


def save_png_with_meta(image: Image.Image, path: Path, metadata: dict[str, Any]) -> None:
    png_info = PngInfo()
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, ensure_ascii=False)
        png_info.add_text(str(key), str(value))
    image.save(path, pnginfo=png_info)


def contact_sheet(
    images: list[Image.Image],
    labels: list[str],
    output: Path,
    png_metadata: dict[str, Any],
) -> None:
    if not images:
        return
    width, height = images[0].size
    label_height = 58
    sheet = Image.new("RGB", (width * len(images), height + label_height), (246, 246, 246))
    draw = ImageDraw.Draw(sheet)
    font = find_font(max(13, round(width / 38)))
    for index, (image, label) in enumerate(zip(images, labels)):
        x = index * width
        sheet.paste(image.resize((width, height), LANCZOS), (x, 0))
        draw.text((x + 8, height + 8), label, fill=(0, 0, 0), font=font)
    save_png_with_meta(sheet, output, png_metadata)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def build_pipeline(cfg: Config, device: str, dtype: torch.dtype, logger: logging.Logger):
    logger.info("[SETUP] 2 ControlNet: evidence-depth + evidence-lineart")
    controls = [
        ControlNetModel.from_pretrained(cfg.depth_controlnet, torch_dtype=dtype),
        ControlNetModel.from_pretrained(cfg.lineart_controlnet, torch_dtype=dtype),
    ]
    vae = AutoencoderKL.from_pretrained(cfg.vae_id, torch_dtype=dtype)
    logger.info(f"[VAE] {cfg.vae_id}")

    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        cfg.base_model,
        controlnet=controls,
        vae=vae,
        torch_dtype=dtype,
        safety_checker=None,
        use_safetensors=cfg.use_safetensors,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
        use_karras_sigmas=True,
    )

    if cfg.enable_xformers:
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            logger.info("[VRAM] xformers aktif")
        except Exception as exc:
            logger.warning(f"[VRAM] xformers açılamadı ({exc}); attention slicing")
            pipeline.enable_attention_slicing("max")
    else:
        pipeline.enable_attention_slicing("max")

    if cfg.enable_vae_slicing:
        pipeline.enable_vae_slicing()
    if cfg.enable_vae_tiling:
        try:
            pipeline.enable_vae_tiling()
        except Exception:
            pass

    ip_active = False
    if cfg.use_ip_adapter:
        if not hasattr(pipeline, "load_ip_adapter"):
            logger.warning("[IP] Bu diffusers sürümünde load_ip_adapter yok; atlandı")
        else:
            try:
                pipeline.load_ip_adapter(
                    cfg.ip_adapter_repo,
                    subfolder=cfg.ip_adapter_subfolder,
                    weight_name=cfg.ip_adapter_weight,
                    cache_dir=cfg.ip_cache_dir,
                )
                pipeline.set_ip_adapter_scale(cfg.ip_adapter_scale)
                ip_active = True
                logger.info(f"[IP] aktif scale={cfg.ip_adapter_scale}")
            except Exception as exc:
                logger.warning(f"[IP] yüklenemedi ({exc}); atlandı")

    if device == "cuda" and cfg.enable_cpu_offload:
        logger.info("[VRAM] model CPU offload aktif")
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device)

    return pipeline, ip_active


def execution_device(pipeline, fallback: str) -> str:
    try:
        return str(pipeline._execution_device)
    except Exception:
        try:
            return str(pipeline.device)
        except Exception:
            return fallback


def generate_one(
    pipeline,
    original_photo: Image.Image,
    evidence_init: Image.Image,
    work_mask: Image.Image,
    final_mask: Image.Image,
    depth_control: Image.Image,
    lineart_control: Image.Image,
    mode: OriginalStateConfig,
    seed: int,
    cfg: Config,
    device: str,
    ip_active: bool,
) -> tuple[Image.Image, Image.Image]:
    prompt = mode.prompt
    if cfg.period_hint.strip():
        prompt = f"{prompt}, expert supplied period: {cfg.period_hint.strip()}"

    generator = torch.Generator(device=execution_device(pipeline, device)).manual_seed(seed)
    kwargs = {
        "prompt": prompt,
        "negative_prompt": mode.negative,
        "image": evidence_init,
        "mask_image": work_mask,
        "control_image": [depth_control, lineart_control],
        "controlnet_conditioning_scale": [mode.depth_scale, mode.lineart_scale],
        "control_guidance_end": list(mode.guidance_end),
        "strength": mode.strength,
        "num_inference_steps": cfg.steps,
        "guidance_scale": cfg.guidance_scale,
        "height": original_photo.height,
        "width": original_photo.width,
        "generator": generator,
    }
    if ip_active:
        kwargs["ip_adapter_image"] = original_photo

    raw = pipeline(**kwargs).images[0].convert("RGB")
    final = Image.composite(raw, original_photo, final_mask.convert("L"))
    return raw, final


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    cfg = Config()

    input_path = Path(cfg.input_photo)
    if not input_path.exists():
        raise FileNotFoundError(f"Ana fotoğraf bulunamadı: {input_path}")

    mask_path = Path(cfg.completion_mask)
    if not mask_path.exists():
        if not cfg.open_mask_editor_if_missing:
            raise FileNotFoundError(
                f"'{cfg.completion_mask}' bulunamadı. Beyaz=tamamlanacak yapı, siyah=korunacak alan."
            )
        print(f"[MASK] '{cfg.completion_mask}' yok; manuel maske editörü açılıyor...")
        saved = create_mask_with_editor(
            photo_path=input_path,
            output_path=mask_path,
            brush_px=cfg.mask_editor_brush_px,
            max_width=cfg.mask_editor_max_width,
            max_height=cfg.mask_editor_max_height,
        )
        if not saved:
            print("[DURDU] Maske kaydedilmedi; üretim başlatılmadı.")
            return
        print(f"[MASK] Kaydedildi: {mask_path}")

    guide_mode = cfg.guide_mode.strip().lower()
    if guide_mode not in {"auto", "manual", "mirror", "none"}:
        raise ValueError("guide_mode yalnız 'auto', 'manual', 'mirror' veya 'none' olabilir")
    if not 0.0 <= cfg.symmetry_axis_x_ratio <= 1.0:
        raise ValueError("symmetry_axis_x_ratio 0.0 ile 1.0 arasında olmalı")
    if guide_mode == "manual" and not Path(cfg.completion_guide).exists():
        raise FileNotFoundError(
            f"guide_mode='manual' fakat '{cfg.completion_guide}' bulunamadı"
        )

    run_id, run_dir = make_run_dir(cfg)
    logger = setup_logger(run_dir, run_id)
    meta_path = run_dir / "run_meta.json"
    started = time.perf_counter()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    script_info = source_info()

    metadata: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "started_at": now_iso(),
        "source_script": script_info,
        "environment": collect_environment(device, dtype),
        "config": asdict(cfg),
        "mode": asdict(MODE),
        "input": {},
        "evidence": {},
        "outputs": [],
        "log_file": "run.log",
    }
    write_json(meta_path, metadata)

    try:
        logger.info("=" * 72)
        logger.info(f"[RUN] {run_id}")
        logger.info(f"[SOURCE] {script_info['filename']} | sha256={script_info['sha256'][:12]}...")
        logger.info(f"[DEVICE] {device} | {metadata['environment'].get('gpu')}")
        logger.info("[MODE] yalnız özgün hâl / yapı içi kanıt")
        logger.info("=" * 72)

        if cfg.save_pip_freeze:
            metadata["pip_freeze_file"] = save_pip_freeze(run_dir, logger)

        photo = load_photo_preserve_ratio(input_path, cfg.max_side)
        hard_mask = load_binary_mask(mask_path, photo.size)
        area_ratio = mask_area_ratio(hard_mask)
        if area_ratio < 0.001:
            raise ValueError(f"completion_mask.png neredeyse boş: %{area_ratio * 100:.3f}")
        if area_ratio > 0.70:
            logger.warning(f"[MASK] Maske çok geniş: %{area_ratio * 100:.1f}; sonuç daha varsayımsal olur")

        photo.save(run_dir / "input.png")
        hard_mask.save(run_dir / "mask_ozgun_hal_hard.png")
        work_mask = expand_and_feather_mask(hard_mask, cfg.mask_expand_px, cfg.mask_feather_px)
        final_mask = feather_mask(hard_mask, cfg.final_feather_px)
        work_mask.save(run_dir / "mask_ozgun_hal_work.png")
        final_mask.save(run_dir / "mask_ozgun_hal_final_blend.png")

        metadata["input"] = {
            "photo": {
                "original_path": str(input_path.resolve()),
                "filename": input_path.name,
                "sha256": sha256_file(input_path),
                "processed_size": list(photo.size),
                "run_copy": "input.png",
            },
            "mask": {
                "original_path": str(mask_path.resolve()),
                "filename": mask_path.name,
                "sha256": sha256_file(mask_path),
                "area_ratio": area_ratio,
                "hard_copy": "mask_ozgun_hal_hard.png",
                "work_copy": "mask_ozgun_hal_work.png",
                "final_blend_copy": "mask_ozgun_hal_final_blend.png",
            },
        }
        logger.info(f"[INPUT] {input_path.name} -> {photo.width}x{photo.height} (crop yok)")
        logger.info(f"[MASK] {mask_path.name}, alan=%{area_ratio * 100:.2f}")

        cfg.seeds = load_selected_seeds(cfg.seeds, cfg.selected_seed_file, logger)
        metadata["resolved_seeds"] = cfg.seeds

        # ── Fotoğrafın kendisinden aynalı kanıt-init üret ──
        mirror_guide, donor_map, mirror_stats = build_mirror_evidence(
            photo,
            hard_mask,
            cfg.symmetry_axis_x_ratio,
            cfg.mirror_fill_radius,
        )
        mirror_guide.save(run_dir / "auto_evidence_guide.png")
        donor_map.save(run_dir / "auto_evidence_donor_map.png")
        logger.info(
            f"[EVIDENCE] simetri ekseni x={mirror_stats['axis_x_px']:.1f}px, "
            f"maskede ayna kapsaması=%{mirror_stats['mirror_coverage_of_mask'] * 100:.1f}"
        )

        manual_guide_path = Path(cfg.completion_guide)
        manual_exists = manual_guide_path.exists()
        if guide_mode == "auto":
            resolved_guide_mode = "manual_lineart+mirror_init" if manual_exists else "mirror"
        elif guide_mode == "manual":
            resolved_guide_mode = "manual_lineart+mirror_init"
        else:
            resolved_guide_mode = guide_mode

        if resolved_guide_mode == "none":
            evidence_init = photo.copy()
        else:
            # Renk/doku başlangıcı daima aynı fotoğrafın aynalı kanıtından gelir.
            evidence_init = mirror_guide.copy()
        evidence_init.save(run_dir / "evidence_init.png")

        manual_guide = None
        if resolved_guide_mode == "manual_lineart+mirror_init":
            manual_guide = Image.open(manual_guide_path).convert("RGB").resize(photo.size, LANCZOS)
            manual_guide.save(run_dir / "completion_guide.png")
            metadata["input"]["completion_guide"] = {
                "original_path": str(manual_guide_path.resolve()),
                "filename": manual_guide_path.name,
                "sha256": sha256_file(manual_guide_path),
                "run_copy": "completion_guide.png",
            }
            logger.info(f"[EVIDENCE] uzman lineart rehberi: {manual_guide_path.name}")
        else:
            metadata["input"]["completion_guide"] = None

        # ── Condition haritaları ──
        depth_original = prepare_depth(photo, cfg.depth_estimator, device, logger)
        lineart_original = prepare_lineart(photo, cfg.annotator_repo, logger, "original")

        if resolved_guide_mode == "none":
            depth_evidence = neutralize_control_inside_mask(depth_original, hard_mask)
            lineart_evidence = neutralize_control_inside_mask(lineart_original, hard_mask)
        else:
            depth_mirror, _depth_donor, depth_stats = build_mirror_evidence(
                depth_original,
                hard_mask,
                cfg.symmetry_axis_x_ratio,
                cfg.mirror_fill_radius,
            )
            depth_evidence = merge_control_inside_mask(depth_original, depth_mirror, hard_mask)

            line_source = manual_guide if manual_guide is not None else mirror_guide
            lineart_guide = prepare_lineart(
                line_source,
                cfg.annotator_repo,
                logger,
                "manual guide" if manual_guide is not None else "mirror evidence",
            )
            lineart_evidence = merge_control_inside_mask(
                lineart_original,
                lineart_guide,
                hard_mask,
            )
            metadata["evidence"]["depth_mirror_stats"] = depth_stats

        depth_original.save(run_dir / "cond_depth_original.png")
        lineart_original.save(run_dir / "cond_lineart_original.png")
        depth_evidence.save(run_dir / "cond_depth_evidence.png")
        lineart_evidence.save(run_dir / "cond_lineart_evidence.png")

        metadata["evidence"].update(
            {
                "requested_guide_mode": guide_mode,
                "resolved_guide_mode": resolved_guide_mode,
                "mirror_stats": mirror_stats,
                "evidence_init_file": "evidence_init.png",
                "mirror_guide_file": "auto_evidence_guide.png",
                "donor_map_file": "auto_evidence_donor_map.png",
                "depth_control_file": "cond_depth_evidence.png",
                "lineart_control_file": "cond_lineart_evidence.png",
                "principle": (
                    "Masked additions are initialized and geometrically guided from surviving "
                    "evidence in the same photograph; external historical motifs are not supplied."
                ),
            }
        )
        write_json(meta_path, metadata)

        pipeline, ip_active = build_pipeline(cfg, device, dtype, logger)
        metadata["resolved_ip_adapter"] = ip_active

        outputs: list[Image.Image] = []
        labels: list[str] = []
        for seed in cfg.seeds:
            logger.info(
                f"[GEN] seed={seed} strength={MODE.strength} "
                f"cn=[{MODE.depth_scale},{MODE.lineart_scale}] end={MODE.guidance_end}"
            )
            raw, final = generate_one(
                pipeline=pipeline,
                original_photo=photo,
                evidence_init=evidence_init,
                work_mask=work_mask,
                final_mask=final_mask,
                depth_control=depth_evidence,
                lineart_control=lineart_evidence,
                mode=MODE,
                seed=seed,
                cfg=cfg,
                device=device,
                ip_active=ip_active,
            )
            stamped = stamp(
                final,
                "AI ÖNGÖRÜSÜ — aynı yapıdaki izlerden türetilmiştir; restitüsyon kararı değildir",
            )
            filename = f"ozgun_hal_seed{seed}.png"
            png_meta = {
                "run_id": run_id,
                "source_script": script_info["filename"],
                "source_script_sha256": script_info["sha256"],
                "input_photo": input_path.name,
                "mode": "ozgun_hal_self_evidence",
                "seed": seed,
                "strength": MODE.strength,
                "depth_scale": MODE.depth_scale,
                "lineart_scale": MODE.lineart_scale,
                "guide_mode": resolved_guide_mode,
                "symmetry_axis_x_ratio": cfg.symmetry_axis_x_ratio,
                "mask_file": mask_path.name,
                "disclaimer": "AI research prediction; not a restitution or restoration decision",
            }
            save_png_with_meta(stamped, run_dir / filename, png_meta)

            raw_name = None
            if cfg.save_raw_debug:
                raw_name = f"debug_raw_ozgun_hal_seed{seed}.png"
                save_png_with_meta(raw, run_dir / raw_name, png_meta)

            metadata["outputs"].append(
                {
                    "mode": "ozgun_hal_self_evidence",
                    "seed": seed,
                    "file": filename,
                    "raw_debug_file": raw_name,
                    "prompt": (
                        f"{MODE.prompt}, expert supplied period: {cfg.period_hint.strip()}"
                        if cfg.period_hint.strip()
                        else MODE.prompt
                    ),
                    "negative_prompt": MODE.negative,
                    "guide_mode": resolved_guide_mode,
                }
            )
            outputs.append(stamped)
            labels.append(f"ÖZGÜN HÂL — yapı içi kanıt | seed {seed}")
            logger.info(f"[SAVE] {filename}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        contact_sheet(
            [photo, evidence_init] + outputs,
            ["ORİJİNAL — mevcut hâl", "KANIT REHBERİ — model girdisi"] + labels,
            run_dir / "set_ozgun_hal.png",
            {
                "run_id": run_id,
                "source_script": script_info["filename"],
                "mode": "ozgun_hal_self_evidence",
                "seeds": cfg.seeds,
                "guide_mode": resolved_guide_mode,
            },
        )
        logger.info("[SAVE] set_ozgun_hal.png")

        metadata["status"] = "completed"
        metadata["finished_at"] = now_iso()
        metadata["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        write_json(meta_path, metadata)

        logger.info("=" * 72)
        logger.info(f"[BİTTİ] {run_dir}")
        logger.info(f"[META] {meta_path.name}")
        logger.info("=" * 72)

    except Exception as exc:
        metadata["status"] = "failed"
        metadata["finished_at"] = now_iso()
        metadata["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        metadata["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            write_json(meta_path, metadata)
        except Exception:
            pass
        logger.error(f"[HATA] {exc}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()