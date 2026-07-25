"""
RestorationForesight_Inpaint_v3.py — MASKELİ RESTORASYON ÖNGÖRÜ ARACI
================================================================================
Tek fotoğraf -> iki ayrı öngörü seti:

  1) ÖZGÜN HÂL ÖNGÖRÜSÜ
     Kayıp/bozuk bölgeler, görünen mimari ritim ve dönem dili devam ettirilerek tamamlanır.

  2) VENEDİK TÜZÜĞÜ RESTORASYON ÖNGÖRÜSÜ
     Sağlam özgün doku korunur; yalnızca maskeli kayıp bölgelerde, uyumlu fakat
     ayırt edilebilir onarım öngörülür.

NEDEN BU SÜRÜM INPAINT KULLANIYOR?
----------------------------------
Eski img2img sürümü, bozuk fotoğrafı başlangıç görüntüsü; aynı fotoğraftan çıkarılan
lineart/depth haritalarını da güçlü geometri kontrolü olarak kullanıyordu. Bu yüzden
model, prompt "tamamla" dese bile mevcut kırıkları ve boşlukları koruyordu.

Bu sürümde:
  - Beyaz bölgeleri yeniden üretilecek bir TAMAMLAMA MASKESİ kullanılır.
  - Manuel maske yoksa basit bir MASKE ÇİZİM PENCERESİ açar; kullanıcı eksik yapı
    bölgelerini işaretleyip completion_mask.png olarak kaydeder ve aynı çalıştırma devam eder.
  - Maskeli alanlardaki bozuk lineart/depth bilgisi nötrlenir.
  - Inpaint yalnız maskeli alanda çalışır.
  - Son kompozitte maskenin dışı orijinal fotoğraftan tekrar alınır; sağlam alanlar
    piksel düzeyinde korunur.
  - Fotoğraf kareye kırpılmaz; en-boy oranı korunarak 8'in katlarına ölçeklenir.

GİRDİ DOSYALARI
---------------
  image.png                         Zorunlu ana fotoğraf.
  completion_mask.png               Tercih edilen ortak uzman maskesi.
                                      Beyaz = üret / tamamla / onar
                                      Siyah = aynen koru

  mask_ozgun_hal.png                İsteğe bağlı moda özel maske.
  mask_venedik_restorasyon.png      İsteğe bağlı moda özel maske.
  completion_guide.png              İsteğe bağlı uzman çizimi / kaba geometri rehberi.

Moda özel maske varsa ortak maskenin önüne geçer. Özgün hâl maskesi daha geniş,
Venedik maskesi daha dar tutulabilir. Gerçek kapı/kemer açıklıklarını beyaza boyama;
yalnızca gerçekten kayıp olduğu düşünülen kısımları maskele. Manuel editörde sol fare
beyaz alan ekler, sağ fare siler; kaydettiğinde üretim otomatik devam eder.

ÇIKTI KLASÖRÜ ADI — DEĞİŞTİRME:
  outputs/restore_YYYYMMDD_HHMMSS

Her çalışmada ayrıca:
  run.log
  run_meta.json
  input ve kondisyon görselleri
  seed bazlı çıktılar
  contact sheet'ler
oluşturulur. Kaynak .py adı, yolu ve SHA256 değeri metadata'ya yazılır; çalışan .py dosyası
çıktı klasörüne kopyalanmaz. Her PNG'nin
kendi içine de run_id, source_script, mode ve seed bilgisi gömülür.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import platform
import shutil
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
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
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


# Pillow sürümünden bağımsız resample sabitleri
_RESAMPLING = getattr(Image, "Resampling", Image)
LANCZOS = _RESAMPLING.LANCZOS
NEAREST = _RESAMPLING.NEAREST
BICUBIC = _RESAMPLING.BICUBIC


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ModeConfig:
    label: str
    mask_file: str
    strength: float
    depth_scale: float
    lineart_scale: float
    guidance_end: tuple[float, float]
    prompt: str
    negative: str


@dataclass
class Config:
    # Girdiler
    input_photo: str = "image.png"
    shared_mask: str = "completion_mask.png"
    completion_guide: str = "completion_guide.png"  # yoksa otomatik atlanır
    period_hint: str = ""  # örn. "late Ottoman", "Seljuk", "Byzantine"

    # Üretim
    seeds: list[int] = field(default_factory=lambda: [42, 1001, 2024])
    selected_seed_file: str = "selected_seed.json"
    max_side: int = 512
    steps: int = 40
    guidance_scale: float = 6.5

    # Maske kenarı
    mask_expand_px: int = 5
    mask_feather_px: float = 3.0

    # Otomatik hasar maskesi ÖNERİSİ. Manuel maske her zaman önceliklidir.
    # "propose_only" = öneriyi kaydet, render etmeden güvenli biçimde dur.
    # "use"          = öneriyi doğrudan iki mod için kullan (daha riskli).
    auto_mask_enabled: bool = False
    open_mask_editor_if_missing: bool = True
    mask_editor_brush_px: int = 34
    mask_editor_max_width: int = 1200
    mask_editor_max_height: int = 780
    auto_mask_behavior: str = "propose_only"  # yalnız bilinçli olarak açılırsa kullanılır
    auto_mask_file: str = "auto_mask_proposal.png"
    auto_mask_preview_file: str = "auto_mask_preview.png"
    auto_mask_dark_percentile: float = 20.0
    auto_mask_max_gray: int = 100
    auto_mask_local_contrast: int = 16
    auto_mask_min_area_ratio: float = 0.0007
    auto_mask_max_component_ratio: float = 0.32
    auto_mask_close_px: int = 5
    auto_mask_expand_px: int = 5
    auto_mask_drop_border_components: bool = True

    # Modeller
    base_model: str = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
    vae_id: str = "stabilityai/sd-vae-ft-mse"
    depth_controlnet: str = "lllyasviel/control_v11f1p_sd15_depth"
    lineart_controlnet: str = "lllyasviel/control_v11p_sd15_lineart"
    depth_estimator: str = "Intel/dpt-hybrid-midas"
    annotator_repo: str = "lllyasviel/Annotators"
    use_safetensors: bool = False

    # İsteğe bağlı IP-Adapter. Diffusers sürümü desteklemiyorsa otomatik atlanır;
    # ortamı yükseltmez ve çalışan pipeline'ı bozmaz.
    use_ip_adapter: bool = False
    ip_adapter_scale: float = 0.25
    ip_adapter_repo: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "models"
    ip_adapter_weight: str = "ip-adapter_sd15.bin"
    ip_cache_dir: str = r"E:\ImageToRender\models"

    # Bellek / kayıt
    enable_xformers: bool = True
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = True
    enable_cpu_offload: bool = True
    save_debug: bool = False
    save_source_snapshot: bool = False  # True yapılırsa source_<dosya>.py kopyası alınır
    save_pip_freeze: bool = False

    output_root: str = "outputs"


COMMON_NEGATIVE = (
    "redesigned building, altered architecture, extra floors, extra windows, duplicate arches, "
    "modern facade, concrete frame, glass curtain wall, steel structure, fantasy architecture, "
    "demolished and rebuilt, changed camera angle, changed viewpoint, changed background, "
    "warped columns, melted masonry, deformed arches, floating stones, low quality, blurry, "
    "cartoon, illustration, painting, text, watermark"
)

MODES: dict[str, ModeConfig] = {
    "ozgun_hal": ModeConfig(
        label="ÖZGÜN HÂL ÖNGÖRÜSÜ",
        mask_file="mask_ozgun_hal.png",
        # Inpaint yüksek strength: özgürlük yalnızca maskeli alanda olduğu için sağlam doku korunur.
        strength=0.96,
        depth_scale=0.24,
        lineart_scale=0.34,
        guidance_end=(0.42, 0.56),
        prompt=(
            "the exact same historical building and the exact same camera view, reconstruct only "
            "the masked missing architectural portions, continue the surviving architectural rhythm, "
            "symmetry, arches, columns, cornices and masonry courses visible in the photograph, "
            "plausible original construction state, intact period-authentic stonework, original lime mortar, "
            "historically coherent materials and colors, photorealistic conservation research visualization, "
            "natural daylight, precise architectural geometry, no redesign"
        ),
        negative=COMMON_NEGATIVE + ", ruin, collapsed wall, missing masonry, open fracture, unfinished repair",
    ),
    "venedik_restorasyon": ModeConfig(
        label="VENEDİK TÜZÜĞÜ RESTORASYON ÖNGÖRÜSÜ",
        mask_file="mask_venedik_restorasyon.png",
        strength=0.90,
        depth_scale=0.32,
        lineart_scale=0.48,
        guidance_end=(0.55, 0.72),
        prompt=(
            "the exact same historical building and the exact same camera view, minimal conservation repair "
            "only inside the masked missing portions, preserve all surviving original fabric and weathered patina, "
            "repair geometry derived from surviving evidence, compatible stone and lime mortar, additions harmonious "
            "but subtly distinguishable by a restrained difference in tone and tooling, no conjectural redesign, "
            "not brand new, not over-restored, photorealistic architectural conservation visualization"
        ),
        negative=COMMON_NEGATIVE + ", pristine new building, polished facade, bright white facade, over-restored",
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# RUN / LOG / METADATA YARDIMCILARI — diğer projelere de taşınabilir yapı
# ══════════════════════════════════════════════════════════════════════════════
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def collect_environment(device: str, dtype: torch.dtype) -> dict[str, Any]:
    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA available"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_name,
        "device": device,
        "dtype": str(dtype),
        "packages": {
            "diffusers": package_version("diffusers"),
            "transformers": package_version("transformers"),
            "controlnet-aux": package_version("controlnet-aux"),
            "Pillow": package_version("Pillow"),
            "numpy": package_version("numpy"),
        },
    }


def make_run_dir(cfg: Config) -> tuple[str, Path]:
    # KULLANICININ SABİT İSTEDİĞİ YAPI: restore_YYYYMMDD_HHMMSS
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def setup_logger(run_dir: Path, run_id: str) -> logging.Logger:
    logger = logging.getLogger(f"restoration_foresight.{run_id}")
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


def collect_source_info(run_dir: Path, save_snapshot: bool = False) -> dict[str, Any]:
    """Çalışan Python dosyasını metadata'da izler; varsayılan olarak klasöre kopyalamaz."""
    source_path = Path(__file__).resolve()
    snapshot_name = None
    if save_snapshot:
        snapshot_name = f"source_{source_path.name}"
        shutil.copy2(source_path, run_dir / snapshot_name)
    return {
        "filename": source_path.name,
        "absolute_path": str(source_path),
        "sha256": sha256_file(source_path),
        "snapshot_file": snapshot_name,
    }


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
    """selected_seed.json içinde {"seed":42} veya {"seeds":[42,1001]} kabul eder."""
    p = Path(path)
    if not p.exists():
        return list(default)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data.get("seeds"), list):
            seeds = [int(x) for x in data["seeds"]]
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
# GÖRSEL / MASKE / CONDITION
# ══════════════════════════════════════════════════════════════════════════════
def rounded_multiple(value: float, multiple: int = 8, minimum: int = 64) -> int:
    return max(minimum, int(round(value / multiple) * multiple))


def load_photo_preserve_ratio(path: Path, max_side: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = max_side / max(w, h)
    nw = rounded_multiple(w * scale)
    nh = rounded_multiple(h * scale)
    return im.resize((nw, nh), LANCZOS)


def load_mask(path: Path, size: tuple[int, int]) -> Image.Image:
    mask = Image.open(path).convert("L").resize(size, NEAREST)
    # Net siyah/beyaz maske: gri bölgeler yanlışlıkla yarım üretim yapmasın.
    arr = np.array(mask)
    arr = np.where(arr >= 128, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, "L")


def expand_and_feather_mask(mask: Image.Image, expand_px: int, feather_px: float) -> Image.Image:
    out = mask.copy()
    for _ in range(max(0, int(expand_px))):
        out = out.filter(ImageFilter.MaxFilter(3))
    if feather_px > 0:
        out = out.filter(ImageFilter.GaussianBlur(float(feather_px)))
    return out


def mask_area_ratio(mask: Image.Image) -> float:
    return float((np.array(mask.convert("L")) > 127).mean())


def manual_mask_path(mode: ModeConfig, cfg: Config) -> Path | None:
    """Moda özel maske > ortak maske. Yoksa None."""
    mode_path = Path(mode.mask_file)
    shared_path = Path(cfg.shared_mask)
    if mode_path.exists():
        return mode_path
    if shared_path.exists():
        return shared_path
    return None


def create_mask_with_editor(
    photo_path: Path,
    output_path: Path,
    brush_px: int = 34,
    max_width: int = 1200,
    max_height: int = 780,
) -> bool:
    """
    Manuel maske yoksa basit bir çizim penceresi açar.

    Sol fare   : beyaza boya (tamamlanacak alan)
    Sağ fare   : sil / siyaha boya (korunacak alan)
    Fare tekeri: fırça boyutunu değiştir

    "Maskeyi Kaydet ve Devam Et" seçilirse completion_mask.png oluşturulur ve True döner.
    Pencere kapatılır veya "İptal" seçilirse hiçbir dosya yazılmaz ve False döner.
    """
    try:
        import tkinter as tk
        from tkinter import messagebox
        from PIL import ImageTk
    except Exception as exc:
        raise RuntimeError(
            "Manuel maske editörü açılamadı. Tkinter/PIL.ImageTk gerekli. "
            f"Ayrıntı: {exc}"
        ) from exc

    original = Image.open(photo_path).convert("RGB")
    ow, oh = original.size
    scale = min(max_width / ow, max_height / oh, 1.0)
    dw = max(1, int(round(ow * scale)))
    dh = max(1, int(round(oh * scale)))
    display_photo = original.resize((dw, dh), LANCZOS)
    display_mask = Image.new("L", (dw, dh), 0)

    root = tk.Tk()
    root.title("Restorasyon Tamamlama Maskesi")
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
            "Kırmızı gösterilen yerler model tarafından tamamlanır. "
            "Sol fareyle boya, sağ fareyle sil. Gerçek kapı ve kemer boşluklarını boyama."
        ),
        fg="white",
        bg="#202020",
        font=("Segoe UI", 10),
        wraplength=max(520, dw),
        justify="left",
    )
    title.pack(fill="x", padx=10, pady=(10, 6))

    canvas = tk.Canvas(root, width=dw, height=dh, highlightthickness=0, bg="black")
    canvas.pack(padx=10, pady=4)

    status_var = tk.StringVar()
    status = tk.Label(root, textvariable=status_var, fg="white", bg="#202020")
    status.pack(fill="x", padx=10, pady=(2, 4))

    tk_image_holder = {"image": None}

    def update_status() -> None:
        original_brush = max(1, int(round(state["brush"] / max(scale, 1e-6))))
        status_var.set(
            f"Fırça: yaklaşık {original_brush}px | Sol: ekle | Sağ: sil | Tekerlek: boyut"
        )

    def redraw() -> None:
        preview = display_photo.convert("RGBA")
        red = Image.new("RGBA", (dw, dh), (255, 36, 36, 0))
        alpha = display_mask.point(lambda p: 150 if p > 0 else 0)
        red.putalpha(alpha)
        preview = Image.alpha_composite(preview, red).convert("RGB")
        tk_img = ImageTk.PhotoImage(preview)
        tk_image_holder["image"] = tk_img
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=tk_img)

    def paint_at(x: int, y: int, value: int) -> None:
        r = state["brush"] / 2
        draw = ImageDraw.Draw(display_mask)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=value)
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
        display_mask.paste(0, (0, 0, dw, dh))
        redraw()

    def save_and_continue() -> None:
        arr = np.array(display_mask)
        if not np.any(arr > 0):
            messagebox.showwarning(
                "Maske boş",
                "Hiç alan boyanmadı. Tamamlanacak yapı bölgelerini sol fareyle işaretle.",
            )
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_mask = display_mask.resize((ow, oh), NEAREST)
        final_arr = np.array(final_mask)
        final_mask = Image.fromarray(
            np.where(final_arr >= 128, 255, 0).astype(np.uint8), "L"
        )
        final_mask.save(output_path)
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


def auto_mask_preview(photo: Image.Image, mask: Image.Image) -> Image.Image:
    """Maske önerisini fotoğraf üstünde yarı saydam kırmızı gösterir."""
    rgb = np.array(photo.convert("RGB"), dtype=np.float32)
    m = np.array(mask.resize(photo.size, NEAREST).convert("L")) > 127
    out = rgb.copy()
    tint = np.array([255.0, 32.0, 32.0], dtype=np.float32)
    out[m] = out[m] * 0.42 + tint * 0.58
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def generate_auto_mask_proposal(
    photo: Image.Image, cfg: Config, logger: logging.Logger
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """
    Karanlık, çevresinden belirgin biçimde daha derin görünen iç bölgeleri hasar adayı sayar.

    Bu yalnızca ÖNERİDİR. Gerçek kapı/pencere/kemer açıklıkları da karanlık olduğu için
    otomatik olarak ayırt edilemez. Bu nedenle varsayılan davranış propose_only'dir.
    """
    arr = np.array(photo.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    gray = np.array(photo.convert("L"), dtype=np.uint8)

    percentile_threshold = float(np.percentile(gray, cfg.auto_mask_dark_percentile))
    dark_threshold = int(min(float(cfg.auto_mask_max_gray), percentile_threshold))
    diagnostics: dict[str, Any] = {
        "method": "dark_local_contrast_connected_components",
        "dark_percentile": cfg.auto_mask_dark_percentile,
        "percentile_threshold": round(percentile_threshold, 3),
        "resolved_dark_threshold": dark_threshold,
        "local_contrast_threshold": cfg.auto_mask_local_contrast,
        "opencv": False,
    }

    try:
        import cv2

        diagnostics["opencv"] = True
        sigma = max(3.0, min(h, w) / 65.0)
        local = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        contrast = local.astype(np.int16) - gray.astype(np.int16)

        dark = gray <= dark_threshold
        very_dark = gray <= max(24, min(52, dark_threshold - 12))
        candidate = (dark & (contrast >= cfg.auto_mask_local_contrast)) | very_dark
        binary = candidate.astype(np.uint8) * 255

        if cfg.auto_mask_close_px > 0:
            k = cfg.auto_mask_close_px * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        n, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
        min_area = max(40, int(h * w * cfg.auto_mask_min_area_ratio))
        max_area = max(min_area + 1, int(h * w * cfg.auto_mask_max_component_ratio))
        kept = np.zeros((h, w), dtype=np.uint8)
        kept_components = []
        rejected_border = 0
        for i in range(1, n):
            x, y, cw, ch, area = [int(v) for v in stats[i]]
            touches_border = x <= 0 or y <= 0 or x + cw >= w or y + ch >= h
            if area < min_area or area > max_area:
                continue
            if cfg.auto_mask_drop_border_components and touches_border:
                rejected_border += 1
                continue
            kept[labels == i] = 255
            kept_components.append({"x": x, "y": y, "w": cw, "h": ch, "area": area})

        if cfg.auto_mask_expand_px > 0:
            k = cfg.auto_mask_expand_px * 2 + 1
            kept = cv2.dilate(
                kept, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1
            )
        proposal = Image.fromarray(kept, "L")
        diagnostics.update(
            {
                "sigma": sigma,
                "min_component_area_px": min_area,
                "max_component_area_px": max_area,
                "kept_component_count": len(kept_components),
                "rejected_border_components": rejected_border,
                "components": kept_components,
            }
        )
    except Exception as exc:
        logger.warning(f"[AUTO MASK] OpenCV yolu kullanılamadı ({exc}); PIL fallback")
        local_img = Image.fromarray(gray, "L").filter(
            ImageFilter.GaussianBlur(max(3.0, min(h, w) / 65.0))
        )
        local = np.array(local_img, dtype=np.int16)
        contrast = local - gray.astype(np.int16)
        candidate = (gray <= dark_threshold) & (contrast >= cfg.auto_mask_local_contrast)
        proposal = Image.fromarray(candidate.astype(np.uint8) * 255, "L")
        for _ in range(max(0, cfg.auto_mask_close_px)):
            proposal = proposal.filter(ImageFilter.MaxFilter(3))
        for _ in range(max(0, cfg.auto_mask_close_px // 2)):
            proposal = proposal.filter(ImageFilter.MinFilter(3))
        for _ in range(max(0, cfg.auto_mask_expand_px)):
            proposal = proposal.filter(ImageFilter.MaxFilter(3))
        diagnostics["fallback_error"] = str(exc)

    # Kesin ikili maske
    mask_arr = np.array(proposal.convert("L"))
    proposal = Image.fromarray(np.where(mask_arr >= 128, 255, 0).astype(np.uint8), "L")
    area = mask_area_ratio(proposal)
    diagnostics["area_ratio"] = area
    if area < 0.0005:
        logger.warning("[AUTO MASK] Öneri neredeyse boş; eşikler bu fotoğrafa uymamış olabilir")
    if area > 0.55:
        logger.warning("[AUTO MASK] Öneri çok geniş; doğrudan kullanma, mutlaka elle düzelt")
    preview = auto_mask_preview(photo, proposal)
    return proposal, preview, diagnostics


def resolve_mode_mask(
    mode: ModeConfig, cfg: Config, size: tuple[int, int], auto_mask_path: Path | None = None
) -> tuple[Image.Image, Path, str]:
    mode_path = Path(mode.mask_file)
    shared_path = Path(cfg.shared_mask)
    if mode_path.exists():
        return load_mask(mode_path, size), mode_path, "mode_specific"
    if shared_path.exists():
        return load_mask(shared_path, size), shared_path, "shared_fallback"
    if auto_mask_path is not None and auto_mask_path.exists():
        return load_mask(auto_mask_path, size), auto_mask_path, "auto_proposal"
    raise FileNotFoundError(
        f"Tamamlama maskesi bulunamadı. '{mode.mask_file}' veya '{cfg.shared_mask}' gerekli. "
        "Beyaz=üretilecek bölge, siyah=korunacak bölge."
    )


def prepare_depth(photo: Image.Image, model_id: str, device: str, logger: logging.Logger) -> Image.Image:
    logger.info(f"[COND] depth modeli: {model_id}")
    model = DPTForDepthEstimation.from_pretrained(model_id)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = model.to(device)
    inputs = processor(images=photo, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        pred = model(**inputs).predicted_depth
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=photo.size[::-1], mode="bicubic", align_corners=False
    )
    arr = pred.squeeze().detach().float().cpu().numpy()
    arr -= arr.min()
    denom = float(arr.max()) or 1.0
    arr = (arr * 255.0 / denom).clip(0, 255).astype(np.uint8)
    depth = Image.fromarray(arr, "L").convert("RGB")

    del model, processor, inputs, pred
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return depth


def prepare_lineart(source: Image.Image, annotator_repo: str, logger: logging.Logger) -> Image.Image:
    logger.info(f"[COND] lineart annotator: {annotator_repo}")
    detector = LineartDetector.from_pretrained(annotator_repo)
    line = detector(source)
    if line.mode != "RGB":
        line = line.convert("RGB")
    del detector
    gc.collect()
    return line.resize(source.size, LANCZOS)


def neutralize_control_inside_mask(control: Image.Image, mask: Image.Image) -> Image.Image:
    """Maskeli bölgedeki bozuk geometri sinyalini dış bölgenin medyan rengiyle nötrler."""
    arr = np.array(control.convert("RGB")).copy()
    m = np.array(mask.resize(control.size, NEAREST).convert("L")) > 127
    outside = arr[~m]
    if outside.size == 0:
        fill = np.array([127, 127, 127], dtype=np.uint8)
    else:
        fill = np.median(outside, axis=0).astype(np.uint8)
    arr[m] = fill
    return Image.fromarray(arr, "RGB")


def find_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        if Path(name).exists():
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                pass
    return ImageFont.load_default()


def stamp(img: Image.Image, text: str) -> Image.Image:
    im = img.copy().convert("RGB")
    draw = ImageDraw.Draw(im)
    font = find_font(max(13, round(im.width / 42)))
    bbox = draw.textbbox((0, 0), text, font=font)
    bar_h = max(30, bbox[3] - bbox[1] + 12)
    draw.rectangle([0, im.height - bar_h, im.width, im.height], fill=(0, 0, 0))
    draw.text((8, im.height - bar_h + 5), text, fill=(255, 255, 255), font=font)
    return im


def save_png_with_meta(img: Image.Image, path: Path, metadata: dict[str, Any]) -> None:
    pnginfo = PngInfo()
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, ensure_ascii=False)
        pnginfo.add_text(str(key), str(value))
    img.save(path, pnginfo=pnginfo)


def contact_sheet(imgs: list[Image.Image], labels: list[str], out: Path, png_meta: dict[str, Any]) -> None:
    if not imgs:
        return
    w, h = imgs[0].size
    label_h = 58
    sheet = Image.new("RGB", (w * len(imgs), h + label_h), (246, 246, 246))
    draw = ImageDraw.Draw(sheet)
    font = find_font(max(13, round(w / 38)))
    for i, (img, label) in enumerate(zip(imgs, labels)):
        x = i * w
        sheet.paste(img.resize((w, h), LANCZOS), (x, 0))
        draw.text((x + 8, h + 8), label, fill=(0, 0, 0), font=font)
    save_png_with_meta(sheet, out, png_meta)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def build_pipeline(cfg: Config, device: str, dtype: torch.dtype, logger: logging.Logger):
    logger.info("[SETUP] 2 ControlNet: depth + lineart")
    controls = [
        ControlNetModel.from_pretrained(cfg.depth_controlnet, torch_dtype=dtype),
        ControlNetModel.from_pretrained(cfg.lineart_controlnet, torch_dtype=dtype),
    ]
    vae = AutoencoderKL.from_pretrained(cfg.vae_id, torch_dtype=dtype)
    logger.info(f"[VAE] {cfg.vae_id}")

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        cfg.base_model,
        controlnet=controls,
        vae=vae,
        torch_dtype=dtype,
        safety_checker=None,
        use_safetensors=cfg.use_safetensors,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )

    if cfg.enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("[VRAM] xformers aktif")
        except Exception as exc:
            logger.warning(f"[VRAM] xformers açılamadı ({exc}); attention slicing")
            pipe.enable_attention_slicing("max")
    else:
        pipe.enable_attention_slicing("max")

    if cfg.enable_vae_slicing:
        pipe.enable_vae_slicing()
    if cfg.enable_vae_tiling:
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass

    ip_active = False
    if cfg.use_ip_adapter:
        if not hasattr(pipe, "load_ip_adapter"):
            logger.warning("[IP] Bu diffusers sürümünde load_ip_adapter yok; yükseltme yapılmadan atlandı")
        else:
            try:
                pipe.load_ip_adapter(
                    cfg.ip_adapter_repo,
                    subfolder=cfg.ip_adapter_subfolder,
                    weight_name=cfg.ip_adapter_weight,
                    cache_dir=cfg.ip_cache_dir,
                )
                pipe.set_ip_adapter_scale(cfg.ip_adapter_scale)
                ip_active = True
                logger.info(f"[IP] aktif scale={cfg.ip_adapter_scale}")
            except Exception as exc:
                logger.warning(f"[IP] yüklenemedi ({exc}); atlandı")

    if device == "cuda" and cfg.enable_cpu_offload:
        logger.info("[VRAM] model CPU offload aktif")
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    return pipe, ip_active


def execution_device(pipe, fallback: str) -> str:
    try:
        return str(pipe._execution_device)
    except Exception:
        try:
            return str(pipe.device)
        except Exception:
            return fallback


def generate_one(
    pipe,
    photo: Image.Image,
    mask: Image.Image,
    depth: Image.Image,
    lineart: Image.Image,
    mode: ModeConfig,
    seed: int,
    cfg: Config,
    device: str,
    ip_active: bool,
) -> tuple[Image.Image, Image.Image]:
    prompt = mode.prompt
    if cfg.period_hint.strip():
        prompt = f"{prompt}, expert-supplied period: {cfg.period_hint.strip()}"

    gen_device = execution_device(pipe, device)
    generator = torch.Generator(device=gen_device).manual_seed(seed)

    kwargs = dict(
        prompt=prompt,
        negative_prompt=mode.negative,
        image=photo,
        mask_image=mask,
        control_image=[depth, lineart],
        controlnet_conditioning_scale=[mode.depth_scale, mode.lineart_scale],
        control_guidance_end=list(mode.guidance_end),
        strength=mode.strength,
        num_inference_steps=cfg.steps,
        guidance_scale=cfg.guidance_scale,
        height=photo.height,
        width=photo.width,
        generator=generator,
    )
    if ip_active:
        kwargs["ip_adapter_image"] = photo

    raw = pipe(**kwargs).images[0].convert("RGB")

    # Kesin koruma: maske dışındaki pikseli modelden değil, doğrudan orijinal fotoğraftan al.
    final = Image.composite(raw, photo, mask.convert("L"))
    return raw, final


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    cfg = Config()

    # PREFLIGHT: eksik zorunlu girdiler için boş output klasörü oluşturma.
    input_preflight = Path(cfg.input_photo)
    if not input_preflight.exists():
        raise FileNotFoundError(f"Ana fotoğraf bulunamadı: {input_preflight}")
    missing_preflight = [
        mode_id for mode_id, mode in MODES.items() if manual_mask_path(mode, cfg) is None
    ]
    if missing_preflight and not cfg.auto_mask_enabled:
        if cfg.open_mask_editor_if_missing:
            print(f"[MASK] '{cfg.shared_mask}' bulunamadı. Manuel maske editörü açılıyor...")
            saved = create_mask_with_editor(
                photo_path=input_preflight,
                output_path=Path(cfg.shared_mask),
                brush_px=cfg.mask_editor_brush_px,
                max_width=cfg.mask_editor_max_width,
                max_height=cfg.mask_editor_max_height,
            )
            if not saved:
                print("[DURDU] Maske kaydedilmedi; render başlatılmadı.")
                return
            print(f"[MASK] Kaydedildi: {cfg.shared_mask}")
        else:
            raise FileNotFoundError(
                f"Tamamlama maskesi yok ({', '.join(missing_preflight)}). "
                f"Proje köküne '{cfg.shared_mask}' koy: beyaz=tamamlanacak yapı, siyah=korunacak alan."
            )

    run_id, run_dir = make_run_dir(cfg)
    logger = setup_logger(run_dir, run_id)
    meta_path = run_dir / "run_meta.json"
    started_perf = time.perf_counter()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    metadata: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "started_at": now_iso(),
        "source_script": None,
        "input": {},
        "environment": collect_environment(device, dtype),
        "config": asdict(cfg),
        "modes": {key: asdict(value) for key, value in MODES.items()},
        "outputs": [],
        "log_file": "run.log",
    }

    try:
        source_info = collect_source_info(run_dir, cfg.save_source_snapshot)
        metadata["source_script"] = source_info
        write_json(meta_path, metadata)

        logger.info(f"{'=' * 72}")
        logger.info(f"[RUN] {run_id}")
        logger.info(f"[SOURCE] {source_info['filename']} | sha256={source_info['sha256'][:12]}...")
        logger.info(f"[DEVICE] {device} | {metadata['environment'].get('gpu')}")
        logger.info(f"{'=' * 72}")

        if cfg.save_pip_freeze:
            metadata["pip_freeze_file"] = save_pip_freeze(run_dir, logger)

        input_path = Path(cfg.input_photo)
        if not input_path.exists():
            raise FileNotFoundError(f"Ana fotoğraf bulunamadı: {input_path}")

        photo = load_photo_preserve_ratio(input_path, cfg.max_side)
        input_copy = run_dir / "input.png"
        photo.save(input_copy)
        metadata["input"]["photo"] = {
            "original_path": str(input_path.resolve()),
            "original_filename": input_path.name,
            "sha256": sha256_file(input_path),
            "run_copy": input_copy.name,
            "processed_size": list(photo.size),
        }
        logger.info(f"[INPUT] {input_path} -> {photo.size[0]}x{photo.size[1]} (oran korundu, crop yok)")

        # Restorasyon maskesi yapısal bir uzman kararıdır. Karanlık alan tespiti bunu güvenilir
        # biçimde çözemez; bu nedenle varsayılan akış manuel maskeyi zorunlu tutar.
        missing_manual_modes = [
            mode_id for mode_id, mode in MODES.items() if manual_mask_path(mode, cfg) is None
        ]
        auto_mask_path: Path | None = None
        if missing_manual_modes and not cfg.auto_mask_enabled:
            missing_names = ", ".join(missing_manual_modes)
            raise FileNotFoundError(
                f"Tamamlama maskesi yok ({missing_names}). '{cfg.shared_mask}' oluştur: "
                "beyaz=gerçekten eksik/tamamlanacak yapı, siyah=aynen korunacak alan. "
                "Otomatik karanlık-alan maskesi varsayılan olarak kapalıdır."
            )

        if missing_manual_modes:
            logger.warning(
                "[AUTO MASK] Deneysel otomatik maske bilinçli olarak açılmış; sonuç yalnız öneridir: "
                + ", ".join(missing_manual_modes)
            )
            proposal, preview, auto_diag = generate_auto_mask_proposal(photo, cfg, logger)
            auto_mask_path = run_dir / cfg.auto_mask_file
            auto_preview_path = run_dir / cfg.auto_mask_preview_file
            proposal.save(auto_mask_path)
            preview.save(auto_preview_path)
            metadata["input"]["auto_mask"] = {
                "behavior": cfg.auto_mask_behavior,
                "missing_manual_modes": missing_manual_modes,
                "proposal_file": auto_mask_path.name,
                "proposal_sha256": sha256_file(auto_mask_path),
                "preview_file": auto_preview_path.name,
                "diagnostics": auto_diag,
            }
            if cfg.auto_mask_behavior == "propose_only":
                metadata["status"] = "needs_mask_review"
                metadata["finished_at"] = now_iso()
                metadata["elapsed_seconds"] = round(time.perf_counter() - started_perf, 2)
                metadata["next_action"] = (
                    f"Öneriyi elle düzeltip proje köküne '{cfg.shared_mask}' adıyla kaydet."
                )
                write_json(meta_path, metadata)
                logger.warning("[DURDU] Deneysel maske yalnız öneri olarak üretildi.")
                return
            if cfg.auto_mask_behavior != "use":
                raise ValueError("auto_mask_behavior yalnız 'propose_only' veya 'use' olabilir")
        else:
            metadata["input"]["auto_mask"] = None

        cfg.seeds = load_selected_seeds(cfg.seeds, cfg.selected_seed_file, logger)
        metadata["resolved_seeds"] = cfg.seeds

        # Ortak condition haritaları
        depth_original = prepare_depth(photo, cfg.depth_estimator, device, logger)

        guide_path = Path(cfg.completion_guide)
        guide_exists = guide_path.exists()
        if guide_exists:
            guide = Image.open(guide_path).convert("RGB").resize(photo.size, LANCZOS)
            guide.save(run_dir / "completion_guide.png")
            line_source = guide
            metadata["input"]["completion_guide"] = {
                "path": str(guide_path.resolve()),
                "filename": guide_path.name,
                "sha256": sha256_file(guide_path),
            }
            logger.info(f"[GUIDE] uzman geometri rehberi kullanılıyor: {guide_path}")
        else:
            line_source = photo
            metadata["input"]["completion_guide"] = None
            logger.info("[GUIDE] completion_guide.png yok; lineart fotoğraftan çıkarılacak")

        line_original = prepare_lineart(line_source, cfg.annotator_repo, logger)
        depth_original.save(run_dir / "cond_depth_original.png")
        line_original.save(run_dir / "cond_lineart_original.png")

        # Her modun maskesi ve nötrlenmiş kontrol haritası farklı olabilir.
        mode_assets: dict[str, dict[str, Any]] = {}
        for mode_id, mode in MODES.items():
            hard_mask, mask_path, mask_source = resolve_mode_mask(mode, cfg, photo.size, auto_mask_path)
            area = mask_area_ratio(hard_mask)
            if area < 0.001:
                raise ValueError(f"{mode_id} maskesi neredeyse boş: %{area * 100:.3f}")
            if area > 0.85:
                logger.warning(f"[MASK] {mode_id} çok geniş: %{area * 100:.1f}; sonuç daha varsayımsal olur")

            work_mask = expand_and_feather_mask(
                hard_mask, cfg.mask_expand_px, cfg.mask_feather_px
            )
            # Depth her durumda nötrlenir. Lineart, uzman guide varsa maskede de değerlidir;
            # guide yoksa bozuk fotoğraftaki kırık sınırlarını taşımaması için nötrlenir.
            depth_mode = neutralize_control_inside_mask(depth_original, hard_mask)
            line_mode = (
                line_original.copy()
                if guide_exists
                else neutralize_control_inside_mask(line_original, hard_mask)
            )

            hard_name = f"mask_{mode_id}_hard.png"
            work_name = f"mask_{mode_id}_work.png"
            depth_name = f"cond_depth_{mode_id}.png"
            line_name = f"cond_lineart_{mode_id}.png"
            hard_mask.save(run_dir / hard_name)
            work_mask.save(run_dir / work_name)
            depth_mode.save(run_dir / depth_name)
            line_mode.save(run_dir / line_name)

            metadata["input"][f"mask_{mode_id}"] = {
                "selected_path": str(mask_path.resolve()),
                "filename": mask_path.name,
                "sha256": sha256_file(mask_path),
                "selection": mask_source,
                "area_ratio": area,
                "hard_copy": hard_name,
                "work_copy": work_name,
            }
            mode_assets[mode_id] = {
                "hard_mask": hard_mask,
                "work_mask": work_mask,
                "depth": depth_mode,
                "lineart": line_mode,
            }
            logger.info(
                f"[MASK] {mode_id}: {mask_path.name} ({mask_source}), alan=%{area * 100:.1f}"
            )

        write_json(meta_path, metadata)
        pipe, ip_active = build_pipeline(cfg, device, dtype, logger)
        metadata["resolved_ip_adapter"] = ip_active

        for mode_id, mode in MODES.items():
            logger.info(f"\n{'=' * 72}\n[MODE] {mode.label}\n{'=' * 72}")
            outputs: list[Image.Image] = []
            labels: list[str] = []
            assets = mode_assets[mode_id]

            for seed in cfg.seeds:
                logger.info(
                    f"[GEN] mode={mode_id} seed={seed} strength={mode.strength} "
                    f"cn=[{mode.depth_scale},{mode.lineart_scale}] end={mode.guidance_end}"
                )
                raw, final = generate_one(
                    pipe=pipe,
                    photo=photo,
                    mask=assets["work_mask"],
                    depth=assets["depth"],
                    lineart=assets["lineart"],
                    mode=mode,
                    seed=seed,
                    cfg=cfg,
                    device=device,
                    ip_active=ip_active,
                )

                stamped = stamp(
                    final,
                    "AI ÖNGÖRÜSÜ — restorasyon kararı değildir / uzman değerlendirmesi gerekir",
                )
                filename = f"{mode_id}_seed{seed}.png"
                output_path = run_dir / filename
                png_meta = {
                    "run_id": run_id,
                    "source_script": source_info["filename"],
                    "source_script_sha256": source_info["sha256"],
                    "input_photo": input_path.name,
                    "mode": mode_id,
                    "mode_label": mode.label,
                    "seed": seed,
                    "strength": mode.strength,
                    "depth_scale": mode.depth_scale,
                    "lineart_scale": mode.lineart_scale,
                    "control_guidance_end": mode.guidance_end,
                    "mask_file": metadata["input"][f"mask_{mode_id}"]["filename"],
                    "disclaimer": "AI prediction; not a restoration decision",
                }
                save_png_with_meta(stamped, output_path, png_meta)

                if cfg.save_debug:
                    raw_name = f"debug_raw_{mode_id}_seed{seed}.png"
                    save_png_with_meta(raw, run_dir / raw_name, png_meta)
                else:
                    raw_name = None

                metadata["outputs"].append(
                    {
                        "mode": mode_id,
                        "seed": seed,
                        "file": filename,
                        "raw_debug_file": raw_name,
                        "prompt": (
                            f"{mode.prompt}, expert-supplied period: {cfg.period_hint.strip()}"
                            if cfg.period_hint.strip()
                            else mode.prompt
                        ),
                        "negative_prompt": mode.negative,
                        "mask_file": metadata["input"][f"mask_{mode_id}"]["filename"],
                    }
                )
                outputs.append(stamped)
                labels.append(f"{mode.label} | seed {seed}")
                logger.info(f"[SAVE] {filename}")

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            sheet_name = f"set_{mode_id}.png"
            contact_sheet(
                [photo] + outputs,
                ["ORİJİNAL — mevcut hâl"] + labels,
                run_dir / sheet_name,
                {
                    "run_id": run_id,
                    "source_script": source_info["filename"],
                    "mode": mode_id,
                    "seeds": cfg.seeds,
                },
            )
            logger.info(f"[SAVE] {sheet_name}")

        metadata["status"] = "completed"
        metadata["finished_at"] = now_iso()
        metadata["elapsed_seconds"] = round(time.perf_counter() - started_perf, 2)
        write_json(meta_path, metadata)

        logger.info(f"\n{'=' * 72}")
        logger.info(f"[BİTTİ] {run_dir}")
        logger.info(f"[META] {meta_path.name}")
        if source_info.get("snapshot_file"):
            logger.info(f"[SOURCE SNAPSHOT] {source_info['snapshot_file']}")
        logger.info(f"{'=' * 72}")

    except Exception as exc:
        metadata["status"] = "failed"
        metadata["finished_at"] = now_iso()
        metadata["elapsed_seconds"] = round(time.perf_counter() - started_perf, 2)
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