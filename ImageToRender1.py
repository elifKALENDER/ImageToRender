"""
ImageToRender_v2.py — RENOVASYON ÖNERİ ARACI (tek fotoğraf → tasarım alternatifleri)
====================================================================================
İş akışı: fotoğraf gir → PROGRAM + SEVIYE seç → çalıştır → comparison_sheet'ten beğendiğini seç.

YENİLİKLER (v1'e göre):
  1) MULTI-CONTROLNET: depth (kütle) + LINEART (pencere/kemer/cephe açıklık düzeni).
     Gotik pencerelerin kaybolma sebebi tek-depth'ti; lineart açıklıkları yerinde tutar.
  2) img2img: fotoğraf artık BAŞLANGIÇ görüntüsü. strength = RENOVASYON ŞİDDETİ KADRANI:
       hafif 0.35 → cephe temizliği/boya, doku tazeleme (orijinal en sadık)
       orta  0.55 → yeni malzeme/pencere müdahalesi
       derin 0.75 → köklü dönüşüm (kütle korunur, dil değişir)
  3) PROGRAM PRESET'leri: bakım evi / kreş / butik otel — prompt+negatif paketleri.
"""
# ─── TAŞINABİLİRLİK: HF ortam ayarlarını kodda sabitle (makineden bağımsız çalışsın) ───
# İMPORTLARDAN ÖNCE olmalı. HF_HUB_ENABLE_HF_TRANSFER sistemde açık olsa bile burada kapatılır
# → 'hf_transfer paketi yok' hatası hiçbir bilgisayarda çıkmaz, davranış her yerde aynı.
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw
from diffusers import (ControlNetModel, StableDiffusionControlNetImg2ImgPipeline,
                       DPMSolverMultistepScheduler, AutoencoderKL)
from controlnet_aux.lineart import LineartDetector
from transformers import DPTForDepthEstimation, AutoImageProcessor

# ═══════════ KULLANICI SEÇİMLERİ ═══════════
INPUT_PHOTO = "image.jpg"
PROGRAM = "bakim_evi"        # "bakim_evi" | "kres" | "butik_otel"
SEVIYE  = "orta"             # "hafif" | "orta" | "derin"
SEEDS   = [42, 1001, 2024, 31415]
SIZE    = 512

RENOVATION_LEVELS = {"hafif": 0.45, "orta": 0.65, "derin": 0.80}

PROGRAM_PRESETS = {
    "bakim_evi": {
        "prompt": ("photorealistic architectural renovation, historical brick building converted to "
                   "modern luxury nursing home, preserved original facade and gothic windows, "
                   "restored brick masonry, serene garden with walking paths and benches, "
                   "elderly-friendly landscape, warm daylight, 8k, sharp details"),
        "negative": ("new modern building, white plaster facade, demolished and rebuilt, "
                     "rectangular windows replacing arches, low quality, blurry, distorted architecture, "
                     "changed facade material, removed windows, modern glass tower, cartoon, dark, night"),
    },
    "kres": {
        "prompt": ("photorealistic architectural renovation, historical brick building converted to "
                   "a cheerful kindergarten, preserved original facade and arched windows, "
                   "colorful playful entrance, playground with soft ground in the yard, "
                   "child-safe landscape, bright daylight, 8k, sharp details"),
        "negative": ("new modern building, white plaster facade, demolished and rebuilt, "
                     "rectangular windows replacing arches, low quality, blurry, distorted architecture, "
                     "changed facade material, removed windows, industrial, dark, night"),
    },
    "butik_otel": {
        "prompt": ("photorealistic architectural renovation, historical brick building converted to "
                   "a boutique hotel, preserved original facade and gothic windows, elegant entrance "
                   "canopy, landscaped courtyard with seating, warm evening ambiance, 8k, sharp details"),
        "negative": ("new modern building, white plaster facade, demolished and rebuilt, "
                     "rectangular windows replacing arches, low quality, blurry, distorted architecture, "
                     "changed facade material, removed windows, cartoon, oversaturated"),
    },
}

# ─── CONDITION HAZIRLIK (fotoğraf ORANI korunur: crop-to-fill, distorsiyon yok) ───
def load_photo(path, size):
    return ImageOps.fit(Image.open(path).convert("RGB"), (size, size), Image.LANCZOS)

def prepare_depth(photo):
    # KRİTİK: model ve processor AYNI checkpoint olmalı (dpt-large + midas processor
    # uyumsuzdu → ağın bir kısmı RASTGELE başlıyordu → bulanık/çöp depth → geometri bozuluyordu)
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
    fe = AutoImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
    with torch.no_grad():
        pred = model(**fe(images=photo, return_tensors="pt")).predicted_depth
    pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=photo.size[::-1],
                                           mode="bicubic", align_corners=False)
    arr = pred.squeeze().cpu().numpy()
    return Image.fromarray((arr * 255 / arr.max()).astype("uint8")).convert("RGB")

def prepare_lineart(photo):
    det = LineartDetector.from_pretrained("lllyasviel/Annotators")
    line = det(photo)
    return line.convert("RGB") if line.mode != "RGB" else line

def contact_sheet(imgs, labels, out):
    w, h = imgs[0].size
    sh = Image.new("RGB", (w*len(imgs), h+40), (255,255,255)); d = ImageDraw.Draw(sh)
    for i,(im,l) in enumerate(zip(imgs,labels)):
        sh.paste(im,(i*w,0)); d.text((i*w+10,h+10),l,fill=(0,0,0))
    sh.save(out)

# ─── MAIN ───
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    run_dir = Path("outputs") / datetime.now().strftime(f"i2r_{PROGRAM}_{SEVIYE}_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)

    photo = load_photo(INPUT_PHOTO, SIZE)
    depth = prepare_depth(photo);   depth.save(run_dir / "cond_depth.png")
    line  = prepare_lineart(photo); line.save(run_dir / "cond_lineart.png")
    photo.save(run_dir / "input_photo.png")

    # MULTI-CONTROLNET: depth (kütle) + lineart (cephe açıklık düzeni)
    cns = [ControlNetModel.from_pretrained(m, torch_dtype=dtype) for m in
           ["lllyasviel/control_v11f1p_sd15_depth", "lllyasviel/control_v11p_sd15_lineart"]]
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V6.0_B1_noVAE", controlnet=cns, vae=vae,
        torch_dtype=dtype, safety_checker=None, use_safetensors=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,
                                                             use_karras_sigmas=True)
    try: pipe.enable_xformers_memory_efficient_attention()
    except Exception: pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()   # 6GB VRAM — iki CN + img2img güvenli

    preset = PROGRAM_PRESETS[PROGRAM]
    strength = RENOVATION_LEVELS[SEVIYE]
    cn_scales = [0.50, 0.60]          # [depth, LINEART] — lineart kemerleri/kimliği tutar (MLSD eğri yakalayamaz!)
    cn_end    = [0.70, 0.85]          # depth erken bırakır (doku serbest), lineart %85'e kadar tutar

    meta = {"photo": INPUT_PHOTO, "program": PROGRAM, "seviye": SEVIYE, "strength": strength,
            "cn_scales": cn_scales, "seeds": SEEDS, "prompt": preset["prompt"],
            "negative": preset["negative"]}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                           encoding="utf-8")

    imgs, labels = [], []
    for i, seed in enumerate(SEEDS, 1):
        gen = torch.Generator(device="cuda" if device=="cuda" else "cpu").manual_seed(seed)
        img = pipe(prompt=preset["prompt"], negative_prompt=preset["negative"],
                   image=photo, control_image=[depth, line],
                   controlnet_conditioning_scale=cn_scales,
                   control_guidance_end=cn_end,
                   strength=strength, num_inference_steps=30, guidance_scale=6.5,
                   generator=gen).images[0]
        img.save(run_dir / f"alt{i:02d}_seed{seed}.png")
        imgs.append(img); labels.append(f"Alt {i} | seed {seed}")
        print("Saved:", f"alt{i:02d}_seed{seed}.png")

    contact_sheet([photo]+imgs, ["ORIJINAL"]+labels, run_dir / "comparison_sheet.png")
    print("Bitti:", run_dir / "comparison_sheet.png")

if __name__ == "__main__":
    main()