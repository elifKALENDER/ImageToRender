import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import torch

from PIL import Image, ImageOps, ImageDraw
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)
from controlnet_aux.lineart import LineartDetector
from transformers import DPTForDepthEstimation, AutoImageProcessor


# --- GPU Kontrol ---
if torch.cuda.is_available():
    print(f"--- GPU AKTİF ---")
    print(f"Cihaz Adı   : {torch.cuda.get_device_name(0)}")
    print(f"Toplam VRAM : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("--- DİKKAT: GPU BULUNAMADI, CPU ÇALIŞIYOR! ---")


# =============================================================
# CONDITION HAZIRLIK FONKSİYONLARI
# =============================================================

def prepare_depth(image_path: str, size: int = 512) -> Image.Image:
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    feature_extractor = AutoImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

    image = Image.open(image_path).convert("RGB").resize((size, size))
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    return Image.fromarray(formatted).convert("RGB")


def prepare_scribble(path: str, size: int = 512, threshold: int = 200) -> Image.Image:
    img = Image.open(path).convert("L")
    img = ImageOps.autocontrast(img)
    img = img.resize((size, size))
    arr = np.array(img)
    arr = np.where(arr > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def prepare_canny(path: str, size: int = 512, low: int = 100, high: int = 200) -> Image.Image:
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.array(img)
    edges = cv2.Canny(arr, low, high)
    edges = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges.astype(np.uint8), mode="RGB")


def prepare_lineart(path: str, size: int = 512, coarse: bool = False) -> Image.Image:
    img = Image.open(path).convert("RGB").resize((size, size))
    detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
    line = detector(img, coarse=coarse)
    if line.mode != "RGB":
        line = line.convert("RGB")
    return line


# =============================================================
# YARDIMCI FONKSİYONLAR
# =============================================================

def compute_condition_stats(img: Image.Image) -> dict:
    arr = np.array(img.convert("L"))
    edge_pixels = np.count_nonzero(arr < 128)
    return {
        "mean": round(float(arr.mean()), 2),
        "std": round(float(arr.std()), 2),
        "edge_density": round(float(edge_pixels / arr.size), 4),
    }


def save_contact_sheet(images: list, labels: list, out_path: Path) -> None:
    w, h = images[0].size
    sheet = Image.new("RGB", (w * len(images), h + 40), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, (img, label) in enumerate(zip(images, labels)):
        sheet.paste(img, (i * w, 0))
        draw.text((i * w + 10, h + 10), label, fill=(0, 0, 0))
    sheet.save(out_path)


# =============================================================
# MAIN
# =============================================================

def main():
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    # --- Model Ayarları ---
    base_model    = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
    controlnet_id = "lllyasviel/control_v11f1p_sd15_depth"
    # controlnet_id = "lllyasviel/sd-controlnet-scribble"
    # controlnet_id = "lllyasviel/sd-controlnet-canny"
    # controlnet_id = "lllyasviel/control_v11p_sd15_lineart"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    # --- LoRA Ayarları ---
    use_lora   = False
    lora_path  = "loras/RealisticVision-LoRA-libr-0.2.safetensors"
    lora_scale = 0.7

    # --- Pipeline Yükle ---
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=dtype,
        safety_checker=None,
        use_safetensors=False,
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
    )

    pipe = pipe.to(device)

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[GPU] xformers aktif.")
    except Exception as e:
        print(f"[Uyarı] xformers yüklenemedi, attention slicing devrede: {e}")
        pipe.enable_attention_slicing()

    # --- LoRA Yükle ---
    if use_lora and os.path.exists(lora_path):
        print(f"[LoRA] Loading: {lora_path}")
        pipe.load_lora_weights(
            os.path.dirname(lora_path),
            weight_name=os.path.basename(lora_path),
            adapter_name="mimari_stil",
        )
        pipe.set_adapters("mimari_stil", adapter_weights=[lora_scale])
        print(f"[LoRA] Loaded — scale: {lora_scale}")
    else:
        print("[LoRA] Disabled")

    # --- Girdi & Condition ---
    input_image_path = "image.jpg"
    depth = prepare_depth(input_image_path, size=512)
    # scribble = prepare_scribble("sketch.png", size=512, threshold=200)
    # canny    = prepare_canny("sketch.png", size=512, low=100, high=200)
    # lineart  = prepare_lineart("sketch.png", size=512, coarse=False)

    depth.save(run_dir / "condition_depth.png")
    condition_stats = compute_condition_stats(depth)

    # --- Prompt ---
    prompt = (
        "Photorealistic architectural render of a historical stone building converted into "
        "a modern luxury nursing home, serene garden with walking paths and ergonomic benches, "
        "large minimalist glass extensions, warm sunlight, elderly-friendly landscape design, "
        "high-quality textures, 8k resolution, cinematic lighting, sharp details, exterior,dark red brick facade, gothic pointed arches," 
        "preserved original stone masonry, keeping original materiality,blue sky, daylight, bright natural lighting."
    )
    negative = (
        "low quality, dark, scary, messy, blurry, industrial, futuristic, distorted architecture,beige, cream, limestone, white walls, changed facade material,red sky, dark sky, dramatic sky, overcast, night, dark shadows."
    )

    # --- Üretim Ayarları ---
    steps           = 30
    guidance_scale  = 6.5
    control_strength = 0.75
    seeds           = [42, 1001, 2024, 31415]

    # --- Run Metadata ---
    run_meta = {
        "run_id": run_id,
        "image_path": input_image_path,
        "prompt": prompt,
        "negative_prompt": negative,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "control_strength": control_strength,
        "model": base_model,
        "controlnet": controlnet_id,
        "device": device,
        "dtype": str(dtype),
        "seeds": seeds,
        "condition_stats": condition_stats,
        "lora": {
            "enabled": use_lora,
            "path": lora_path if use_lora else None,
            "scale": lora_scale if use_lora else None,
        },
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- Üretim Döngüsü ---
    generated_images = []
    labels = []

    for i, seed in enumerate(seeds, start=1):
        gen = torch.Generator(device=device).manual_seed(seed)

        img = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=depth,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=control_strength,
            generator=gen,
        ).images[0]

        img_path  = run_dir / f"a{i:02d}_seed{seed}.png"
        meta_path = run_dir / f"a{i:02d}_seed{seed}.json"

        img.save(img_path)

        meta = {
            "run_id": run_id,
            "index": i,
            "seed": seed,
            "image_file": img_path.name,
            "image_path": input_image_path,
            "prompt": prompt,
            "negative_prompt": negative,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "control_strength": control_strength,
            "model": base_model,
            "controlnet": controlnet_id,
            "experiment_type": "controlnet_depth",
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        generated_images.append(img)
        labels.append(f"Alt {i} | seed {seed}")
        print("Saved:", img_path)

    if not generated_images:
        raise ValueError("No images were generated; cannot create contact sheet.")

    contact_path = run_dir / "comparison_sheet.png"
    save_contact_sheet(generated_images, labels, contact_path)
    print("Comparison sheet saved:", contact_path)


if __name__ == "__main__":
    main()

# ---------- NOTLAR ----------
# pip freeze > requirements.txt          → ortamı dondur
# pip install -r requirements.txt        → ortamı geri yükle
# pip install -U diffusers               → YAPMA, pipeline bozuluyor
# setx HF_HOME E:\hf_cache              → HF cache konumunu sabitle