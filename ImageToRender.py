import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps,ImageDraw, ImageFont


from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers import UniPCMultistepScheduler
import os


from datetime import datetime

import cv2 # canny için ekledik

from controlnet_aux.lineart import LineartDetector #lineart için

from transformers import DPTForDepthEstimation, DPTFeatureExtractor
from diffusers import AutoencoderKL
from diffusers import DPMSolverMultistepScheduler




import torch

if torch.cuda.is_available():
    print(f"--- GPU AKTİF ---")
    print(f"Cihaz Adı: {torch.cuda.get_device_name(0)}")
    print(f"Toplam VRAM: {torch.cuda.get_memory_info(0).total / 1024**3:.2f} GB")
else:
    print("--- DİKKAT: GPU BULUNAMADI, CPU ÇALIŞIYOR! ---")


###############################################
# -----İleride başka detector'lar gerekirse:
# - controlnet_aux.<detector> şeklinde import edilecek
# - gerekirse genel import'a geri dönülebilir
###############################################
# from diffusers import StableDiffusionPipeline
# import torch
# pipe = StableDiffusionPipeline.from_pretrained(
#     "Lykon/dreamshaper-xl-1-0",  # DOĞRU ID
#     torch_dtype=torch.float16
# )
# # Lokalde saklamak için:
# pipe.save_pretrained(r"E:\Models\dreamshaper-xl-1-0")

def prepare_depth(image_path: str, size=512):
    #model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")


    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

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
    depth_img = Image.fromarray(formatted)
    # Depth haritasını smoothing yap
    kernel = np.ones((5, 5), np.float32) / 25
    smoothed = cv2.filter2D(np.array(depth_img), -1, kernel)

    return depth_img.convert("RGB")




def prepare_scribble(path: str, size=512, threshold=200) -> Image.Image:
    """
    Eskizi ControlNet Scribble için temizle:
    - 512x512 ölçekle
    - griye çevir
    - eşikleyip (binarize) çizgileri belirginleştir
    - tekrar RGB'ye çevir (pipeline RGB bekliyor)
    """
    img = Image.open(path).convert("L")
    img = ImageOps.autocontrast(img)
    img = img.resize((size, size))

    arr = np.array(img)
    # beyaz zemin + siyah çizgi varsayımı: çizgileri daha net yapmak için eşikleme
    arr = np.where(arr > threshold, 255, 0).astype(np.uint8)
    out = Image.fromarray(arr, mode="L").convert("RGB")
    return out

def prepare_canny(path: str, size=512, low=100, high=200) -> Image.Image:
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.array(img)

    edges = cv2.Canny(arr, low, high)          # 0-255 tek kanal
    edges = np.stack([edges]*3, axis=-1)       # (H,W) -> (H,W,3)

    return Image.fromarray(edges.astype(np.uint8), mode="RGB")

def prepare_lineart(path: str, size=512, coarse=False) -> Image.Image:
    """
    ControlNet Lineart için condition image üretir (önerilen yöntem).
    coarse=False: daha ince/temiz çizgiler
    coarse=True : daha kalın/sert çizgiler (bazı eskizlerde daha iyi)
    """

    img = Image.open(path).convert("RGB").resize((size, size))

    # Detector ilk çağrıda modeli indirir/cache'ler
    detector = LineartDetector.from_pretrained("lllyasviel/Annotators")

    line = detector(img, coarse=coarse)  # PIL döner (genelde gri ton)
    if line.mode != "RGB":
        line = line.convert("RGB")
    return line

def compute_condition_stats(img: Image.Image):
    """
    Condition image için basit ama güçlü istatistikler.
    Yorum yapmaz, sadece ölçer.
    """
    arr = np.array(img.convert("L"))  # grayscale
    mean = float(arr.mean())
    std = float(arr.std())

    # edge density: 0-255 aralığında "çizgi" oranı
    edge_pixels = np.count_nonzero(arr < 128)
    edge_density = float(edge_pixels / arr.size)

    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "edge_density": round(edge_density, 4)
    }

def save_contact_sheet(images, labels, out_path):
    """
    images: [PIL.Image, PIL.Image, PIL.Image]
    labels: ["A | seed 42", "B | seed 43", "C | seed 44"]
    """
    w, h = images[0].size
    sheet = Image.new("RGB", (w * len(images), h + 40), color=(255, 255, 255))

    draw = ImageDraw.Draw(sheet)

    for i, (img, label) in enumerate(zip(images, labels)):
        sheet.paste(img, (i * w, 0))
        draw.text((i * w + 10, h + 10), label, fill=(0, 0, 0))

    sheet.save(out_path)

def main():
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # ✅ Her çalıştırmada benzersiz run klasörü
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    # --- Model ayarları (senin kodun aynı) ---
    base_model = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
    controlnet_id = "lllyasviel/control_v11f1p_sd15_depth"
    #controlnet_id = "lllyasviel/sd-controlnet-scribble" #Bu kısım 'scribble' için olması gereken kısım.
    #controlnet_id = "lllyasviel/sd-controlnet-canny" #Bu kısım 'canny' için . pip install opencv-python ile openCV yi kur.
    #controlnet_id = "lllyasviel/control_v11p_sd15_lineart" #pip install controlnet-aux

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # ======================
    # LoRA CONFIG (SAFE TEST)
    # ======================
    use_lora = False  # örn: "loras/motif_fins_v0.1.safetensors"
    lora_path = "loras/RealisticVision-LoRA-libr-0.2.safetensors"
    lora_scale = 0.7  # 0.0-1.0 arası; dosya yoksa zaten kullanılmayacak

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=dtype
    )
    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=dtype,
        safety_checker=None,
        use_safetensors=False,#True yapmalısın eğer sd 1.5 olunca true yap
    )
    #pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # Mevcut UniPC yerine bunu ekle:
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True  # Karras iyileştirmesini aktif eder
    )
    pipe = pipe.to(device)
    #pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[GPU] xformers aktif edildi, performans optimize edildi.")
    except Exception as e:
        print(f"[Uyarı] xformers yüklenemedi, alternatif bellek yönetimi devrede: {e}")
        pipe.enable_attention_slicing()  # Sadece xformers hata verirse çalışır

    # if use_lora and os.path.exists(lora_path):
    #     print(f"[LoRA] Loading: {lora_path}")
    #     pipe.load_lora_weights(
    #         os.path.dirname(lora_path),
    #         weight_name=os.path.basename(lora_path)
    #     )
    #     print(f"[LoRA] Loaded (scale will be applied at inference)")
    # else:
    #     print("[LoRA] Disabled")
    if use_lora and os.path.exists(lora_path):
        print(f"[LoRA] Loading: {lora_path}")
        pipe.load_lora_weights(
            os.path.dirname(lora_path),
            weight_name=os.path.basename(lora_path),
            adapter_name="mimari_stil"  # Bir isim veriyoruz
        )
        # Ölçeği burada set ediyoruz, böylece pipe içinde tekrar göndermeye gerek kalmaz
        pipe.set_adapters("mimari_stil", adapter_weights=[lora_scale])
        print(f"[LoRA] Loaded with scale: {lora_scale}")

    # --- Girdi ---
    input_image_path = "image.jpg"  # veya dış cephe
    depth = prepare_depth(input_image_path, size=512)

    #sketch_path = "sketch.png"
    #scribble = prepare_scribble(sketch_path, size=512, threshold=200) #scrible image ayarı

    #canny = prepare_canny(sketch_path, size=512, low=100, high=200) # canny image ayarı

    #lineart = prepare_lineart(sketch_path, size=512,coarse=False) #lneart için image ayarım
    ## --- Condition artefact olarak kaydet ---
    condition_path = run_dir / "condition_depth.png"
    depth.save(condition_path)
    ## --- Condition stats hesapla ---
    condition_stats = compute_condition_stats(depth)

    prompt = (
        "Photorealistic architectural render of a historical stone building converted into a modern luxury nursing home, serene garden with walking paths and ergonomic benches, large minimalist glass extensions, warm sunlight, elderly-friendly landscape design, high-quality textures, 8k resolution, cinematic lighting, sharp details,exterior."
    )
    negative = "low quality, dark, scary, messy, blurry, industrial, futuristic, distorted architecture."

    # --- Üretim ayarları ---
    steps = 20
    guidance_scale = 7.0
    control_strength = 0.7

    # ✅ 3 alternatif için seed listesi
    #base_seed = 42
    #seeds = [base_seed, base_seed + 1, base_seed + 2]
    seeds = [42, 1001, 2024, 7777]

    # ✅ Run genel metadata
    run_meta = {
        "run_id": run_id,
        #"sketch_path": sketch_path,
        "image_path":input_image_path,
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
        #"condition_type": "lineart",
        #"condition_image": "condition_lineart.png",
        "condition_stats": condition_stats,
        "lora": {
            "enabled": use_lora,
            "path": lora_path if use_lora else None,
            "scale": lora_scale if use_lora else None
        }

    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    generated_images = []
    labels = []

    def _decode_latents_to_pil(latents: torch.Tensor) -> Image.Image:
        # latents: (1,4,H/8,W/8)
        lat = latents.detach()
        lat = lat.to(pipe.vae.dtype)

        # VAE scaling
        scale = getattr(pipe.vae.config, "scaling_factor", 0.18215)
        lat = lat / scale

        with torch.no_grad():
            image = pipe.vae.decode(lat).sample  # (1,3,H,W)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).round().astype("uint8")
        return Image.fromarray(image)

    # ✅ Her seed için üret ve kaydet
    for i, seed in enumerate(seeds, start=1):
        gen = torch.Generator(device=device).manual_seed(seed)

        img = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=depth,
            #image=canny,
            #image=lineart,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=control_strength,
            generator=gen,
            #cross_attention_kwargs={"scale": lora_scale},

        ).images[0]


        # dosya isimleri
        img_path = run_dir / f"a{i:02d}_seed{seed}.png"
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
            #"changed_parameter": "controlnet_type=0.7",
            # "notes": {
            #     "geometry": "çok iyi",
            #     "line_fidelity": "yüksek, biraz fazla baskın",
            #     "creativity": "orta",
            #     "materials": "kontrollü ama biraz tekdüze",
            #     "overall": "lineart cephe oranları için çok güçlü"
            # }


        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        generated_images.append(img)
        labels.append(f"Alt {i} | seed {seed}")

        print("Saved:", img_path)


    if not generated_images:
        raise ValueError("No images were generated; cannot create contact sheet.")

    contact_path = run_dir / "comparison_sheet.png"
    save_contact_sheet(
        generated_images,
        labels,
        contact_path
    )

    print("Comparison sheet saved:", contact_path)


if __name__ == "__main__":
    main()

#pip freeze > requirements_facadegen_v0.1.txt  eğer kod kusursuz çalışırsa güncelleme almamak için bu kısmı çalıştır ve pip i dondur
#bu süreçte tekrar hata alırsan pip install -r requirements_facadegen_v0.1.txt çalıştırarak çalışan hale tekrar getirebilirsin.
#pip install -U diffusers asla bu şekilde güncelleme yapma diffuser ve controlnet pipline'ı bozluyuyor bu da görüntünün işlenmemesine ve bulanık görüntü çıkmasına nedne oluyor
# setx HF_HOME E:\hf_cache_facadegen_v0.1 bu şekilde huggingfacei sabit bir cache de tutabilirsin. diğer türlü otoımatik c'de oluşacaktır.

