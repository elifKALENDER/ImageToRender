# ImageToRender — Müşteri Proje El Kitabı

## 1. HIZLI BAŞLANGIÇ — Yeni Müşteri Geldi, Ne Yapacaksın?

### Adım 1 — Müşteriden Al
- [ ] Bina fotoğrafı (cephe, mümkünse düz açı)
- [ ] İstek: ne olmasını istiyor? (otel, ofis, bakım evi, rezidans...)
- [ ] Stil tercihi: modern / klasik / minimalist / lüks

### Adım 2 — Kodu Ayarla (değiştirmen gereken tek dosya: ImageToRender.py)

```python
# 1. Fotoğrafı proje klasörüne koy, adını buraya yaz:
input_image_path = "musteri_bina.jpg"

# 2. Prompt'u güncelle (aşağıdaki şablonlardan birini kullan)

# 3. Seed sayısını ayarla:
seeds = [42, 1001, 2024, 7777]   # 4 alternatif
# seeds = [42, 1001, 2024]       # 3 alternatif (daha hızlı)

# 4. Çalıştır → outputs/run_TARIH/ klasörüne düşer
```

---

## 2. PROMPT ŞABLONLARı

### Temel Yapı (her prompt'ta olmalı)
```
photorealistic architectural visualization, [NE OLACAK],
[MALZEME/STİL], [PEYZAJ], [IŞIK],
high-quality textures, 8k resolution, cinematic lighting, sharp details, exterior view
```

### Negative (değiştirme, her zaman aynı kullan)
```
low quality, blurry, dark, cartoonish, distorted architecture,
unrealistic textures, bad proportions, watermark, sketch, drawing
```

---

### Hazır Prompt Şablonları

#### 🏥 Bakım Evi / Huzurevi
```
photorealistic architectural render, historical building converted into a modern
luxury nursing home, warm beige facade, serene garden with walking paths and
ergonomic benches, large windows, warm sunlight, elderly-friendly landscape,
blooming flowers, accessible entrance, high-quality textures, 8k resolution,
cinematic lighting, sharp details, exterior view
```

#### 🏨 Butik Otel
```
photorealistic architectural visualization, historic building transformed into
a boutique luxury hotel, elegant stone facade, manicured entrance garden,
soft warm evening lighting, hotel signage, potted plants at entrance,
valet area, high-quality textures, 8k resolution, cinematic lighting,
sharp details, exterior view
```

#### 🏢 Modern Ofis / Kurumsal
```
photorealistic architectural render, building renovated into a modern
corporate office, glass extensions, clean minimalist facade, landscaped
entrance plaza, professional lighting, people walking at entrance,
high-quality textures, 8k resolution, sharp details, exterior view
```

#### 🏘️ Lüks Rezidans / Konut
```
photorealistic architectural visualization, historical building converted into
luxury residential apartments, warm stone facade, private garden, ornamental
trees, soft golden hour lighting, elegant entrance, high-quality textures,
8k resolution, cinematic lighting, sharp details, exterior view
```

#### 🏫 Kültür Merkezi / Müze
```
photorealistic architectural render, historic building transformed into a
contemporary cultural center, preserved original facade details, modern
glass canopy entrance, public plaza with sculptures, evening illumination,
people gathering, high-quality textures, 8k resolution, cinematic lighting,
sharp details, exterior view
```

---

## 3. MALZEME KORUMA — Orijinal Bina Görünümü Korunacaksa

Müşteri "binanın görünümü korunsun ama içerik değişsin" diyorsa:

```python
# Prompt'a ekle:
"preserving original brick facade, keeping original materiality and color,
maintaining historical architectural character"

# Negative'e ekle:
"changed facade, different material, repainted, white walls, beige walls"

# Kodu ayarla:
control_strength = 0.85   # 0.7 yerine — orijinale daha sadık
```

---

## 4. CONTROLNET MODU SEÇİMİ

| Mod | Ne Zaman Kullan | Kod |
|---|---|---|
| **depth** (varsayılan) | Gerçek fotoğraf girişi | `control_v11f1p_sd15_depth` |
| **lineart** | El çizimi / eskiz girişi | `control_v11p_sd15_lineart` |
| **canny** | Keskin kenarlı teknik çizim | `sd-controlnet-canny` |
| **scribble** | Kaba karalama / konsept eskiz | `sd-controlnet-scribble` |

```python
# Fotoğraf → depth (şu an aktif)
controlnet_id = "lllyasviel/control_v11f1p_sd15_depth"
depth = prepare_depth(input_image_path, size=512)
image = depth

# Eskiz → lineart
# controlnet_id = "lllyasviel/control_v11p_sd15_lineart"
# lineart = prepare_lineart(sketch_path, size=512, coarse=False)
# image = lineart
```

---

## 5. PARAMETRE AYARLARI

### Hız vs Kalite
```python
steps = 20    # hızlı test (4-5 sn/görsel)
steps = 30    # denge (6-8 sn/görsel)
steps = 40    # maksimum kalite (9-12 sn/görsel)
```

### Yaratıcılık Kontrolü
```python
guidance_scale = 5.0   # daha özgür, daha yaratıcı
guidance_scale = 7.0   # denge (varsayılan)
guidance_scale = 9.0   # prompt'a çok sadık, sert
```

### Orijinal Yapıya Bağlılık
```python
control_strength = 0.5   # özgür — bina kütlesi gevşek korunur
control_strength = 0.7   # denge (varsayılan)
control_strength = 0.85  # sıkı — orijinal form korunur
control_strength = 1.0   # maksimum — neredeyse değişmez
```

### Kaç Alternatif?
```python
seeds = [42]                      # 1 görsel — hızlı test
seeds = [42, 1001, 2024]          # 3 alternatif
seeds = [42, 1001, 2024, 7777]    # 4 alternatif (varsayılan)
seeds = [42, 1001, 2024, 7777, 9999, 1234]  # 6 alternatif — sunum için
```

---

## 6. LoRA — GEREKLİ Mİ?

### Kısa Cevap
**Şu an için HAYIR.** Realistic Vision V6 base model olarak zaten çok güçlü.
LoRA'yı şu durumlarda ekle:

| Durum | LoRA Gerekli mi? |
|---|---|
| Genel fotogerçekçi mimari | ❌ base model yeterli |
| Belirli bir mimari stil (Osmanlı, Art Deco...) | ✅ stil LoRA |
| Belirli bir malzeme (ahşap, beton, cam...) | ✅ malzeme LoRA |
| Gece / dramatik aydınlatma | ✅ lighting LoRA |
| Müşteri markası / tekrar eden stil | ✅ custom LoRA |

### LoRA Eklemek İstersen
```python
use_lora = True
lora_path = "loras/senin_loran.safetensors"
lora_scale = 0.6   # 0.4-0.8 arası dene, fazlası bozar
```

### İyi LoRA Kaynakları (SD 1.5 uyumlu)
- CivitAI → "architectural" filtrele → SD 1.5 seç
- "Realistic architecture exterior" araması yap
- `lora_scale` düşük başla (0.4), yavaş artır

---

## 7. MÜŞTERI SUNUM AKIŞI

```
1. Müşteri fotoğraf verir
         ↓
2. control_strength = 0.85 ile ilk test (3 seed)
         ↓
3. Müşteri "daha yaratıcı olsun" derse → control_strength = 0.6
   Müşteri "bina korunsun" derse      → control_strength = 0.9
         ↓
4. Beğenilen seed'i seç
         ↓
5. O seed ile steps=40 yap → final kalite
         ↓
6. comparison_sheet.png müşteriye sun
```

---

## 8. SORUN GİDERME

| Sorun | Sebebi | Çözüm |
|---|---|---|
| Görsel bulanık | control_strength çok düşük | 0.7 → 0.85 yap |
| Bina tanınmıyor | control_strength çok yüksek | 0.85 → 0.7 yap |
| Köşelerde yeşil/renkli leke | VRAM taşması | O seed'i kullanma |
| Malzeme tamamen değişmiş | Prompt malzeme belirtmiyor | "preserving original brick" ekle |
| Çok tekdüze 4 görsel | Seed'ler birbirine yakın | Uzak seed'ler seç (42, 5000, 9999, 50000) |
| Işık çok sert | guidance_scale yüksek | 7.0 → 6.0 yap |

---

## 9. ÖRNEK ÇALIŞMA AKIŞI (Gerçek Senaryo)

```
Müşteri: Kırmızı tuğlalı eski kilisem var, butik otel olsun, modern cam
         uzantılar eklensin ama gotik karakteri korunsun.

Sen:
  input_image_path = "kilise.jpg"
  controlnet_id = depth (fotoğraf var)
  control_strength = 0.80
  
  prompt = "photorealistic architectural render, historic red brick gothic
            church converted into boutique luxury hotel, preserving original
            pointed arches and brick facade, modern glass extension on side,
            manicured entrance garden, warm evening lighting, 8k, sharp details"
  
  negative += "beige, cream, changed facade material, modern facade"
  
  seeds = [42, 1001, 2024, 7777]
  steps = 30
  
→ 4 alternatif üret, müşteriye sun
→ Beğenilen seed ile steps=40 tekrar çalıştır
```

---

## 10. VERSİYON KİLİDİ (DOKUNMA)

| Paket | Versiyon |
|---|---|
| torch | 2.4.0+cu121 |
| xformers | 0.0.27.post2 |
| diffusers | 0.20.2 |
| transformers | 4.33.2 |
| huggingface-hub | 0.16.4 |
| accelerate | 0.22.0 |
| numpy | 1.26.4 |
| opencv-python | 4.8.1.78 |

```bash
# Ortamı dondur (çalışır haldeyken yap):
pip freeze > requirements.txt

# Bozulursa geri yükle:
pip install -r requirements.txt
```

## 10. KARARLI PARAMETRELER
 # --- Üretim Ayarları ---
    steps           = 30
    guidance_scale  = 6.5
    control_strength = 0.75
    seeds           = [42, 1001, 2024, 31415]
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