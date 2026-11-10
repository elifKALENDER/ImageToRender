# ImageToRender

Bina cephe fotoğraflarını fotogerçekçi mimari görselleştirmelere dönüştüren, Stable Diffusion
(diffusers + ControlNet + IP-Adapter/LoRA) tabanlı bir araştırma/prototip projesi. Ayrıca cephe
fotoğraflarından yapısal analiz ve geometrik rekonstrüksiyon yapan aşamalı bir pipeline içerir.

> Durum: aktif geliştirme. Depo yapısı yeniden düzenleniyor.

## Gereksinimler

- Windows/Linux, NVIDIA GPU (CUDA 12.1)
- Conda, Python 3.10
- Model ağırlıkları depoya dahil değildir (aşağıya bakın)

## Kurulum

Çalışan ortamın tam yedeği repo dışında tutulur; aşağıdaki adımlar sıfırdan kurulum için
referanstır. Ayrıntılar ve sürüm kilitleri: [`setup/SETUP.md`](setup/SETUP.md).

```bash
cd setup
conda env create -f environment.yml
conda activate ImageToRender_GPU
pip install -r torch_install.txt      # torch 2.4.0 + xformers (cu121 index'inden)
pip install -r requirements.txt
```

Sürüm kilitlerine (torch, xformers, numpy, opencv, diffusers) dikkat edin: bunları
güncellemek ortamı bozar.

## Model dosyaları

`models/` ve `loras/` klasörleri `.gitignore` ile dışarıda tutulur. Yerelde şunlar beklenir:

- `models/ip-adapter_sd15.bin`, `models/models--h94--IP-Adapter` (IP-Adapter)
- `models/mobile_sam.pt` (MobileSAM)
- `loras/*.safetensors` (LoRA ağırlıkları)

## Kullanım

Görsel üretim akışı ve prompt şablonları için [`GUIDE.md`](GUIDE.md). Çıktılar `outputs/`
altına yazılır (versiyon kontrolü dışındadır).

Betikler aşama sırasıyla ilerler: tespit (Stage 1) → tekrar/eksen analizi (Stage 2) →
iskelet, perspektif aileleri, mimari ilişkiler ve cephe bölümleri (Stage 3) → geometrik
rekonstrüksiyon (Stage 4). Stage 5 henüz aktif değildir.

## Lisans

Henüz belirtilmedi.
