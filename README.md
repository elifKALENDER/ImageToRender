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

## Durum

Repo yeniden yapılandırılıyor. Plan ve görevler: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
[`docs/TASKS.md`](docs/TASKS.md). Eski denemelerin dökümü: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).
Önceki tüm betikler [`archive/`](archive/README.md) altında, çalıştı/çalışmadı bilgisiyle duruyor.
Yeni pipeline (`run.py`) henüz yazılmadı.

Ortamı denetlemek için: `python tools/check_env.py`.

## Lisans

Henüz belirtilmedi.
