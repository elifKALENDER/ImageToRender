# ImageToRender

Araştırma/prototip projesi. Girdi: hasarlı bir tarihi yapının tek fotoğrafı ve hasar maskesi. Çıktı:
yapının kendi sağlam elemanlarından (donör) kanıta dayalı bir "restorasyon görünümü" (Venedik Tüzüğü),
bir "ilk inşa görünümü" ve her eklenen bölge için kaynak kaydı. Erken aşama karar desteğidir; hiçbir
çıktı restorasyon kararı değildir.

> Durum: aktif geliştirme (dal `refactor/structure`). Bu README geçicidir; örnek görseller ve sonuç
> tablolarıyla birlikte T24'te yeniden yazılacak (`docs/TASKS.md`).

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

Model ağırlıkları depoda yoktur. Pipeline'ın kullandığı modeller (OWL-ViT, SAM, DINOv2; kimlikleri
`config.yaml` içinde) ilk kullanımda Hugging Face önbelleğine indirilir. `models/` ve `loras/`
klasörleri yalnızca arşivdeki eski denemelere aittir ve `.gitignore` ile dışarıda tutulur.

## Durum

Plan ve görevler: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/TASKS.md`](docs/TASKS.md).
Eski denemelerin dökümü: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). Önceki tüm betikler
[`archive/`](archive/README.md) altında, çalıştı/çalışmadı bilgisiyle duruyor.

Yeni pipeline'ın altı adımı (`detect`, `review`, `damage`, `evidence`, `complete`, `render`) çalışıyor
(T1-T18); sentetik hasar değerlendirmesi (T19) ve eşik kalibrasyonu (T20) henüz yapılmadı. Sonuçlar tek
bir test yapısına aittir (n = 1) ve eşikler doğrulanmamış ilk tahminlerdir. Ham tespit uzman
envanterinin yaklaşık yarısını buluyor (`docs/rnd/RND-001-detection-baseline.md`); bu yüzden kullanılabilir
bir sonuç için envanterin `tools/editor_adapter.py` ile düzeltilmesi gerekir.

```bash
python tools/check_env.py                      # ortam denetimi
python -m unittest discover tests              # testler (GPU gerekmez)
python run.py --structure <id> --no-review     # data/structures/<id>/ için tüm zincir, iki görünüm
```

Fotoğraflar depoda değildir (kaynak ve lisans bilinmiyor); klasör düzeni: [`data/README.md`](data/README.md).

## Lisans

Henüz belirtilmedi.
