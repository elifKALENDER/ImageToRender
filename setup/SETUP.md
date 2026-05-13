# ImageToRender Kurulum Rehberi

## Dosyalar
- `environment.yml` → conda env + python sürümü
- `torch_install.txt` → torch + torchvision + xformers (doğru index ile)
- `requirements.txt` → geri kalan tüm pip paketleri

## Kurulum Sırası (BUNU TAKİP ET)

### 1. Conda env oluştur
```
conda env create -f environment.yml
conda activate ImageToRender_GPU
```

### 2. Torch'u DOĞRU index'ten kur (en kritik adım)
```
pip install -r torch_install.txt
```

### 3. Geri kalan paketleri kur
```
pip install -r requirements.txt
```

### 4. Kontrol et
```
python -c "import torch; import xformers; import cv2; import numpy; print(torch.__version__); print(xformers.__version__); print(numpy.__version__); print(torch.cuda.is_available())"
```
Çıktı şöyle görünmeli:
```
2.4.0+cu121
0.0.27.post2
1.26.4
True
```

---

## ÖNEMLİ NOTLAR

- `torch_install.txt` içindeki `--index-url` satırını SİLME.
  Bu satır torch'un PyPI yerine pytorch'un kendi sunucusundan gelmesini sağlar.
  Silinirse binary uyumsuzluk hatası alırsın.

- xformers versiyon kilidi: `xformers==0.0.27.post2` sadece `torch==2.4.0` ile çalışır.
  torch'u güncellersen xformers da patlar. İkisini birlikte değiştir ya da hiç değiştirme.

- numpy kilidi: `numpy==1.26.4` (1.x) zorunlu.
  numpy 2.x torch 2.4.0 ile çakışıyor. `pip install numpy --upgrade` YAPMA.

- opencv kilidi: `opencv-python==4.8.1.78` zorunlu.
  opencv 4.9+ numpy 2.x istiyor, bu ortamda çalışmaz.

- `pip install -U diffusers` YAPMA.
  Diffusers güncellemesi controlnet pipeline'ını bozuyor, bulanık/bozuk görüntü çıkıyor.

- HuggingFace cache konumunu sabitlemek istersen:
  ```
  setx HF_HOME E:\hf_cache
  ```
  Yapılmazsa otomatik olarak C: sürücüsünde oluşur.

## Versiyon Özeti (Çalışan Kombinasyon)

| Paket | Versiyon |
|---|---|
| Python | 3.10.19 |
| torch | 2.4.0+cu121 |
| torchvision | 0.19.0+cu121 |
| torchaudio | 2.4.0+cu121 |
| xformers | 0.0.27.post2 |
| numpy | 1.26.4 |
| opencv-python | 4.8.1.78 |
| diffusers | 0.20.2 |
| transformers | 4.33.2 |