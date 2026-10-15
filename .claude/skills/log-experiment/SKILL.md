---
name: log-experiment
description: Tamamlanan bir eğitim ya da değerlendirmeyi docs/EXPERIMENTS.md deney kaydına işlemek için kullan.
---

# Deney kaydı

1. `outputs/results/<run_id>/metrics.json` ve `config.yaml` dosyalarını oku.
2. `docs/EXPERIMENTS.md` tablosuna bir satır ekle: run_id, tarih, model, imgsz, epoch, tohum, ortam (Colab T4 / yerel 3060), mAP50, mAP50:95, APs, kısa not.
3. Sayıları metrics.json'dan kopyala; yuvarlama 3 ondalık.
4. Değişiklikleri anlamlı bir commit mesajıyla kaydet: `exp: <run_id> sonuçları`.
