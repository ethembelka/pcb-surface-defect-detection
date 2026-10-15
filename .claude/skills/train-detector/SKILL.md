---
name: train-detector
description: DsPCBSD+ üzerinde YOLO dedektörünü yerelde ya da Colab'da eğitmek veya yarım kalan eğitimi sürdürmek için kullan.
---

# Dedektör eğitimi

1. `docs/ARCHITECTURE.md` ve ilgili `configs/train/*.yaml` dosyasını oku. Yeni bir ayar gerekiyorsa mevcut dosyayı değiştirme, yeni bir YAML oluştur (örn. `yolo11s_640_s1.yaml`).
2. Ortamı belirle:
   - **Colab (varsayılan):** kod değişikliklerini commit + push et, kullanıcıya `notebooks/colab_train.ipynb` defterini çalıştırmasını söyle. Defter checkpoint'leri Drive'a yazar ve kopma sonrası `resume` ile devam eder.
   - **Yerel:** 6 GB VRAM için `batch` 8–16, `workers` ≤ 4. `python scripts/train.py --config configs/train/<dosya>.yaml`
3. Eğitimden önce kontrol et: veri yolu var mı, `configs/dspcbsd.yaml` sınıf sırası etiketlerle uyumlu mu, `val_internal` ayrımı oluşturulmuş mu (`scripts/dataset_stats.py`).
4. `run_id` biçimi: `{YYYYMMDD}_{model}_{imgsz}_s{seed}`. Çıktı `outputs/runs/<run_id>/`.
5. Eğitim bitince `evaluate-detector`, ardından `log-experiment` skill'ini çalıştır.
6. Yeni model türü eklemek gerekirse `models/base.py` arayüzünü uygula ve `models/registry.py`'ye kaydet; eğitim/değerlendirme kodunu değiştirme.
