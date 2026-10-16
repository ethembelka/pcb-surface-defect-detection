# Durum
_Son güncelleme: 2026-10-16 (cc)_

## Hazır
- Depo yapısı, mimari, değerlendirme protokolü, skill'ler, Colab defteri
- Veri `data/DsPCBSD/DsPCBSD+/` altında, doğrulandı; zip Drive'da `MyDrive/CMPE7144_PCB/data/`
- Proje önerisi PDF'i hazır (kullanıcı okuyup gönderecek)
- Depo GitHub'da (public): https://github.com/ethembelka/pcb-surface-defect-detection; defterde `GITHUB_USER` dolduruldu

## Sıradaki
1. Eğitimden önce aşağıdaki açık sorunlardan 1 ve 2'yi çöz
2. Ana eğitim: YOLO11s, 640, tohum 0 (Colab) → süreyi ölç
3. Değerlendirme + hata analizi

## Açık sorunlar (cc incelemesi, 2026-10-16)
1. **`val_internal` kurulmamış:** `configs/dspcbsd.yaml` → `val: images/val` (resmi val). Defter de bunu kullanıyor, yani `best.pt` resmi val'e göre seçilir; protokole aykırı. `train.yaml`'daki `val_split` alanını Ultralytics okumaz. `splits.py` + ayrı eğitim yaml'ı gerekli.
2. **pycocotools `image_id` eşlemesi:** dosya adları sayısal değil (`Y_011791.jpg`, COCO id 0). Ultralytics `save_json` `image_id` olarak dosya kökünü yazar → `export_predictions` içinde dosya adı → COCO id eşlemesi şart; test yazılmalı.
3. Defter ayarları (`EPOCHS`, `IMGSZ`, `BATCH`, `SEED`, `MODEL`) elle yazılı; `configs/train/*.yaml`'dan okunmalı.
4. Görüntülerin hepsi 226×226 değil: 111 görüntü 108×108 (train 86, val 25). CLAUDE.md'deki ifade düzeltilmeli.
5. Doğrulandı, sorun yok: COCO `area` = bbox alanı (APs/m/l tutarlı); train–val arasında birebir kopya yok, dhash ile yalnız 2 yakın kopya.

## Bekleyen
- Hocanın DsPCBSD+ onayı
- Araştırma önerisi (4 SCIE makale, 1 sayfa) — teslim 1 Kasım
