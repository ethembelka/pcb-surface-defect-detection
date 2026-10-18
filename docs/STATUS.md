# Durum
_Son güncelleme: 2026-10-18 (cc)_

**Aşama: A1 (Altyapı) — adım 1–2 tamam, 3–6 sırada.** Plan: `docs/ROADMAP.md`. Yerel ortam hazır: Python 3.13, torch 2.6 + CUDA (RTX 3060 6 GB), ultralytics 8.3.101, pycocotools.

## Son yapılan (2026-10-18, cc)
- `src/pcbdet/` iskeleti: `config.py` (YAML→dataclass, yolları köke göre mutlaklaştırır), `data/classes.py`, `data/splits.py`, `utils/{io,seed,run}.py`; `pyproject.toml` (pytest `pythonpath=src`).
- **val_internal kuruldu (E2):** `scripts/make_splits.py` → train 8.208 = train_fit **7.387** + val_internal **821** (tohum 0, en nadir sınıfa göre tabakalı; her sınıfın %9,2–10,9'u val_internal'da). Listeler `data/splits/` (git'te) + kopyası `Data_YOLO/` kökünde (Ultralytics okur).
- `configs/dspcbsd.yaml` → train_fit / val_internal; yeni `configs/dspcbsd_test.yaml` → resmi val (sadece nihai rapor). Dört liste de Ultralytics ile yüklenip doğrulandı (7.387 / 821 / 2.051).
- `tests/test_splits.py`: 9 test geçiyor (kesişim yok, sayı korunuyor, her sınıf iki tarafta, deterministik, `../` yasağı, uçtan uca).
- E5 ultralytics==8.3.101 sabitlendi · E6 CLAUDE.md 108×108 notu · E7 EXPERIMENTS.md "Bulgular" başlığı.
- Öğrenilen Ultralytics ayrıntıları (kodda yorumlandı): liste satırındaki `./` listenin kendi klasörüne göre çözülür ve satırdaki **tüm** `./` değiştirilir (`../` kırılır); yaml'daki göreli `path` depoya değil Ultralytics'in datasets klasörüne göre çözülür → `DatasetConfig.to_ultralytics_dict()` mutlak yol verir. Liste tabanlı iki ayrım aynı `labels/train.cache` dosyasını paylaşır; başlangıçta yeniden taranır (saniyeler), sorun değil.

## Hazır
- Depo yapısı, mimari, değerlendirme protokolü, skill'ler, Colab defteri
- Veri `data/DsPCBSD/DsPCBSD+/` altında, doğrulandı; zip Drive'da `MyDrive/CMPE7144_PCB/data/`
- Proje önerisi PDF'i hazır (kullanıcı okuyup gönderecek)
- Depo GitHub'da (public): https://github.com/ethembelka/pcb-surface-defect-detection; defterde `GITHUB_USER` dolduruldu

## Sıradaki (ayrıntı: ROADMAP.md → A1)
1. Öneriyi gönder (kullanıcı, 18 Ekim)
2. A1.3 `data/stats.py` + `scripts/dataset_stats.py` (sınıf × boyut tablosu)
3. A1.4 `models/base.py`, `ultralytics_detector.py`, `registry.py`, `engine/trainer.py`, `scripts/train.py` (E4: defter bunu çağırsın)
4. A1.6 yerel duman testi (2 epoch, küçük `fraction`) → epoch süresi → karar kapısı: ana eğitim yerelde mi Colab'da mı

## Açık sorunlar (cc incelemesi, 2026-10-16; güncelleme 2026-10-18)
1. ~~`val_internal` kurulmamış~~ **Çözüldü (18 Eki):** `scripts/make_splits.py`, `configs/dspcbsd.yaml` (train_fit/val_internal), `configs/dspcbsd_test.yaml` (resmi val). Defter hâlâ eski akışta → E4 ile birlikte düzelecek.
2. **pycocotools `image_id` eşlemesi:** dosya adları sayısal değil (`Y_011791.jpg`, COCO id 0). Ultralytics `save_json` `image_id` olarak dosya kökünü yazar → `export_predictions` içinde dosya adı → COCO id eşlemesi şart; test yazılmalı. (A3)
3. Defter ayarları (`EPOCHS`, `IMGSZ`, `BATCH`, `SEED`, `MODEL`) elle yazılı; `scripts/train.py --config` çağırmalı. (A1.4 / E4)
4. ~~226×226 ifadesi~~ **Çözüldü:** CLAUDE.md'de 108×108 notu var; raporda da belirtilecek.
5. Doğrulandı, sorun yok: COCO `area` = bbox alanı (APs/m/l tutarlı); train–val arasında birebir kopya yok, dhash ile yalnız 2 yakın kopya.

## Bekleyen
- Hocanın DsPCBSD+ onayı
- Araştırma önerisi (4 SCIE makale, 1 sayfa) — teslim 1 Kasım
