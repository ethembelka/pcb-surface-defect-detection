# Durum
_Son güncelleme: 2026-10-18 (cc)_

**Aşama: A1 (Altyapı) tamamlandı — karar kapısı bekliyor (ana eğitim yerel mi Colab mı).** Plan: `docs/ROADMAP.md`. Yerel ortam hazır: Python 3.13, torch 2.6 + CUDA (RTX 3060 6 GB), ultralytics 8.3.101, pycocotools.

## Son yapılan (2026-10-18 akşam, cc) — A1 ikinci yarı
- `data/stats.py` + `scripts/dataset_stats.py` → `reports/tables/dataset_stats.csv` (sınıf × boyut, üç ayrım). Çarpıcı: SP %97,7 ve MB %95,3 küçük; HB %1,3 küçük (hemen hepsi orta); CS %29,5 küçük, büyük kutuların çoğu CS'te.
- Eğitim akışı: `models/base.py` (Detector protokolü), `models/ultralytics_detector.py`, `models/registry.py`, `engine/trainer.py`, `scripts/train.py` (`--data-root`, `--project`, `--run-id`, `--resume`). `TrainConfig` config.py'de. Her run: `outputs/runs/<run_id>/{config.yaml,git_hash.txt,data.yaml,train_summary.json}`.
- E4: Colab defteri yeniden yazıldı; ayar içermiyor, `make_splits.py --yolo-root` + `train.py --config ... --project <Drive>/runs --run-id ... [--resume]` çağırıyor. (Colab'da henüz denenmedi.)
- **Duman testi (yerel, RTX 3060):** `configs/train/smoke_local.yaml` — YOLO11s, 640, batch 16, train_fit'in %10'u (739 görüntü), val_internal tam (821), 2 epoch. Hatasız bitti. `results.csv`: epoch başına **~48 s** (train 739 + val 821 görüntü); tepe VRAM **4,8 GB / 6 GB** (batch 16 sığıyor, 32 sığmaz). 2 epoch sonunda val_internal mAP50 0,20 (sadece "öğreniyor" kanıtı; rapora girmez). Run: `outputs/runs/20261018_yolo11s_640_s0_smoke/`.
- **Ana eğitim süre tahmini (yerel):** train kısmı ≈ 35 s / 739 görüntü → 7.387 görüntü ≈ 350 s, + val ≈ 15 s → **≈ 6 dk/epoch** → 100 epoch ≈ 10 saat; patience 30 ile erken durursa (tipik 50–70 epoch) **≈ 5–7 saat**. Colab T4 fp16'da 3060 Laptop'tan hızlı değildir (benzer ya da biraz yavaş); avantajı bilgisayarı meşgul etmemesi ve batch 32'nin sığması; dezavantajı oturum kopması (resume var).
- Testler: 16 geçiyor (`test_splits`, `test_stats`, `test_run`).

## Son yapılan (2026-10-18 öğlen, cc) — A1 ilk yarı
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

## Sıradaki (ayrıntı: ROADMAP.md → A2)
1. Öneriyi gönder (kullanıcı, 18 Ekim)
2. **Karar kapısı (kullanıcı):** ana eğitim yerelde (gece, ≈5–10 saat, batch 16) mi Colab'da (batch 32, oturum kopabilir) mı? Karar DECISIONS.md'ye.
3. A2 ana eğitim: `configs/train/yolo11s_640.yaml` (Colab) ya da yerel için batch 16 kopyası; `run_id` sabit → `yolo11s_640_s0`
4. A2 sürerken A3 değerlendirme kodunu yaz (`export_predictions` + image_id eşlemesi, pycocotools, per_class, confusion)

## Açık sorunlar (cc incelemesi, 2026-10-16; güncelleme 2026-10-18)
1. ~~`val_internal` kurulmamış~~ **Çözüldü (18 Eki):** `scripts/make_splits.py`, `configs/dspcbsd.yaml` (train_fit/val_internal), `configs/dspcbsd_test.yaml` (resmi val). Defter hâlâ eski akışta → E4 ile birlikte düzelecek.
2. **pycocotools `image_id` eşlemesi:** dosya adları sayısal değil (`Y_011791.jpg`, COCO id 0). Ultralytics `save_json` `image_id` olarak dosya kökünü yazar → `export_predictions` içinde dosya adı → COCO id eşlemesi şart; test yazılmalı. (A3)
3. ~~Defter ayarları elle yazılı~~ **Çözüldü (18 Eki):** defter `scripts/train.py --config` çağırıyor; Colab'da ilk gerçek çalıştırmada doğrulanacak.
4. ~~226×226 ifadesi~~ **Çözüldü:** CLAUDE.md'de 108×108 notu var; raporda da belirtilecek.
5. Doğrulandı, sorun yok: COCO `area` = bbox alanı (APs/m/l tutarlı); train–val arasında birebir kopya yok, dhash ile yalnız 2 yakın kopya.

## Bekleyen
- Hocanın DsPCBSD+ onayı
- Araştırma önerisi (4 SCIE makale, 1 sayfa) — teslim 1 Kasım
