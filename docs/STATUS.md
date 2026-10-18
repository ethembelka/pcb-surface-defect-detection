# Durum
_Son güncelleme: 2026-10-18 (cc)_

**Aşama: A2 ana eğitim Colab'da koşuyor · A3 değerlendirme kodu hazır (test'te henüz çalıştırılmadı).** Plan: `docs/ROADMAP.md`.

## Son yapılan (2026-10-18 gece, cc) — A3 değerlendirme kodu
- `scripts/evaluate.py --run <run_dir> --split {val_internal,test} [--efficiency]` → `outputs/results/<run_id>/<split>/{predictions.json, metrics.json, config.yaml, git_hash.txt}` + `reports/tables/{main_results,per_class}_<run>_<split>.csv` + `reports/figures/confusion_matrix_<run>_<split>.png`.
- Modüller: `data/coco.py` (ayrım görüntüleri, **dosya adı → COCO image_id eşlemesi — E3 çözüldü**, filtrelenmiş GT), `evaluation/matching.py` (IoU, açgözlü sınıf-bazlı eşleme), `coco_eval.py` (pycocotools: AP50/75/50:95, APs/m/l, AR, sınıf bazlı AP), `per_class.py` (F1-maks eşik **val_internal'da seçilir, test'te aynen uygulanır**), `confusion.py` (arka plan dahil 10×10), `efficiency.py` (param, GFLOPs, FPS), `evaluator.py`, `reporting/tables.py` (referanslar `configs/references.yaml`'dan). `UltralyticsDetector.export_predictions` COCO sonuç json'u yazar (conf 0,001, NMS IoU 0,7, max_det 300).
- **Sağlama (duman modeli, val_internal):** pycocotools AP50 **0,1935** / AP50:95 **0,0892** ↔ Ultralytics'in aynı model için verdiği 0,199 / 0,0898. Sınıf bazında da tutarlı (HB 0,451 ↔ 0,452). Yani id eşlemesi doğru. F1-maks eşik 0,28. Verimlilik (3060, fp16, batch 1): 9,43 M param, 21,6 GFLOPs, 25,7 ms/görüntü (38,9 FPS).
- Test ayrımı için GT yükleme doğrulandı (2.051 görüntü, 4.092 kutu); **test'te tahmin henüz çalıştırılmadı** (eğitim bitince, bir kez).
- Testler: 24 geçiyor (`test_eval.py`: IoU, eşleme, F1-maks eşik, karışıklık matrisi, id eşlemesi, pycocotools).
- Not: `predictions.json` ~23 MB (821 görüntü × 300 kutu); `outputs/` git dışı. Smoke run'ın rapor dosyaları `reports/` altından silindi (gerçek run'lar kalacak).

## A2 ana eğitim — Colab (2026-10-18 ~18:55 başladı)
- Tesla T4 (15 GB), torch 2.11+cu130 (yerelde 2.6), ultralytics 8.3.101, 2 dataloader worker. `run_id = yolo11s_640_s0`, çıktılar Drive `CMPE7144_PCB/runs/yolo11s_640_s0/`.
- Epoch 1: 2 dk 26 sn + val 9 sn ≈ **2,6 dk/epoch** → 100 epoch ≈ 4 sa 20 dk. GPU 7,7–9,9 GB. Epoch 1 val_internal mAP50 (Ultralytics) **0,441**.
- Colab defteri ilk gerçek çalıştırmada sorunsuz (hücre 6'daki fark yalnızca CRLF idi → `io.py` LF'ye sabitlendi, f38dfc9).
- Bitince: `runs/yolo11s_640_s0/` Drive'dan → `outputs/runs/yolo11s_640_s0/` (best.pt, results.csv, grafikler); sonra `evaluate.py --split val_internal` → `--split test --efficiency` (bir kez, kullanıcı onayıyla).

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

## Sıradaki (ayrıntı: ROADMAP.md → A3/A4)
1. Öneriyi gönder (kullanıcı, 18 Ekim)
2. Colab eğitimi bitince run klasörünü `outputs/runs/yolo11s_640_s0/` altına indir (kullanıcı); `train_summary.json` ve `results.csv`'yi kaydet → `log-experiment`
3. `evaluate.py --split val_internal` (eşik seçimi) → kullanıcı onayı → `evaluate.py --split test --efficiency` (**bir kez**) → EXPERIMENTS.md
4. A4 hata analizi: sınıf × boyut AP, karışma çiftleri, FN/FP galerileri, IoU dağılımı

## Açık sorunlar (cc incelemesi, 2026-10-16; güncelleme 2026-10-18)
1. ~~`val_internal` kurulmamış~~ **Çözüldü (18 Eki):** `scripts/make_splits.py`, `configs/dspcbsd.yaml` (train_fit/val_internal), `configs/dspcbsd_test.yaml` (resmi val). Defter hâlâ eski akışta → E4 ile birlikte düzelecek.
2. ~~pycocotools `image_id` eşlemesi~~ **Çözüldü (18 Eki):** `data/coco.py::stem_to_image_id` + `export_predictions`; pycocotools ↔ Ultralytics mAP sağlaması tutuyor; `test_eval.py` ile korunuyor.
3. ~~Defter ayarları elle yazılı~~ **Çözüldü (18 Eki):** defter `scripts/train.py --config` çağırıyor; Colab'da ilk gerçek çalıştırmada doğrulanacak.
4. ~~226×226 ifadesi~~ **Çözüldü:** CLAUDE.md'de 108×108 notu var; raporda da belirtilecek.
5. Doğrulandı, sorun yok: COCO `area` = bbox alanı (APs/m/l tutarlı); train–val arasında birebir kopya yok, dhash ile yalnız 2 yakın kopya.

## Bekleyen
- Hocanın DsPCBSD+ onayı
- Araştırma önerisi (4 SCIE makale, 1 sayfa) — teslim 1 Kasım
