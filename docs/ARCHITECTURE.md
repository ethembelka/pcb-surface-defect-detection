# Mimari

## Tasarım ilkeleri
- **Tek sorumluluk (SRP):** Her modül tek bir iş yapar: veri, model, eğitim, değerlendirme, analiz, raporlama ayrıdır.
- **Açık/kapalı (OCP):** Yeni bir dedektör eklemek mevcut kodu değiştirmeyi gerektirmez; `Detector` arayüzünü uygulayan yeni bir sınıf yazılır ve kayıt defterine (registry) eklenir.
- **Bağımlılığın tersine çevrilmesi (DIP):** Değerlendirme ve analiz kodu somut bir kütüphaneye (Ultralytics) değil, `Detector` arayüzüne ve standart tahmin formatına (COCO JSON) bağımlıdır.
- **Konfigürasyonla yönetim:** Bütün ayarlar YAML → tipli `dataclass` olarak okunur. Kodda sabit yol ya da sayı yoktur.
- **Tekrarlanabilirlik:** Tohum sabitlenir, her çalıştırma konfigürasyonunu ve git hash'ini kaydeder.
- **Temiz kod:** Tip ipuçları, kısa fonksiyonlar, anlamlı adlar, docstring; yan etkiler (dosya yazma) uç katmanda.

## Klasörler
```
src/pcbdet/
├── config.py            # YAML → dataclass (DataConfig, TrainConfig, EvalConfig); load_config()
├── data/
│   ├── classes.py       # Sınıf adları/kısaltmaları, tek kaynak
│   ├── stats.py         # Veri seti istatistikleri: sınıf × boyut sayıları, kutu/görüntü dağılımı
│   └── splits.py        # Resmi ayrım + eğitimden iç doğrulama ayırma (tohumlu)
├── models/
│   ├── base.py          # Detector (Protocol/ABC): train(), predict(), export_predictions()
│   ├── ultralytics_detector.py  # YOLO11 vb. için uygulama
│   └── registry.py      # "yolo11s" → sınıf eşlemesi
├── engine/
│   └── trainer.py       # Eğitim akışı: config al, Detector oluştur, eğit, çıktıyı kaydet
├── evaluation/
│   ├── coco_eval.py     # pycocotools ile AP, AP50, AP75, APs/m/l, AR
│   ├── per_class.py     # Sınıf bazlı P, R, F1, AP
│   ├── confusion.py     # Karışıklık matrisi (IoU eşiğiyle eşleme)
│   ├── efficiency.py    # Parametre, GFLOPs, FPS ölçümü
│   └── evaluator.py     # Hepsini birleştirip metrics.json üretir
├── analysis/
│   ├── error_gallery.py # Kaçırılan/yanlış tespitlerin görsel galerileri
│   └── size_analysis.py # Boyut grubuna göre başarı
├── reporting/
│   └── tables.py        # Rapor için LaTeX/CSV tablolar, makale referanslarıyla karşılaştırma
└── utils/
    ├── io.py            # JSON/YAML okuma-yazma
    ├── seed.py          # Tohum sabitleme
    └── run.py           # run_id üretimi, çıktı klasörü, git hash kaydı
scripts/                 # İnce CLI'ler: train.py, evaluate.py, analyze.py, dataset_stats.py
tests/                   # Metrik ve veri dönüşümü testleri (pytest)
configs/                 # dspcbsd.yaml, train/*.yaml, eval/*.yaml
```

## Veri akışı
```
configs/*.yaml ─► scripts/train.py ─► engine.Trainer ─► models.Detector ─► outputs/runs/<run_id>/weights
                                                                             │
configs/eval/*.yaml ─► scripts/evaluate.py ─► Detector.export_predictions() ─► predictions.json (COCO)
                                         └─► evaluation.Evaluator ─► outputs/results/<run_id>/metrics.json
                                                                  └─► reporting.tables ─► reports/tables/*.csv|.tex
scripts/analyze.py ─► analysis.* ─► reports/figures/*.png
```

## Arayüz sözleşmesi
```python
class Detector(Protocol):
    def train(self, cfg: TrainConfig, out_dir: Path) -> Path: ...          # en iyi ağırlığın yolu
    def predict(self, images: Sequence[Path], conf: float) -> list[Detection]: ...
    def export_predictions(self, split: str, out_json: Path) -> Path: ...   # COCO results formatı
```
Değerlendirme yalnızca **COCO formatındaki tahmin dosyasına** bakar; böylece farklı bir dedektör eklemek değerlendirme kodunu değiştirmez.

## Adlandırma
- `run_id`: `{YYYYMMDD}_{model}_{imgsz}_s{seed}` (örn. `20261101_yolo11s_640_s0`)
- Metrik anahtarları: `map50`, `map50_95`, `ap75`, `ap_small`, `ap_medium`, `ap_large`, `precision`, `recall`, `f1`, `per_class.{SH..BMFO}.{ap50,ap50_95,p,r,f1}`
