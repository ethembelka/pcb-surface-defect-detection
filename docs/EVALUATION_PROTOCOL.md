# Değerlendirme Protokolü

## Ayrımlar
| Ayrım | İçerik | Kullanım |
|---|---|---|
| `train_fit` | Resmi train'in %90'ı (tohumlu, sınıf varlığına göre dengeli) | Eğitim |
| `val_internal` | Resmi train'in %10'u | Eğitim sırasında en iyi ağırlığı seçmek |
| `val_official` | Resmi val (2.051 görüntü) | **Nihai raporlama (test seti gibi)** |

Not: Literatürde çoğu çalışma resmi val'i hem model seçimi hem raporlama için kullanıyor. Biz model seçimini `val_internal` ile
yapıp `val_official`'ı yalnızca raporlamada kullanıyoruz; karşılaştırma için ayrıca "resmi val ile seçilmiş" sonucu da (ek tablo) verebiliriz.

## Tahminlerin üretilmesi (19.10.2026'da sabitlendi)
Tahminler **Ultralytics doğrulama hattıyla** üretilir (`model.val(save_json=True)`): dikdörtgen letterbox (640 + kenar dolgusu),
**sınıf bazlı (multi-label) NMS**, conf 0,001, NMS IoU 0,7, max_det 300, FP16. Gerekçe: (1) checkpoint seçimi bu hatla yapıldı;
(2) referans dedektörler (Co-DETR/mmdet, YOLOv6) de sınıf bazlı NMS ile değerlendirilir; (3) aynı ağırlıklarla `model.predict`
hattı (kutu başına tek etiket, dolgusuz) val_internal'da 2,6 AP50 puanı daha düşük verdi (0,834 vs 0,860) — iki hattı
karıştırmak sonuçları karşılaştırılamaz yapar. Ultralytics'in json çıktısındaki `image_id` dosya adıdır (sayısal adlarda baştaki
sıfırlar düşer); `data/coco.py::remap_ultralytics_ids` bunu COCO id'lerine çevirir. Metrikler her zaman pycocotools ile hesaplanır;
Ultralytics'in kendi mAP'i yalnızca sağlama içindir (fark ≈ 0,4 puan).

## Metrikler
| Grup | Metrik | Kaynak |
|---|---|---|
| Ana | mAP@0.5, mAP@0.5:0.95 | pycocotools (COCO tanımı) |
| Ek | AP75, AR@100 | pycocotools |
| Boyut | AP_small (<32²), AP_medium, AP_large (>96²) — orijinal 226 px görüntü alanına göre | pycocotools |
| Sınıf | Sınıf başına AP50, AP50:95, P, R, F1 | per_class.py |
| Karar noktası | P, R, F1: F1'i maksimize eden güven eşiğinde (eşik raporlanır) | per_class.py |
| Karışıklık | Normalize karışıklık matrisi (IoU ≥ 0.5, conf ≥ seçilen eşik), arka plan satır/sütunu dahil | confusion.py |
| Verimlilik | Parametre sayısı, GFLOPs, FPS (batch=1, imgsz eğitimdekiyle aynı, FP16, 50 ısınma + 500 ölçüm, medyan; donanım belirtilir) | efficiency.py |
| Kararlılık | Birden çok tohum varsa ortalama ± std | reporting/tables.py |

## Referans sonuçlar (Lv vd., 2024 — resmi val)
| Model | AP50 | AP75 | AP50:95 | APs | APm | APl |
|---|---|---|---|---|---|---|
| Co-DETR (1333×800) | 0.848 | 0.490 | 0.492 | 0.425 | 0.554 | 0.671 |
| YOLOv6-L6 (1280×1280) | 0.851 | 0.525 | 0.514 | 0.405 | 0.597 | 0.681 |

Sınıf bazlı AP50 (YOLOv6-L6): SH .898 · SP .826 · SC .830 · OP .923 · MB .843 · HB .971 · CS .746 · CFO .743 · BMFO .881
Sınıf bazlı AP50 (Co-DETR):  SH .885 · SP .842 · SC .825 · OP .898 · MB .836 · HB .973 · CS .727 · CFO .741 · BMFO .900

Literatürdeki en iyi tam veri seti sonuçları (farklı ayrımlarla, dikkatli karşılaştır): PCB-FS (Symmetry 2025) 86.2 / 52.4.

## Raporlama biçimi
- `outputs/results/<run_id>/metrics.json` (yukarıdaki anahtarlarla)
- `reports/tables/main_results.csv|.tex` → bizim sonuç + referanslar aynı tabloda
- `reports/tables/per_class.csv|.tex`, `reports/tables/per_size.csv|.tex`
- `reports/figures/confusion_matrix.png`, `pr_curves.png`, `error_gallery_<class>.png`
