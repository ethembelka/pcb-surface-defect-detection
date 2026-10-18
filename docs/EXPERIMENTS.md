# Deney Kaydı

Her deney bitince `log-experiment` skill'i ile bir satır eklenir. Sonuç sayıları `outputs/results/<run_id>/metrics.json` dosyasından alınır, elle yazılmaz.

| run_id | tarih | model | imgsz | epoch | tohum | ortam | mAP50 | mAP50:95 | APs | not |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo11s_640_s0 | 2026-10-19 | YOLO11s | 640 | 100 (en iyi 70) | 0 | Colab T4 | 0.847 | 0.505 | 0.402 | test = resmi val, pycocotools; val_internal AP50 0.860; AP75 0.525, APm 0.573, APl 0.605; eşik conf 0.33 → P 0.806 / R 0.813 / F1 0.810; 9.43 M param, 21.57 GFLOPs, 25.61 ms/img (3060, fp16); eğitim 4.2 sa |

## Bulgular
_Hata analizi (`error-analysis` skill'i) buraya kısa maddeler ekler: görüntü düzeyinde gözlem + sayı._
