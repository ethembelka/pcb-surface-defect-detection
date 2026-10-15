---
name: evaluate-detector
description: Eğitilmiş bir PCB hata dedektörünü değerlendirme protokolüne göre ölçmek, metrics.json ve rapor tablolarını üretmek için kullan.
---

# Değerlendirme

1. `docs/EVALUATION_PROTOCOL.md` dosyasını oku; metrik tanımları ve ayrımlar oradadır. Protokolden sapma gerekiyorsa önce kullanıcıya sor.
2. Tahminleri COCO formatında dışa aktar (`Detector.export_predictions`, `conf=0.001`).
3. `scripts/evaluate.py --run <run_id> --config configs/eval/default.yaml` ile:
   - pycocotools: AP, AP50, AP75, APs, APm, APl, AR@100
   - Sınıf bazlı AP50, AP50:95, P, R, F1 (F1-maksimum eşikte; eşiği de kaydet)
   - Normalize karışıklık matrisi (arka plan dahil)
   - Parametre, GFLOPs, FPS (protokoldeki ölçüm ayarlarıyla; GPU adını kaydet)
4. Çıktılar: `outputs/results/<run_id>/metrics.json`, `reports/tables/main_results.(csv|tex)`, `per_class.*`, `per_size.*`, `reports/figures/confusion_matrix.png`.
5. Ana tabloda bizim sonucu Lv vd. (2024) referanslarıyla yan yana koy; farklı ayrım kullanan literatür sonuçlarını ayrı ve uyarılı göster.
6. Birden çok tohum varsa ortalama ± std hesapla. Tek sonuç üzerinden "iyileşme" iddiasında bulunma.
7. Sayıları raporlarken her zaman metrics.json'dan oku; elle yazma.
