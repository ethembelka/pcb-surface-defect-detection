---
name: error-analysis
description: PCB dedektörünün hatalarını sınıf, boyut ve karışıklık açısından analiz etmek, hata galerileri ve rapor şekilleri üretmek için kullan.
---

# Hata analizi (yeniden eğitim gerektirmez)

1. Girdi: `outputs/results/<run_id>/` altındaki tahminler ve metrics.json. Yoksa önce `evaluate-detector`.
2. Analizler:
   - **Boyut:** her sınıf için küçük/orta/büyük kutu sayısı ve AP; "Sınıf zorluğu boyuttan mı geliyor?" sorusunu sayılarla cevapla.
   - **Karışıklık çiftleri:** SP–MB–SC ve CS–CFO çiftleri için karışma oranları; en sık 5 karışma.
   - **Kaçırılanlar (FN) ve yanlış alarmlar (FP):** sınıf başına en kötü örneklerden 3×4'lük görsel galeriler (gerçek kutu yeşil, tahmin kırmızı, skor yazılı).
   - **Kutu kalitesi:** IoU dağılımı (0.5–0.95) → AP50 ile AP50:95 farkının kaynağı.
3. Şekiller `reports/figures/`, tablolar `reports/tables/` altına; dosya adları açıklayıcı olsun.
4. Her bulguyu `docs/EXPERIMENTS.md` altındaki "Bulgular" bölümüne kısa maddelerle ekle (görüntü düzeyinde gözlem + sayı).
