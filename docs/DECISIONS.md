# Karar Günlüğü
Yeni kararlar en üste, tarihli. Kaynak: (sohbet) = claude.ai sohbeti, (cc) = Claude Code, (hoca) = hocayla görüşme.

- 2026-10-20 (kullanıcı+cc): Colab Pro alındı; **tohum 1 Colab L4'te koşuldu** (saf tekrar). Test: 640 px 2 tohum AP50 0,849 ± 0,002. **Deneyler kapandı**; 1024 px, YOLO11n/m, RT-DETR yapılmayacak (gelecek çalışma). Sırada A6 rapor/sunum.
- 2026-10-19 (kullanıcı): **Ek eğitim bütçesi en fazla 2, hedef 1; deneyler 1 hafta içinde bitmeli.** Seçilen: **320 px** (yerel 3060, ~3 sa) → çözünürlük sorusu (Q4a, B4). İkinci eğitim (tohum 1, 640) yalnızca Colab kotası açılırsa; 1024 px, YOLO11n/m, RT-DETR yapılmayacak → raporda sınırlılık / gelecek çalışma. Öneri metnindeki "a subset" ifadesi bunu kapsıyor.
- 2026-10-19 (cc): **Tahminler Ultralytics doğrulama hattıyla üretilir** (multi-label NMS, dikdörtgen letterbox), metrikler pycocotools ile. `model.predict` hattı aynı ağırlıklarda 2,6 AP50 puanı düşük verdi; checkpoint seçimi ve referanslarla tutarlılık için val hattı seçildi (EVALUATION_PROTOCOL.md).
- 2026-10-19 (cc): Ana eğitim tamamlandı (Colab T4, 100 epoch, 4,2 sa, en iyi epoch 70). val_internal pycocotools AP50 0,860 / AP50:95 0,520; karar eşiği conf 0,33.
- 2026-10-18 (cc): **Ana eğitim Colab'da** (kullanıcı kararı). Gerekçe: yerel 3060'ta ≈6 dk/epoch → 5–10 saat ve bilgisayar meşgul; Colab T4'te batch 32 sığar, oturum koparsa `--resume`. Duman testi yerelde yapıldı (48 s/epoch @ %10 veri, 4,8 GB VRAM).
- 2026-10-18 (cc): Yol haritası `docs/ROADMAP.md`: A1 altyapı → A2 ana eğitim → A3 değerlendirme → A4 hata analizi → A5 ek eğitimler → A6 rapor; kod dondurma 6 Aralık.
- 2026-10-16 (sohbet): Öneri revize: 9 kusur türü tabloyla açıklandı; ayrım kuralı (train %90 / iç doğrulama %10 / resmi val = test) ve 640 px büyütmenin gerekçesi yazıldı; ek eğitimler ana eğitim süresi ölçüldükten sonra seçilecek.
- 2026-10-16 (cc): Depo **public** yapıldı (kullanıcı isteği). Colab defteri `GITHUB_TOKEN` olmadan da çeker; veri ve ağırlıklar depoda yok.
- 2026-10-16 (cc): Depo GitHub'da **private**: https://github.com/ethembelka/pcb-surface-defect-detection (`origin/main`). Colab defteri depoyu `GITHUB_TOKEN` gizli anahtarıyla çeker.
- 2026-10-16 (sohbet): Proje önerisi yazıldı → `../Proposal/CMPE7144_Proposal_Ethem_Belka_Sahin.pdf` (teslim 18 Ekim). Başlık: "Where Do Detectors Fail on Small PCB Defects?"
- 2026-10-16 (sohbet): Veri doğrulandı: 8.208/2.051 görüntü, 16.184/4.092 kutu; YOLO id = COCO id − 1; kutuların %67'si küçük (<32²).
- 2026-10-15 (sohbet): "Tek eğitim" kısıtı kaldırıldı; eğitimler süreye göre öncelik sırasıyla (CLAUDE.md → Eğitim planı).
- 2026-10-15 (hoca): Proje görüntü iyileştirme değil, nesne tespiti; uygulama değil bilimsel çalışma; CV projesinden ayrı.
- 2026-10-15 (sohbet): Veri seti DsPCBSD+; araştırma ödevi için 4 SCIE makale: Lv 2024, Wang 2025 (PCB-FS), Jin 2026 (PCB-MMF), Xia 2026 (VisDefectNet). Hoca onayı bekleniyor.
- 2026-10-15 (sohbet): Eğitim Colab'da (veri 123 MB), analizler yerelde.
