# Karar Günlüğü
Yeni kararlar en üste, tarihli. Kaynak: (sohbet) = claude.ai sohbeti, (cc) = Claude Code, (hoca) = hocayla görüşme.

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
