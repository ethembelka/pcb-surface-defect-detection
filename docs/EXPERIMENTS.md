# Deney Kaydı

Her deney bitince `log-experiment` skill'i ile bir satır eklenir. Sonuç sayıları `outputs/results/<run_id>/metrics.json` dosyasından alınır, elle yazılmaz.

| run_id | tarih | model | imgsz | epoch | tohum | ortam | mAP50 | mAP50:95 | APs | not |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo11s_640_s0 | 2026-10-19 | YOLO11s | 640 | 100 (en iyi 70) | 0 | Colab T4 | 0.847 | 0.505 | 0.402 | test = resmi val, pycocotools; val_internal AP50 0.860; AP75 0.525, APm 0.573, APl 0.605; eşik conf 0.33 → P 0.806 / R 0.813 / F1 0.810; 9.43 M param, 21.57 GFLOPs, 25.61 ms/img (3060, fp16); eğitim 4.2 sa |
| yolo11s_320_s0 | 2026-10-20 | YOLO11s | 320 | 100 (en iyi 87) | 0 | yerel 3060 (ısıl kısıtlı) | 0.833 | 0.503 | 0.422 | test = resmi val, pycocotools; val_internal AP50 0.865; AP75 0.521, APm 0.567, APl 0.725; eşik conf 0.35 → P 0.795 / R 0.790 / F1 0.792; 9.43 M param, 5.39 GFLOPs, 17.67 ms/img (3060, fp16); eğitim duvar süresi ~4 sa (ısıl kısıtlama; normalde ~2,5) |

## Bulgular
_Hata analizi (`error-analysis` skill'i) buraya kısa maddeler ekler: görüntü düzeyinde gözlem + sayı._

### yolo11s_640_s0 — test (resmi val), conf ≥ 0,33, IoU ≥ 0,5 (2026-10-19, `scripts/analyze.py`; sayılar `outputs/results/yolo11s_640_s0/test/analysis.json`)
Toplam: 4.092 GT kutu → 3.326 bulundu, **766 kaçırıldı**, **799 yanlış alarm**.

- **B1 Boyut etkisi var ama tek başına açıklamıyor.** Kaçırılanların %73'ü küçük (559/766), oysa küçük kutuların GT'deki payı %66. Duyarlılık: küçük **0,794**, orta 0,852, büyük 0,830. Küçük nesneler biraz daha çok kaçıyor; asıl fark sınıfa bağlı (B2, B3).
- **B2 CS (çizik): zorluk boyuttan geliyor.** Küçük CS AP50 **0,586** / duyarlılık **0,49**; büyük CS AP50 0,864 / duyarlılık 0,82. Galeri (`gallery_fn_CS`): uzun ince çizikler tek bir büyük kutuyla etiketlenmiş, model aynı çiziği 2–3 parça hâlinde buluyor → her parça IoU < 0,5 kalıyor → "kaçırıldı" sayılıyor. Yani CS'teki kayıp kısmen **kutu tanımı (granülerlik)** sorunu, saf tespit hatası değil. Büyük kutuların 132/176'sı CS olduğu için referanslara göre APl açığımız (0,605 vs 0,67–0,68) da büyük ölçüde buradan.
- **B3 CFO (iletken üstü yabancı madde): zorluk boyuttan değil görünümden.** AP50 küçükte 0,715, ortada 0,705 — boyutla değişmiyor; sınıf heterojen (kir, leke, kabarcık, parmak izi). Karışmaları iki yönlü: CFO↔MB (10 + 14, hepsi küçük kutular) ve tek yönlü CFO→BMFO (9, orta/büyük; ters yön 0): aynı yabancı madde iletken–taban sınırındayken BMFO deniyor.
- **B4 Lokalizasyon kalitesi AP50:95'i belirliyor.** Bulunan kutuların medyan IoU'su: HB **0,92** (IoU ≥ 0,9 olanlar %68) ↔ SP **0,747** (%51'i IoU < 0,75, yalnız %3'ü ≥ 0,9), MB 0,767 (%44 < 0,75), BMFO 0,78. SP/MB kutuları ~10–15 px; birkaç piksel kayma IoU'yu 0,2 düşürüyor. Bu yüzden SP AP50 0,849 ama AP50:95 0,389; HB'de 0,979 / 0,813. Şekil: `iou_boxplot`.
- **B5 Karışmalar.** Simetrik çiftler **MB↔CFO** (14/10) ve **SC↔SP** (12/9): hepsi küçük kutular, medyan skor 0,53–0,65 (model kararsız). Galeri (`gallery_pair_MB_as_CFO`): iletken kenarındaki koyu, yarım daire çentik ile kenara yapışmış koyu leke aynı görünüyor. Tek yönlü: **CFO→BMFO** (9/0). Önerideki SP/MB/SC ve CS/CFO çiftlerine ek olarak MB↔CFO en sık çift çıktı; CS↔CFO beklendiği kadar sık değil (2–3).
- **B6 Yanlış alarmların yarısı aslında etiket karışması.** 799 FP'nin **411'i (%51)** başka sınıftan bir GT kutusuyla IoU ≥ 0,3 örtüşüyor: kutu doğru, sınıf yanlış (aynı hata hem FN hem FP sayılıyor; multi-label NMS'in ikinci etiketleri de buna katkıda bulunuyor). **246'sı (%31)** yakınında hiç GT olmayan "gerçek" yanlış alarm.
- **B7 HB yanlış alarmları etiket eksikliğine işaret ediyor.** HB'nin 41 FP'sinden 39'unun yakınında GT yok; galeri (`gallery_fp_HB`) 0,70–0,86 güvenle işaretlenmiş, **gözle de merkezden kaçık görünen** delikler gösteriyor. Aday açıklama: veri setinde etiketlenmemiş HB'ler (etiket gürültüsü). Raporda "veri seti üzerine gözlem" olarak verilecek; doğrulamak için bu 39 örnek elle incelenmeli.
- **B8 Düşük destekli hücreler yorumlanmadı:** HB-küçük (7 GT, AP50 0,33), MB-orta (16), SP-orta (25), SC-büyük (2), BMFO-büyük (13), CFO-büyük (29). Tablolarda `low_support` ile işaretli.

Rapora girecek şekiller: `iou_boxplot_*`, `gallery_fn_CS_*`, `gallery_pair_MB_as_CFO_*`, `gallery_pair_SC_as_SP_*`, `gallery_fp_HB_*`; tablolar: `size_ap_*`, `size_recall_*`, `iou_*`, `confusion_pairs_*`.

### Çözünürlük deneyi: yolo11s_320_s0 vs yolo11s_640_s0 — test (2026-10-20; `reports/tables/resolution_comparison_640_vs_320_test.csv`, `analysis.json` her iki run için)
- **B9 Çözünürlüğün toplam etkisi küçük.** 320 px test'te AP50 **0,833** (640: 0,847; −1,5), AP50:95 0,503 (−0,3), AP75 0,521 (−0,4). İç doğrulamada sıralama tersti (320: 0,865, 640: 0,860). Yani 226 px görüntüyü 640'a büyütmek 320'ye göre ~1,5 AP50 puanı kazandırıyor; bu fark tek tohumlu deneyde gürültü sınırına yakın. Önerideki "büyütme küçük hatalar için şart" varsayımı **bu ölçekte doğrulanmadı**: 226→320 zaten yeterli, 320→640 marjinal.
- **B10 Etki sınıfa göre zıt yönlü.** Kenar ayrıntısı gerektiren küçük sınıflar 320'de kaybediyor: MB **−5,0** AP50 (0,842→0,792), OP −2,6, SH −2,4, BMFO −2,4, SP −2,1. Buna karşılık SC +2,5 ve CS +0,3; **APl +12,0** (0,605→0,725) ve büyük CS'te AP50:95 0,619→0,678: düşük çözünürlükte uzun çizikler daha az parçalanıyor (alıcı alan çiziğe göre daha büyük) → B2'deki "kutu granülerliği" sorunu hafifliyor. HB etkilenmiyor (0,978).
- **B11 Sabit eşikte duyarlılık düşüyor, AP'de görünmüyor.** conf 0,35'te 320 px 861 kutu kaçırıyor (640: 766), küçüklerde 678 vs 559; ama APs 0,422 (640: 0,402) daha yüksek. AP sıralamayı ödüllendirir, sabit eşik kullanılabilirliği: fabrikada 320 px modeli aynı eşikte daha çok küçük hata kaçırır.
- **B12 Hız.** 320 px: 5,4 GFLOPs (640: 21,6, 4×) ve 3060'ta 17,7 ms/görüntü (640: 25,6; 1,45×; ön/son işleme sabit maliyet). Doğruluk–hız tablosu için.
- **Sınırlılıklar:** tek tohum; 320 px yerelde ısıl kısıtlama altında eğitildi (sonucu değil süreyi etkiler); val_internal→test düşüşü 320'de daha büyük (−3,2 vs −1,3) → 320 modelinin iç doğrulamaya daha çok uyduğu ya da tohum gürültüsü; ayırt etmek için tohum tekrarı gerekir.
