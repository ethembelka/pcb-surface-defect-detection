# Yol Haritası
_Oluşturma: 2026-10-18 (cc). Aşama bitince kutuyu işaretle; güncel durum `docs/STATUS.md`'de._

Hedef: öneride verilen sözleri ("Where Do Detectors Fail on Small PCB Defects?") eksiksiz yerine getirmek.
Not ağırlığı: uygulama %50 · rapor %25 · sunum %25.

## Takvim
| Hafta | Tarih | Aşama | Sabit tarih |
|---|---|---|---|
| 0 | 18 Eki | Öneri teslimi | **18 Eki öneri** |
| 1 | 19–25 Eki | A1 Altyapı + ayrım + duman testi | |
| 2 | 26 Eki–1 Kas | A2 Ana eğitim · A3 değerlendirme kodu (paralel) | **1 Kas araştırma önerisi** (ayrı ödev) |
| 3–4 | 2–15 Kas | A3 Değerlendirme + A4 hata analizi · A5 ek eğitimler başlar | |
| 5 | 16–22 Kas | A5 ek eğitimler · bulgular kesinleşir | |
| 6–7 | 23 Kas–6 Ara | A6 Rapor + şekiller · **6 Ara kod dondurma** | |
| 8 | 7–13 Ara | Rapor son okuma, sunum, zip | **13 Ara teslim 23:59** |
| 9 | 14–16 Ara | Sunum provası | **16 Ara sunum** |

## Eksikler — nerede, ne yapılacak
| # | Eksik | Nerede | Ne yapılacak | Aşama |
|---|---|---|---|---|
| E1 | Hiç kod yok | `src/pcbdet/`, `scripts/`, `tests/` | ARCHITECTURE.md'deki modülleri sırayla yaz | A1–A4 |
| E2 | `val_internal` kurulmamış; eğitim resmi val'e bakıyor | `configs/dspcbsd.yaml`, `src/pcbdet/data/splits.py` | Train'i %90/%10 böl (tohum 0, sınıf varlığına göre dengeli) → `data/splits/{train_fit,val_internal}.txt`; eğitim yaml'ı `train: train_fit.txt`, `val: val_internal.txt`. Resmi val sadece `configs/dspcbsd_test.yaml` ile | A1 |
| E3 | pycocotools `image_id` eşlemesi yok (dosya adları `Y_011791.jpg`, COCO id sayısal) | `models/ultralytics_detector.py → export_predictions` | Dosya adı → COCO id sözlüğü `instances_val2017.json`'dan; test yaz | A3 |
| E4 | Colab defteri ayarları elle yazılı | `notebooks/colab_train.ipynb` | `scripts/train.py --config configs/train/<x>.yaml` çağırsın; ayar tek kaynakta | A1 |
| E5 | Ultralytics sürümü sabit değil (yerel 8.3.101) | `requirements.txt` | `ultralytics==8.3.101`; Colab da aynı sürümü kursun | A1 |
| E6 | "226×226" ifadesi eksik: 111 görüntü 108×108 | `CLAUDE.md`, rapor | İfadeyi düzelt; boyut analizinde ayrıca belirt | A1 |
| E7 | `docs/EXPERIMENTS.md`'de "Bulgular" bölümü yok (error-analysis skill'i oraya yazıyor) | `docs/EXPERIMENTS.md` | Başlığı ekle | A1 |
| E8 | IEEE rapor şablonu ve sunum iskeleti yok | `reports/paper/` (yeni) | IEEEtran şablonu; bölüm başlıkları öneriden | A6 |
| E9 | Hoca onayı bekleniyor | — | Öneri gönderildikten sonra cevabı DECISIONS.md'ye yaz | 0 |

## A1 — Altyapı (19–25 Eki)
1. `config.py` (YAML → dataclass), `data/classes.py`, `utils/{io,seed,run}.py` (run_id + git hash).
2. `data/splits.py` + `scripts/make_splits.py` → E2. `tests/test_splits.py`: kesişim yok, toplam 8.208, her sınıf iki tarafta da var.
3. `data/stats.py` + `scripts/dataset_stats.py`: sınıf × boyut tablosu (rapor Tablo 1 olur).
4. `models/base.py`, `models/ultralytics_detector.py` (train), `models/registry.py`, `engine/trainer.py`, `scripts/train.py`.
5. E4, E5, E6, E7.
6. **Duman testi (yerel):** 2 epoch, `fraction` küçük → hata yok, çıktılar `outputs/runs/<run_id>/`'de, **epoch süresi ölçüldü**.
- **Bitti sayılır:** `pytest -q` geçiyor; duman testi çalıştı; epoch süresi STATUS.md'de.
- **Karar kapısı:** epoch süresine göre ana eğitim yerelde mi Colab'da mı (6 GB'ta YOLO11s 640 için batch 16 civarı).

## A2 — Ana eğitim (26 Eki–1 Kas)
- YOLO11s, 640, tohum 0, 100 epoch, patience 30; seçim `val_internal`'da. Tahmini süre A1 ölçümünden hesaplanır, kullanıcıya sorulur.
- **Bitti sayılır:** `best.pt` + eğitim eğrileri; toplam süre EXPERIMENTS.md'de.
- **Karar kapısı:** toplam süre × kalan haftalar → A5'te hangi ek eğitimlerin sığacağı.

## A3 — Değerlendirme (A2 sürerken yazılır; 2–8 Kas'ta koşulur)
1. `export_predictions` (E3), `evaluation/coco_eval.py`, `per_class.py` (F1-maks eşik), `confusion.py` (arka plan dahil), `efficiency.py`, `evaluator.py`, `scripts/evaluate.py`.
2. Testler: id eşlemesi, F1-maks eşik seçimi, karışıklık eşlemesi (elle hesaplanmış küçük örnekler).
3. Resmi val (test) üzerinde **bir kez** koş → `metrics.json`; `reporting/tables.py` → referanslarla ana tablo.
- **Bitti sayılır:** AP50, AP50:95, APs/m/l, sınıf bazlı P/R/F1, karışıklık matrisi, FPS hazır; sayılar Ultralytics'in kendi `val` çıktısıyla kabaca tutarlı (büyük fark varsa E3'ü kontrol et).

## A4 — Hata analizi (9–15 Kas; yeniden eğitim yok)
- Sınıf × boyut AP; SP–MB–SC ve CS–CFO karışma oranları; sınıf başına FN/FP galerileri; IoU dağılımı (AP50 ile AP50:95 farkı).
- Her bulgu: sayı + görüntü örneği → EXPERIMENTS.md "Bulgular".
- **Bitti sayılır:** Q2 ve Q3 sayılarla cevaplandı; rapora girecek şekiller `reports/figures/`'da.

## A5 — Ek eğitimler (Kasım; A2 süresine göre seçilir)
Öncelik: (1) tohum 1, 2 → ortalama ± std · (2) girdi boyutu 320 / 1024 → Q4 · (3) YOLO11n / m · (4) RT-DETR.
Her biri için: süre tahmini → STATUS/EXPERIMENTS → kullanıcı onayı → eğitim → A3 değerlendirmesi (otomatik) → kayıt.
- **Bitti sayılır:** en az (1) tamam; öneride söz verilen "subset" anlamlı bir sonuç veriyor.

## A6 — Rapor ve sunum (23 Kas–13 Ara)
- IEEE çift sütun: Giriş · İlgili çalışmalar (4 makale) · Veri (sınıf tablosu, ayrım) · Yöntem · Deney düzeni · Sonuçlar (ana tablo, boyut, karışıklık, ek eğitimler) · Tartışma (nerede başarısız, neden) · Sınırlılıklar · Sonuç.
- Sınırlılıklarda: referanslarla çözünürlük ve model seçim farkı; literatürdeki farklı ayrımlar; 108 px görüntüler.
- **6 Ara kod dondurma:** README'de yeniden üretim adımları, testler geçiyor, `git tag v1.0`.
- Zip: `CMPE7144_2026_fall_project_<studentID>_Ethem_Belka_Sahin.zip` → rapor PDF + sunum PDF + kaynak kod (veri yok, 128 MB > 5 MB; DOI bağlantısı README'de).

## Riskler
| Risk | Önlem |
|---|---|
| Colab oturumu kopar / kota biter | Checkpoint Drive'da, `resume`; yerel 3060 yedek |
| Ek eğitimlere zaman yetmez | A2 sonrası karar kapısı; tohum tekrarları öncelikli, RT-DETR isteğe bağlı |
| Sonuç referansların altında kalır | Bu bir başarısızlık değil: proje sorusu "nerede başarısız" — analiz derinliği asıl katkı |
| Test setine dolaylı bakma | Tüm seçimler `val_internal`'da; resmi val her model için bir kez |
