# pcb-surface-defect-detection

CMPE 7144 Digital Image Processing dönem projesi (İstanbul Üniversitesi, 2026 Güz).
Öğrenci: Ethem Belka Şahin · Hoca: Dr. Öğr. Üyesi Abbas Memiş · **Yanıtları Türkçe ver.** Kod, değişken adları ve yorumlar İngilizce.

## Proje tek cümlede
DsPCBSD+ veri setinde (Lv vd., *Scientific Data* 2024) baskılı devre kartı yüzey hatalarını **güncel bir dedektörle (YOLO11)**
tespit etmek, sonucu makaledeki referanslarla karşılaştırmak ve başarıyı **hata türü, hata boyutu ve sınıflar arası karışıklık**
açısından ayrıntılı analiz etmek. Hoca "uygulama değil bilimsel proje" istiyor; amaç bir makaleye dönüşebilecek temiz bir çalışma.

## Kısıtlar (önemli)
- **Eğitim sayısı sabit değil; zamana göre planlanır.** Önce ana model (YOLO11s, 640) eğitilir ve süresi ölçülür.
  Sonraki eğitimler bu süreye ve teslim tarihine göre aşağıdaki öncelik sırasıyla eklenir.
- Her yeni eğitimden önce: tahmini süreyi hesapla, `docs/EXPERIMENTS.md`'deki plana yaz, kullanıcıya sor.
- Yeniden eğitim gerektirmeyen analizler (boyut, karışıklık, hata galerisi) her eğitilmiş modelde ücretsizdir; önce onları tüket.

## Eğitim planı (öncelik sırası)
1. YOLO11s, 640, tohum 0 — ana sonuç (zorunlu)
2. YOLO11s, tohum 1 ve 2 — ortalama ± std (makale için güçlü öneri)
3. Model boyutu: YOLO11n ve YOLO11m — doğruluk / hız ödünleşimi
4. Girdi boyutu: 320 ve 1024 — küçük nesneler üzerine etkisi (görüntüler 226×226)
5. Zaman kalırsa: farklı bir aile (ör. RT-DETR) ile karşılaştırma
- Donanım: RTX 3060 Laptop **6 GB VRAM**, 16 GB RAM. Ana eğitim ortamı **Google Colab** (notebooks/colab_train.ipynb).

## Araştırma soruları
1. YOLO11, resmi ayrımda referanslara (YOLOv6-L6 / Co-DETR) göre nerede?
2. Başarı hata türüne ve boyutuna (COCO small/medium/large) göre nasıl değişiyor?
3. Hangi sınıflar karışıyor (SP–MB–SC, CS–CFO) ve görüntü düzeyinde neden?
4. Tohumlar arası dalgalanma (ortalama ± std) ve model/girdi boyutunun etkisi.

## Belgeler (kod yazmadan önce oku)
- `docs/ARCHITECTURE.md` → modüller, sorumluluklar, arayüzler, kod standartları
- `docs/EVALUATION_PROTOCOL.md` → ayrımlar, metrikler, referans sonuçlar, raporlama biçimi
- `docs/EXPERIMENTS.md` → deney kaydı (her deney buraya işlenir)

## Skills (.claude/skills/)
| Skill | Ne zaman |
|---|---|
| `train-detector` | Yerelde ya da Colab'da eğitim başlatırken / devam ettirirken |
| `evaluate-detector` | Eğitilmiş bir modeli protokole göre değerlendirirken |
| `error-analysis` | Sınıf/boyut/karışıklık analizi ve hata galerileri üretirken |
| `log-experiment` | Her deneyden sonra kaydı güncellerken |

## Çalışma kuralları
- Yol, eşik, hiperparametre **koda gömülmez**; `configs/` altındaki YAML'lardan okunur.
- `scripts/` sadece ince komut satırı girişleridir; iş mantığı `src/pcbdet/` içindedir.
- Her çalıştırma `outputs/results/<run_id>/` altına `config.yaml`, `metrics.json` ve git commit hash'ini yazar.
- Metrik kodu değişirse `tests/` altındaki testler çalıştırılır (`pytest -q`).
- Veri ve model ağırlıkları git'e girmez (`.gitignore`).

## Veri yerleşimi
```
data/raw/DsPCBSD+.zip          # indirilen zip (git'e girmez)
data/DsPCBSD/DsPCBSD+/
  Data_YOLO/{images,labels}/{train,val}   # YOLO eğitimi bunu kullanır
  Data_COCO/annotations/instances_{train,val}2017.json   # COCO değerlendirmesi (pycocotools) için gerçek etiketler
  Data_COCO/{train2017,val2017}/
  Hash.py                                  # yazarların benzer görüntü ayıklama betiği; ÇALIŞTIRMA (dosya taşır)
```
Klasör adı farklı çıkarsa `configs/dspcbsd.yaml` içindeki `path` güncellenir. Colab için aynı zip Drive'da `MyDrive/CMPE7144_PCB/data/` altına.

## Veri
DsPCBSD+: 10.259 görüntü (226×226), 20.276 kutu, 9 sınıf: SH, SP, SC, OP, MB, HB, CS, CFO, BMFO.
Resmi ayrım train 8.208 / val 2.051, ayrı test yok. %67 küçük nesne. CC BY 4.0.
İndirme: https://doi.org/10.6084/m9.figshare.24970329 → `data/raw/`, açılmış hali `data/DsPCBSD/`.
Sınıf sırası doğrulandı: YOLO id = COCO category_id − 1. Kutu sayıları train 16.184 / val 4.092; en az SH (746), en çok SP (3.655).

## Teslimler
Öneri 18 Ekim 2026 · Teslim 13 Aralık 2026 (IEEE rapor, sunum, kod; zip: `CMPE7144_2026_fall_project_studentID_name_surname.zip`) · Sunum 16 Aralık 2026.
Araştırma makaleleri: `../../03_Research_Paper/` (Lv 2024, Wang 2025 PCB-FS, Jin 2026 PCB-MMF, Xia 2026 VisDefectNet).
