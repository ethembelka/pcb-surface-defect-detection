# pcb-surface-defect-detection

CMPE 7144 Digital Image Processing dönem projesi (İstanbul Üniversitesi, 2026 Güz).
Öğrenci: Ethem Belka Şahin · Hoca: Dr. Öğr. Üyesi Abbas Memiş · **Yanıtları Türkçe ver.** Kod, değişken adları ve yorumlar İngilizce.

## Proje tek cümlede
DsPCBSD+ veri setinde (Lv vd., *Scientific Data* 2024) baskılı devre kartı yüzey hatalarını **tek bir güncel dedektörle (YOLO11)**
tespit etmek, sonucu makaledeki referanslarla karşılaştırmak ve başarıyı **hata türü, hata boyutu ve sınıflar arası karışıklık**
açısından ayrıntılı analiz etmek. Hoca "uygulama değil bilimsel proje" istiyor; amaç bir makaleye dönüşebilecek temiz bir çalışma.

## Kısıtlar (önemli)
- **Çok sayıda model eğitme.** Ana deney: 1 model (YOLO11s). İsteğe bağlı: aynı modelin 3 tohumla tekrarı.
- Ek deneyler mümkün olduğunca **yeniden eğitim gerektirmeyen** (sadece test/inference) deneyler olmalı.
- Donanım: RTX 3060 Laptop **6 GB VRAM**, 16 GB RAM. Ana eğitim ortamı **Google Colab** (notebooks/colab_train.ipynb).

## Araştırma soruları
1. YOLO11, resmi ayrımda referanslara (YOLOv6-L6 / Co-DETR) göre nerede?
2. Başarı hata türüne ve boyutuna (COCO small/medium/large) göre nasıl değişiyor?
3. Hangi sınıflar karışıyor (SP–MB–SC, CS–CFO) ve görüntü düzeyinde neden?
4. (İsteğe bağlı) Tohumlar arası dalgalanma: ortalama ± std.

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

## Veri
DsPCBSD+: 10.259 görüntü (226×226), 20.276 kutu, 9 sınıf: SH, SP, SC, OP, MB, HB, CS, CFO, BMFO.
Resmi ayrım train 8.208 / val 2.051, ayrı test yok. %67 küçük nesne. CC BY 4.0.
İndirme: https://doi.org/10.6084/m9.figshare.24970329 → `data/raw/`, açılmış hali `data/DsPCBSD/`.
**`configs/dspcbsd.yaml` sınıf sırası, veri setindeki etiket indeksleriyle doğrulanmalı.**

## Teslimler
Öneri 18 Ekim 2026 · Teslim 13 Aralık 2026 (IEEE rapor, sunum, kod; zip: `CMPE7144_2026_fall_project_studentID_name_surname.zip`) · Sunum 16 Aralık 2026.
Araştırma makaleleri: `../../03_Research_Paper/` (Lv 2024, Wang 2025 PCB-FS, Jin 2026 PCB-MMF, Xia 2026 VisDefectNet).
