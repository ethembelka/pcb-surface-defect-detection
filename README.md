# PCB Surface Defect Detection (DsPCBSD+)

DsPCBSD+ veri setinde YOLO11 ile baskılı devre kartı yüzey hatası tespiti ve hata türü/boyutu analizi. (CMPE 7144 dönem projesi) Ayrıntılar için `CLAUDE.md`.

## Kurulum (yerel)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Veri
1. https://doi.org/10.6084/m9.figshare.24970329 adresinden `DsPCBSD+.zip` dosyasını indir → `data/raw/`
2. Zip'i `data/DsPCBSD/` içine aç.

## Colab
`notebooks/colab_train.ipynb` defterini Colab'da aç. Defter kodu GitHub'dan çeker, veriyi Drive'dan kopyalar,
çıktıları Drive'a kaydeder.

## Yapı
```
configs/    YAML ayarları (yol, model, eğitim, değerlendirme) — kodda sabit değer yok
src/pcbdet/  iş mantığı (veri, model, değerlendirme, raporlama)
scripts/    ince komut satırı girişleri
notebooks/  Colab defteri
docs/       mimari, değerlendirme protokolü, deney kaydı
tests/      metrik testleri (pytest)
outputs/    çalıştırma çıktıları (git'e girmez)
reports/    rapora girecek tablo ve şekiller
.claude/skills/  Claude Code skill'leri
```

## Claude Code ile çalışma
Bu klasörü Claude Code ile aç; `CLAUDE.md` ve `.claude/skills/` otomatik okunur.
Örnek istekler: "YOLO11s eğitimini Colab için hazırla", "son modeli değerlendir", "hata analizi yap", "deneyi kaydet".
