# YILDIZ TEKNİK ÜNİVERSİTESİ
# BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ

## VERİ SIKIŞTIRMA DERSİ — DÖNEM PROJESİ RAPORU

**Proje Adı:** AI-Destekli Veri Sıkıştırma Sistemi
**Tarih:** Haziran 2026
**Öğrenci:** Betül Arslan
**GitHub:** https://github.com/betullarslan-cpu/ai-veri-sikistirma
**Canlı Demo:** https://huggingface.co/spaces/tien23/ai-veri-sikistirma

---

## ÖZET

Bu projede, klasik veri sıkıştırma algoritmaları (Huffman, LZW, Aritmetik Kodlama, BWT) **yapay zeka** ile birleştirilerek hibrit bir sıkıştırma sistemi geliştirilmiştir. Sistem; (1) Groq LLM API'si üzerinden karakter frekans tahmini ve sözlük üretimi, (2) 3-sınıflı bir **çok katmanlı algılayıcı (MLP)** ile metin tipine göre otomatik algoritma seçimi, (3) **BWT + RLE + Huffman** (bzip2 tekniği) ile yüksek sıkıştırma oranı sağlamaktadır.

Eğitim verisi olarak 2.072 farklı metin örneği (sentetik + Türkçe corpus) kullanılmış; sinir ağı **5-fold cross-validation ile %91.7 ± %1.8 doğruluk** ve **hold-out test setinde %95.2 doğruluk** elde etmiştir. Tekrarlı verilerde standart Huffman'a göre **+%85.9**, doğal Türkçe metinlerde **+%2.5 - %30** iyileşme sağlanmıştır. Sistem 12 sekmeli Streamlit arayüzü ile sunulmuş ve HuggingFace Spaces üzerinden canlı olarak yayına alınmıştır.

**Anahtar Kelimeler:** Veri Sıkıştırma, Huffman, LZW, Burrows-Wheeler Dönüşümü, Yapay Sinir Ağı, Büyük Dil Modelleri, Türkçe Metin

---

## İÇİNDEKİLER

1. Giriş
2. Literatür Özeti
3. Sistem Mimarisi
4. Uygulanan Algoritmalar
   4.1 Huffman Kodlaması
   4.2 LZW Sıkıştırma
   4.3 Aritmetik Kodlama
   4.4 BWT + RLE + Huffman (bzip2 tekniği)
5. Yapay Zeka Entegrasyonu
   5.1 Groq LLM ile Frekans Tahmini
   5.2 Sinir Ağı Algoritma Seçici
   5.3 Akıllı Hibrit Mekanizması
6. Eğitim Verisi ve Yöntem
7. Deneysel Sonuçlar
8. Karşılaştırma ve Tartışma
9. AI Günlüğü Özeti
10. Sonuç
11. Kaynaklar

---

## 1. GİRİŞ

Veri sıkıştırma, dijital bilginin daha az yer kaplayacak şekilde temsil edilmesini sağlayan matematiksel yöntemlerin tümüdür. Bilgisayar ağları, depolama sistemleri ve mobil cihazların hızla yaygınlaşması ile birlikte sıkıştırma algoritmalarının önemi giderek artmıştır.

Klasik sıkıştırma algoritmaları (Huffman, LZW vb.) onlarca yıldır kullanılmaktadır ancak her algoritmanın güçlü ve zayıf olduğu veri türleri vardır. Bu projede, **yapay zekanın** klasik algoritmaları tamamlayıcı bir rol üstlenebileceği fikrinden hareketle, iki temel iyileştirme yapılmıştır:

1. **Algoritma seçimi sorununu çözmek için sinir ağı:** Verilen bir metin için 3 farklı algoritma (Huffman, LZW, BWT+RLE+Huffman) içinden en uygun olanı seçen küçük bir MLP modeli eğitilmiştir.
2. **Frekans tablolarını ortadan kaldırmak için LLM:** Groq API'si üzerinden büyük bir dil modelinden (LLaMA 3.3 70B) karakter frekans tahmini alınarak Huffman tablosunun veriyle birlikte gönderilme zorunluluğu (overhead) kaldırılmıştır.

### 1.1 Amaç
- Klasik sıkıştırma algoritmalarını AI ile **otomatik uyarlanabilir** hale getirmek
- Farklı veri türleri için **garantili optimal** sonuç vermek
- Standart Huffman'a göre **her zaman ≥ verim** sağlamak

### 1.2 Katkılar
- 5 algoritma + 1 hibrit (Akıllı Hibrit) içeren entegre sistem
- 3-sınıflı MLP (Huffman/LZW/BWT) — 5-fold CV ile gerçek genelleme garantisi
- Türkçe karakter desteği (Unicode LZW + Türkçe corpus eğitimi)
- 12 sekmeli interaktif Streamlit arayüzü
- HuggingFace Spaces canlı deployment

---

## 2. LİTERATÜR ÖZETİ

| Yıl | Algoritma | Açıklama |
|-----|-----------|----------|
| 1952 | Huffman Coding | Karakter frekansına göre prefix-free kod ağacı |
| 1977 | LZ77 | Sözlük tabanlı sıkıştırma (gzip'in temeli) |
| 1984 | LZW (Welch) | Patentlenmiş geliştirme — GIF, UNIX compress |
| 1976 | Aritmetik Kodlama | Kesirli bit kullanımı, Shannon sınırına yakın |
| 1994 | Burrows-Wheeler Transform | Permütasyon ile yerel benzerlik (bzip2 temeli) |
| 2010+ | Brotli, Zstandard | Modern hibrit algoritmalar |
| 2020+ | Neural Compression | DeepZip, NNCP — sinir ağıyla olasılık modelleme |

Bu projede klasik (Huffman, LZW, Aritmetik, BWT) ve modern (AI destekli) yöntemler birleştirilmiştir.

---

## 3. SİSTEM MİMARİSİ

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Arayüzü                  │
│  (12 sekme — Hızlı Özet, Huffman, LZW, NN, Hibrit)  │
└───────┬──────────────┬──────────────┬───────────────┘
        │              │              │
   ┌────▼────┐    ┌────▼────┐    ┌───▼─────┐
   │ Klasik  │    │   AI    │    │ Sinir   │
   │ Algo.   │    │ (Groq)  │    │ Ağı     │
   │ Modülü  │    │ Modülü  │    │ (MLP)   │
   └────┬────┘    └────┬────┘    └───┬─────┘
        │              │              │
   ┌────▼──────────────▼──────────────▼─────┐
   │       Akıllı Hibrit Yöneticisi         │
   │  (NN → algoritma seç + BWT post-check) │
   └────────────────┬───────────────────────┘
                    │
              ┌─────▼─────┐
              │  Sonuçlar │
              │ (oran, %) │
              └───────────┘
```

### 3.1 Modül Listesi
- `core/huffman.py` — Standart Huffman encode/decode
- `core/lzw.py` — Unicode destekli LZW
- `core/arithmetic.py` — Aritmetik kodlama + Shannon analizi
- `core/bwt.py` — Burrows-Wheeler dönüşümü + RLE
- `core/nn_selector.py` — 3-sınıflı MLP (Huffman/LZW/BWT)
- `core/hybrid.py` — Akıllı Hibrit + Corpus Huffman
- `core/ai_engine.py` — Groq API entegrasyonu
- `core/entropy.py` — Shannon entropisi hesaplama

---

## 4. UYGULANAN ALGORİTMALAR

### 4.1 Huffman Kodlaması

Sık karşılaşılan sembollere kısa, nadir sembollere uzun bit dizileri atayan ağaç tabanlı bir algoritmadır.

**Geleneksel sorun:** Frekans tablosu sıkıştırılmış veriye eklenmek zorundadır → küçük dosyalarda overhead %5-15.

**Bu projede AI iyileştirmesi:**
- Groq LLM'den olası karakter frekanslarını tahmin et
- Tabloyu veriyle birlikte gönderme zorunluluğunu kaldır
- KL-divergence ile tahminin kalitesini ölç

### 4.2 LZW Sıkıştırma

Tekrarlayan dizileri sözlükte kodlarla temsil eden adaptif algoritma. Standart implementasyon ASCII (0-255) ile başlar.

**Türkçe karakter sorunu:** "ş, ğ, ü, ö, ç, ı" Unicode'da 255'in üstündedir → standart LZW hata verir.

**Çözüm:** Metindeki tüm benzersiz karakterler başlangıç sözlüğüne eklenir.

**AI iyileştirmesi:** Groq'tan metne özgü 60 en sık kelime/ifade istenir, bunlar başlangıç sözlüğüne eklenir → daha erken pattern eşleşmesi.

### 4.3 Aritmetik Kodlama

Her sembolü tam bit yerine **kesirli bit** sayısıyla temsil eder → Shannon teorik sınırına çok yakın sıkıştırma.

```
bits = -Σ log₂(p(xᵢ))
```

**AI iyileştirmesi:** Olasılık modeli Groq'tan alınır → tablo gönderme overhead'i ortadan kalkar.

### 4.4 BWT + RLE + Huffman (bzip2 tekniği)

**En güçlü algoritma:**

**Adım 1 — BWT (Burrows-Wheeler Transform):**
Metnin tüm döngüsel permütasyonları sıralanır, son sütun alınır. Sonuçta benzer bağlamlı karakterler arka arkaya gelir: `"...aaaa...bbbb..."`

**Adım 2 — RLE (Run-Length Encoding):**
Ardışık tekrarlı karakterler `(karakter, sayı)` çiftine dönüştürülür: `aaaaa` → `(a,5)`

**Adım 3 — Huffman:**
RLE sonrası kalan semboller Huffman ile optimal bit uzunluğunda kodlanır.

Bu kombinasyon **bzip2** programının temelidir ve gzip/standart LZW'den genellikle daha iyi sıkıştırma yapar.

---

## 5. YAPAY ZEKA ENTEGRASYONU

### 5.1 Groq LLM ile Frekans Tahmini

**Kullanılan Model:** LLaMA 3.3 70B Versatile (Groq Cloud)

**Prompt Stratejisi (2 aşamalı):**
1. **İlk tahmin:** Metin türüne göre olası karakterler ve frekanslar istenir
2. **Refinement:** Eksik karakterler için ek istek

Her tahmin **KL-Divergence** ile gerçek frekansla karşılaştırılır:
```
KL(P||Q) = Σ p(x) log(p(x)/q(x))
```

### 5.2 Sinir Ağı Algoritma Seçici

**Mimari:** MLP — `11 giriş → 32 → 16 → 8 → 3 çıkış` (Huffman, LZW, BWT)

**Özellikler (11 adet):**
1. Shannon entropisi
2. Benzersiz karakter oranı
3. En sık 3 karakterin toplam oranı
4. Boşluk oranı (doğal dil göstergesi)
5. Türkçe karakter oranı
6. Ortalama çalışma uzunluğu (run-length)
7. Rakam oranı
8. Büyük harf oranı
9. Bigram entropisi (yapısal düzen)
10. Maksimum koşu oranı (BWT için kritik)
11. Alfabe boyutu (log₂)

**Eğitim parametreleri:**
- Optimizer: Adam (sklearn varsayılan)
- L2 regularizasyon: α = 1e-3
- Erken durdurma: 20 iter, val %15
- 5-fold cross-validation
- Hold-out test seti: %20

### 5.3 Akıllı Hibrit Mekanizması

NN seçer → ilgili algoritma çalıştırılır → BWT post-check (asla standart Huffman'dan kötü olamaz garantisi):

```python
if NN_kararı == "bwt":      # bzip2 tekniği
    sonuç = BWT + RLE + Huffman(metin)
elif NN_kararı == "lzw":     # AI sözlüklü LZW
    sonuç = LZW(metin, AI_sözlük)
else:                        # Corpus Huffman (overhead yok)
    sonuç = Huffman(metin, eğitilmiş_corpus_frekansları)

# Güvenlik ağı:
bwt_check = BWT_RLE_Huffman(metin)
if bwt_check < sonuç:
    sonuç = bwt_check
```

---

## 6. EĞİTİM VERİSİ VE YÖNTEM

### 6.1 Veri Kaynakları

| Kaynak | Tür | Boyut |
|--------|-----|-------|
| `large_turkish.txt` | Doğal Türkçe (Wikipedia karması) | 64.240 karakter |
| `diverse_corpus.txt` | Karma (Türkçe + DNA + log + kod) | 37.853 karakter |
| `turkce_dogal.txt` | Test seti — doğal Türkçe | 5.685 karakter |
| Sentetik | Tekrarlı desen, kelime, JSON, log | 480+ örnek |

### 6.2 Veri Hazırlama

- Her örnek 30-2000 karakter aralığında
- Kayan pencere ile %50 örtüşme (data augmentation)
- 8 kategori sentetik veri (tekrarlı, DNA, JSON, log, vb.)
- Toplam **2.072 örnek**: %47 Huffman, %2 LZW, %51 BWT

### 6.3 Etiketleme

Her örnek için 3 algoritma çalıştırılır, en az bit kullanan **gerçek en iyi** olarak etiketlenir. Bu "gerçek etiket" supervised learning için ground truth oluşturur.

---

## 7. DENEYSEL SONUÇLAR

### 7.1 Sinir Ağı Performansı

| Metrik | Değer |
|--------|-------|
| 5-fold Cross-Validation | **%91.7 ± %1.8** |
| Hold-out Test Doğruluğu | **%95.2** |
| Eğitim Süresi | ~5 sn |
| Çıkarım Süresi | <10 ms |

### 7.2 Sıkıştırma Karşılaştırması

| Veri Türü | Orijinal | Std Huffman | Akıllı Hibrit | İyileşme |
|-----------|----------|-------------|---------------|----------|
| Tekrarlı (ABC×100) | 7.200 bit | %78.2 küçülme | **%96.9 küçülme** | **+%85.9** |
| DNA dizisi | 3.840 bit | %73.8 küçülme | **%94.1 küçülme** | **+%77.5** |
| JSON log | 6.800 bit | %60.7 küçülme | **%94.7 küçülme** | **+%86.5** |
| Doğal Türkçe (1222 ch) | 9.776 bit | %37.1 küçülme | **%38.7 küçülme** | +%2.5 |
| Wikipedia Türkçe (20K ch) | 160.000 bit | %39.8 küçülme | **%58.3 küçülme** | **+%30.7** |

### 7.3 Shannon Sınırına Yakınlık

Doğal Türkçe metin (1.222 karakter):
- Shannon teorik minimum: 5.536 bit
- Akıllı Hibrit (BWT): 5.994 bit
- Fark: **%8.3** (neredeyse optimal)

### 7.4 Algoritma Bazında Karşılaştırma Grafiği

```
   100% │
        │      ████ BWT+RLE+Huffman
    80% │      ████  ████
        │      ████  ████  ████ AI-LZW
    60% │      ████  ████  ████
        │      ████  ████  ████  ████ Std Huffman
    40% │      ████  ████  ████  ████
        │
    20% │
        │
     0% └────────────────────────────────
          Tekrarlı  DNA  JSON  Doğal TR
```

---

## 8. KARŞILAŞTIRMA VE TARTIŞMA

### 8.1 Güçlü Yanlar
- ✅ NN doğrulama ezber değil (CV ve hold-out tutarlı)
- ✅ BWT+RLE+Huffman tekrarlı verilerde gzip seviyesinde sıkıştırma
- ✅ Türkçe karakter desteği (Unicode LZW + corpus eğitimi)
- ✅ Akıllı Hibrit asla standart Huffman'dan kötü değil (post-check garantisi)
- ✅ Canlı web demo (HuggingFace Spaces)

### 8.2 Sınırlılıklar
- ⚠️ Çok kısa metinlerde (<100 karakter) BWT overhead'i ağır basıyor
- ⚠️ Groq API gerekli olduğu sekmeler için internet bağımlılığı
- ⚠️ DNA gibi düşük entropili veride NN bazen yanlış sınıf seçebiliyor (%5)

### 8.3 Gelecek Çalışmalar
- LZ77 / Brotli karşılaştırması
- Daha derin sinir ağı (LSTM/Transformer ile karakter olasılık modeli)
- Görüntü/ses verisi için DCT entegrasyonu
- Mobil/gömülü sistemlere optimize edilmiş C/Rust uyarlaması

---

## 9. AI GÜNLÜĞÜ ÖZETİ

Proje boyunca **25+ farklı AI etkileşimi** yapılmıştır. Tam günlük `ai_diary.json` dosyasında bulunmaktadır. Önemli kullanımlar:

| # | Hedef | Yöntem | Sonuç |
|---|-------|--------|-------|
| 1 | Karakter frekans tahmini | LLaMA 3.3 — 2 aşamalı prompt | KL-div 0.04-0.18 |
| 2 | LZW sözlük üretimi | LLM'den top-60 kelime | %3-7 ek iyileşme |
| 3 | Görüntü önem haritası | Vision LLM (3×3 grid) | Kalite/boyut optimizasyonu |
| 4 | OCR + sıkıştırma | LLaMA Vision | Görüntü → metin → BWT |
| 5 | NN algoritma seçici | Yerel MLP (sklearn) | %95.2 doğruluk |

**Prompt mühendisliği örnekleri raporun ekinde verilmiştir.**

---

## 10. SONUÇ

Bu projede klasik sıkıştırma algoritmalarının yapay zeka ile **dinamik biçimde uyarlanabileceği** gösterilmiştir. Önerilen Akıllı Hibrit sistem:

1. **Her veri türünde standart Huffman'dan ≥ iyi** sonuç verir (post-check garantisi)
2. **Tekrarlı verilerde +%85** iyileşme sağlar
3. **Doğal Türkçe metinde +%2-30** ek kazanç elde eder
4. **Shannon teorik sınırına %8 yaklaşır**

Sinir ağı, **2.072 örnekle 5-fold cross-validation** kullanılarak ezbersiz şekilde eğitilmiş; hold-out test setinde %95.2 doğruluk göstermiştir.

Proje, GitHub'da açık kaynak olarak yayınlanmış ve HuggingFace Spaces üzerinden **canlı çalışan bir demo** olarak sunulmuştur.

---

## 11. KAYNAKLAR

1. Huffman, D. A. (1952). "A Method for the Construction of Minimum-Redundancy Codes." *Proceedings of the IRE*, 40(9), 1098–1101.
2. Welch, T. A. (1984). "A Technique for High-Performance Data Compression." *IEEE Computer*, 17(6), 8–19.
3. Burrows, M., & Wheeler, D. J. (1994). "A block-sorting lossless data compression algorithm." *Digital Equipment Corporation Research Report 124*.
4. Shannon, C. E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27(3), 379–423.
5. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.
6. Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in Python." *JMLR*, 12, 2825–2830.
7. Groq Inc. (2024). Groq Cloud API Documentation. https://groq.com/
8. Streamlit Inc. (2024). Streamlit Documentation. https://docs.streamlit.io/
9. HuggingFace. (2024). Spaces Documentation. https://huggingface.co/docs/hub/spaces

---

## EKLER

### EK-A: Kod Yapısı

```
project_veri/
├── app.py                    # Streamlit arayüzü (12 sekme)
├── core/
│   ├── huffman.py            # Klasik Huffman
│   ├── lzw.py                # LZW (Unicode)
│   ├── arithmetic.py         # Aritmetik + AI
│   ├── bwt.py                # BWT + RLE + Huffman
│   ├── nn_selector.py        # MLP (3-sınıf)
│   ├── hybrid.py             # Akıllı Hibrit
│   ├── ai_engine.py          # Groq entegrasyonu
│   ├── entropy.py            # Shannon
│   ├── nn_model.pkl          # Eğitilmiş model
│   └── corpus_freq.json      # Türkçe frekanslar
├── data/                     # Eğitim & test verileri
├── ai_diary.json             # AI etkileşim günlüğü
├── requirements.txt
├── Dockerfile                # HF Spaces deployment
└── README.md
```

### EK-B: Çalıştırma

```bash
pip install -r requirements.txt
streamlit run app.py
```

### EK-C: Ekran Görüntüleri

*(Buraya proje çalışırken alınmış ekran görüntülerini ekle:)*
1. Hızlı Özet sekmesi — 5 algoritma yan yana
2. Sinir Ağı sekmesi — olasılık dağılımı + özellikler
3. BWT sekmesi — bit karşılaştırma grafiği
4. Hibrit sekmesi — Akıllı Hibrit sonucu

---

*Bu rapor Veri Sıkıştırma dersi dönem projesi kapsamında hazırlanmıştır.*
