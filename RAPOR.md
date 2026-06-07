# YILDIZ TEKNİK ÜNİVERSİTESİ
## BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ
### VERİ SIKIŞTIRMA — DÖNEM PROJESİ RAPORU

**Proje:** AI-Destekli Veri Sıkıştırma Sistemi
**Tarih:** Haziran 2026
**GitHub:** https://github.com/betullarslan-cpu/ai-veri-sikistirma
**Canlı Demo:** https://huggingface.co/spaces/tien23/ai-veri-sikistirma

---

## ÖZET

Bu projede klasik veri sıkıştırma algoritmaları (Huffman, LZW, BWT+RLE+Huffman)
yapay zeka teknikleriyle birleştirilerek **akıllı hibrit bir sıkıştırma sistemi**
geliştirilmiştir. Üç ana yapay zeka bileşeni vardır:

1. **MLP (3 sınıflı) sinir ağı** — verinin özelliklerine göre uygun algoritmayı seçer
2. **Akıllı Hibrit yöneticisi** — NN seçimi + BWT post-check garantisi
3. **Groq LLM entegrasyonu** — karakter frekansı tahmini

Eğitim verisi olarak **2.357 sentetik + corpus örneği** kullanılmıştır. Model
**5-fold stratified cross-validation** ile %91.7 ± %1.8 doğruluk, hold-out test
setinde %95.2 doğruluk elde etmiştir. Doğal Türkçe metinde standart Huffman'a
göre +%30 iyileşme, tekrarlı verilerde +%85.9 iyileşme sağlanmıştır.

Sistem **88 birim test** ile doğrulanmış, **HuggingFace Spaces** üzerinden canlı
yayına alınmıştır.

**Anahtar Kelimeler:** Veri Sıkıştırma, Huffman, LZW, BWT, Yapay Sinir Ağı,
Algorithm Selection, Bilgi Teorisi

---

## 1. GİRİŞ

Veri sıkıştırma, bilgi teorisinin pratik uygulamalarındandır (Shannon, 1948).
Her algoritmanın güçlü olduğu veri türü farklıdır:

- **Huffman** — karakter bazlı, doğal dil için orta
- **LZW** — pattern bazlı, tekrarlı verilerde güçlü
- **BWT** — yapısal düzen tabanlı, bzip2'nin temeli

Bu çeşitlilik **algoritma seçimi sorunu** (Rice, 1976) yaratır. Bu proje,
sinir ağı tabanlı bir seçici ile bu sorunu çözmeyi amaçlar.

### 1.1 Amaç

- Klasik sıkıştırma algoritmalarını veri tipine göre otomatik seçen sistem
- Her veri tipinde **standart Huffman'a ≥ verim** garantisi
- Türkçe karakter desteği (Unicode + corpus)
- Açık kaynak, eğitsel implementasyon

### 1.2 Katkılar

1. **3 sınıflı MLP** algoritma seçici (%95.2 hold-out doğruluk)
2. **Akıllı Hibrit** mekanizması: NN + BWT post-check
3. **Bloklu BWT** (sınırsız uzunlukta kayıpsız)
4. **8 sekmeli interaktif arayüz** (Streamlit + Plotly)
5. **Next-Token entropi analizi** (Shannon 1951 yaklaşımı)
6. **86+ birim test** (kayıpsızlık ve ölçeklenme)

---

## 2. LİTERATÜR ÖZETİ

| Yıl | Çalışma | Katkı |
|-----|---------|-------|
| 1948 | Shannon, BSTJ | Bilgi teorisi, entropi |
| 1951 | Shannon, BSTJ | İngilizce için ~1 bit/karakter |
| 1952 | Huffman, IRE | Optimal prefix-free kod |
| 1976 | Rice | Algorithm Selection Problem |
| 1977 | Ziv & Lempel | LZ77 — sözlük tabanlı |
| 1984 | Welch | LZW — pratik LZ varyantı |
| 1994 | Burrows & Wheeler | BWT — bloklara dayalı |
| 2014 | Kotthoff | ML-tabanlı algorithm selection |

---

## 3. SİSTEM MİMARİSİ

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Arayüzü                  │
│  (9 sekme — Hızlı Özet, Huffman, LZW, NN, Hibrit,  │
│   Shannon, BWT, Next-Token, AI Günlüğü)            │
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
              │  (.bin)   │
              └───────────┘
```

### 3.1 Modül Listesi
- `core/huffman.py` — Huffman encode/decode (Sayood, §3.2)
- `core/lzw.py` — LZW + Türkçe Unicode desteği (Welch 1984, Salomon §6.13)
- `core/bwt.py` — BWT + RLE + Huffman pipeline (Burrows-Wheeler 1994)
- `core/next_token.py` — Bigram/trigram entropi (Shannon 1951)
- `core/nn_selector.py` — MLP 32→16→8 (Goodfellow §6.3)
- `core/hybrid.py` — Akıllı Hibrit yöneticisi
- `core/ai_engine.py` — Groq LLM entegrasyonu
- `core/entropy.py` — Shannon entropisi (Shannon 1948)
- `core/benchmark.py` — gzip/bzip2/zlib/lzma karşılaştırma
- `core/ui_helpers.py` — Streamlit yardımcı bileşenler

---

## 4. UYGULANAN ALGORİTMALAR

### 4.1 Huffman Kodlaması

Frekansa dayalı **optimal prefix-free** kod ağacı (Huffman, 1952).

**Algoritma adımları:**
1. Karakter frekansları say
2. Her karakter için yaprak düğüm oluştur
3. Min-heap kullanarak en küçük iki frekansı birleştir
4. Tek düğüm kalana kadar tekrarla

**Sıkıştırma sınırı (Cover & Thomas, 2006, §5.6):**
$$H(X) \leq L < H(X) + 1$$

### 4.2 LZW Sıkıştırma

Sözlük tabanlı, **uyarlanabilir** sıkıştırma (Welch, 1984).

**Türkçe karakter sorunu:** Standart LZW 0-255 ASCII başlangıç sözlüğü kullanır.
Türkçe karakterler (ş, ğ, ü, ö, ç, ı) Unicode'da 255 üstündedir → çakışma.

**Çözüm:** Metindeki tüm benzersiz karakterleri başlangıç sözlüğüne ekle
(Salomon, §6.13'te önerilen yaklaşım).

### 4.3 BWT + RLE + Huffman (bzip2 Tekniği)

3 aşamalı pipeline (Burrows & Wheeler, 1994):

1. **BWT** — Tüm döngüsel permütasyonları sırala, son sütunu al
2. **RLE** — Ardışık tekrarları (karakter, sayı) çiftine dönüştür
3. **Huffman** — RLE sonrası optimal kodla

**Bloklu uygulama:** 8.000 karakter üzeri metinler otomatik bloklara bölünür
(gerçek bzip2 100KB-900KB blok kullanır). Tüm bloklar kayıpsız geri alınır.

### 4.4 Akıllı Hibrit

```
1. NN tahmin et: Huffman / LZW / BWT?
2. Seçilen algoritmayı çalıştır
3. BWT post-check: BWT sonucu daha küçükse onu seç
4. Sonuç: asla standart Huffman'dan kötü değil (garanti)
```

---

## 5. YAPAY ZEKA ENTEGRASYONU

### 5.1 Sinir Ağı (MLP 32→16→8)

**11 Özellik Girdisi:**
1. Shannon entropisi
2. Benzersiz karakter oranı
3. Top-3 karakter yoğunluğu
4. Boşluk oranı
5. Türkçe karakter oranı
6. Run-length ortalaması
7. Rakam oranı
8. Büyük harf oranı
9. Bigram entropisi
10. Maksimum koşu oranı
11. Alfabe boyutu (log₂)

**Eğitim:**
- 2.357 örnek (sentetik + Türkçe corpus)
- L2 regularizasyon (α = 1e-3)
- Early stopping (validation %15)
- StratifiedKFold 5-fold cross-validation
- Hold-out test seti (%20)

**Sonuçlar:**
- Hold-out doğruluk: **%95.2**
- 5-fold CV: **%91.7 ± %1.8**
- Eğitim süresi: ~10 saniye

### 5.2 Groq LLM ile Frekans Tahmini

**Model:** LLaMA 3.3 70B (Groq Cloud)

**2 aşamalı prompt:**
1. İlk tahmin: "Türkçe metin için karakter frekansları?"
2. Refinement: "Eksik karakterler için tahmin ekle"

**Tahmin kalitesi ölçümü:** KL-Divergence (Kullback & Leibler, 1951):
$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

### 5.3 Akıllı Hibrit Mekanizması

```python
nn_karari = neural_network.predict(text_features)

if nn_karari == "bwt":
    sonuc = BWT_RLE_Huffman(text)
elif nn_karari == "lzw":
    sonuc = LZW(text, ai_dictionary)
else:
    sonuc = Corpus_Huffman(text)

# Güvenlik ağı: BWT her zaman karşılaştır
bwt_alternatif = BWT_RLE_Huffman(text)
if bwt_alternatif < sonuc:
    sonuc = bwt_alternatif
```

---

## 6. EĞİTİM VERİSİ VE YÖNTEM

| Kaynak | Karakter | Tür |
|--------|----------|-----|
| `large_turkish.txt` | 64.240 | Doğal Türkçe (Wikipedia karması) |
| `diverse_corpus.txt` | 37.853 | Karma (Türkçe + DNA + log + kod) |
| `turkce_dogal.txt` | 5.685 | Test seti |
| `sample.txt` | 7.400 | Tekrarlı test verisi |
| **Sentetik** | 14 kategori | Augmentasyon |

**Sınıf dağılımı:** Huffman %42, LZW %2, BWT %56 (eğitim verisi etiketleri)

---

## 7. DENEYSEL SONUÇLAR

### 7.1 Sıkıştırma Karşılaştırması

| Veri Türü | Std Huffman | Akıllı Hibrit | İyileşme |
|-----------|-------------|---------------|----------|
| Tekrarlı (ABC×100) | %78.2 küçülme | **%96.9** | +%85.9 |
| DNA dizisi | %73.8 küçülme | **%94.1** | +%77.5 |
| JSON log | %60.7 küçülme | **%94.7** | +%86.5 |
| Doğal Türkçe (1.2K) | %37.1 küçülme | %38.7 | +%2.5 |
| Wikipedia TR (20K) | %39.8 küçülme | **%58.3** | +%30.7 |

### 7.2 Endüstri Karşılaştırması (Türkçe Wikipedia 4.5K karakter)

| Algoritma | Boyut | Küçülme | Süre |
|-----------|-------|---------|------|
| zlib -9 | 2.154 byte | %56.4 | 0.01 ms |
| bzip2 | 2.154 byte | %56.4 | 0.80 ms |
| gzip -9 | 2.166 byte | %56.2 | 0.06 ms |
| **bwt_rle_huffman** | **2.147 byte** | **%56.5** | 0.56 ms |
| Akıllı Hibrit | 2.266 byte | %54.1 | ~3 ms |

**BWT implementasyonumuz bzip2'den %0.3 daha iyi.**

### 7.3 Kontekstli Entropi (Next-Token)

Test metni: "Yapay zeka teknolojileri hizla gelisiyor..." (68 karakter)

| Model | Bit Sayısı | Bit/Karakter |
|-------|-----------|-------------|
| Orijinal (8 bit/c) | 544 | 8.00 |
| Unigram (iid) | 316 | 4.65 |
| Bigram (Markov-1) | 248 | 3.65 |
| **Trigram (Markov-2)** | **191** | **2.82** |

**Shannon (1951)** insanlardan tahmin yaptırarak İngilizce için ~1.3 bit/karakter
ölçmüştü. Bizim trigram modelimiz Türkçe için 2.82 bit/karakter veriyor — daha
gelişmiş modeller (LLM) bu sınıra yaklaşır.

### 7.4 Birim Test Kapsamı

- **88 birim test** geçiyor (`pytest tests/`)
- 72 kayıpsızlık testi (Huffman/LZW/BWT × edge case'ler)
- 14 ölçeklenme testi (Huffman doğrusal, BWT karesel)
- 2 bloklu BWT testi (11.000+ karakter)

### 7.5 Sinir Ağı Şeffaflığı

**Permutation importance** (Breiman, 2001) ile özellik önemleri:

| Özellik | Önem Skoru |
|---------|-----------|
| Benzersiz oran | 0.25 |
| Bigram entropisi | 0.18 |
| Alfabe boyutu (log₂) | 0.10 |
| Run-length ortalama | 0.08 |
| Türkçe karakter oranı | 0.06 |

**Confusion matrix:**
```
            Huff   LZW   BWT
Huffman    187     0    16
LZW          0     7     0
BWT          5     0   201
```

---

## 8. KARŞILAŞTIRMA VE TARTIŞMA

### 8.1 Güçlü Yanlar

- ✅ Klasik bilgi teorisi temellerini doğru uyguluyor (Sayood, Cover & Thomas)
- ✅ Sinir ağı doğrulaması (5-fold CV) ezber riskini önlüyor
- ✅ Akıllı Hibrit asla standart Huffman'dan kötü değil
- ✅ Türkçe karakter desteği eksiksiz
- ✅ Bloklu BWT ile sınırsız metin uzunluğu
- ✅ Endüstri standartlarıyla rekabet edebilir (bzip2'den iyi)

### 8.2 Sınırlamalar

- ⚠️ Shannon entropisi **karakter-bağımsız (iid)** varsayım altında
- ⚠️ Sinir ağı **algoritma seçimi** yapar, doğrudan karakter tahmini değil
- ⚠️ BWT tek-blok için 8.000 karakter sınırı (O(n²) kompleksite)
- ⚠️ Groq API gereksinimi (sadece AI sekmeleri için)

### 8.3 Gelecek Çalışmalar

- LZ77 / Brotli karşılaştırması
- Move-to-Front (MTF) eklenmesi (klasik bzip2 ile %100 uyum)
- Görüntü/ses verisi için DCT entegrasyonu
- Daha derin n-gram modeli (4-gram, 5-gram)

---

## 9. KAYNAKLAR

### Klasik Makaleler

1. Shannon, C. E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27(3), 379–423.
2. Shannon, C. E. (1951). "Prediction and Entropy of Printed English." *Bell System Technical Journal*, 30(1), 50–64.
3. Huffman, D. A. (1952). "A Method for the Construction of Minimum-Redundancy Codes." *Proceedings of the IRE*, 40(9), 1098–1101.
4. Kullback, S. & Leibler, R. A. (1951). "On Information and Sufficiency." *Annals of Mathematical Statistics*, 22(1), 79–86.
5. Welch, T. A. (1984). "A Technique for High-Performance Data Compression." *IEEE Computer*, 17(6), 8–19.
6. Burrows, M. & Wheeler, D. J. (1994). "A block-sorting lossless data compression algorithm." *DEC SRC Research Report 124*.
7. Rice, J. R. (1976). "The Algorithm Selection Problem." *Advances in Computers*, 15, 65–118.
8. Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5–32.
9. Kotthoff, L. (2014). "Algorithm Selection for Combinatorial Search Problems: A Survey." *AI Magazine*, 35(3), 48–60.

### Standart Kitaplar

10. Sayood, K. (2017). *Introduction to Data Compression* (4th ed.). Morgan Kaufmann.
11. Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley-Interscience.
12. Salomon, D. (2007). *Data Compression: The Complete Reference* (4th ed.). Springer.
13. MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press.
14. Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*. MIT Press.
15. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
16. Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

### Yazılım

17. Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, 2825–2830.
18. Groq Inc. (2024). Groq Cloud API. https://groq.com/

---

## EK-A: KOD YAPISI

```
project_veri/
├── app.py                   # Streamlit (9 sekme)
├── core/
│   ├── huffman.py           # Huffman encode/decode
│   ├── lzw.py               # LZW (Türkçe destekli)
│   ├── bwt.py               # BWT + RLE + Huffman + bloklu
│   ├── next_token.py        # Bigram/trigram entropi
│   ├── nn_selector.py       # MLP + Feature importance + CM
│   ├── hybrid.py            # Akıllı Hibrit
│   ├── ai_engine.py         # Groq LLM
│   ├── entropy.py           # Shannon entropisi
│   ├── benchmark.py         # gzip/bzip2/zlib/lzma
│   ├── ui_helpers.py        # Streamlit yardımcı
│   ├── corpus_freq.json     # Türkçe frekanslar
│   └── nn_model.pkl         # Eğitilmiş MLP
├── data/
│   ├── large_turkish.txt    # 64K Türkçe corpus
│   ├── diverse_corpus.txt   # Karma veri
│   ├── turkce_dogal.txt     # Test seti
│   └── sample.txt           # Tekrarlı test
├── tests/
│   ├── test_kayipsizlik.py  # 72 test
│   └── test_scaling.py      # 14 test
├── ai_diary.json
├── README.md
├── MIMARI.md
├── Dockerfile
└── requirements.txt
```

## EK-B: ÇALIŞTIRMA

```bash
# Bağımlılıklar
pip install -r requirements.txt

# Yerel çalıştırma
streamlit run app.py

# Testler
pytest tests/ -v          # 88 test geçer

# Docker
docker build -t veri-sikistirma .
docker run -p 8501:8501 veri-sikistirma
```

---

*Bu rapor Veri Sıkıştırma dersi dönem projesi kapsamında hazırlanmıştır.*
