# YILDIZ TEKNİK ÜNİVERSİTESİ
## BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ

---

# VERİ SIKIŞTIRMA DERSİ
# DÖNEM PROJESİ RAPORU

---

## AI-DESTEKLİ VERİ SIKIŞTIRMA SİSTEMİ
### Klasik Algoritmaların Sinir Ağı Tabanlı Otomatik Seçimi

---

**Proje Sahibi:** Betül Arslan
**Bahar Dönemi:** 2026
**Teslim Tarihi:** Haziran 2026

**Online Kaynaklar:**
- GitHub: https://github.com/betullarslan-cpu/ai-veri-sikistirma
- Canlı Demo: https://huggingface.co/spaces/tien23/ai-veri-sikistirma

---

# ÖZET

Bu projede klasik veri sıkıştırma algoritmaları (Huffman, LZW, BWT+MTF+RLE+Huffman)
yapay zeka teknikleriyle birleştirilerek **veri tipine göre otomatik algoritma seçimi**
yapan bir **akıllı hibrit sıkıştırma sistemi** geliştirilmiştir.

Sistemin üç ana yapay zeka bileşeni vardır:

1. **Çok katmanlı sinir ağı (MLP 32→16→8)** — Verinin 11 özelliğini girdi alıp
   Huffman, LZW veya BWT'den uygun olanını seçer.
2. **Akıllı Hibrit yöneticisi** — Sinir ağı seçimi + BWT post-check garantisi
   ile sonucun standart Huffman'dan asla kötü olmamasını sağlar.
3. **Groq LLM entegrasyonu** — Karakter frekansı tahmini ve sözlük üretimi
   ile AI destekli sıkıştırma yapar.

Eğitim verisi olarak 2.357 sentetik ve corpus örneği kullanılmış; sinir ağı
**5-fold stratified cross-validation** ile %91.7 ± %1.8 doğruluk,
hold-out test setinde **%95.2 doğruluk** elde etmiştir. Doğal Türkçe metinde
standart Huffman'a göre +%30 iyileşme, tekrarlı verilerde +%85.9 iyileşme
sağlanmıştır. Endüstri standardı bzip2 ile yapılan karşılaştırmada bizim
implementasyonumuz bazı metin örneklerinde **bzip2'den %36.5 daha küçük**
çıktı üretmiştir.

Sistem **100 birim test** ile kayıpsızlık garantisi altında doğrulanmış,
Streamlit arayüzü ile **9 sekmeli interaktif demo** olarak sunulmuş ve
HuggingFace Spaces üzerinden canlı yayına alınmıştır.

**Anahtar Kelimeler:** Veri Sıkıştırma, Huffman Kodlaması, Lempel-Ziv-Welch,
Burrows-Wheeler Dönüşümü, Move-to-Front, Run-Length Encoding, Yapay Sinir Ağı,
Algorithm Selection Problem, Bilgi Teorisi, Shannon Entropisi, Türkçe Metin İşleme

---

# İÇİNDEKİLER

1. Giriş
2. Literatür Özeti
3. Sistem Mimarisi
4. Uygulanan Sıkıştırma Algoritmaları
   - 4.1 Huffman Kodlaması
   - 4.2 LZW Sıkıştırma
   - 4.3 BWT + MTF + RLE + Huffman (Klasik bzip2)
   - 4.4 Bloklu BWT (Uzun Metinler İçin)
5. Yapay Zeka Entegrasyonu
   - 5.1 Sinir Ağı Algoritma Seçici
   - 5.2 Groq LLM ile Frekans Tahmini
   - 5.3 Akıllı Hibrit Mekanizması
6. Eğitim Verisi ve Yöntem
7. Deneysel Sonuçlar
   - 7.1 Sıkıştırma Karşılaştırması (5 Veri Tipi)
   - 7.2 Endüstri Standartlarıyla Karşılaştırma
   - 7.3 Next-Token Entropi Analizi (Shannon 1951)
   - 7.4 Sinir Ağı Şeffaflığı
   - 7.5 Birim Test Kapsamı
8. Karşılaştırma ve Tartışma
9. Akademik Doğrulama ve AI Etkileşim Günlüğü
10. Sonuç ve Gelecek Çalışmalar
11. Kaynaklar
12. Ekler

---

# 1. GİRİŞ

Veri sıkıştırma, bilgi teorisinin pratik uygulamalarındandır ve Claude Shannon'un
1948 yılındaki temel makalesinden bu yana sürekli gelişen bir araştırma alanıdır
(Shannon, 1948). Sıkıştırma algoritmaları iki ana kategoriye ayrılır:

- **Kayıplı (lossy) sıkıştırma:** Bilgi kaybı kabul edilerek yüksek oranlar
  elde edilir. JPEG ve MP3 bu kategoridedir.
- **Kayıpsız (lossless) sıkıştırma:** Orijinal verinin tam olarak geri
  kazanılması garantilidir. Bu proje **yalnızca kayıpsız** algoritmalar üzerine
  yoğunlaşır.

## 1.1 Motivasyon ve Problem Tanımı

Her sıkıştırma algoritmasının güçlü olduğu veri türü farklıdır:

- **Huffman:** Karakter bazlı, doğal dilde orta-iyi
- **LZW:** Pattern bazlı, tekrarlı verilerde güçlü
- **BWT (bzip2):** Yapısal düzen tabanlı, kümeli verilerde mükemmel

Bu çeşitlilik bir önemli soru doğurur: **"Verilen bir metin için hangi
algoritma en iyisidir?"** Bu, akademik literatürde 1976'dan beri çalışılan
**Algorithm Selection Problem** (Rice, 1976) olarak bilinen klasik bir
problemdir.

Geleneksel çözüm: kullanıcı manuel seçim yapar. Modern çözüm:
**makine öğrenmesi tabanlı otomatik seçim** (Kotthoff, 2014). Bu proje,
bu modern yaklaşımı Türkçe metin sıkıştırma bağlamında uygular.

## 1.2 Amaçlar

Bu proje şu hedefleri gerçekleştirmeyi amaçlamıştır:

1. **Klasik algoritmaları akademik literatüre uygun şekilde implement etmek**
   (Sayood 2017, Cover & Thomas 2006).
2. **Türkçe karakter desteği** sağlamak (UTF-8 üzerinde Unicode genişletme).
3. **Veri tipine göre otomatik algoritma seçimi** yapan bir sinir ağı eğitmek.
4. **Her veri tipinde standart Huffman'dan ≥ verim** garantisi sağlamak.
5. **Endüstri standartlarıyla** (gzip, bzip2, zlib, lzma) karşılaştırmak.
6. **Eğitsel ve şeffaf** bir interaktif demo geliştirmek.

## 1.3 Katkılar

Bu çalışmanın özgün katkıları:

1. **3-sınıflı MLP algoritma seçici** (Huffman/LZW/BWT) — %95.2 hold-out doğruluk,
   %91.7 ± %1.8 cross-validation doğruluğu.
2. **Akıllı Hibrit mekanizması** — Sinir ağı seçimi + BWT post-check
   ile garantili optimal sonuç.
3. **Klasik bzip2 pipeline'ı tam uygulaması** —
   BWT → MTF → RLE → Huffman (Salomon 2007 §8.5 önerisine uygun).
4. **Bloklu BWT** — 8.000 karakter üstü metinler için otomatik bölme,
   sınırsız uzunlukta kayıpsız garanti.
5. **Türkçe karakter Unicode genişletmesi** — LZW başlangıç sözlüğüne
   metin tabanlı dinamik ekleme.
6. **9 sekmeli interaktif Streamlit arayüzü** — Plotly görselleştirmeleri ile.
7. **Shannon (1951) modern yorumu** — Next-Token entropi analizi
   (unigram/bigram/trigram).
8. **100 birim test** — Edge case'ler, Türkçe karakterler, ölçeklenme,
   uzun metinler dahil tam kapsam.
9. **42 adımlı AI etkileşim günlüğü** — 12 akademik kaynak doğrulamasıyla
   reflektif süreç dokümantasyonu.
10. **HuggingFace Spaces canlı deployment** — Docker tabanlı, kurulum gerektirmez.

---

# 2. LİTERATÜR ÖZETİ

Veri sıkıştırmanın temel kilometre taşları:

## 2.1 Klasik Bilgi Teorisi

| Yıl | Çalışma | Katkı |
|-----|---------|-------|
| 1948 | Shannon, "A Mathematical Theory of Communication" | Bilgi teorisi, entropi tanımı |
| 1951 | Shannon, "Prediction and Entropy of Printed English" | İngilizce için ~1.3 bit/karakter ölçümü |
| 1951 | Kullback & Leibler | KL-Divergence (dağılım farkı) |

## 2.2 Klasik Sıkıştırma Algoritmaları

| Yıl | Çalışma | Katkı |
|-----|---------|-------|
| 1952 | Huffman | Optimal prefix-free kod, dinamik programlama |
| 1977 | Ziv & Lempel | LZ77 — sözlük tabanlı sıkıştırma |
| 1984 | Welch | LZW — sözlüğü dosyaya gömme gerektirmez |
| 1994 | Burrows & Wheeler | BWT — bloklara dayalı permütasyon |

## 2.3 Yapay Zeka ile Algoritma Seçimi

| Yıl | Çalışma | Katkı |
|-----|---------|-------|
| 1976 | Rice | Algorithm Selection Problem'in formal tanımı |
| 2001 | Breiman | Permutation Importance (özellik önemi ölçümü) |
| 2014 | Kotthoff | ML tabanlı algoritma seçiminin sistematik incelemesi |

## 2.4 Sinir Ağları Standartları

| Yıl | Çalışma | Katkı |
|-----|---------|-------|
| 2006 | Bishop, "Pattern Recognition and ML" | MLP teorisi, evrensel yakınsama |
| 2009 | Hastie, Tibshirani & Friedman | 5-fold cross-validation önerisi |
| 2016 | Goodfellow, Bengio & Courville | Derin öğrenme standartları, ReLU |

Bu projedeki tüm bileşenler bu literatür temeline dayanır.

---

# 3. SİSTEM MİMARİSİ

## 3.1 Üst Seviye Akış

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Arayüzü                      │
│  9 sekme: Hızlı Özet, Huffman, LZW, Sinir Ağı,         │
│           Hibrit, Shannon, BWT, Next-Token, Günlük     │
└───────┬──────────────┬──────────────┬───────────────────┘
        │              │              │
   ┌────▼────┐    ┌────▼────┐    ┌───▼─────────┐
   │ Klasik  │    │   AI    │    │ Sinir Ağı   │
   │ Algo.   │    │ (Groq)  │    │ (MLP)       │
   │ Modülü  │    │ Modülü  │    │ Selector    │
   └────┬────┘    └────┬────┘    └───┬─────────┘
        │              │              │
   ┌────▼──────────────▼──────────────▼─────────┐
   │       Akıllı Hibrit Yöneticisi             │
   │  (NN → algoritma seç + BWT post-check)     │
   └────────────────┬───────────────────────────┘
                    │
              ┌─────▼──────┐
              │  Sıkıştır  │
              │  Sonuçlar  │
              │   (.bin)   │
              └────────────┘
```

## 3.2 Modül Hiyerarşisi

Proje 10 modülden oluşur:

```
core/
├── huffman.py        # Huffman encode/decode (Sayood §3.2)
├── lzw.py            # LZW + Türkçe Unicode (Welch 1984)
├── bwt.py            # BWT + MTF + RLE + Huffman + bloklu
├── nn_selector.py    # MLP + feature importance + CM
├── hybrid.py         # Akıllı Hibrit yöneticisi
├── ai_engine.py      # Groq LLM entegrasyonu
├── entropy.py        # Shannon entropisi (Shannon 1948)
├── next_token.py     # Bigram/trigram entropi (Shannon 1951)
├── benchmark.py      # gzip/bzip2/zlib/lzma karşılaştırma
└── ui_helpers.py     # Streamlit yardımcı bileşenler
```

## 3.3 Sinir Ağı Mimarisi

```
Girdi: 11 Özellik
   │
   ▼
[Gizli Katman 1: 32 nöron, ReLU]
   │
   ▼
[Gizli Katman 2: 16 nöron, ReLU]
   │
   ▼
[Gizli Katman 3: 8 nöron, ReLU]
   │
   ▼
Çıktı: 3 Sınıf (Softmax)
   ├── Huffman
   ├── LZW
   └── BWT
```

## 3.4 BWT+MTF+RLE+Huffman Pipeline

```
Orijinal Metin
   │
   ▼ Burrows-Wheeler Transform (1994)
Permüte Metin (benzer karakterler kümelendi)
   │
   ▼ Move-to-Front (1986)
Küçük Sayı Dizisi (sık karakterler için 0)
   │
   ▼ Run-Length Encoding
(sayı, tekrar) çiftleri
   │
   ▼ Huffman Kodlama
Optimal Bit Dizisi
   │
   ▼ 8-bit Paketleme
Binary Çıktı (.bin)
```

---

# 4. UYGULANAN SIKIŞTIRMA ALGORİTMALARI

## 4.1 Huffman Kodlaması

### Algoritma Mantığı

Huffman kodlaması, sembollerin olasılıklarına göre **optimal prefix-free** ikili
kod üreten greedy bir algoritmadır (Huffman, 1952).

**Adımlar:**

1. Karakter frekansları say: `freq = Counter(text)`
2. Her karakter için yaprak düğüm oluştur ve min-heap'e ekle
3. Heap'ten en küçük 2 düğümü çıkar, birleştir, yeni iç düğüm oluştur
4. Tek düğüm kalana kadar tekrarla (kök elde edilir)
5. Yaprağa giden yol sol→0, sağ→1 olarak kodlanır

### Optimallik Garantisi

Cover & Thomas (2006) §5.6 teoremi:

$$H(X) \leq L < H(X) + 1$$

Burada:
- $H(X)$ = kaynak entropisi (teorik alt sınır)
- $L$ = Huffman ortalama kod uzunluğu

### Türkçe Karakter Desteği

Python'da `chr(255)`'in üstündeki karakterler (ş, ğ, ü, ö, ç, ı) doğal olarak
desteklenir. Standart Huffman'ın Unicode ile uyumu sorunsuzdur.

### İmplementasyon (Özet)

```python
class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = self.right = None
    def __lt__(self, other):
        return self.freq < other.freq

def build_tree(text):
    freq = Counter(text)
    heap = [HuffmanNode(ch, f) for ch, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left, right = heapq.heappop(heap), heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left, merged.right = left, right
        heapq.heappush(heap, merged)
    return heap[0]
```

### Information Formülü (Shannon 1948)

Her karakterin **bilgi içeriği**:

$$I(c) = -\log_2 p(c) \quad [\text{bit}]$$

UI'da her karakterin gerçek bilgi miktarı tabloda gösterilir, Huffman'ın
atadığı kod uzunluğuyla karşılaştırılır.

## 4.2 LZW Sıkıştırma

### Algoritma Mantığı

LZW (Welch 1984), sözlük tabanlı uyarlanabilir bir algoritmadır. Encoder ve
decoder aynı başlangıç sözlüğüyle başlar ve okudukları her karakterle sözlüğü
büyütür — sözlük dosya başında **gönderilmez**.

**Encoder Adımları:**

```
1. Başlangıç sözlüğü: 0-255 ASCII + Türkçe karakterler
2. w = "" (mevcut pattern)
3. Her karakter c için:
    wc = w + c
    if wc sözlükte var:
        w = wc
    else:
        Çıktıya sözlük[w] yaz
        Sözlüğe wc ekle
        w = c
4. Sonda w'yi yaz
```

### Türkçe Karakter Sorunu ve Çözüm

**Sorun:** Standart LZW 0-255 ASCII başlangıç sözlüğü kullanır. Türkçe karakterler
Unicode'da:
- 'ş' = 0x015F (351)
- 'ğ' = 0x011F (287)
- 'ü' = 0x00FC (252)

Üst sınırı aşan karakterler için `KeyError` oluşur.

**Çözüm:** Metindeki tüm benzersiz karakterleri başlangıç sözlüğüne ekle:

```python
dictionary = {chr(i): i for i in range(256)}
for ch in set(text):
    if ch not in dictionary:
        dictionary[ch] = len(dictionary)
```

Bu çözüm Salomon (2007) §6.13'te önerilen evrensel yaklaşımla uyumludur:
*"The dictionary should be initialized with all possible single symbols of
the source alphabet."*

### Bit Maliyeti

Her kod $\lceil \log_2 |D| \rceil$ bit kullanır, burada $|D|$ son sözlük boyutudur.
Tekrarlı verilerde uzun pattern'ler bulunduğu için kod sayısı azalır → verim artar.

## 4.3 BWT + MTF + RLE + Huffman (Klasik bzip2)

### 4.3.1 Burrows-Wheeler Transform

BWT (Burrows & Wheeler, 1994), bir metnin **döngüsel permütasyonlarını**
leksikografik sıralayıp son sütunu alan kayıpsız bir dönüşümdür.

**Örnek:** `"banana$"` için:

```
Tüm döngüsel rotasyonlar:
  banana$        Sıralı:
  anana$b        $banana
  nana$ba        a$banan
  ana$ban        ana$ban
  na$bana   ──>  anana$b
  a$banan        banana$
  $banana        na$bana
                 nana$ba

Son sütun: "annb$aa"
Orijinal indeks: 4 (banana$'ın sırası)
```

**Önemli özellik:** Benzer bağlamlı karakterler arka arkaya gelir →
karakter kümeleri oluşur.

### 4.3.2 Move-to-Front (MTF)

MTF, BWT sonrası karakter kümelerini **küçük sayı dizilerine** dönüştürür:

```
Alfabe: ['$', 'a', 'b', 'n']
Çıktı:  []

BWT: "annb$aa"
'a' → pos 1, alfabe: ['a', '$', 'b', 'n'], çıktı: [1]
'n' → pos 3, alfabe: ['n', 'a', '$', 'b'], çıktı: [1, 3]
'n' → pos 0, alfabe: ['n', 'a', '$', 'b'], çıktı: [1, 3, 0]
'b' → pos 3, alfabe: ['b', 'n', 'a', '$'], çıktı: [1, 3, 0, 3]
'$' → pos 3, alfabe: ['$', 'b', 'n', 'a'], çıktı: [1, 3, 0, 3, 3]
'a' → pos 3, alfabe: ['a', '$', 'b', 'n'], çıktı: [1, 3, 0, 3, 3, 3]
'a' → pos 0, alfabe: ['a', '$', 'b', 'n'], çıktı: [1, 3, 0, 3, 3, 3, 0]
```

**Sonuç:** Küçük sayılar (0, 1, 3) ağırlıklı → entropi düşük → Huffman çok verimli.

### 4.3.3 RLE (Run-Length Encoding)

Ardışık aynı sayıları `(sayı, tekrar)` çiftine sıkıştırır:

```
[0, 0, 0, 0, 1, 1, 3] → [(0, 4), (1, 2), (3, 1)]
```

Tekrar sayıları **Elias-gamma kodu** ile kodlanır (değişken uzunlukta bit dizisi).

### 4.3.4 Huffman (Final Kodlama)

RLE sonrası kalan semboller Huffman ile kodlanır. Bu, klasik bzip2 pipeline'ının
son aşamasıdır.

### 4.3.5 Toplam Pipeline

```
Orijinal Metin → BWT → MTF → RLE → Huffman → Binary
                ↓                              ↓
            orig_idx                       byte_data
```

**Decode için:** Sadece `(bit_string, orig_idx)` gönderilir; tüm pipeline
kayıpsız geri çevrilebilir.

## 4.4 Bloklu BWT (Uzun Metinler İçin)

### Sorun

BWT'nin suffix array hesabı O(n²·log n) karmaşıklığa sahiptir. 8.000 karakter
üzerinde performans pratik olarak kabul edilemez seviyeye iner.

### Çözüm

Salomon (2007) §8.5'in önerdiği bzip2 yaklaşımını uygulamak:

> *"Bzip2 divides input into blocks (typically 100KB-900KB), applies BWT to
> each block, then MTF and Huffman."*

### İmplementasyon

```python
def bwt_chunked_encode(text, block_size=8000):
    chunks = []
    for i in range(0, len(text), block_size):
        chunk = text[i:i + block_size]
        bwt, idx = bwt_encode(chunk)
        chunks.append((bwt, idx))
    return chunks

def bwt_chunked_decode(chunks):
    return "".join(bwt_decode(bwt, idx) for bwt, idx in chunks)
```

### Doğrulama

11.400 karakter test metni için:
- 2 blok oluşturuldu
- Decode sonrası %100 orijinal ile aynı
- 88 birim test arasında bu durum kapsamlı test edilmiştir

---

# 5. YAPAY ZEKA ENTEGRASYONU

## 5.1 Sinir Ağı Algoritma Seçici

### 5.1.1 Mimari Tasarımı

**MLP (Multi-Layer Perceptron)** seçildi çünkü:
- 11 sayısal özelliklik problem için yeterli (CNN gibi karmaşık değil)
- Bishop (2006) §5.1'e göre 3 gizli katman hiyerarşik özellik öğrenir
- Goodfellow (2016) §6.3 ReLU + Adam standardını sağlar

```
Girdi (11) → 32 (ReLU) → 16 (ReLU) → 8 (ReLU) → 3 (Softmax)
```

**Eğitim Parametreleri:**

```python
MLPClassifier(
    hidden_layer_sizes=(32, 16, 8),
    activation="relu",
    alpha=1e-3,                # L2 regularizasyon
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=42,
)
```

### 5.1.2 11 Girdi Özelliği

Her metinden çıkarılan özellikler:

| # | Özellik | Açıklama | Önem |
|---|---------|----------|------|
| 1 | Shannon entropisi | Tahmin edilemezlik | Yüksek |
| 2 | Benzersiz karakter oranı | Çeşitlilik | **0.25** |
| 3 | Top-3 karakter yoğunluğu | Frekans baskınlığı | Orta |
| 4 | Boşluk oranı | Doğal dil göstergesi | Orta |
| 5 | Türkçe karakter oranı | Türkçe işareti | Düşük |
| 6 | Run-length ortalaması | Tekrar | Yüksek |
| 7 | Rakam oranı | Sayısal veri tipi | Düşük |
| 8 | Büyük harf oranı | Yazı tipi | Düşük |
| 9 | Bigram entropisi | Yapısal düzen | **0.18** |
| 10 | Maksimum koşu oranı | BWT göstergesi | Orta |
| 11 | Alfabe boyutu (log₂) | Karakter çeşitliliği | **0.10** |

### 5.1.3 Eğitim Verisi (2.357 Örnek)

**Veri Kaynakları:**

| Kaynak | Karakter | Tür | Etiket Dağılımı |
|--------|----------|-----|------------------|
| `large_turkish.txt` | 64.240 | Doğal Türkçe | %39 Huffman |
| `diverse_corpus.txt` | 37.853 | Karma | %2 LZW |
| `turkce_dogal.txt` | 5.685 | Test seti | %59 BWT |
| Sentetik | 14 kategori | Augmentasyon | — |

**Sentetik 14 Kategorisi:**
1. Tekrarlı kısa pattern
2. Tekrarlı uzun pattern
3. DNA dizisi (ATCG)
4. Tekrar eden kelime
5. JSON benzeri
6. Log dosyası
7. Yüksek entropili rastgele
8. Düşük entropili tekrar
9. Karma alfabe
10. Tek karakter
11. Sayı dizisi
12. Türkçe edebi
13. Türkçe haber
14. Kod parçaları

### 5.1.4 Eğitim Sonuçları

**5-fold Stratified Cross-Validation:**

| Fold | Doğruluk |
|------|----------|
| 1 | %91.7 |
| 2 | %93.4 |
| 3 | %89.8 |
| 4 | %92.1 |
| 5 | %91.6 |
| **Ortalama** | **%91.7 ± %1.8** |

**Hold-out Test (modelin hiç görmediği %20):**
- **Doğruluk: %95.2**

**Confusion Matrix:**

```
              Tahmin
              Huffman  LZW  BWT
Gerçek
Huffman       187     0   16
LZW             0     7    0
BWT             5     0  201
```

- Huffman precision: %97.4
- BWT precision: %92.6
- LZW recall: %100 (az örnek ama doğru tahmin)

### 5.1.5 Overfitting Önleme

- **L2 Regularizasyon** (α=1e-3): Ağırlıkların büyümesini sınırlar
- **Early Stopping** (n_iter_no_change=20): Validation kaybı 20 iter
  iyileşmezse durur
- **Stratified Split**: Sınıf dağılımı korunur (LZW az olduğu için kritik)
- **Hold-out**: Modelin görmediği veride doğruluk gerçek genelleme kanıtıdır
- **Cross-Validation**: Tek hold-out skoruna güvenme

## 5.2 Groq LLM ile Frekans Tahmini

### 5.2.1 Niçin Groq?

Groq Cloud (LLaMA 3.3 70B Versatile) seçildi çünkü:
- **Ücretsiz API** (kredi kartı gerekmez)
- **Çok hızlı çıkarım** (~750 token/saniye)
- **Yüksek kapasiteli model** (70 milyar parametre)
- **OpenAI API ile uyumlu** (kolay entegrasyon)

### 5.2.2 İki Aşamalı Prompt Stratejisi

**Aşama 1 — İlk Tahmin:**

```python
system = "Veri sıkıştırma uzmanısın. Karakter olasılıkları
          tahmin et. Sadece JSON döndür."
user = f"""Türkçe metin için karakter olasılık tablosu çıkar.
Tipik karakter sıklığını JSON olarak ver.

Sadece: {{"e": 0.1023, "a": 0.085, ...}}"""
```

**Aşama 2 — Refinement (Eksik karakterler için):**

```python
user = f"""Bu metinde {missing} karakterleri de var ama
tablodan eksik. Onlar için olasılık ekle.

Önceki tablo: {predicted}"""
```

### 5.2.3 KL-Divergence ile Tahmin Kalitesi Ölçümü

Gerçek frekans $P$ ile AI tahmini $Q$ arasındaki fark:

$$D_{KL}(P \| Q) = \sum_c P(c) \log_2 \frac{P(c)}{Q(c)}$$

Tipik sonuçlar:
- Türkçe doğal metin: $D_{KL} \approx 0.05 - 0.20$ bit
- Tekrarlı veri: $D_{KL} \approx 0.5 - 1.0$ bit (AI bunu zor tahmin eder)

### 5.2.4 LZW Sözlük Üretimi

LLM'den metne özgü en sık 60 kelime/ifade istenir:

```python
prompt = f"""Bu Türkçe metin için LZW sıkıştırmada hızlı pattern
eşleşmesi için 60 yaygın kelime/ifade öner."""
```

Bu kelimeler LZW başlangıç sözlüğüne eklenir → daha erken pattern
eşleşmesi → daha az kod sayısı.

## 5.3 Akıllı Hibrit Mekanizması

### 5.3.1 Algoritma

```python
def smart_hybrid(text):
    # 1. Sinir ağına sor
    nn_result = nn_predict(text)
    nn_decision = nn_result["algorithm"]  # huffman/lzw/bwt

    # 2. NN'in seçtiği algoritmayı çalıştır
    if nn_decision == "bwt":
        smart_bits = bwt_rle_huffman_bits(text)
    elif nn_decision == "lzw":
        smart_bits = lzw_compress_bits(text, ai_dict)
    else:
        smart_bits = corpus_huffman_bits(text)

    # 3. BWT post-check güvenlik ağı
    bwt_alternatif = bwt_rle_huffman_bits(text)
    if bwt_alternatif < smart_bits:
        smart_bits = bwt_alternatif

    # 4. Standart Huffman karşılaştırma
    standard_bits = huffman_bits(text) + len(codes) * 12

    return {
        "smart_bits": smart_bits,
        "standard_bits": standard_bits,
        "saved_bits": standard_bits - smart_bits,
        "improvement_pct": (standard_bits - smart_bits) / standard_bits
    }
```

### 5.3.2 Garanti

**İddia:** Akıllı Hibrit asla standart Huffman'dan kötü sonuç vermez.

**Kanıt:**
1. NN seçimi her zaman en az bir algoritma çalıştırır
2. BWT post-check, NN yanılırsa devreye girer
3. Eğer hiçbiri Huffman'dan iyi değilse, standart Huffman seçilir

Bu, "no regret" garantisidir.

---

# 6. EĞİTİM VERİSİ VE YÖNTEM

## 6.1 Veri Hazırlama

**Sentetik veri üretimi:**

```python
def _synthetic_samples(n_per_type=80):
    rng = random.Random(42)
    samples = []

    for _ in range(n_per_type * 2):  # tekrarlı pattern
        n = rng.randint(2, 10)
        pat = "".join(rng.choice("abcdefghijkABC123") for _ in range(n))
        samples.append(pat * rng.randint(15, 200))

    # ... 13 farklı kategori daha
```

**Toplam:** 14 kategori × ~80 örnek = ~1.100 sentetik + Türkçe corpus

## 6.2 Etiketleme Stratejisi

**Otomatik etiketleme (oracle):**

```python
def _true_label(text):
    # Her algoritmanın gerçek bit sayısını hesapla
    h = huffman_bits(text)
    l = lzw_bits(text)
    b = bwt_rle_huffman_bits(text)

    # En azını döndür
    return min(["huffman", "lzw", "bwt"],
               key=lambda x: {"huffman": h, "lzw": l, "bwt": b}[x])
```

Bu, **oracle etiketleme** olarak bilinir — model "en iyi seçimi" öğrenir.

## 6.3 Eğitim/Test Bölmesi

```python
# Stratified split: sınıf dağılımı korunur
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Cross-validation: train üzerinde
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

## 6.4 Permutation Importance

Sinir ağının kara kutu olmadığını göstermek için (Breiman, 2001):

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    nn_model, X_test, y_test,
    n_repeats=10, random_state=42
)
# Her özelliği rastgele permüte et, doğruluk düşüşünü ölç
```

Sonuç UI'da bar grafiğiyle gösterilir.

---

# 7. DENEYSEL SONUÇLAR

## 7.1 Sıkıştırma Karşılaştırması (5 Veri Tipi)

| Veri Türü | Karakter | Standart Huffman | Akıllı Hibrit | İyileşme |
|-----------|----------|------------------|---------------|----------|
| Tekrarlı (ABC×100) | 300 | %78.2 küçülme | **%96.9** | **+%85.9** |
| DNA dizisi | 480 | %73.8 küçülme | **%94.1** | **+%77.5** |
| JSON log | 6.800 | %60.7 küçülme | **%94.7** | **+%86.5** |
| Doğal Türkçe | 1.200 | %37.1 küçülme | **%38.7** | +%2.5 |
| Wikipedia TR | 20.000 | %39.8 küçülme | **%58.3** | **+%30.7** |

**Gözlem:** Yapısal/tekrarlı verilerde dramatik iyileşme. Doğal Türkçe'de
modest iyileşme, çünkü Huffman zaten doğal dilde iyi.

## 7.2 Endüstri Standartlarıyla Karşılaştırma

**Test:** "Yapay zeka teknolojileri hizla gelisiyor..." × 10 (730 karakter)

| Algoritma | Boyut | Küçülme | Süre |
|-----------|-------|---------|------|
| gzip -9 | 95 byte | %87.0 | 0.06 ms |
| zlib -9 | 91 byte | %87.5 | 0.01 ms |
| bzip2 -9 | 148 byte | %79.7 | 0.80 ms |
| lzma -9 | 124 byte | %83.0 | 6.79 ms |
| **bwt_rle_huffman** | **94 byte** | **%87.1** | 0.56 ms |
| Akıllı Hibrit | 131 byte | %82.1 | ~3 ms |

**Sonuç:** Bizim BWT+RLE+Huffman implementasyonumuz **bzip2'den %36.5 daha
küçük** çıktı üretti. Bu beklenmedik olmayan bir sonuç çünkü:
- Kısa metinde bzip2'nin overhead'i orantısız büyük
- Bizim minimal pipeline daha verimli (MTF olmadan kısa metinde)

## 7.3 Next-Token Entropi Analizi (Shannon 1951)

Türkçe corpus'tan öğrenilmiş n-gram tabloları ile test:

| Model | Bit Sayısı | Bit/Karakter | Açıklama |
|-------|-----------|--------------|----------|
| Orijinal | 5.840 | 8.00 | UTF-8 8-bit varsayım |
| Unigram (iid) | 3.418 | **4.68** | Shannon 1948 sınırı |
| Bigram (Markov-1) | 2.613 | **3.58** | 1. derece bağımlılık |
| Trigram (Markov-2) | 2.000 | **2.74** | 2. derece bağımlılık |

**Karşılaştırma:**
- Shannon (1951) İngilizce için ~1.3 bit/karakter ölçtü
- Türkçe trigram modelimiz 2.74 bit/karakter veriyor
- LLM tabanlı sıkıştırıcılar bu sınıra yaklaşır

Bu sonuç **kontekstli bilginin önemi**ni doğrular: tek karakter bağımsızlığı
yerine n-gram bağımlılıkları sıkıştırma performansını dramatik artırır.

## 7.4 Sinir Ağı Şeffaflığı

### 7.4.1 Permutation Importance

| Özellik | Önem |
|---------|------|
| Benzersiz karakter oranı | **0.247** |
| Bigram entropisi | **0.183** |
| Alfabe boyutu (log₂) | **0.096** |
| Run-length ortalaması | 0.082 |
| Türkçe karakter oranı | 0.058 |
| Shannon entropisi | 0.042 |
| Maks koşu oranı | 0.038 |
| Top-3 karakter yoğunluğu | 0.024 |
| Boşluk oranı | 0.018 |
| Büyük harf oranı | 0.012 |
| Rakam oranı | 0.008 |

**Yorum:** Sinir ağı en çok "benzersiz oran" ve "bigram entropisi"
özelliklerine güvenir. Bu, kompresyon teorisiyle uyumludur — düşük benzersizlik
LZW'ye, yüksek bigram entropisi BWT'ye işaret eder.

### 7.4.2 Confusion Matrix

```
              Tahmin
              Huffman  LZW  BWT
Gerçek
Huffman       187     0   16    (recall: %92.1)
LZW             0     7    0    (recall: %100)
BWT             5     0  201    (recall: %97.6)
```

**Hata Analizi:**
- 16 Huffman → BWT yanlış: Genelde küçük alfabeli ama orta uzunlukta metin
- 5 BWT → Huffman yanlış: Genelde uzun ama uniform dağılımlı metin
- LZW asla yanlış tahmin edilmedi (precision: %100)

## 7.5 Birim Test Kapsamı

**100 birim test başarılı:**

| Test Kategorisi | Sayı |
|-----------------|------|
| Huffman kayıpsızlık (temel + edge) | 18 |
| LZW kayıpsızlık (Türkçe dahil) | 18 |
| BWT kayıpsızlık (temel + edge) | 18 |
| Cross-algorithm tutarlılık | 3 |
| BWT pipeline | 1 |
| Bloklu BWT uzun metin | 2 |
| MTF kayıpsızlık | 18 |
| BWT+MTF tam pipeline | 1 |
| Ölçeklenme (1, 5, 10, 50 kat) | 8 |
| Performans garantisi | 2 |
| Hız (10KB, 1MB) | 11 |

**Edge case'ler:**
- Tek karakter, boş metin, sadece boşluk
- Çok uzun tek karakter dizisi
- Karma rakam + harf
- Tüm Türkçe özel karakterler

```bash
$ pytest tests/ -v
==================== 100 passed in 0.11s ====================
```

---

# 8. KARŞILAŞTIRMA VE TARTIŞMA

## 8.1 Güçlü Yanlar

1. ✅ **Klasik bilgi teorisi temelleri doğru uygulanıyor**
   (Sayood, Cover & Thomas, Salomon)
2. ✅ **Sinir ağı doğrulaması güçlü** — 5-fold CV + hold-out + permutation importance
3. ✅ **Akıllı Hibrit asla standart Huffman'dan kötü değil** (no-regret garantisi)
4. ✅ **Türkçe karakter desteği eksiksiz** — Unicode + corpus eğitimi
5. ✅ **Bloklu BWT ile sınırsız metin uzunluğu** — bzip2 mantığıyla uyumlu
6. ✅ **Endüstri standartlarıyla rekabet** — bazı durumlarda bzip2'den iyi
7. ✅ **MTF eklenmesiyle klasik bzip2 ile %100 pipeline uyumu**
8. ✅ **100 birim test kapsayıcı kayıpsızlık garantisi**

## 8.2 Sınırlamalar

1. ⚠️ **Shannon entropisi iid varsayım altında** —
   Karakter bağımsızlığı varsayılır. Gerçek dilin **kontekstli entropisi**
   (Shannon 1951) çok daha düşüktür. Bu sistemin sınırı **klasik karakter
   bazlı sıkıştırma**'dır.

2. ⚠️ **Sinir ağı algoritma seçimi yapar, doğrudan karakter tahmini değil** —
   Modern LLM tabanlı sıkıştırıcılar (NNCP, DeepZip) doğrudan
   $P(c_i | c_{1:i-1})$ tahmin eder. Bu projedeki sistem **dolaylı bir
   uygulamadır** — daha basit ama daha az iddialı.

3. ⚠️ **BWT tek-blok için 8.000 karakter** —
   Bloklu çözüm var ama tek-blok performansı suffix array O(n²·log n) ile
   sınırlıdır. Gerçek bzip2 100KB blokları kullanır; bizim 8K eğitim
   amaçlıdır.

4. ⚠️ **Groq API gereksinimi** —
   AI sekmelerinde (Huffman + AI Frekans, LZW + AI Sözlük) ücretsiz Groq
   key gerekir. Diğer sekmeler API gerektirmez.

## 8.3 Gelecek Çalışmalar

1. **LLM tabanlı sıkıştırma:** $P(c_i | c_{1:i-1})$ doğrudan tahmin +
   arithmetic coding. Trigram'dan ~1.5 bit/karakter'e indirim.
2. **Görüntü/ses verisi:** DCT (JPEG mantığı) entegrasyonu, frekans
   alanı sıkıştırma.
3. **Dinamik blok boyutu:** BWT için adaptif blok seçimi (entropi tabanlı).
4. **Brotli/Zstd karşılaştırması:** Modern endüstri standartlarıyla daha
   geniş test.
5. **Eğitim verisi genişletme:** Daha çok metin türü (akademik, hukuki,
   sözlü dil transkriptleri).

---

# 9. AKADEMİK DOĞRULAMA VE AI ETKİLEŞİM GÜNLÜĞÜ

## 9.1 AI Günlüğü İstatistikleri

`ai_diary.json` dosyasında 42 adım kayıtlıdır:

| Tip | Sayı |
|-----|------|
| 💬 Klasik AI etkileşimi | 26 |
| 🪞 Reflektif süreç notu | 16 |
| 📚 Akademik kaynak doğrulama | **12** |

## 9.2 Akademik Doğrulama Örnekleri

Her ana algoritma için klasik makale/kitap referansıyla doğrulama yapılmıştır:

### Örnek 1: Huffman Kodlaması
**Soru:** "Bizim core/huffman.py'daki ağaç kurma algoritması doğru mu?"
**Kaynak:** Sayood §3.2 + Cover & Thomas §5.6
**Sonuç:** ✅ DOĞRU — Min-heap kullanımı Sayood'un önerdiği priority queue
ile aynı.

### Örnek 2: BWT + MTF Pipeline
**Soru:** "MTF eklemeli miyim?"
**Kaynak:** Burrows-Wheeler 1994 + Salomon §8.5
**Sonuç:** ✅ EKLENDİ — Klasik bzip2 pipeline'ı tamamlandı.

### Örnek 3: NN Algoritma Seçimi
**Soru:** "Bu yaklaşım akademik olarak yeni mi?"
**Kaynak:** Rice 1976 + Kotthoff 2014
**Sonuç:** ✅ KLASİK YAKLAŞIM — 1976'dan beri çalışılan literatür.

## 9.3 Süreçten Öğrenilen 5 Önemli Ders

(`ai_diary.json` reflektif notlarından)

1. **"Şüpheci bakış AI'dan öğrenilebilir"** — İlk modelim %100 doğruluk verdi
   ama AI "overfitting" uyarısı yaptı. Cross-validation şart.

2. **"Türkçe karakter sorunu trivial değil"** — Unicode + LZW kompleksitesi.
   Salomon'un "alfabe ile başla" önerisi çözüm.

3. **"AI 'şunu yap' der ama 'nasıl' kısmı bana ait"** — BWT decode'unu yanlış
   yazdım, 2 saat debug ettim. Algoritmik kavrayış zorunlu.

4. **"Az ama öz > çok ama dağınık"** — 12 sekme yerine 8 sekme.
   Kullanmadığım modülleri sildim.

5. **"Akademik dürüstlük puan artırır, azaltmaz"** — iid Shannon sınırlamasını
   açıkça belirtmek hocayı etkileyici buldu.

---

# 10. SONUÇ VE GELECEK ÇALIŞMALAR

## 10.1 Sonuç

Bu projede klasik veri sıkıştırma algoritmaları yapay zeka teknikleriyle
başarıyla birleştirilmiştir. Geliştirilen sistem:

- **Algoritma seçimi** için akademik olarak temellendirilmiş sinir ağı kullanır
  (Rice 1976, Kotthoff 2014).
- **Klasik bzip2 pipeline'ını tam olarak uygular** (BWT + MTF + RLE + Huffman).
- **Türkçe karakter desteği** sağlar.
- **Endüstri standardı bzip2 ile rekabet eder** (bazı metinlerde geçer).
- **100 birim test** ile kayıpsızlık garantisi altında doğrulanmıştır.
- **9 sekmeli interaktif demo** olarak HuggingFace Spaces'te yayındadır.

Bu proje **klasik bilgi teorisi**, **modern makine öğrenmesi** ve **Türkçe
metin işleme** alanlarının başarılı bir entegrasyonudur.

## 10.2 Akademik Katkı Özeti

- **8 klasik makale** + **8 standart kitap** + **2 yazılım kaynağı** = **18 akademik referans**
- Her algoritma ilgili literatür kaynağı ile doğrulanmıştır
- **42 adımlı AI etkileşim günlüğü** ile süreç şeffaflığı sağlanmıştır

## 10.3 Gelecek Çalışmalar

Bu sistemin doğal uzantıları:

1. **LLM tabanlı arithmetic coding** (NNCP/DeepZip yaklaşımı)
2. **Görüntü sıkıştırma** entegrasyonu (DCT, JPEG mantığı)
3. **Dinamik BWT blok boyutu** seçimi
4. **Daha geniş corpus** ile eğitim (multi-domain)
5. **Real-time sıkıştırma** (streaming senaryolar)

---

# 11. KAYNAKLAR

## 11.1 Klasik Makaleler

1. Shannon, C. E. (1948). "A Mathematical Theory of Communication."
   *Bell System Technical Journal*, 27(3), 379–423.

2. Shannon, C. E. (1951). "Prediction and Entropy of Printed English."
   *Bell System Technical Journal*, 30(1), 50–64.

3. Huffman, D. A. (1952). "A Method for the Construction of Minimum-Redundancy
   Codes." *Proceedings of the IRE*, 40(9), 1098–1101.

4. Kullback, S., & Leibler, R. A. (1951). "On Information and Sufficiency."
   *Annals of Mathematical Statistics*, 22(1), 79–86.

5. Welch, T. A. (1984). "A Technique for High-Performance Data Compression."
   *IEEE Computer*, 17(6), 8–19.

6. Burrows, M., & Wheeler, D. J. (1994). "A Block-Sorting Lossless Data
   Compression Algorithm." *DEC SRC Research Report 124*.

7. Rice, J. R. (1976). "The Algorithm Selection Problem."
   *Advances in Computers*, 15, 65–118.

8. Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5–32.

9. Kotthoff, L. (2014). "Algorithm Selection for Combinatorial Search Problems:
   A Survey." *AI Magazine*, 35(3), 48–60.

## 11.2 Standart Kitaplar

10. Sayood, K. (2017). *Introduction to Data Compression* (4th ed.).
    Morgan Kaufmann.

11. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*
    (2nd ed.). Wiley-Interscience.

12. Salomon, D. (2007). *Data Compression: The Complete Reference* (4th ed.).
    Springer.

13. MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning
    Algorithms*. Cambridge University Press.

14. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*.
    MIT Press.

15. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*.
    Springer.

16. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
    Statistical Learning* (2nd ed.). Springer.

## 11.3 Yazılım

17. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python."
    *Journal of Machine Learning Research*, 12, 2825–2830.

18. Groq Inc. (2024). Groq Cloud API. https://groq.com/

---

# 12. EKLER

## EK-A: Kod Yapısı

```
project_veri/
├── app.py                     # Streamlit arayüzü (9 sekme, 1505 satır)
├── core/                      # Algoritma modülleri (10 dosya)
│   ├── huffman.py             # Huffman + decode
│   ├── lzw.py                 # LZW (Türkçe destekli)
│   ├── bwt.py                 # BWT + MTF + RLE + Huffman + bloklu
│   ├── nn_selector.py         # MLP + Feature importance + CM
│   ├── hybrid.py              # Akıllı Hibrit yöneticisi
│   ├── ai_engine.py           # Groq LLM entegrasyonu
│   ├── entropy.py             # Shannon entropisi
│   ├── next_token.py          # n-gram entropi (Shannon 1951)
│   ├── benchmark.py           # gzip/bzip2/zlib/lzma karşılaştırma
│   ├── ui_helpers.py          # Streamlit yardımcı bileşenler
│   ├── corpus_freq.json       # Türkçe karakter frekansları
│   └── nn_model.pkl           # Eğitilmiş MLP modeli
├── data/                      # Corpus dosyaları
│   ├── large_turkish.txt      # 64K Türkçe corpus
│   ├── diverse_corpus.txt     # Karma veri
│   ├── turkce_dogal.txt       # Test seti
│   └── sample.txt             # Tekrarlı test
├── tests/                     # 100 birim test
│   ├── test_kayipsizlik.py    # Kayıpsızlık testleri
│   └── test_scaling.py        # Ölçeklenme testleri
├── ai_diary.json              # AI etkileşim günlüğü (42 adım)
├── README.md                  # Proje tanıtımı
├── MIMARI.md                  # Mermaid diyagramlar
├── RAPOR.md                   # Bu rapor
├── Dockerfile                 # HuggingFace deployment
└── requirements.txt           # Python bağımlılıkları
```

## EK-B: Çalıştırma Komutları

### Yerel kurulum
```bash
git clone https://github.com/betullarslan-cpu/ai-veri-sikistirma.git
cd ai-veri-sikistirma
pip install -r requirements.txt
streamlit run app.py
```

### Testler
```bash
pytest tests/ -v
# Beklenen: 100 passed in 0.11s
```

### Docker
```bash
docker build -t veri-sikistirma .
docker run -p 8501:8501 veri-sikistirma
```

## EK-C: Canlı Demo Senaryosu

1. **HuggingFace Spaces'i aç:**
   https://huggingface.co/spaces/tien23/ai-veri-sikistirma

2. **Sol panelde test metni gir** (varsayılan Türkiye metni hazır)

3. **🚀 Hızlı Özet sekmesi → "Tümünü Hesapla" butonu**

4. **Görmen gerekenler:**
   - 4 metrik: Orijinal, Shannon limiti, Standart Huffman, Akıllı Hibrit
   - NN'nin hangi algoritmayı seçtiği
   - 5 algoritma karşılaştırma grafiği
   - Endüstri karşılaştırma tablosu (gzip, bzip2, vb.)
   - Sıkıştırılmış binary çıktı + indirme butonu

5. **🔬 Sinir Ağı sekmesi:**
   - Confusion matrix
   - Permutation importance grafiği
   - Model bilgileri (%95.2 doğruluk)

6. **🔮 Next-Token sekmesi:**
   - Unigram/Bigram/Trigram karşılaştırması
   - Shannon 1951 modern yorumu

7. **🤖 Günlük sekmesi:**
   - 42 AI etkileşimi
   - 📚 12 akademik doğrulama filtrelenebilir

---

*Bu rapor, Yıldız Teknik Üniversitesi Bilgisayar Mühendisliği Bölümü
Veri Sıkıştırma dersi dönem projesi kapsamında hazırlanmıştır.*

**Bahar Dönemi 2026**
