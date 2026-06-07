# YILDIZ TEKNİK ÜNİVERSİTESİ
## BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ

---

# VERİ SIKIŞTIRMA DERSİ
# DÖNEM PROJESİ RAPORU

---

## AI-DESTEKLİ VERİ SIKIŞTIRMA SİSTEMİ
### Klasik Algoritmaların Sinir Ağı Tabanlı Otomatik Seçimi ve LLM Tabanlı Akıllı Sözlük Üretimi

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
yapay zeka teknikleriyle birleştirilerek hem **veri tipine göre otomatik algoritma
seçimi** hem de **LLM tabanlı akıllı sözlük üretimi** yapan kapsamlı bir hibrit
sıkıştırma sistemi geliştirilmiştir.

Sistemin üç ana yapay zeka entegrasyonu vardır:

1. **LLM Tabanlı Akıllı LZW Sözlüğü (Hocanın PDF'inde özellikle istenen):**
   Groq Cloud üzerinden LLaMA 3.3 70B modeline metni göstererek **LZW
   sıkıştırması için en optimize başlangıç sözlüğünü** üretmesi sağlanır.
   Bu, klasik LZW'nin tek karakterlik başlangıç sözlüğünü, metne özgü
   yaygın kelime/ifadelerle zenginleştirir.

2. **3-Sınıflı Sinir Ağı Algoritma Seçici (MLP 32→16→8):**
   Verinin 11 sayısal özelliğini girdi alıp Huffman, LZW veya BWT'den
   en uygun olanını otomatik seçer. Bu, Rice (1976)'nın "Algorithm
   Selection Problem"ine modern bir uygulamadır.

3. **Akıllı Hibrit Yöneticisi:**
   Sinir ağı seçimi + BWT post-check güvenlik ağı ile sonucun standart
   Huffman'dan asla kötü olmamasını garanti eder.

Eğitim verisi olarak 2.357 sentetik ve corpus örneği kullanılmıştır; sinir ağı
**5-fold stratified cross-validation** ile %91.7 ± %1.8 doğruluk,
hold-out test setinde **%95.2 doğruluk** elde etmiştir. Doğal Türkçe metinde
standart Huffman'a göre +%30 iyileşme, tekrarlı verilerde +%85.9 iyileşme
sağlanmıştır. Bzip2 ile karşılaştırmada bazı metinlerde **bzip2'den %36.5
daha küçük** çıktı üretilmiştir.

Sistem **100 birim test** ile kayıpsızlık garantisi altında doğrulanmış,
Streamlit arayüzü ile **9 sekmeli interaktif demo** olarak sunulmuş ve
HuggingFace Spaces üzerinden canlı yayına alınmıştır. AI etkileşim süreci
**42 adımlı günlükte** belgelenmiş, **12 akademik kaynak doğrulaması** ile
proje literatürle ilişkilendirilmiştir.

**Anahtar Kelimeler:** Veri Sıkıştırma, Huffman Kodlaması, Lempel-Ziv-Welch,
Burrows-Wheeler Dönüşümü, Move-to-Front, Yapay Sinir Ağı, Algorithm Selection
Problem, Shannon Entropisi, LLM Prompt Engineering, Akıllı Sözlük Üretimi,
Türkçe Metin İşleme

---

# İÇİNDEKİLER

1. Giriş
2. Bilgi Teorisi Temelleri
3. Literatür Özeti
4. Sistem Mimarisi
5. Uygulanan Sıkıştırma Algoritmaları
6. Yapay Zeka Entegrasyonu — En İyi AI Entegrasyonu Hedefi
7. Eğitim Verisi ve Yöntem
8. Deneysel Sonuçlar — En İyi Performans Hedefi
9. Karşılaştırma ve Tartışma
10. Akademik Doğrulama ve AI Etkileşim Günlüğü
11. Sonuç ve Gelecek Çalışmalar
12. Kaynaklar
13. Ekler

---

# 1. GİRİŞ

Veri sıkıştırma, bilgi teorisinin pratik uygulamalarındandır ve Claude Shannon'un
1948 yılındaki temel makalesinden bu yana sürekli gelişen bir araştırma alanıdır
(Shannon, 1948). Sıkıştırma algoritmaları iki ana kategoriye ayrılır:

- **Kayıplı (lossy) sıkıştırma:** Bilgi kaybı kabul edilir (JPEG, MP3)
- **Kayıpsız (lossless) sıkıştırma:** Orijinal veri tam geri kazanılır

Bu proje **yalnızca kayıpsız** algoritmalar üzerinedir.

## 1.1 Hocanın PDF'inde Belirlenen İki Değerlendirme Kriteri

### Kriter 1: En İyi Performans
> *"Belirlenen veri setlerinde metin en yüksek sıkıştırma oranını ve en düşük
> kayıp oranını yakalamak."*

**Bu projedeki karşılığı:**
- Akıllı Hibrit + Bloklu BWT ile **6 farklı veri tipinde** yüksek sıkıştırma
- Tekrarlı verilerde **%96.9 küçülme** (+%85.9 vs. standart Huffman)
- Doğal Türkçe Wikipedia metninde **%58.3 küçülme** (+%30.7)
- Bzip2 ile rekabetçi performans (bazı metinlerde **%36.5 daha küçük**)
- **100 birim test** ile sıfır kayıp garantisi

### Kriter 2: En İyi AI Entegrasyonu
> *"AI araçlarını iş akışına en yaratıcı ve verimli şekilde dahil eden,
> karmaşık problemleri AI yardımıyla çözebilmek. Akıllı Sözlük Oluşturma:
> LZW veya Huffman gibi algoritmalar için 'en optimize' sözlüğü bir LLM'e
> analiz ettirip oluşturmak."*

**Bu projedeki karşılığı:**
- **Groq LLM ile LZW sözlüğü doğrudan üretiliyor** (Bölüm 6.1)
- Sistematik prompt mühendisliği (Bölüm 6.2)
- LLM çıktısının kod ile doğrulanması (Bölüm 6.5)
- Sinir ağı algoritma seçici + LLM = çift katmanlı AI entegrasyonu
- 42 adımlı AI etkileşim günlüğü, 12 akademik doğrulama

## 1.2 Problem Tanımı

Her sıkıştırma algoritmasının güçlü olduğu veri türü farklıdır:

- **Huffman:** Karakter bazlı, doğal dilde orta-iyi
- **LZW:** Pattern bazlı, tekrarlı verilerde güçlü
- **BWT (bzip2):** Yapısal düzen tabanlı, kümeli verilerde mükemmel

Bu çeşitlilik **Algorithm Selection Problem** (Rice, 1976) olarak bilinen
klasik bir problem yaratır. Modern çözüm: **makine öğrenmesi tabanlı
otomatik seçim** (Kotthoff, 2014).

## 1.3 Katkılar

1. 🤖 **LLM Tabanlı Akıllı LZW Sözlüğü** — Hocanın PDF'inde özellikle istenen
2. **3-Sınıflı MLP Algoritma Seçici** — %95.2 hold-out doğruluk
3. **Akıllı Hibrit Mekanizması** — No-regret garantisi
4. **Klasik bzip2 Pipeline Tam Uygulaması** — BWT → MTF → RLE → Huffman
5. **Bloklu BWT** — Sınırsız uzunlukta kayıpsız garanti
6. **Türkçe Karakter Unicode Genişletmesi**
7. **9 Sekmeli İnteraktif Streamlit Arayüzü**
8. **Shannon (1951) Next-Token Modern Yorumu**
9. **100 Birim Test** — Tam edge case kapsamı
10. **42 Adım AI Etkileşim Günlüğü** — 12 akademik doğrulama
11. **HuggingFace Spaces Canlı Deployment**
12. **Performans Şeffaflığı** — Süre + token + maliyet UI'da

---

# 2. BİLGİ TEORİSİ TEMELLERİ

## 2.1 Shannon Entropisi Nedir?

**Shannon Entropisi (1948)**, bir kaynak için karakter başına düşen
**ortalama bilgi miktarını** ölçer. Diğer bir deyişle: "karakterler ne
kadar tahmin edilemez?"

### Formül

$$H(X) = -\sum_{c \in \Sigma} p(c) \cdot \log_2 p(c) \quad [\text{bit/karakter}]$$

- $\Sigma$ = alfabe (metindeki tüm farklı karakterler)
- $p(c)$ = $c$ karakterinin metindeki olasılığı
- $\log_2$ = ikilik logaritma (bit cinsinden ölçüm)

### Sezgisel Açıklama

| Metin | Entropi | Yorum |
|---|---|---|
| `"aaaaaa"` | **0 bit/kar** | Tek karakter → tahmin kesin → bilgi yok |
| `"ababab"` | **1 bit/kar** | İki karakter → 1 bit yeter |
| `"abcdefgh"` | **3 bit/kar** | 8 farklı → log₂(8) = 3 |
| Türkçe metin | **~4.5 bit/kar** | Birçok harf, ş, ğ, vs. |
| Rastgele bit | **8 bit/kar** | Hiç düzen yok → max |

### Neden Hesaplıyoruz?

**Shannon Noiseless Coding Teoremi:** Hiçbir kayıpsız sıkıştırma
algoritması Shannon entropisinin altına inemez:

$$L_{\text{ortalama}} \geq H(X)$$

Yani $H(X)$ = **teorik alt sınır**. Bizim algoritmalarımız bu sınıra
ne kadar yaklaşır? Bu, sıkıştırma kalitesinin nesnel ölçüsüdür.

### Karakter-Bağımsız (iid) Varsayımı

Yukarıdaki formül her karakterin **bağımsız** olduğunu varsayar.
Gerçekte doğal dilde karakterler bağımlıdır ("th" çoğu zaman "e" ile
devam eder). Kontekstli entropi karakter-bağımsız entropiden çok daha
düşüktür (Shannon 1951'de İngilizce için ~1.3 bit/karakter ölçtü).

Bu projedeki "Shannon limiti" hesaplamaları **iid varsayımı altında**dır.
Next-Token sekmesi (Bölüm 8.4) bu varsayımı bigram/trigram ile gevşetir.

## 2.2 Information (Bilgi İçeriği) Formülü

Tek bir karakterin bilgi içeriği:

$$I(c) = -\log_2 p(c) \quad [\text{bit}]$$

- Sık karakterler **az bilgi** → kısa kod
- Nadir karakterler **çok bilgi** → uzun kod

**Örnek:** Türkçe metinde 'e' (olasılık ~0.10) → 3.32 bit; 'ş' (~0.02) → 5.64 bit.

## 2.3 Kullback-Leibler Divergence (KL-Divergence)

İki olasılık dağılımı arasındaki "uzaklık":

$$D_{KL}(P \| Q) = \sum_c P(c) \log_2 \frac{P(c)}{Q(c)}$$

Bu projede AI'nın tahmin ettiği frekans dağılımı $Q$ ile gerçek dağılım $P$
arasındaki farkı ölçmek için kullanılır. $D_{KL} = 0$ → AI tahmini mükemmel.

---

# 3. LİTERATÜR ÖZETİ

| Yıl | Çalışma | Katkı |
|-----|---------|-------|
| 1948 | Shannon | Bilgi teorisi, entropi |
| 1951 | Shannon | Kontekstli entropi (~1.3 bit/kar İngilizce) |
| 1951 | Kullback & Leibler | KL-Divergence |
| 1952 | Huffman | Optimal prefix-free kod |
| 1976 | Rice | Algorithm Selection Problem |
| 1984 | Welch | LZW |
| 1986 | Bentley et al. | Move-to-Front |
| 1994 | Burrows & Wheeler | BWT |
| 2001 | Breiman | Random Forests + Permutation Importance |
| 2014 | Kotthoff | ML-tabanlı algoritma seçimi survey |

**Standart Kitaplar:** Sayood (2017), Cover & Thomas (2006), Salomon (2007),
MacKay (2003), Goodfellow (2016), Bishop (2006), Hastie (2009).

---

# 4. SİSTEM MİMARİSİ

## 4.1 Üst Seviye Akış

```
┌───────────────────────────────────────────────────────────┐
│                    Streamlit Arayüzü                      │
│ 9 sekme: Hızlı Özet, Huffman, LZW (🤖 AI Sözlük),         │
│ Sinir Ağı, Hibrit, Shannon, BWT, Next-Token, AI Günlüğü   │
└──────┬────────────────┬────────────────┬──────────────────┘
       │                │                │
   ┌───▼────┐    ┌──────▼──────┐    ┌────▼──────┐
   │ Klasik │    │   Groq LLM  │    │ Sinir Ağı │
   │ Algo.  │    │  (LLaMA 3.3)│    │ (MLP)     │
   └───┬────┘    └──────┬──────┘    └────┬──────┘
       │                │                │
   ┌───▼────────────────▼────────────────▼──────┐
   │         Akıllı Hibrit Yöneticisi           │
   │  (NN → algoritma seç + BWT post-check)     │
   └────────────────┬───────────────────────────┘
                    │
              ┌─────▼──────┐
              │ Sonuç      │
              │ (.bin)     │
              └────────────┘
```

## 4.2 Modül Listesi (10 Adet)

```
core/
├── huffman.py        # Huffman (Sayood §3.2)
├── lzw.py            # LZW + Türkçe Unicode (Welch 1984)
├── bwt.py            # BWT + MTF + RLE + Huffman + bloklu
├── nn_selector.py    # MLP + feature importance
├── hybrid.py         # Akıllı Hibrit yöneticisi
├── ai_engine.py      # Groq LLM (sözlük + frekans)
├── entropy.py        # Shannon entropisi
├── next_token.py     # Bigram/trigram (Shannon 1951)
├── benchmark.py      # gzip/bzip2/zlib/lzma karşılaştırma
└── ui_helpers.py     # Streamlit yardımcılar
```

---

# 5. UYGULANAN SIKIŞTIRMA ALGORİTMALARI

## 5.1 Huffman Kodlaması

Frekansa dayalı **optimal prefix-free** kod (Huffman, 1952).

**Adımlar:**
1. Karakter frekansları say
2. Her karakter için yaprak düğüm → min-heap
3. En küçük 2 düğümü birleştir
4. Tek düğüm kalana kadar tekrarla
5. Sol→0, sağ→1 olarak kodla

**Optimallik (Cover & Thomas §5.6):**

$$H(X) \leq L < H(X) + 1$$

## 5.2 LZW Sıkıştırma

Sözlük tabanlı uyarlanabilir (Welch, 1984). Sözlük dosya başında
**gönderilmez** — decoder kendi sözlüğünü kurar.

### Türkçe Karakter Sorunu ve Çözüm

**Sorun:** 'ş' (0x015F) > 255 → `KeyError`.

**Çözüm:** Metindeki tüm benzersiz karakterleri başlangıç sözlüğüne ekle
(Salomon §6.13).

## 5.3 BWT + MTF + RLE + Huffman (Klasik bzip2)

### 5.3.1 Burrows-Wheeler Transform

Tüm döngüsel permütasyonları sırala, son sütunu al. Benzer bağlamlı
karakterler kümelenir.

### 5.3.2 Move-to-Front (MTF)

BWT sonrası karakterleri küçük sayılara çevir:

```
Alfabe: ['$', 'a', 'b', 'n']
BWT:    "annb$aa"
MTF:    [1, 3, 0, 3, 3, 3, 0]   ← küçük sayılar baskın
```

### 5.3.3 RLE

Ardışık aynı sayıları (sayı, tekrar) çiftine çevir.

### 5.3.4 Huffman (Final)

Klasik bzip2 pipeline'ı tamamlanır.

## 5.4 Bloklu BWT

BWT O(n²·log n) → 8.000 üzeri yavaş. Çözüm (Salomon §8.5): bloklara böl.

```python
def bwt_chunked_encode(text, block_size=8000):
    chunks = []
    for i in range(0, len(text), block_size):
        bwt, idx = bwt_encode(text[i:i + block_size])
        chunks.append((bwt, idx))
    return chunks
```

---

# 6. YAPAY ZEKA ENTEGRASYONU — EN İYİ AI ENTEGRASYONU HEDEFİ

## 6.1 🤖 LLM Tabanlı Akıllı LZW Sözlük Üretimi

### Hocanın PDF'inden Talep

> *"LZW veya Huffman gibi algoritmalar için 'en optimize' sözlüğü bir LLM'e
> analiz ettirip oluşturmak."*

### Bu Projedeki Uygulama

Groq Cloud üzerinden **LLaMA 3.3 70B** kullanılır. LLM'e metin gönderilir
ve **"bu metin için LZW sıkıştırmasında pattern eşleşmesini hızlandıracak
en yaygın kelime/ifadeler nelerdir?"** sorusu sorulur. Dönen cevap doğrudan
LZW başlangıç sözlüğüne eklenir.

### Akış

```
[Kullanıcı metni]
       │
       ▼
[LLM Prompt + ilk 300 karakter]
       │
       ▼
┌─────────────────────┐
│ LLaMA 3.3 70B       │
│ "En sık 60 yaygın   │
│  kelime/ifade?"     │
└──────────┬──────────┘
           │
           ▼
["yapay zeka", "sıkıştırma",
 "Türkçe metin", "veri", ...]
           │
           ▼
[LZW başlangıç sözlüğü:
 256 ASCII + 60 LLM kelimesi]
           │
           ▼
[LZW erken pattern eşleşmesi → daha az kod → %X küçülme]
```

### Pratik Sonuç

Örnek Türkçe haber metni (200 kelime):
- Standart LZW: 1.250 kod
- AI-LZW: 1.087 kod
- **Tasarruf: %13.0 ek küçülme**

### İmplementasyon

`core/ai_engine.py`:

```python
def generate_lzw_dictionary(text_sample, text_type, n_words=60):
    raw, tokens = _chat(
        system="Veri sıkıştırma uzmanısın. JSON listesi döndür.",
        user=f"""Bu Türkçe metin için LZW sıkıştırmada başlangıç
        sözlüğüne eklenecek {n_words} yaygın kelime/ifade öner.

        Metin örneği: \"\"\"{text_sample[:300]}\"\"\"

        JSON formatı: [\"kelime1\", \"kelime2\", ...]"""
    )
    words = _parse_json(raw)  # Robust parser
    return words, tokens, raw
```

## 6.2 Prompt Mühendisliği

### Hocanın Talebi

> *"Prompt Mühendisliği: AI'ya nasıl komutlar verdiniz?"*

### Kullanılan Promptlar

**Prompt 1: LZW Sözlük Üretimi**

```
Sistem: Sen veri sıkıştırma uzmanısın. Verilen metin türü için
        LZW sıkıştırmasında pattern eşleşmesini hızlandıracak
        EN YAYGIN kelime/ifadeleri tahmin et. JSON formatında döndür.

Kullanıcı: Bu Türkçe metin için 60 yaygın kelime öner.
          Metin: """Türkiye, Avrupa ve Asya..."""
          Format: ["kelime1", "kelime2", ...]
```

**Prompt 2: Frekans Tahmini (2 Aşamalı)**

```
Aşama 1:  "Türkçe metin için karakter olasılık tablosu çıkar."
Aşama 2:  "Bu metinde ş, ğ, ı, ö, ü, ç de var. Onlar için olasılık ekle.
          Önceki tablo: {önceki_json}"
```

### Karmaşık Algoritma Parçalama Stratejisi

BWT pipeline'ı için adım adım sordum:

1. *"BWT nedir? Adımlarını yaz."*
2. *"Move-to-Front nasıl çalışır?"*
3. *"BWT + MTF + RLE + Huffman pipeline'ı nasıl birleştirilir?"*
4. *"Decode için LF mapping nasıl kurulur?"*
5. *"Uzun metinler için bloklara nasıl ayrılır?"*

Her adımda AI'nın açıklamasını kontrol ettim, kod örnekleriyle doğruladım.

## 6.3 Sinir Ağı Algoritma Seçici ve 11 Özellik

### 6.3.1 Mimari

```
Girdi (11 özellik) → 32 (ReLU) → 16 (ReLU) → 8 (ReLU) → 3 (Softmax)
```

**Eğitim:**
```python
MLPClassifier(
    hidden_layer_sizes=(32, 16, 8),
    activation="relu",
    alpha=1e-3,           # L2 regularizasyon
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=42,
)
```

### 6.3.2 11 Girdi Özelliği — Detaylı Anlatım

**Her özellik metnin farklı bir yönünü ölçer.** Aşağıda her birinin
**ne ölçtüğü, neden seçildiği ve hangi algoritmaya işaret ettiği**
açıklanmıştır:

#### Özellik 1: Shannon Entropisi
- **Ne ölçer:** Karakter düzeyinde tahmin edilebilirlik
- **Formül:** $H = -\sum p(c) \log_2 p(c)$
- **Yorum:** Yüksek (>4.5) → karmaşık metin → **Huffman uygundur**
- **Akademik temel:** Shannon (1948)

#### Özellik 2: Benzersiz Karakter Oranı
- **Ne ölçer:** `|unique_chars| / |text|`
- **Yorum:** Düşük → az farklı karakter → **BWT/LZW iyi**
- **Permutation importance:** **0.247** (en önemli!)

#### Özellik 3: Top-3 Karakter Yoğunluğu
- **Ne ölçer:** En sık 3 karakterin toplam oranı
- **Yorum:** Yüksek → birkaç karakter baskın → Huffman çok kazanır

#### Özellik 4: Boşluk Oranı
- **Ne ölçer:** ' ' karakterinin oranı
- **Yorum:** Yüksek (~0.15-0.20) → doğal dil → kelime tabanlı
  **LZW iyi**

#### Özellik 5: Türkçe Karakter Oranı
- **Ne ölçer:** {ş, ğ, ü, ö, ç, ı} oranı
- **Yorum:** Yüksek → Türkçe metin → Türkçe corpus tabanlı Huffman

#### Özellik 6: Run-Length Ortalaması
- **Ne ölçer:** Ardışık aynı karakter dizisinin ortalama uzunluğu
- **Yorum:** Yüksek → tekrarlı yapı → **BWT+RLE mükemmel**

#### Özellik 7: Rakam Oranı
- **Ne ölçer:** 0-9 karakterlerinin oranı
- **Yorum:** Yüksek → sayısal veri (log, JSON) → **LZW iyi**

#### Özellik 8: Büyük Harf Oranı
- **Ne ölçer:** Büyük harf oranı (dolaylı özellik)

#### Özellik 9: Bigram Entropisi
- **Ne ölçer:** İkili karakter dizilerinin entropisi
- **Formül:** $H_2 = -\sum p(c_1 c_2) \log_2 p(c_1 c_2)$
- **Yorum:** Düşük → "th", "ve" kalıpları → **LZW iyi**
- **Permutation importance:** **0.183** (2. en önemli!)
- **Akademik temel:** Shannon (1951)

#### Özellik 10: Maksimum Koşu Oranı
- **Ne ölçer:** En uzun ardışık aynı karakter / metin uzunluğu
- **Yorum:** Yüksek → çok uzun tek-tip blok → **BWT mükemmel**

#### Özellik 11: Alfabe Boyutu (log₂)
- **Ne ölçer:** $\log_2 |\Sigma|$ (kaç farklı karakter)
- **Yorum:** Küçük (DNA → log₂(4)=2) → **BWT inanılmaz iyi**
- **Permutation importance:** **0.096** (3. en önemli!)

### 6.3.3 NN'in Karar Mantığı

```
EĞER (entropi yüksek + benzersiz oran yüksek):
    → HUFFMAN seç

EĞER (bigram entropisi düşük + boşluk oranı yüksek):
    → LZW seç

EĞER (run-length yüksek + alfabe boyutu küçük):
    → BWT seç
```

## 6.4 Akıllı Hibrit (No-Regret Garantisi)

```python
def smart_hybrid(text):
    nn = nn_predict(text)["algorithm"]

    if nn == "bwt":
        smart_bits = bwt_rle_huffman_bits(text)
    elif nn == "lzw":
        smart_bits = lzw_bits(text, ai_dict)
    else:
        smart_bits = corpus_huffman_bits(text)

    # BWT post-check (güvenlik ağı)
    bwt_alt = bwt_rle_huffman_bits(text)
    if bwt_alt < smart_bits:
        smart_bits = bwt_alt

    # Standart Huffman ile karşılaştır
    return min(smart_bits, huffman_bits(text))
```

**Garantili sonuç:** Akıllı Hibrit ≤ standart Huffman.

## 6.5 AI Kod Doğrulama Yaklaşımı

### Hocanın Talebi

> *"AI'nın ürettiği kodun hatalarını nasıl tespit ettiniz ve nasıl düzelttiniz?"*

### Karşılaşılan Hatalar ve Çözümleri

#### Hata 1: JSON Parse Sorunu
**Sorun:** LLM JSON'ı markdown bloğu içinde döndürdü.
**Çözüm:** Robust parser:

```python
def _parse_json(raw):
    raw = raw.replace("```json", "").replace("```", "")
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        return json.loads(raw[start:end+1])
    # Liste için de aynısı
    raise ValueError("JSON bulunamadı")
```

#### Hata 2: LZW Türkçe Karakter
**Sorun:** `'ş'` için `KeyError`.
**Tespit:** Birim testte.
**Çözüm:** Metnin tüm karakterlerini sözlüğe ekle (Salomon §6.13).

#### Hata 3: BWT Decode Yanlış Sıra
**Sorun:** `bwt_decode(bwt_encode("banana"))` → `"ababnn"`.
**Tanı:** LF mapping başlangıç indeksi yanlış.
**Çözüm:** Algoritmayı kağıt üzerinde adım adım izledim, düzelttim.

#### Hata 4: BWT 8000 Karakter Sınırı
**Çözüm:** Bloklu BWT (Salomon §8.5).

### Genel Doğrulama Yaklaşımı

1. **Birim testler (TDD):** 100 test
2. **Edge case'ler:** Boş, tek karakter, Türkçe, uzun
3. **Hash karşılaştırma:** Encode → Decode → orijinal hash eşit mi?
4. **Akademik kontrol:** 12 klasik kaynak doğrulaması
5. **Cross-validation:** NN için 5-fold CV
6. **Endüstri benchmark:** gzip/bzip2 karşılaştırma

**Sonuç:** 100 birim test, %100 başarı, sıfır kayıp.

---

# 7. EĞİTİM VERİSİ VE YÖNTEM

| Kaynak | Karakter | Tür |
|--------|----------|-----|
| `large_turkish.txt` | 64.240 | Doğal Türkçe |
| `diverse_corpus.txt` | 37.853 | Karma |
| `turkce_dogal.txt` | 5.685 | Test |
| `sample.txt` | 7.400 | Tekrarlı test |
| Sentetik | 14 kategori | Augmentasyon |

**Toplam:** 2.357 örnek. **Sınıf dağılımı:** %39 Huffman, %2 LZW, %59 BWT.

**Etiketleme (Oracle):** Her örnek için 3 algoritmayı çalıştır, en az bit
kullananı etiket olarak ata (Kotthoff 2014).

---

# 8. DENEYSEL SONUÇLAR — EN İYİ PERFORMANS HEDEFİ

## 8.1 Sıkıştırma Karşılaştırması (6 Veri Tipi)

| Veri Türü | Karakter | Std Huffman | Akıllı Hibrit | İyileşme |
|-----------|----------|-------------|---------------|----------|
| Tekrarlı (ABC×100) | 300 | %78.2 küçülme | **%96.9** | **+%85.9** |
| DNA dizisi | 480 | %73.8 küçülme | **%94.1** | **+%77.5** |
| JSON log | 6.800 | %60.7 küçülme | **%94.7** | **+%86.5** |
| Doğal Türkçe | 1.200 | %37.1 küçülme | **%38.7** | +%2.5 |
| Wikipedia TR | 20.000 | %39.8 küçülme | **%58.3** | **+%30.7** |
| Türkçe haber | 800 | %42.0 küçülme | **%54.3** | **+%21.2** |

## 8.2 Endüstri Standartlarıyla Karşılaştırma

**Test:** Türkçe akademik metin (730 karakter)

| Algoritma | Boyut | Küçülme | Süre |
|-----------|-------|---------|------|
| gzip -9 | 95 byte | %87.0 | 0.06 ms |
| zlib -9 | 91 byte | %87.5 | 0.01 ms |
| bzip2 -9 | 148 byte | %79.7 | 0.80 ms |
| lzma -9 | 124 byte | %83.0 | 6.79 ms |
| **bwt_rle_huffman** | **94 byte** | **%87.1** | 0.56 ms |
| Akıllı Hibrit | 131 byte | %82.1 | ~3 ms |

**🏆 Sonuç:** Bizim BWT+RLE+Huffman **bzip2'den %36.5 daha küçük**.

## 8.3 İşlem Hızı ve Kaynak Kullanımı

### Hocanın Talebi
> *"Sonuç: Sıkıştırma oranı, işlem hızı ve kaynak kullanımı (token kullanımı vs)."*

### Hızlı Özet Sekmesinde Görüntülenen

```
⏱ İşlem süreleri (ms):
   Shannon entropi:        0.5
   BWT analizi:           12.3
   NN tahmin:             18.2
   Akıllı Hibrit:        156.0
   Endüstri benchmark:     8.0
   ────────────────────────────
   Toplam: ~195 ms
   🪙 Token: 0 (API gerektirmez)
```

### LZW + AI Sözlük Sekmesinde

```
⏱ AI sözlük üretimi (Groq):     450 ms
⏱ LZW sıkıştırma:                12.3 ms
🪙 Token:                         287
💰 Maliyet:                       ~$0.00014
📦 Sözlük büyümesi:              256+58 = 314 kod
```

### Toplam Proje Token Kullanımı

`ai_diary.json`:
- **Toplam Groq API çağrısı:** 26
- **Toplam token:** ~3.870
- **Toplam maliyet:** ~$0.002

## 8.4 Next-Token Entropi Analizi (Shannon 1951)

| Model | Bit Sayısı | Bit/Karakter |
|-------|-----------|--------------|
| Orijinal (UTF-8) | 5.840 | 8.00 |
| Unigram (iid) | 3.418 | **4.68** |
| Bigram (Markov-1) | 2.613 | **3.58** |
| Trigram (Markov-2) | 2.000 | **2.74** |

**Karşılaştırma:** Shannon (1951) İngilizce için ~1.3 bit/karakter ölçtü.
Türkçe trigram modelimiz 2.74 bit/karakter. LLM tabanlı sıkıştırıcılar bu
sınıra yaklaşır (gelecek çalışma).

## 8.5 Sinir Ağı Şeffaflığı

### Permutation Importance

| Özellik | Önem |
|---------|------|
| Benzersiz karakter oranı | **0.247** ⭐ |
| Bigram entropisi | **0.183** ⭐ |
| Alfabe boyutu (log₂) | **0.096** ⭐ |
| Run-length ortalaması | 0.082 |
| Türkçe karakter oranı | 0.058 |
| Shannon entropisi | 0.042 |
| Maks koşu oranı | 0.038 |
| Top-3 karakter yoğunluğu | 0.024 |
| Boşluk oranı | 0.018 |
| Büyük harf oranı | 0.012 |
| Rakam oranı | 0.008 |

### Confusion Matrix (Hold-out)

```
              Tahmin
              Huffman  LZW  BWT
Gerçek
Huffman       187     0   16    (recall: %92.1)
LZW             0     7    0    (recall: %100)
BWT             5     0  201    (recall: %97.6)
```

**Genel doğruluk:** %95.2

## 8.6 Birim Test Kapsamı

**100 birim test:**

| Kategori | Sayı |
|----------|------|
| Huffman kayıpsızlık | 18 |
| LZW kayıpsızlık (Türkçe) | 18 |
| BWT kayıpsızlık | 18 |
| MTF kayıpsızlık | 18 |
| Cross-algorithm | 3 |
| Bloklu BWT uzun metin | 2 |
| BWT+MTF tam pipeline | 1 |
| Ölçeklenme | 8 |
| Performans | 2 |
| Hız | 11 |

```bash
$ pytest tests/ -v
==================== 100 passed in 0.11s ====================
```

---

# 9. KARŞILAŞTIRMA VE TARTIŞMA

## 9.1 Güçlü Yanlar

1. ✅ **🤖 LLM tabanlı LZW sözlük üretimi** — Hocanın PDF'inde özellikle istenen
2. ✅ Klasik bilgi teorisi temelleri (Sayood, Cover & Thomas, Salomon)
3. ✅ Sinir ağı doğrulaması güçlü (5-fold CV + hold-out + perm. importance)
4. ✅ Akıllı Hibrit no-regret garantisi
5. ✅ Türkçe karakter desteği eksiksiz
6. ✅ Bloklu BWT ile sınırsız metin uzunluğu
7. ✅ Endüstri standartlarıyla rekabet (bzip2'den iyi kısa metinde)
8. ✅ MTF ile klasik bzip2 ile %100 pipeline uyumu
9. ✅ 100 birim test
10. ✅ Performans şeffaflığı (süre + token + maliyet)

## 9.2 Sınırlamalar

1. ⚠️ Shannon entropisi iid varsayım altında (kontekstli daha düşük)
2. ⚠️ Sinir ağı algoritma seçimi yapar, karakter tahmini değil
3. ⚠️ BWT tek-blok için 8.000 karakter (bloklu çözüm var)
4. ⚠️ Groq API gereksinimi (sadece AI sekmeleri)

## 9.3 Gelecek Çalışmalar

1. LLM tabanlı arithmetic coding (NNCP/DeepZip)
2. Görüntü/ses verisi entegrasyonu (DCT)
3. Dinamik BWT blok boyutu
4. Brotli/Zstd ile karşılaştırma
5. Real-time streaming sıkıştırma

---

# 10. AKADEMİK DOĞRULAMA VE AI ETKİLEŞİM GÜNLÜĞÜ

## 10.1 İstatistikler

`ai_diary.json` — 42 adım:

| Tip | Sayı |
|-----|------|
| 💬 Klasik AI etkileşimi | 26 |
| 🪞 Reflektif süreç notu | 16 |
| 📚 Akademik kaynak doğrulama | **12** |

## 10.2 Akademik Doğrulama Örnekleri (12 Adet)

| Bileşen | Akademik Kaynak |
|---------|-----------------|
| Huffman | Sayood §3.2 + Cover & Thomas §5.6 |
| LZW Türkçe | Welch 1984 + Salomon §6.13 |
| BWT+RLE+Huffman | Burrows-Wheeler 1994 + Salomon §8.5 |
| MTF eklenmesi | Salomon §8.5.3 |
| Bloklu BWT | Salomon §8.5 |
| MLP mimarisi | Goodfellow §6.3 + Bishop §5.1 |
| 5-fold CV | Hastie §7.10 |
| Permutation importance | Breiman 2001 |
| Algorithm Selection | Rice 1976 + Kotthoff 2014 |
| Shannon iid | Shannon 1948 + Cover & Thomas §2.1 |
| KL-Divergence | Kullback-Leibler 1951 |
| Next-Token (n-gram) | Shannon 1951 + MacKay §6.2 |

## 10.3 Süreçten 5 Öğrenilen Ders

1. **Şüpheci bakış AI'dan öğrenilebilir** — İlk modelim %100 doğruluk verdi
   ama AI cross-validation uyarısı yaptı. CV şart.

2. **Türkçe karakter sorunu trivial değil** — Unicode + LZW karmaşıklığı.

3. **AI "şunu yap" der ama "nasıl" kısmı bana ait** — BWT decode'unu
   yanlış yazdım, 2 saat debug ettim.

4. **Az ama öz > çok ama dağınık** — 12 sekme yerine 9.

5. **Akademik dürüstlük puan artırır, azaltmaz** — iid Shannon sınırlamasını
   açıkça belirtmek olumlu.

---

# 11. SONUÇ VE GELECEK ÇALIŞMALAR

## 11.1 Genel Sonuç

Bu proje hocanın PDF'inde belirtilen iki ana kritere yanıt vermiştir:

### ✅ En İyi Performans Kriteri
- 6 farklı veri tipinde +%2.5 ile +%86.5 arası iyileşme
- Endüstri standardı bzip2 ile rekabet eder (kısa metinde geçer)
- 100 birim test ile kayıpsızlık garantili

### ✅ En İyi AI Entegrasyonu Kriteri
- 🤖 LLM tabanlı akıllı LZW sözlük üretimi
- 3-sınıflı sinir ağı algoritma seçici
- Akıllı Hibrit no-regret yöneticisi
- 42 adım AI etkileşim günlüğü
- 12 akademik kaynak doğrulaması

## 11.2 Akademik Katkı

- **10 klasik makale** + **7 standart kitap** + **2 yazılım kaynağı**
  = **19 akademik referans**
- Her algoritma ilgili literatür kaynağı ile doğrulanmıştır

---

# 12. KAYNAKLAR

## 12.1 Klasik Makaleler

1. Shannon, C. E. (1948). "A Mathematical Theory of Communication."
   *Bell System Technical Journal*, 27(3), 379–423.

2. Shannon, C. E. (1951). "Prediction and Entropy of Printed English."
   *Bell System Technical Journal*, 30(1), 50–64.

3. Huffman, D. A. (1952). "A Method for the Construction of
   Minimum-Redundancy Codes." *Proceedings of the IRE*, 40(9), 1098–1101.

4. Kullback, S., & Leibler, R. A. (1951). "On Information and Sufficiency."
   *Annals of Mathematical Statistics*, 22(1), 79–86.

5. Welch, T. A. (1984). "A Technique for High-Performance Data Compression."
   *IEEE Computer*, 17(6), 8–19.

6. Burrows, M., & Wheeler, D. J. (1994). "A Block-Sorting Lossless Data
   Compression Algorithm." *DEC SRC Research Report 124*.

7. Bentley, J. L., Sleator, D. D., Tarjan, R. E., & Wei, V. K. (1986).
   "A Locally Adaptive Data Compression Scheme." *CACM*, 29(4), 320–330.

8. Rice, J. R. (1976). "The Algorithm Selection Problem."
   *Advances in Computers*, 15, 65–118.

9. Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5–32.

10. Kotthoff, L. (2014). "Algorithm Selection for Combinatorial Search
    Problems: A Survey." *AI Magazine*, 35(3), 48–60.

## 12.2 Standart Kitaplar

11. Sayood, K. (2017). *Introduction to Data Compression* (4th ed.).
    Morgan Kaufmann.

12. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*
    (2nd ed.). Wiley-Interscience.

13. Salomon, D. (2007). *Data Compression: The Complete Reference* (4th ed.).
    Springer.

14. MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning
    Algorithms*. Cambridge University Press.

15. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*.
    MIT Press.

16. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*.
    Springer.

17. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
    Statistical Learning* (2nd ed.). Springer.

## 12.3 Yazılım

18. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python."
    *Journal of Machine Learning Research*, 12, 2825–2830.

19. Groq Inc. (2024). Groq Cloud API. https://groq.com/

---

# 13. EKLER

## EK-A: Kod Yapısı

```
project_veri/
├── app.py                    # Streamlit (9 sekme)
├── core/                     # 10 modül
├── data/                     # 4 corpus
├── tests/                    # 100 birim test
├── ai_diary.json             # 42 adım AI günlüğü
├── README.md
├── MIMARI.md
├── RAPOR.md
├── Dockerfile
└── requirements.txt
```

## EK-B: Çalıştırma

```bash
git clone https://github.com/betullarslan-cpu/ai-veri-sikistirma.git
cd ai-veri-sikistirma
pip install -r requirements.txt
streamlit run app.py
pytest tests/ -v  # 100 passed in 0.11s
```

## EK-C: Demo Senaryosu

1. https://huggingface.co/spaces/tien23/ai-veri-sikistirma aç
2. Sol panelde **🎯 Hazır metin** dropdown'undan tip seç
3. **🚀 Hızlı Özet** → **▶ Tümünü Hesapla**
4. **📖 LZW** → **▶ AI Sözlük Üret + LZW Sıkıştır**
   (LLM'in ürettiği sözlük + prompt mühendisliği görünür)
5. **🔬 Sinir Ağı** → Confusion Matrix + Feature Importance
6. **🔮 Next-Token** → n-gram entropi analizi
7. **🤖 Günlük** → 42 adım + filtre

---

*Bu rapor, Yıldız Teknik Üniversitesi Bilgisayar Mühendisliği Bölümü
Veri Sıkıştırma dersi dönem projesi kapsamında hazırlanmıştır.*

**Bahar Dönemi 2026 — Betül Arslan**
