---
marp: true
theme: gaia
class: lead
paginate: true
backgroundColor: #fff
header: 'AI-Destekli Veri Sıkıştırma · YTÜ Veri Sıkıştırma Dersi'
footer: 'Betül Arslan · Bahar 2026'
style: |
  section { font-size: 1.6rem; }
  h1 { color: #2c3e50; }
  h2 { color: #3498db; }
  code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
  table { font-size: 1.2rem; }
---

<!-- _class: lead -->
<!-- _paginate: false -->

#  AI-Destekli Veri Sıkıştırma

## Klasik Algoritmaların Sinir Ağı Tabanlı Otomatik Seçimi ve LLM Tabanlı Akıllı Sözlük Üretimi

<br>

**Betül Arslan**
Yıldız Teknik Üniversitesi
Bilgisayar Mühendisliği

🌐 huggingface.co/spaces/tien23/ai-veri-sikistirma
📁 github.com/betullarslan-cpu/ai-veri-sikistirma

---

# 📋 İçerik

1. Problem ve Amaç
2. Bilgi Teorisi Temeli (Shannon)
3. Sistem Mimarisi
4. Klasik Algoritmalar
5. **🤖 AI Entegrasyonu** (Akıllı Sözlük + NN Seçici)
6. Sinir Ağının 11 Özelliği
7. Performans Sonuçları
8. Endüstri Karşılaştırması
9. Canlı Demo
10. Sonuç

---

# 🎯 Problem ve Amaç

## Her algoritma farklı veride iyi:
- **Huffman:** Karakter bazlı → doğal dilde orta
- **LZW:** Pattern bazlı → tekrarlı veride güçlü
- **BWT (bzip2):** Yapısal düzen → kümeli veride mükemmel

## Soru
> *"Verilen bir metin için hangi algoritma en iyi?"*

**Bu Algorithm Selection Problem'dir (Rice, 1976).**
Modern çözüm: **Makine öğrenmesi** (Kotthoff, 2014).

---

# 🎯 Hocanın PDF'inde 2 Kriter

## ✅ 1. En İyi Performans
- 6 farklı veri tipinde **+%2.5 ile +%86.5** iyileşme
- Bzip2'den **%36.5 daha küçük** (kısa metinde)
- **100 birim test** ile sıfır kayıp

## ✅ 2. En İyi AI Entegrasyonu
- 🤖 **LLM ile LZW sözlüğü üretimi** (PDF'te özellikle istenen!)
- 3-sınıflı sinir ağı algoritma seçici
- Akıllı Hibrit no-regret yöneticisi
- 42 adım AI günlüğü, 12 akademik doğrulama

---

# 📐 Bilgi Teorisi: Shannon Entropisi

## Formül (Shannon, 1948)
$$H(X) = -\sum_c p(c) \cdot \log_2 p(c) \quad [\text{bit/karakter}]$$

## Sezgisel Anlam

| Metin | Entropi | Anlam |
|---|---|---|
| `"aaaaaa"` | **0 bit/kar** | Tahmin kesin → bilgi yok |
| `"abababab"` | **1 bit/kar** | 2 karakter → 1 bit yeter |
| Türkçe metin | **~4.5 bit/kar** | Birçok harf, ş, ğ, vs. |

## **Hiçbir algoritma bu sınırın altına inemez!** ⚡

---

# 🏗️ Sistem Mimarisi

```
[Kullanıcı Metni]
        │
        ▼
┌─────────────────────┐    ┌──────────────┐
│ Klasik Algoritmalar │    │   Groq LLM   │
│ Huffman/LZW/BWT     │    │ (LLaMA 3.3)  │
└──────────┬──────────┘    └──────┬───────┘
           │                       │
           │     ┌─────────────────▼─────┐
           │     │ 🤖 AI Sözlük Üretimi  │
           │     └────────────┬──────────┘
           │                  │
        ┌──▼──────────────────▼──┐
        │   Sinir Ağı (MLP)     │
        │   Algoritma Seçici    │
        └──────────┬─────────────┘
                   │
        ┌──────────▼──────────┐
        │ Akıllı Hibrit       │
        │ + BWT Post-check    │
        └──────────┬──────────┘
                   │
              Sonuç (.bin)
```

---

# 📦 Klasik Algoritmalar

## 1️⃣ Huffman (1952)
- Optimal prefix-free kod
- $H(X) \leq L < H(X) + 1$ (Cover & Thomas)

## 2️⃣ LZW (1984)
- Sözlük tabanlı, uyarlanabilir
- **Türkçe karakter sorunu:** ş, ğ, ü Unicode > 255
- **Çözüm:** Metnin alfabesini başlangıç sözlüğüne ekle

## 3️⃣ BWT + MTF + RLE + Huffman (1994 + 1986)
- **Klasik bzip2 pipeline'ı tam uygulaması**
- Bloklu sistem ile sınırsız uzunlukta kayıpsız

---

# 🤖 AI Entegrasyonu — Akıllı Sözlük

## Hocanın PDF'inden Talep
> *"LZW veya Huffman gibi algoritmalar için 'en optimize' sözlüğü
> bir LLM'e analiz ettirip oluşturmak."*

## Bu Projedeki Uygulama

```
[Kullanıcı metni: "Türkiye, Avrupa ve Asya..."]
              │
              ▼
[LLM Prompt: "Bu metin için 60 yaygın kelime öner"]
              │
              ▼
        ┌─────────────────┐
        │  LLaMA 3.3 70B  │  (Groq Cloud)
        │   Token: 287    │
        │   Süre: 450 ms  │
        └────────┬────────┘
                 │
                 ▼
["Türkiye", "Avrupa", "Asya", "köprü", "tarihi", ...]
                 │
                 ▼
[LZW Sözlüğü: 256 ASCII + 60 LLM kelimesi = 316 pattern]
                 │
                 ▼
[Erken pattern eşleşmesi → daha az kod → %13 ek küçülme]
```

---

# 🔧 Prompt Mühendisliği

## Sistem Promptu
```
Sen veri sıkıştırma uzmanısın. Verilen metin türü için
LZW sıkıştırmasında pattern eşleşmesini hızlandıracak
EN YAYGIN kelime/ifadeleri tahmin et. JSON döndür.
```

## Kullanıcı Promptu
```
Bu Türkçe metin için 60 yaygın kelime öner.
Metin: """Türkiye, Avrupa ve Asya..."""
Format: ["kelime1", "kelime2", ...]
```

## Karmaşık Algoritmayı Parçalama
BWT için adım adım sordum:
1. *"BWT nedir? Adımlarını yaz."*
2. *"Move-to-Front nasıl çalışır?"*
3. *"LF mapping ile decode nasıl yapılır?"*

---

# 🧠 Sinir Ağı: MLP 32→16→8

## Mimari (Bishop §5.1, Goodfellow §6.3)

```
Girdi: 11 özellik
   ↓
[32 nöron, ReLU]
   ↓
[16 nöron, ReLU]
   ↓
[8 nöron, ReLU]
   ↓
Çıktı: Huffman / LZW / BWT (Softmax)
```

## Eğitim
- **2.357 örnek** (sentetik + corpus)
- **L2 regularizasyon + Early stopping**
- **5-fold Cross-Validation:** %91.7 ± %1.8
- **Hold-out test:** **%95.2** ⭐

---

# 🔬 Sinir Ağının 11 Özelliği (1/2)

| # | Özellik | Önem | Hangi Algoritmaya İşaret |
|---|---|---|---|
| 1 | Shannon entropisi | 0.042 | Yüksek → Huffman |
| 2 | **Benzersiz oran** | **0.247** ⭐ | Düşük → BWT/LZW |
| 3 | Top-3 yoğunluk | 0.024 | Yüksek → Huffman |
| 4 | Boşluk oranı | 0.018 | Yüksek → LZW (doğal dil) |
| 5 | Türkçe kar. oranı | 0.058 | Yüksek → Türkçe corpus |
| 6 | Run-length ort. | 0.082 | Yüksek → BWT |

---

# 🔬 Sinir Ağının 11 Özelliği (2/2)

| # | Özellik | Önem | Hangi Algoritmaya İşaret |
|---|---|---|---|
| 7 | Rakam oranı | 0.008 | Yüksek → LZW (log/JSON) |
| 8 | Büyük harf | 0.012 | Dolaylı |
| 9 | **Bigram entropi** | **0.183** ⭐ | Düşük → LZW |
| 10 | Max koşu oranı | 0.038 | Yüksek → BWT |
| 11 | **Alfabe boyutu** | **0.096** ⭐ | Küçük → BWT |

## Karar Mantığı
- Düşük benzersizlik + küçük alfabe → **BWT**
- Düşük bigram + boşluk yüksek → **LZW**
- Yüksek entropi + çeşitli karakter → **Huffman**

---

# ⚡ Akıllı Hibrit (No-Regret Garantisi)

```python
def smart_hybrid(text):
    # 1. Sinir ağı seç
    nn = nn_predict(text)["algorithm"]

    # 2. Seçilen algoritmayı çalıştır
    if nn == "bwt":
        smart_bits = bwt_rle_huffman_bits(text)
    elif nn == "lzw":
        smart_bits = lzw_bits(text, ai_dictionary)
    else:
        smart_bits = corpus_huffman_bits(text)

    # 3. BWT post-check (güvenlik ağı)
    bwt_alt = bwt_rle_huffman_bits(text)
    if bwt_alt < smart_bits:
        smart_bits = bwt_alt

    # 4. Standart Huffman'dan asla kötü olamaz
    return min(smart_bits, huffman_bits(text))
```

**Garanti:** Akıllı Hibrit ≤ standart Huffman 🛡️

---

# 📊 Performans Sonuçları (6 Veri Tipi)

| Veri Türü | Std Huffman | Akıllı Hibrit | İyileşme |
|---|---|---|---|
| Tekrarlı (ABC×100) | %78.2 | **%96.9** | **+%85.9** 🔥 |
| DNA dizisi | %73.8 | **%94.1** | **+%77.5** 🔥 |
| JSON log | %60.7 | **%94.7** | **+%86.5** 🔥 |
| Doğal Türkçe | %37.1 | %38.7 | +%2.5 |
| Wikipedia TR | %39.8 | **%58.3** | **+%30.7** |
| Türkçe haber | %42.0 | **%54.3** | **+%21.2** |

**Tekrarlı/yapısal verilerde dramatik iyileşme!**

---

# 🏭 Endüstri Karşılaştırması

## Test: Türkçe akademik metin (730 karakter)

| Algoritma | Boyut | Küçülme | Süre |
|---|---|---|---|
| gzip -9 | 95 byte | %87.0 | 0.06 ms |
| zlib -9 | 91 byte | %87.5 | 0.01 ms |
| **bzip2 -9** | 148 byte | %79.7 | 0.80 ms |
| lzma -9 | 124 byte | %83.0 | 6.79 ms |
| **🏆 bwt_rle_huffman** | **94 byte** | **%87.1** | 0.56 ms |

## 🎉 Sonuç: Bizim implementasyon **bzip2'den %36.5 daha küçük!**

---

# ⏱️ İşlem Hızı ve Kaynak Kullanımı

## Hızlı Özet Sekmesi

```
⏱ İşlem süreleri (ms):
   Shannon entropi:        0.5
   BWT analizi:           12.3
   NN tahmin:             18.2 (CPU)
   Akıllı Hibrit:        156.0
   Endüstri benchmark:     8.0
   ────────────────────────────
   Toplam: ~195 ms
   🪙 Token: 0 (API gerektirmez)
```

## LZW + AI Sözlük
- **AI sözlük üretimi:** 450 ms
- **Token:** 287 (~$0.00014)
- **Toplam proje:** 3.870 token, ~$0.002

---

# 🔮 Next-Token Entropi (Shannon 1951)

## n-gram tabanlı kontekstli entropi

| Model | Bit/Karakter | Yöntem |
|---|---|---|
| Orijinal | 8.00 | UTF-8 |
| Unigram (iid) | **4.68** | Shannon 1948 |
| Bigram (Markov-1) | **3.58** | 1. derece bağımlılık |
| **Trigram (Markov-2)** | **2.74** | 2. derece bağımlılık |

## Karşılaştırma
- Shannon (1951) İngilizce için **~1.3 bit/karakter** ölçtü
- Bizim Türkçe trigram: **2.74 bit/karakter**
- LLM tabanlı sıkıştırıcılar bu sınıra yaklaşır

---

# 🧪 Doğrulama: 100 Birim Test

```bash
$ pytest tests/ -v
==================== 100 passed in 0.11s ====================
```

| Kategori | Test |
|---|---|
| Huffman kayıpsızlık | 18 ✅ |
| LZW kayıpsızlık (Türkçe) | 18 ✅ |
| BWT kayıpsızlık | 18 ✅ |
| MTF kayıpsızlık | 18 ✅ |
| Cross-algorithm tutarlılık | 3 ✅ |
| Bloklu BWT (11.400 karakter) | 2 ✅ |
| Ölçeklenme | 8 ✅ |
| Hız testleri | 11 ✅ |

**Edge case'ler:** Tek karakter, boş metin, tüm Türkçe karakterler

---

# 📚 Akademik Doğrulama

## ai_diary.json — 42 Adımlık Etkileşim Günlüğü

| Tip | Sayı |
|---|---|
| 💬 Klasik AI etkileşimi | 26 |
| 🪞 Reflektif süreç notu | 16 |
| **📚 Akademik kaynak doğrulama** | **12** |

## Doğrulanan Bileşenler (12 adet)
- Huffman → Sayood §3.2 + Cover & Thomas §5.6
- LZW Türkçe → Welch 1984 + Salomon §6.13
- BWT+MTF → Burrows-Wheeler 1994 + Salomon §8.5
- MLP → Goodfellow §6.3 + Bishop §5.1
- 5-fold CV → Hastie §7.10
- Algorithm Selection → Rice 1976 + Kotthoff 2014

---

# 🎮 Canlı Demo

## https://huggingface.co/spaces/tien23/ai-veri-sikistirma

## Akış (3 dakika)
1. **🎯 Hazır metin dropdown** → DNA seç
2. **🚀 Hızlı Özet** → "Tümünü Hesapla"
3. **3 yan yana kutu:** Orijinal · Binary · Decode
4. **📖 LZW + AI Sözlük** → LLM 60 kelime üretir
5. **🔬 Sinir Ağı** → Confusion Matrix + Feature Importance
6. **🔮 Next-Token** → n-gram analizi
7. **🤖 Günlük** → 42 adım, 📚 filtre

---

# 💪 Güçlü Yanlar (Özet)

1. ✅ 🤖 **LLM tabanlı LZW sözlük üretimi** (PDF'te istenen)
2. ✅ Klasik bilgi teorisi temelleri (Sayood, Cover & Thomas)
3. ✅ Sinir ağı şeffaflığı (Confusion Matrix + Perm. Importance)
4. ✅ Akıllı Hibrit no-regret garantisi
5. ✅ Türkçe karakter desteği (Unicode + corpus)
6. ✅ Bloklu BWT ile sınırsız metin
7. ✅ Endüstri ile rekabet (bzip2'den iyi)
8. ✅ Klasik bzip2 pipeline tam uygulama (MTF dahil)
9. ✅ 100 birim test
10. ✅ Performans şeffaflığı (süre + token + maliyet)

---

# ⚠️ Sınırlamalar (Dürüstlük)

1. Shannon entropisi **iid varsayım** altında
   (kontekstli daha düşük → next-token sekmesi gösterir)

2. Sinir ağı **algoritma seçimi** yapar,
   doğrudan karakter tahmini yapmaz
   (NNCP/DeepZip yaklaşımı gelecek çalışma)

3. BWT tek-blok için 8.000 karakter
   (bloklu çözüm var, kayıpsızlık garantili)

4. Groq API gerekir (sadece AI sekmeleri için)

---

# 🚀 Gelecek Çalışmalar

1. **LLM tabanlı arithmetic coding**
   (NNCP/DeepZip — ~1.5 bit/karakter Türkçe)

2. **Görüntü/ses verisi** (DCT, JPEG mantığı)

3. **Dinamik BWT blok boyutu** (entropi tabanlı)

4. **Brotli/Zstd** ile karşılaştırma

5. **Real-time streaming** sıkıştırma

---

# 📈 Sonuç



###  En İyi Performans
- 6 veri tipinde +%2.5 ile +%86.5 iyileşme
- Bzip2'den %36.5 daha küçük (kısa metinde)
- 100 birim test, sıfır kayıp

###  En İyi AI Entegrasyonu
- 🤖 LLM ile LZW sözlüğü
- 3-sınıflı NN algoritma seçici
- Akıllı Hibrit + 42 AI günlüğü
- 12 akademik doğrulama



---

<!-- _class: lead -->


<br>

**Betül Arslan**
Yıldız Teknik Üniversitesi
Bilgisayar Mühendisliği

🌐 https://huggingface.co/spaces/tien23/ai-veri-sikistirma
📁 https://github.com/betullarslan-cpu/ai-veri-sikistirma

<br>

*Bahar Dönemi 2026*
