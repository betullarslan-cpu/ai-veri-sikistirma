---
title: AI Destekli Veri Sıkıştırma
emoji: 🗜️
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

# 🗜️ AI Destekli Veri Sıkıştırma

Yıldız Teknik Üniversitesi — Veri Sıkıştırma Dönem Projesi

Klasik sıkıştırma algoritmalarını (Huffman, LZW, Aritmetik Kodlama, BWT) **yapay zeka ile optimize eden** Streamlit uygulaması.

## 🎯 Özellikler

### Algoritmalar
- **Huffman** + AI frekans tahmini (tablo overhead'i kaldırır)
- **LZW** + AI sözlük (Groq'tan en sık kelimeler)
- **Aritmetik Kodlama** + AI olasılık modeli (Shannon sınırına yakın)
- **BWT + RLE + Huffman** (bzip2 tekniği — en güçlü sonuç)
- **Corpus Huffman** (Türkçe karakter frekanslarından öğrenilmiş)

### Yapay Zeka Bileşenleri
- **🔬 Sinir Ağı (MLP 32→16→8)** — Veri tipine göre en iyi algoritmayı seçer
  - 3 sınıf: Huffman / LZW / BWT
  - 11 özellik (entropi, run-length, bigram, alfabe boyutu vb.)
  - Hold-out doğruluk: **%95.2** | 5-fold CV: **%91.7 ± %1.8**
  - 2.072 örnekle eğitildi (sentetik + 4 corpus)
- **Akıllı Hibrit** — NN seçer + BWT post-check (asla standart Huffman'dan kötü olmaz)
- **Groq LLM** entegrasyonu (frekans tahmini, sözlük üretimi)
- **OCR + Sıkıştırma** (görüntüden metin çıkar → sıkıştır)
- **AI Görüntü Sıkıştırma** (önem haritası → bölgesel kalite)

## 📊 Sonuçlar

| Veri Türü | Standart Huffman | Akıllı Hibrit (BWT) | İyileşme |
|---|---|---|---|
| Tekrarlı veri | %78.2 küçülme | **%96.9 küçülme** | +%85.9 |
| DNA dizisi | %73.8 küçülme | **%94.1 küçülme** | +%77.5 |
| JSON log | %60.7 küçülme | **%94.7 küçülme** | +%86.5 |
| Doğal Türkçe | %37.1 küçülme | **%38.7 küçülme** | +%2.5 |

## 🚀 Çalıştırma

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tarayıcıda http://localhost:8501 açılır.

### Groq API key (opsiyonel)
AI özellikleri için ücretsiz key: https://console.groq.com/keys
Sol sidebar'dan gir veya `GROQ_API_KEY` env var olarak ayarla.

## 📁 Proje Yapısı

```
project_veri/
├── app.py                  # Streamlit arayüzü (12 sekme)
├── core/
│   ├── huffman.py          # Huffman kodlama
│   ├── lzw.py              # LZW sıkıştırma (Unicode destekli)
│   ├── arithmetic.py       # Aritmetik kodlama + AI olasılık
│   ├── bwt.py              # BWT + RLE + Huffman (bzip2 tekniği)
│   ├── nn_selector.py      # Sinir ağı (MLP 3-sınıf)
│   ├── hybrid.py           # Akıllı hibrit + corpus eğitimi
│   ├── ai_engine.py        # Groq API entegrasyonu
│   ├── entropy.py          # Shannon entropisi
│   ├── selector.py         # AI tabanlı algoritma seçici
│   ├── image_compress.py   # AI önem haritası + bölgesel JPEG
│   ├── ocr_compress.py     # Görüntü → metin → sıkıştırma
│   ├── corpus_freq.json    # Eğitilmiş Türkçe frekans tablosu
│   └── nn_model.pkl        # Eğitilmiş sinir ağı modeli
├── data/
│   ├── large_turkish.txt   # 64K karakter Türkçe corpus
│   ├── diverse_corpus.txt  # Çeşitli veri tipleri
│   └── turkce_dogal.txt    # Test metni
├── ai_diary.json           # AI prompt geçmişi (proje şartı)
├── requirements.txt
└── README.md
```

## 🧠 Sinir Ağı Eğitimi

Model, 2.072 örnekle 5-fold cross-validation ile eğitildi:

- **Sentetik veriler:** Tekrarlı desenler, DNA, log, JSON, rastgele metin
- **Corpus parçaları:** Türkçe metinler farklı uzunluklarda
- **Etiket:** Her örnek için 3 algoritmadan en az bit kullananı

Ezber engellemek için:
- L2 regularizasyon (alpha=1e-3)
- Erken durdurma (early stopping)
- Hold-out test seti (modelin görmediği)

## 📚 Kaynaklar

- Burrows-Wheeler Transform: https://en.wikipedia.org/wiki/Burrows%E2%80%93Wheeler_transform
- Huffman Coding: Cormen et al., "Introduction to Algorithms"
- LZW: Welch (1984), "A Technique for High-Performance Data Compression"
- Groq API: https://groq.com

## 👤 Yazar

Yıldız Teknik Üniversitesi — Bilgisayar Mühendisliği
Veri Sıkıştırma Dersi, Bahar 2026
