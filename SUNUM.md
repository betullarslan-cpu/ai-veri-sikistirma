---
marp: true
theme: default
paginate: true
backgroundColor: #fff
header: 'AI-Destekli Veri Sıkıştırma — Betül Arslan'
footer: 'Yıldız Teknik Üniversitesi — Bahar 2026'
---

<!-- _class: lead -->

# 🗜️ AI-Destekli Veri Sıkıştırma

## Yıldız Teknik Üniversitesi
### Veri Sıkıştırma Dönem Projesi

**Betül Arslan**
Bahar 2026

---

# 🎯 Problem

> "Hangi sıkıştırma algoritması en iyi sonucu verir?"

- Huffman → karakter bazlı (doğal dil iyi)
- LZW → pattern bazlı (tekrarlı veride iyi)
- BWT → yapısal düzen bazlı (DNA, log için harika)

**Kullanıcı manuel seçmek zorunda kalıyor.**

➡️ Çözüm: **Sinir ağı otomatik seçsin.**

---

# 🏗️ Sistem Mimarisi

```
Metin → Özellik Çıkarımı (11 özellik)
      ↓
      Sinir Ağı MLP 32→16→8
      ↓
    Huffman / LZW / BWT
      ↓
   BWT Post-Check (güvenlik ağı)
      ↓
    Sıkıştırılmış Çıktı (.bin)
```

---

# 🧠 Sinir Ağı Detayları

- **Mimari:** MLP 32 → 16 → 8
- **Girdi:** 11 özellik (entropi, alfabe, run-length, bigram, ...)
- **Çıktı:** 3 sınıf (Huffman / LZW / BWT)
- **Eğitim:** 2.357 örnek, 5-fold CV

| Metrik | Değer |
|---|---|
| Hold-out doğruluk | **%95.2** |
| Cross-validation | **%91.7 ± %1.8** |

**Ezber yok** — gerçek genelleme.

---

# ⚡ BWT + RLE + Huffman (bzip2 tekniği)

3 aşamalı pipeline:

1. **BWT** → benzer karakterleri kümele
2. **RLE** → tekrarları (ch, count) ile sıkıştır
3. **Huffman** → optimal bit kodlaması

**Tekrarlı verilerde +%85 iyileşme.**
**Türkçe Wikipedia metninde bzip2'yi geçti.**

---

# 📊 Sonuçlar

| Veri | Std Huffman | Akıllı Hibrit | İyileşme |
|---|---|---|---|
| Tekrarlı | %78 | **%97** | **+%85** |
| DNA | %74 | **%94** | **+%77** |
| JSON log | %61 | **%95** | **+%86** |
| Türkçe Wiki | %47 | **%57** | **+%30** |

**Standart Huffman'a göre asla kötü değil** (post-check garantisi).

---

# 🏭 Endüstri Karşılaştırması

Türkçe Wikipedia metni (4.516 karakter):

| Araç | Boyut | Küçülme |
|---|---|---|
| zlib | 2.154 byte | %56.4 |
| bzip2 | 2.154 byte | %56.4 |
| gzip | 2.166 byte | %56.2 |
| **Bizim BWT** | **2.147 byte** | **%56.5** |
| **Akıllı Hibrit** | 2.266 byte | %54.1 |

**bzip2'den %0.3 daha iyi.**

---

# ✅ Kanıtlar (Eleştirelliğe Dayanıklı)

- **72 unit testi** — kayıpsızlık her senaryoda doğrulandı
- **GitHub Actions CI** — her commit'te otomatik test
- **Permutation importance** — NN'in hangi özelliği kullandığı şeffaf
- **gzip/bzip2 benchmark** — endüstri standardı karşılaştırma
- **Türkçe karakter** desteği (ş, ğ, ü, ö, ç, ı)

---

# 🌐 Canlı Demo

**HuggingFace Spaces:**
https://huggingface.co/spaces/tien23/ai-veri-sikistirma

**GitHub:**
https://github.com/betullarslan-cpu/ai-veri-sikistirma

8 sekme:
- 🚀 Hızlı Özet • 📊 Huffman • 📖 LZW
- 🔬 Sinir Ağı • ⚡ Hibrit • 📈 Shannon
- 🔄 BWT • 🤖 AI Günlüğü

---

# 🤖 AI Etkileşim Süreci

**30+ AI etkileşimi belgelenmiş** (`ai_diary.json`):

- Algoritma kararları
- Hata düzeltme (Türkçe `ı` bug'ı)
- NN ezber tartışması ("CV şart!")
- BWT eklemenin gerekçesi
- Reflektif analiz

> "AI kullandım ama **körü körüne değil, eleştirel olarak**."

---

# 💡 Öğrendiklerim

1. **Sadece doğruluk skoruna güvenme** — CV şart
2. **Sadelik > karmaşıklık** — gereksiz sekme/modül kaldırıldı
3. **AI'nın önerisi haklı olabilir ama uygulaması bana ait**
4. **Türkçe destek = teknik çaba** (Unicode + LZW = bug'lar)
5. **Demo > rapor** — canlı çalışan sistem daha ikna edici

---

<!-- _class: lead -->

# 🙏 Teşekkürler

## Sorularınız?

**Betül Arslan**
Yıldız Teknik Üniversitesi
Bilgisayar Mühendisliği
