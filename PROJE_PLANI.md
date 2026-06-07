# 📅 PROJE PLANI — Sıfırdan Teslime Yol Haritası

## AI-Destekli Veri Sıkıştırma Projesi
**Toplam süre:** 10-14 gün (her gün ~3-5 saat)

---

# AŞAMA 0: HAZIRLIK (1 gün)

## 0.1 Konu Analizi
- [ ] Ödev şartlarını oku (hocanın PDF'i)
- [ ] Değerlendirme kriterleri: **Performans** + **AI Entegrasyonu**
- [ ] Teslim formatı: rapor + kod + AI günlüğü
- [ ] Deadline: 15 Haziran 2026

## 0.2 Çevre Hazırlığı
- [ ] Python 3.9+ kur (zaten var)
- [ ] VS Code + Python eklentisi
- [ ] GitHub hesabı (kod yedekleme)
- [ ] HuggingFace hesabı (canlı demo için)
- [ ] Groq API hesabı (ücretsiz AI için) — https://console.groq.com

## 0.3 Konu Seçimi
**Soru:** Hangi algoritmayı geliştireceğim?
**Cevap:** AI-Destekli Uyarlamalı Huffman → sonra LZW → sonra BWT eklenir

**Karar mantığı:**
- Huffman → klasik temel ✓
- LZW → tekrarlı veride iyi ✓
- BWT → bzip2 mantığı, çok güçlü ✓
- Sinir ağı → algoritma seçer (en özgün katkı) ⭐

---

# AŞAMA 1: TEMEL ALTYAPI (2-3 gün)

## 1.1 Klasik Algoritmaları İmplement Et
**Hedef:** AI olmadan da çalışan temel sistem

### Gün 1
- [ ] `core/huffman.py` — Standart Huffman ağacı + encode
  - HuffmanNode sınıfı
  - Frekans hesabı → ağaç → kod tablosu
  - Test: "merhaba" → bit dizisi

- [ ] `core/lzw.py` — LZW sıkıştırma
  - Sözlük tabanlı encode/decode
  - **Türkçe karakter sorunu yakala** (Unicode > 255)
  - Çözüm: metindeki tüm karakterleri başlangıç sözlüğüne ekle

### Gün 2
- [ ] `core/entropy.py` — Shannon entropisi
  - Bilgi teorisinin teorik minimumu
  - Karşılaştırma referansı

- [ ] `core/arithmetic.py` — Aritmetik kodlama (teorik)
  - bits = -Σ log₂(p(xᵢ))
  - Shannon limitine yakın

## 1.2 Test Verisi Hazırla
- [ ] Türkçe corpus: Wikipedia paragrafları (data/large_turkish.txt)
- [ ] Tekrarlı veri: ABCABC... (data/sample.txt)
- [ ] Doğal cümleler (data/turkce_dogal.txt)
- [ ] Karma corpus: DNA + log + Türkçe (data/diverse_corpus.txt)

## 1.3 İlk Sonuçları Görmek
- [ ] Basit script ile her algoritmayı çalıştır
- [ ] %küçülme yüzdesini hesapla
- [ ] **Gözlem:** Hangi veri hangi algoritmaya iyi geliyor?

**Milestone 1:** ✅ 4 algoritma çalışıyor, sonuçlar elde edildi

---

# AŞAMA 2: AI ENTEGRASYONU (2 gün)

## 2.1 LLM API Seçimi
**Aday API'ler:**
- Anthropic Claude — kart gerekli ❌
- Google Gemini — quota düşük ❌
- **Groq — ücretsiz, hızlı** ✓
- OpenAI — kart gerekli ❌

**Karar:** Groq (LLaMA 3.3 70B)

## 2.2 AI Modülü
- [ ] `core/ai_engine.py` — Groq client + prompt'lar
- [ ] **Prompt mühendisliği:**
  - Frekans tahmini için 2 aşamalı prompt
  - JSON çıktı zorunluluğu
  - Türkçe karakter desteği

## 2.3 AI Hata Yönetimi
**Karşılaşılacak hatalar:**
- Model decommissioned (model adı değişebilir)
- JSON parse hatası (LLM kod bloğu ekler, single quote kullanır)
- Rate limit (429)
- Invalid API key (401)

**Çözüm:** Robust parser + try/except + Türkçe hata mesajları

## 2.4 KL-Divergence
- [ ] AI tahmininin gerçeğe ne kadar yakın olduğunu ölç
- [ ] `entropy.py` içine kl_divergence ekle

**Milestone 2:** ✅ AI frekans tahmini çalışıyor, KL ölçülüyor

---

# AŞAMA 3: SİNİR AĞI (2-3 gün)

## 3.1 Problem Tanımı
**Soru:** Verilen metin için hangi algoritma en iyi?
**Cevap:** Bu bir **sınıflandırma problemi** — 3 sınıf (Huffman/LZW/BWT)

## 3.2 Özellik Mühendisliği
- [ ] Hangi özellikler ayırt edici?
  1. Shannon entropisi
  2. Karakter çeşitliliği
  3. Türkçe karakter oranı
  4. Run-length (BWT için kritik)
  5. Bigram entropisi (LZW için)
  6. Alfabe boyutu (BWT için)
  7. Boşluk/rakam/büyük harf oranı
  8. Top-3 karakter yoğunluğu

**Toplam:** 11 özellik

## 3.3 Eğitim Verisi
**Problem:** Doğal Türkçe metinlerde her zaman Huffman/LZW kazanıyor
**Çözüm:** Sentetik veri ekle (DNA, JSON, log, tekrarlı)

- [ ] `core/nn_selector.py` → `_synthetic_samples()`
- [ ] 9 kategori sentetik veri
- [ ] Veri augmentasyonu: kayan pencere ile %50 örtüşme
- [ ] Hedef: ~2.000 örnek

## 3.4 Model
**Mimari karar:**
- Çok küçük → underfit
- Çok büyük → overfit
- **MLP 32→16→8** (3 gizli katman)

**Düzenleyiciler:**
- L2 regularizasyon (alpha=1e-3)
- Early stopping (20 iter, val %15)
- StandardScaler

## 3.5 Doğrulama
**Tek skor yetmez — ezber olabilir!**
- [ ] 5-fold cross-validation (CV)
- [ ] Hold-out test seti (%20)
- [ ] Hedef: CV %85+, hold-out %85+

**Milestone 3:** ✅ NN doğrulanmış, ezbersiz

---

# AŞAMA 4: BWT EKLEME (1-2 gün) ⭐

## 4.1 Motivasyon
**Gözlem:** Klasik algoritmalar %30-40 kaldı, zlib %88 yapıyor.
**Hipotez:** BWT eklemek farkı kapatır.

## 4.2 İmplementasyon
- [ ] `core/bwt.py`
  - bwt_encode (suffix array ile)
  - bwt_decode (LF mapping)
  - rle_compress / rle_decompress
  - bwt_rle_huffman_bits (bit hesabı)
  - bwt_rle_huffman_encode (gerçek bit dizisi)

## 4.3 Test
- [ ] Encode → decode kayıpsız mı? (banana, merhaba dunya)
- [ ] Tekrarlı veride sonuç ne? **Beklenen: %95+ küçülme**

## 4.4 Entegrasyon
- [ ] NN'i 2 sınıf → 3 sınıf yap (BWT eklendi)
- [ ] Yeniden eğit
- [ ] `smart_hybrid()` BWT'yi de seçebilsin

**Milestone 4:** ✅ BWT entegre, NN 3 sınıflı

---

# AŞAMA 5: HİBRİT SİSTEM (1 gün)

## 5.1 Akıllı Hibrit Mantığı
```
1. NN tahmini al → algoritma seç
2. Seçilen algoritmayı çalıştır
3. BWT'yi de çalıştır (post-check)
4. Bit sayısı küçük olanı al
5. Standart Huffman'dan asla kötü değil!
```

## 5.2 Güvenlik Garantisi
- [ ] Her sonucu standart Huffman ile kıyasla
- [ ] Eğer akıllı seçim daha kötüyse standart kullan
- [ ] Test: tüm corpuslarla doğrula

**Milestone 5:** ✅ Hibrit sistem garanti optimal

---

# AŞAMA 6: WEB ARAYÜZÜ (2 gün)

## 6.1 Streamlit Seçimi
**Neden Streamlit?**
- Veri bilimi için optimize
- Plotly entegrasyonu hazır
- 1 dosyada UI
- HF Spaces destekli

## 6.2 Sekme Yapısı
**İlk versiyon:** 10+ sekme (her algoritma için ayrı)
**Sadeleştirme:** Gereksizleri çıkar (Görüntü, OCR, AI Seçici, Aritmetik)
**Son versiyon:** 8 sekme

**Ana sekme:** 🚀 Hızlı Özet — tek butonla her şey

## 6.3 Görselleştirme
- [ ] Plotly bar grafikleri
- [ ] Shannon limiti çizgisi
- [ ] Renk kodlu sonuçlar (sarı=kazanan)
- [ ] Sıkıştırılmış binary önizleme + download

## 6.4 UX İyileştirmeleri
- [ ] Her sonuç altına 💡 açıklama
- [ ] Tooltip'ler (help=)
- [ ] Sidebar kompakt
- [ ] API key sadece gerekli sekmelerde

**Milestone 6:** ✅ Arayüz hazır, kullanıcı dostu

---

# AŞAMA 7: DEPLOYMENT (1 gün)

## 7.1 GitHub
- [ ] git init + repo oluştur
- [ ] .gitignore (venv, __pycache__)
- [ ] README.md (HF metadata header'ı dahil)
- [ ] requirements.txt
- [ ] İlk commit + push

## 7.2 HuggingFace Spaces
**SDK seçimi:**
- ~~Streamlit~~ (yeni arayüzde kaldırıldı)
- **Docker** ✓ (Dockerfile ile)
- Gradio (UI yeniden yazmak gerek)

- [ ] Dockerfile (python:3.11-slim + Streamlit komutu)
- [ ] HF Space oluştur (Docker → Streamlit template)
- [ ] HF token al (Write yetki)
- [ ] git push (force, ilk seferinde)

## 7.3 Karşılaşılan Sorunlar
- Python sürüm uyumsuzluğu (3.13 vs 3.11)
- sklearn versiyon uyumsuzluğu (pickle)
- **Çözüm:** Otomatik model yeniden eğitimi

**Milestone 7:** ✅ Canlı demo URL hazır

---

# AŞAMA 8: TEST VE DOĞRULAMA (1 gün)

## 8.1 Algoritma Testleri
- [ ] Encode → decode kayıpsız (BWT)
- [ ] Türkçe karakterler (LZW)
- [ ] Edge cases (boş metin, tek karakter)

## 8.2 NN Testleri
- [ ] Hold-out doğruluk
- [ ] CV doğruluk
- [ ] Farklı veri tiplerinde tahmin
- [ ] Confidence dağılımı

## 8.3 Performans Karşılaştırması
**Test seti:**
| Veri tipi | Std Huffman | Akıllı Hibrit | İyileşme |
|-----------|-------------|---------------|----------|
| Doğal Türkçe | %37 | %39 | +%2 |
| Tekrarlı | %78 | %97 | +%85 |
| DNA | %74 | %94 | +%77 |
| Log dosyası | %43 | %91 | +%55 |
| Wikipedia | %40 | %58 | +%30 |

**Milestone 8:** ✅ Sonuçlar belgelendi

---

# AŞAMA 9: RAPOR (1-2 gün)

## 9.1 Yapı
1. Özet (1 paragraf)
2. Giriş (amaç + katkılar)
3. Literatür özeti
4. Sistem mimarisi (diyagram)
5. Algoritmalar (detay)
6. AI entegrasyonu
7. Eğitim yöntemi
8. Deneysel sonuçlar
9. AI günlüğü özeti
10. Sonuç
11. Kaynaklar
12. Ekler (kod yapısı, ekran görüntüleri)

## 9.2 Görseller
- [ ] Sistem mimarisi şeması
- [ ] NN mimari diyagramı
- [ ] Sonuç tablosu
- [ ] Bar grafik karşılaştırması
- [ ] Streamlit ekran görüntüleri (4 adet)

## 9.3 PDF Dönüşümü
- [ ] RAPOR.md hazırla
- [ ] Markdown → PDF (md2pdf.netlify.app)
- [ ] Kapak sayfası ekle (Word'de)
- [ ] Ekran görüntülerini yerleştir

**Milestone 9:** ✅ RAPOR.pdf hazır

---

# AŞAMA 10: AI GÜNLÜĞÜ (Sürekli)

## 10.1 Her Aşamada Kayıt
**Hangi promptu verdim → hangi cevabı aldım → ne işime yaradı?**

- [ ] `ai_diary.json` formatında saklan
- [ ] Her LLM çağrısında otomatik log
- [ ] Manuel "fikir" girişleri için tab10

## 10.2 İçerik
- Konu seçimi yardımı
- Algoritma karşılaştırması
- Hata düzeltme (Türkçe karakter bug'ı)
- NN mimari önerisi
- BWT eklenmesi tartışması

**Milestone 10:** ✅ Günlük ZIP'e dahil edilebilir

---

# AŞAMA 11: TESLİM (1 gün)

## 11.1 Son Kontroller
- [ ] Kod GitHub'da temiz
- [ ] HF Space çalışıyor
- [ ] RAPOR.pdf düzgün görünüyor
- [ ] AI günlüğü dolu
- [ ] Ekran görüntüleri eklendi

## 11.2 ZIP Yapısı
```
no_BetulArslan.zip
├── RAPOR.pdf
├── ai_diary.json
├── README.md
├── linkler.txt (GitHub + HF)
└── kod/
    ├── app.py
    ├── core/
    ├── data/
    ├── requirements.txt
    └── Dockerfile
```

## 11.3 Sisteme Yükle
- [ ] online.yildiz.edu.tr giriş
- [ ] Doğru ödev sayfası
- [ ] ZIP yükle
- [ ] Onay sayfası ekran görüntüsü al (kanıt)

**Milestone 11:** ✅ ÖDEV TESLİM EDİLDİ 🎉

---

# 📊 ZAMAN ÇİZELGESİ ÖNERİSİ

| Hafta | Gün | Aşama | Süre |
|-------|-----|-------|------|
| 1 | 1 | Hazırlık + Konu | 3 saat |
| 1 | 2-3 | Klasik algoritmalar | 8 saat |
| 1 | 4-5 | AI entegrasyonu | 6 saat |
| 1 | 6-7 | Sinir ağı | 8 saat |
| 2 | 8 | BWT ekle | 4 saat |
| 2 | 9 | Hibrit | 3 saat |
| 2 | 10-11 | Web arayüzü | 6 saat |
| 2 | 12 | Deployment | 4 saat |
| 2 | 13 | Test + rapor başlangıç | 5 saat |
| 2 | 14 | Rapor + teslim | 5 saat |

**Toplam:** ~50 saat çalışma

---

# 🎯 KARAR NOKTALARI (Kritik Anlar)

## K1: Konu seçimi
**Karar:** AI-Destekli Huffman → genişletilebilir mimari

## K2: AI API
**Karar:** Groq (ücretsiz + hızlı)

## K3: NN sınıf sayısı
**Karar:** 2 → 3 (BWT eklenince)

## K4: SDK seçimi
**Karar:** Streamlit (Docker üzerinden HF'de)

## K5: Performans iyileştirmesi
**Karar:** BWT+RLE+Huffman ekle (bzip2 tekniği)

## K6: Sekme sayısı
**Karar:** 12 → 8 (gereksizleri çıkar)

---

# 💡 ÖĞRENDİKLERİM (Reflektif)

1. **Sadelik önemli** — başta çok özellik ekledim, sonra kullanıcı kafası karıştı → sadeleştirdim
2. **NN ezber olabilir** — tek bir doğruluk skoru güvenilmez, CV şart
3. **AI sadece bonus** — sistem AI olmadan da çalışmalı
4. **Türkçe için Unicode hatası tipiktir** — başta kaçınılmaz
5. **Deployment'ta versiyon farkı** — pickle dosyası farklı sürümde açılmayabilir → otomatik retrain çözüm
6. **Sıkıştırma orijinal dosya görmek lazım** — sadece "% küçülme" yetmez, gerçek bit'i göster

---

# ✅ TESLİM ÖNCESİ SON CHECKLIST

- [ ] **Kod:** GitHub + HF Space güncel
- [ ] **Rapor:** PDF hazır, ekran görüntüleri ekli
- [ ] **AI Günlüğü:** ai_diary.json dolu
- [ ] **Demo:** HF Space test edildi
- [ ] **ZIP:** doğru isimle paketlendi
- [ ] **Sunum:** akış prova edildi (SUNUM_YAPISI.md)
- [ ] **Linkler:** linkler.txt hazır
- [ ] **Yıldız:** sisteme yüklendi
- [ ] **Onay:** kanıt ekran görüntüsü alındı

---

*Bu plan referans amaçlıdır — proje sürecinin retrospektif analizidir.*
