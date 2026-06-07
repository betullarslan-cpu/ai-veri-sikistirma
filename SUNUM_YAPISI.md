# 📋 ÖDEV TESLİM VE SUNUM YAPISI

## AI-Destekli Veri Sıkıştırma Projesi
**Öğrenci:** Betül Arslan
**Ders:** Veri Sıkıştırma — Dönem Projesi
**Teslim Tarihi:** 15 Haziran 2026, 23:59

---

# 1. TESLİM PAKETİ YAPISI

## 1.1 Yıldız Online Sistemine Yüklenecekler

```
no_BetulArslan.zip
├── RAPOR.pdf                    ← Ana belge (10-15 sayfa)
├── ai_diary.json                ← AI prompt geçmişi
├── README.md                    ← Proje tanıtımı
├── linkler.txt                  ← GitHub + HF linki
└── kod/                         ← Tüm kaynak kod
    ├── app.py
    ├── core/ (8 modül)
    ├── data/ (4 corpus dosyası)
    ├── requirements.txt
    └── Dockerfile
```

## 1.2 Online Linkler (linkler.txt içeriği)

```
📁 Kaynak Kod (GitHub):
https://github.com/betullarslan-cpu/ai-veri-sikistirma

🌐 Canlı Demo (HuggingFace Spaces):
https://huggingface.co/spaces/tien23/ai-veri-sikistirma

🤖 AI Etkileşim Günlüğü:
ai_diary.json (ZIP içinde)
```

---

# 2. SUNUM AKIŞI (10-12 Dakika)

## Bölüm 1: Giriş (1 dk)
> "Veri sıkıştırmada her algoritmanın güçlü olduğu veri tipi farklıdır. Klasik algoritmalar bu seçimi yapamaz — kullanıcı manuel seçmek zorundadır. Ben bu kararı **bir sinir ağına** verdim. Sistem 5 algoritmayı eşzamanlı çalıştırıp en iyisini otomatik seçer."

**Slayt:** Proje başlığı + 3 link (GitHub, HF, AI günlüğü)

## Bölüm 2: Sistem Mimarisi (2 dk)
> "Sistemde 3 ana katman var: Klasik algoritmalar (Huffman/LZW/Aritmetik/BWT), AI katmanı (Groq LLM frekans tahmini), Karar katmanı (sinir ağı + akıllı hibrit)."

**Slayt:** Mimari diyagramı (RAPOR.md'deki ASCII şema)

## Bölüm 3: Sinir Ağı (3 dk) ⭐
> "Bu projenin en özgün parçası. MLP 32→16→8 mimarisi, 11 özellikten Huffman/LZW/BWT seçimi yapıyor. Entropi, run-length, bigram entropisi gibi metrikleri girdi alıyor. 5-fold cross-validation ile **%91.7 doğruluk** — yani ezber değil, gerçek genelleme."

**Slayt:** NN mimarisi şeması + doğruluk metrikleri
**Demo:** HF Space → Sinir Ağı sekmesi → "Sinir Ağı Tahmin Et"

## Bölüm 4: BWT + RLE + Huffman (2 dk) ⭐
> "Klasik Huffman + LZW yetmiyordu. bzip2'nin temeli olan BWT'yi ekledim. Permütasyon → RLE → Huffman ile **tekrarlı verilerde %96.9 küçülme** sağlıyoruz. Standart Huffman'a göre **+%85 iyileşme**."

**Slayt:** BWT örneği — `"banana"` → permütasyon → kümeleme
**Demo:** HF Space → BWT sekmesi

## Bölüm 5: AI Entegrasyonu (1.5 dk)
> "Groq LLM (LLaMA 3.3 70B) ile Huffman tablosu önceden tahmin ediliyor → veriyle birlikte gönderme zorunluluğu (overhead) ortadan kalkıyor. Tahmin kalitesi KL-Divergence ile ölçülüyor."

**Slayt:** AI prompt örneği + dönen JSON
**Demo:** HF Space → Hızlı Özet sekmesi → "Tümünü Hesapla"

## Bölüm 6: Sonuçlar (2 dk)
> "Doğal Türkçe metinlerde +%30 iyileşme. Tekrarlı verilerde +%85 iyileşme. Shannon teorik limitine **%8** uzaklık — neredeyse optimal."

**Slayt:** Sonuç tablosu (5 veri tipi × 3 algoritma karşılaştırması)
**Demo:** Sıkıştırılmış `.bin` dosyasını indir → boyutu Finder'da göster

## Bölüm 7: Kapanış (0.5 dk)
> "Sistem GitHub'da açık kaynak ve HuggingFace Spaces'te canlı çalışıyor. Herkes tarayıcıdan açıp test edebilir."

**Slayt:** Final ekran görüntüsü + linkler

---

# 3. DEMO SENARYOSU (HuggingFace Space üzerinden)

## Hazırlık (sunum öncesi)
- HF Space'i açık tut: https://huggingface.co/spaces/tien23/ai-veri-sikistirma
- Test metni hazır olsun (aşağıdaki örnek)

## Adım Adım Demo

### Adım 1: Hızlı Özet — Genel Bakış
1. **Hızlı Özet** sekmesi
2. Şu metni yapıştır:
```
Yapay zeka teknolojileri, son yıllarda büyük gelişmeler kaydetmiştir.
Veri sıkıştırma alanında da yapay zekanın katkıları artmaktadır.
Huffman ve LZW gibi klasik algoritmalar hala temel taşı olarak kullanılır.
Burrows-Wheeler dönüşümü ise modern sıkıştırma sistemlerinin omurgasıdır.
```
3. **▶ Tümünü Hesapla** bas
4. Göster:
   - 4 metrik (Orijinal, Shannon, Huffman, **Akıllı Hibrit**)
   - NN kararı satırı: "BWT seçildi, %X iyileşme"
   - Grafik: 5 algoritma karşılaştırması
   - **💡 expander'ları aç** → her sonucun ne demek olduğunu hocaya göster
   - Sağ kolonda: NN olasılıkları + sıkıştırılmış bit dizisi
5. **⬇ BWT sıkıştırılmış (.bin)** butonuna bas → dosya indirilsin
6. Finder'da göster: orijinal 400 byte vs sıkışmış 60 byte

### Adım 2: Sinir Ağı — Karar Mekanizması
1. **Sinir Ağı** sekmesi
2. Hold-out %95.2 ve CV %91.7 metriklerini göster
3. **▶ Sinir Ağı Tahmin Et** bas
4. Olasılık grafiği + 11 özellik göster
5. Açıkla: "Bu modeli 2.072 örnek ile eğittim, ezber değil — gerçek doğruluk"

### Adım 3: BWT — En Güçlü Algoritma
1. **BWT** sekmesi
2. BWT dönüşüm etkisi göster (koşu azalması metrikleri)
3. **▶ BWT Sıkıştırma Analizi Başlat** bas
4. 5 algoritmadan BWT+RLE+Huffman'ın kazanmasını göster
5. **🔬 BWT Doğruluk Testi** expander'ı aç:
   - Test metni gir
   - Encode → Decode kayıpsız çalıştığını göster

### Adım 4: Hibrit — Akıllı Seçim
1. **Hibrit** sekmesi
2. **▶ Akıllı Hibrit Çalıştır** bas
3. NN kararı + standart Huffman'a göre iyileşme yüzdesi

### Adım 5: AI Günlüğü
1. **Günlük** sekmesi
2. Birkaç prompt örneği göster
3. Toplam token sayısını göster

---

# 4. OLASI HOCAM SORULARI ve CEVAPLAR

### Soru 1: "Sinir ağın ne kadar veriyle eğitildi?"
> "2.072 örnek — bu sentetik veri ve 4 farklı Türkçe corpus'tan veri augmentasyonuyla oluşturuldu. Sınıf dağılımı: Huffman 943, LZW 35, BWT 1054. Dengesiz olmasın diye stratified split kullandım."

### Soru 2: "Bu sinir ağı gerçekten öğrendiğini nasıl biliyorsun? Ezber olabilir mi?"
> "Üç yöntemle kanıtladım:
> 1. **5-fold cross-validation** ile %91.7 ± %1.8 — standart sapma çok düşük, tutarlı
> 2. **Hold-out test seti** (%20, eğitimde hiç görmediği) — %95.2 doğruluk
> 3. **L2 regularizasyon + early stopping** ile overfitting önlemi aldım"

### Soru 3: "BWT'yi neden ekledin? Huffman/LZW yetmiyor mu?"
> "Doğal dilde Huffman/LZW %30-40 küçülme yapıyor. Ama tekrarlı verilerde, log dosyalarında, DNA dizilerinde çok zayıf kalıyorlar. BWT permütasyonu benzer karakterleri kümeleyince RLE harika çalışıyor → %85+ iyileşme. bzip2'nin temel mantığı bu."

### Soru 4: "AI olmadan da çalışıyor mu?"
> "Evet. AI sadece bonus özellik — Groq LLM ile frekans tahmini ve sözlük üretimi. Sinir ağı tamamen yerel (scikit-learn, internet gerekmez). Hızlı Özet, Sinir Ağı, BWT, Hibrit, Shannon sekmeleri API olmadan çalışıyor."

### Soru 5: "Türkçe karakter desteği nasıl?"
> "Klasik LZW 0-255 ASCII başlatıyor — Türkçe karakterler (ş, ğ, ü, ö, ç, ı) Unicode'da 255'ten büyük olduğu için hata veriyordu. Metindeki tüm benzersiz karakterleri başlangıç sözlüğüne ekleyerek çözdüm."

### Soru 6: "Gzip/Bzip2'ye göre nasıl?"
> "Standart Huffman vs bizim BWT+RLE+Huffman karşılaştırması doğal Türkçe'de +%30 iyileşme. Gerçek bzip2'ye (referans implementasyon) yakın ama eşit değil — bizimki saf eğitim amaçlı. Hız değil sıkıştırma oranı odaklı."

### Soru 7: "Decompress edebiliyor mu? Lossless mi?"
> "Evet. BWT encode → decode test ettim, kayıpsız. BWT sekmesindeki Doğruluk Testi bölümünden canlı görebilirsiniz."

### Soru 8: "Bu çıktıyı disk üzerinde gerçek dosya olarak nasıl gösteriyor?"
> "Hızlı Özet'teki '⬇ BWT sıkıştırılmış (.bin)' butonu gerçek binary dosyayı indirir. Finder'da boyutunu görebilirsiniz — 1200 byte Türkçe metin → 350 byte binary."

### Soru 9: "Sinir ağı için neden 11 özellik?"
> "Her özellik bir hipotezi temsil ediyor: entropi (yüksek = Huffman iyi), max koşu (yüksek = BWT iyi), alfabe boyutu (küçük = BWT iyi), bigram entropisi (düzen = LZW iyi). 11 özellik dengeli — daha azı yetersiz, daha çoğu overfitting riski."

### Soru 10: "Streamlit yerine neden Gradio/Flask değil?"
> "Streamlit veri bilimi projeleri için optimize, hızlı prototyping sağlıyor. Plotly entegrasyonu hazır. HuggingFace Spaces'te Docker SDK ile sorunsuz çalışıyor."

---

# 5. KONTROL LİSTESİ (Teslim Öncesi)

## Kod Kontrolü
- [x] GitHub'da kod güncel: https://github.com/betullarslan-cpu/ai-veri-sikistirma
- [x] HF Space çalışıyor: https://huggingface.co/spaces/tien23/ai-veri-sikistirma
- [x] requirements.txt eksiksiz
- [x] README.md proje açıklaması içeriyor

## Rapor Kontrolü
- [ ] RAPOR.md → PDF'e dönüştürüldü (https://md2pdf.netlify.app/)
- [ ] Kapak sayfası eklendi (okul logosu, öğrenci no, tarih)
- [ ] Ekran görüntüleri eklendi (EK-C kısmı):
  - [ ] Hızlı Özet sonucu
  - [ ] Sinir Ağı sekmesi
  - [ ] BWT sekmesi grafiği
  - [ ] Hibrit sekmesi sonucu
- [ ] Kaynaklar bölümü tam

## AI Günlüğü
- [x] ai_diary.json güncel
- [ ] Önemli prompt örnekleri rapora eklendi

## ZIP Hazırlığı
- [ ] Finder → project_veri klasörü → Sağ tık → Sıkıştır
- [ ] ZIP adı: `no_BetulArslan.zip` (no = öğrenci no)
- [ ] RAPOR.pdf ZIP'in içinde
- [ ] linkler.txt eklendi

## Yıldız Sistemi
- [ ] online.yildiz.edu.tr'de doğru ödev sayfasına giriş
- [ ] ZIP yüklendi
- [ ] Onay alındı

---

# 6. SUNUM ÖNCESİ HAZIRLIK

## Bilgisayar Hazırlığı
1. HF Space açık ve test edilmiş
2. Yedek olarak yerelde Streamlit çalışsın: `streamlit run app.py`
3. Test metni hazır (kopyala-yapıştır için)
4. İnternet bağlantısı stabil (Groq API için)

## Yedek Plan
- HF Space çökerse → yerel Streamlit
- Yerel de çalışmazsa → GitHub'dan ekran görüntüleri göster
- Hiçbiri olmazsa → RAPOR.pdf'teki grafikleri göster

## Test Metinleri (Demo İçin)

### Doğal Türkçe (BWT kazanır)
```
Yapay zeka teknolojileri hizla gelisiyor. Makine ogrenmesi modelleri
verilerden ornek alir ve tahmin yapar. Sinir aglari karmasik desenleri
yakalamada cok basarilidir.
```

### Tekrarlı (BWT %95 küçülme)
```
ABCABCABCABCABCABCABCABCABCABCABCABCABCABCABCABCABCABCABCABCABCABCABCABC
```

### Log dosyası (BWT mükemmel)
```
[INFO 2026-01-01] islem basarili
[INFO 2026-01-02] islem basarili
[INFO 2026-01-03] islem basarili
[INFO 2026-01-04] islem basarili
[INFO 2026-01-05] islem basarili
```

---

# 7. SİSTEM GENEL ÖZELLİKLERİ (Bir Bakışta)

| Özellik | Değer |
|---------|-------|
| **Kod satırı** | ~3.000+ Python |
| **Modül sayısı** | 9 (core/) |
| **Sekme sayısı** | 8 (Streamlit) |
| **Algoritma sayısı** | 5 (Huffman, LZW, Aritmetik, BWT, Akıllı Hibrit) |
| **AI entegrasyonu** | Groq LLaMA 3.3 70B + scikit-learn MLP |
| **NN doğruluk** | %95.2 hold-out / %91.7 CV |
| **Eğitim örneği** | 2.072 |
| **Sıkıştırma performansı** | %85+ (tekrarlı), %30+ (Türkçe) |
| **Türkçe destek** | ✅ Unicode + corpus eğitimi |
| **Canlı demo** | ✅ HuggingFace Spaces |
| **Açık kaynak** | ✅ GitHub |
| **AI günlüğü** | ✅ 25+ prompt |
| **Test edilmiş** | ✅ Encode/decode kayıpsız |

---

*Bu doküman ödev hazırlığı için kişisel notlardır — teslim paketine eklemen gerekmez.*
