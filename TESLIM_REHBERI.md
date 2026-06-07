# 🎓 ÖDEV TESLİMİ — Adım Adım Rehber

> **Bu doküman teslime kadar yapılacak son işleri sırayla anlatır.**
> Tahmini toplam süre: **45-60 dakika**

---

## 📋 GENEL DURUM

✅ **Hazır olanlar:**
- Kod (GitHub + HuggingFace güncel)
- 86 unit test (kayıpsızlık + scaling)
- Belgeler: RAPOR.md, SUNUM.md, MIMARI.md, README.md
- Eğitilmiş NN modeli (%95.2 doğruluk)
- AI günlüğü (`ai_diary.json`, 30+ adım)

⏳ **Bu rehberle tamamlayacaklar:**
1. RAPOR.md → PDF
2. SUNUM.md → PDF (slayt)
3. 5 ekran görüntüsü al
4. PDF'lere ekran görüntüleri yerleştir
5. ZIP hazırla
6. Yıldız sistemine yükle
7. Onay ekran görüntüsü al

---

# ADIM 1 — RAPOR.md → PDF (10 dakika)

## Yöntem A — Online (Kurulum yok, hızlı)

1. **VS Code'da `RAPOR.md` dosyasını aç**
2. **Tümünü kopyala** (`Cmd+A`, sonra `Cmd+C`)
3. Tarayıcıda aç: **https://md2pdf.netlify.app/**
4. Sol panele yapıştır
5. Sağda canlı önizleme görünür
6. Üst köşeden **"Download PDF"** bas → `RAPOR.pdf` indirilir
7. İndirilenler klasöründen `project_veri` klasörüne taşı

## Yöntem B — VS Code eklentisi (Daha düzenli görsel)

1. VS Code → Extensions → **"Markdown PDF"** (yoshinobu yamashita) ara → Install
2. `RAPOR.md`'yi aç
3. Sağ tık → **"Markdown PDF: Export (pdf)"**
4. Aynı klasörde `RAPOR.pdf` oluşur

## 📝 Word'de Düzenleme (Opsiyonel ama ÖNERİLİR)

PDF'i Word'de açıp şunları ekle:
- **Kapak sayfası:**
  - Yıldız Teknik Üniversitesi logosu (web'den indir)
  - "Veri Sıkıştırma Dönem Projesi"
  - Senin ismin + öğrenci numaran
  - Danışman ismi
  - Tarih (Haziran 2026)
- **İçindekiler** (Word otomatik oluşturur: Refs → TOC)
- **Sayfa numaraları** (Insert → Page Number)

---

# ADIM 2 — SUNUM.md → PDF Slayt (10 dakika)

## Yöntem A — Marp Online

1. Tarayıcıda aç: **https://marp.app/**
2. **"Try Marp Online"** veya **"Web"** seç
3. `SUNUM.md` içeriğini kopyala → yapıştır
4. Sağda slaytlar görünür
5. **"Export"** → **PDF** seç
6. `SUNUM.pdf` indir

## Yöntem B — VS Code Marp eklentisi

1. VS Code Extensions → **"Marp for VS Code"** kur
2. `SUNUM.md`'yi aç
3. Sağ üstte 🎯 ikonuna tıkla → **"Export Slide Deck"**
4. PDF veya HTML olarak indir

---

# ADIM 3 — 5 Ekran Görüntüsü Al (15 dakika)

> **HuggingFace Space'e gir:** https://huggingface.co/spaces/tien23/ai-veri-sikistirma
> Sayfada Streamlit uygulaması açılır.

## Mac'te ekran görüntüsü kısayolu
- **`Cmd + Shift + 4`** → ekrandan seç (alan seçimi)
- **`Cmd + Shift + 4` + `Space`** → tüm pencere
- Görüntü Desktop'a `Screen Shot 2026-...png` olarak kaydedilir

## 5 Mutlaka Alınması Gereken Ekran Görüntüsü

### 1️⃣ Hızlı Özet Sekmesi — Tüm sonuçlar
- Test metni yapıştır:
```
Yapay zeka, son yillarda hayatimizin her alanini etkileyen onemli bir teknolojidir.
Veri sikistirma alaninda da yapay zeka kullanimi yayginlaşmaktadir.
Klasik algoritmalar onlarca yildir kullaniliyor ancak yapay zeka ile daha verimli sonuclar elde edilebilir.
```
- **▶ Tümünü Hesapla** bas
- Beklenmesi: 4 metric + sinir ağı kararı + grafik + benchmark tablosu görünür
- **Tüm görünür alanı çek** (scroll'sız tek ekran)
- İsim: `01_hizli_ozet.png`

### 2️⃣ Sinir Ağı Sekmesi — Confusion Matrix
- 🔬 Sinir Ağı sekmesine git
- **📐 Confusion Matrix Göster** bas
- Heatmap + sınıf bazlı tablo görünsün
- **Tüm görünür alanı çek**
- İsim: `02_confusion_matrix.png`

### 3️⃣ Sinir Ağı Sekmesi — Feature Importance
- Aynı sekmede aşağı in
- **📊 Önem analizini başlat** bas
- Yatay bar grafik görünür
- **Tüm görünür alanı çek**
- İsim: `03_feature_importance.png`

### 4️⃣ BWT Sekmesi — Sıkıştırılmış çıktı + Decode aşamaları
- 🔄 BWT sekmesine git
- **▶ BWT Sıkıştırma Analizi Başlat** bas
- Aşağı in, "💾 Sıkıştırılmış Çıktı" + "🔓 Açma" görünsün
- Decode aşamalarından birini aç (Aşama 5 — LF zinciri)
- **Tüm görünür alanı çek**
- İsim: `04_bwt_decode_asamalari.png`

### 5️⃣ Hibrit Sekmesi — Akıllı Hibrit sonucu
- ⚡ Hibrit sekmesine git
- **▶ Akıllı Hibrit Çalıştır** bas
- NN kararı + standart Huffman'a göre iyileşme % görünsün
- **Tüm görünür alanı çek**
- İsim: `05_akilli_hibrit.png`

### Bonus (İsteğe bağlı)
- 6️⃣ Huffman sekmesi — sıkıştırılmış bit dizisi + hex + decode
- 7️⃣ Shannon sekmesi — entropi grafiği

---

# ADIM 4 — Ekran Görüntülerini Rapora Ekle (10 dakika)

Word'de açtığın `RAPOR.pdf`'ye ekleyeceğin yer:

## EK-C: Ekran Görüntüleri bölümü

Mevcut RAPOR.md'de "EK-C: Ekran Görüntüleri" bölümü var. Word'de:

1. EK-C başlığını bul
2. Insert → Picture → File ile her PNG'yi ekle
3. Her görüntü altına başlık yaz:
   ```
   Şekil 1: Hızlı Özet sekmesi — tüm sıkıştırma sonuçları tek ekranda
   Şekil 2: Confusion matrix — sinir ağı doğruluğu
   Şekil 3: Feature importance — NN'in en önemli özellikleri
   Şekil 4: BWT decode aşamaları — kayıpsızlık kanıtı
   Şekil 5: Akıllı Hibrit — NN seçimi + iyileşme
   ```

4. PDF olarak kaydet (File → Save As → PDF)

---

# ADIM 5 — linkler.txt Oluştur (1 dakika)

VS Code'da yeni dosya: `linkler.txt`

İçerik:
```
=================================================
AI-Destekli Veri Sıkıştırma Projesi
Betül Arslan - YTÜ Bilgisayar Mühendisliği
=================================================

🌐 CANLI DEMO (HuggingFace Spaces):
https://huggingface.co/spaces/tien23/ai-veri-sikistirma

📁 KAYNAK KOD (GitHub):
https://github.com/betullarslan-cpu/ai-veri-sikistirma

📄 ÖNEMLI DOSYALAR:
- RAPOR.pdf       → Proje raporu
- SUNUM.pdf       → Sunum slaytları
- ai_diary.json   → AI etkileşim günlüğü
- README.md       → Proje tanıtımı
- MIMARI.md       → Sistem diyagramları
- tests/          → 86 birim test

✅ HAZIR ÖZELLİKLER:
- 5 algoritma (Huffman, LZW, Aritmetik, BWT, Akıllı Hibrit)
- Sinir ağı %95.2 doğruluk (5-fold CV %91.7)
- gzip/bzip2 ile karşılaştırma
- Türkçe karakter desteği
- 8 sekmeli Streamlit arayüz
- Canlı HuggingFace deployment
```

---

# ADIM 6 — ZIP Hazırla (3 dakika)

## Mac'te Finder ile

1. Finder aç
2. Desktop → `project_veri` klasörüne git
3. Klasörün **ÜZERİNE** sağ tık
4. **"project_veri'yi Sıkıştır"** seç
5. Aynı dizine `project_veri.zip` oluşur

## Adlandırma

```bash
# Terminal'de proje klasöründeyken:
mv ~/Desktop/project_veri.zip ~/Desktop/no_BetulArslan.zip
```
(`no` → öğrenci numaran)

## ZIP içeriği kontrol

```bash
unzip -l ~/Desktop/no_BetulArslan.zip | head -30
```

Görmesi gerekenler:
```
- app.py
- core/*.py (8-9 dosya)
- data/*.txt (5 dosya)
- tests/*.py
- RAPOR.pdf       ← bu olmalı (sen ekleyeceksin)
- SUNUM.pdf
- ai_diary.json
- README.md
- MIMARI.md
- linkler.txt
- Dockerfile
- requirements.txt
```

## ZIP'e PDF'leri Eklemek (sıkıştırdıktan sonra)

ZIP'in içine ek dosya eklemek için:
1. Finder → ZIP'i çift tıkla → klasör açılır
2. PDF dosyalarını içine sürükle
3. Yeniden ZIP'le (sağ tık → Sıkıştır)

VEYA önce dosyaları project_veri içine kopyalayıp sonra zip yap (daha temiz).

---

# ADIM 7 — Yıldız Online Sistemine Yükle (5 dakika)

1. Tarayıcıda aç: **https://online.yildiz.edu.tr**
2. Giriş yap (öğrenci numaran + şifren)
3. Sol menüden **Derslerim** → **Veri Sıkıştırma**
4. Sağ tarafta **Ödevler** veya **Görevler** linki
5. **Dönem Projesi** ödevini bul
6. **"Ödev Yükle"** veya **"Teslim Et"** butonu
7. `no_BetulArslan.zip` dosyasını seç
8. **Onay/Gönder** butonuna bas

## ÖNEMLI: Onay sayfasının ekran görüntüsünü al!

Yükleme tamamlanınca **"Başarıyla teslim edildi"** veya benzer mesaj çıkar.
- **`Cmd + Shift + 4`** ile çek
- Kanıt olarak sakla (sistem sorunlarına karşı)

---

# ADIM 8 — Son Kontrol Listesi

Teslim öncesi her şeyi tekrar gözden geçir:

## Kod ✅
- [ ] HuggingFace Space açılıyor: https://huggingface.co/spaces/tien23/ai-veri-sikistirma
- [ ] GitHub repo güncel: https://github.com/betullarslan-cpu/ai-veri-sikistirma
- [ ] `streamlit run app.py` yerel olarak çalışıyor

## Belgeler ✅
- [ ] RAPOR.pdf hazır (kapak + içindekiler + ekran görüntüleri)
- [ ] SUNUM.pdf hazır
- [ ] linkler.txt oluşturuldu
- [ ] ai_diary.json güncel (30 adım var)

## ZIP ✅
- [ ] `no_BetulArslan.zip` oluştu
- [ ] İçinde RAPOR.pdf, SUNUM.pdf, kod, ai_diary.json var
- [ ] Boyut makul (~10-50 MB)

## Yükleme ✅
- [ ] Yıldız sistemine yüklendi
- [ ] Onay alındı
- [ ] Onay ekran görüntüsü kaydedildi

---

# 🎯 ZAMAN ÇİZELGESİ

| Adım | İş | Süre |
|---|---|---|
| 1 | RAPOR.md → PDF | 10 dk |
| 2 | SUNUM.md → PDF | 10 dk |
| 3 | 5 ekran görüntüsü | 15 dk |
| 4 | PDF'lere ekran görüntüleri | 10 dk |
| 5 | linkler.txt | 1 dk |
| 6 | ZIP | 3 dk |
| 7 | Sisteme yükleme | 5 dk |
| 8 | Son kontrol | 5 dk |
| **Toplam** | | **~60 dk** |

---

# 💡 SON İPUÇLARI

## Yüklemeden Önce Test
HuggingFace Space'ini son bir kez aç ve şu metni yapıştır:
```
Bu, hocanın benzer şekilde verebileceği bir test metnidir.
Türkçe karakterlerle birlikte çalışmayı doğrular: ş, ğ, ü, ö, ç, ı.
```

Hızlı Özet → **▶ Tümünü Hesapla** → sorunsuz çalışmalı.

## Yedek Plan
Eğer Yıldız sistemine yükleme başarısız olursa:
1. Hocaya **email** at: `betulla...@yildiz.edu.tr`'den hoca email'i
2. Email içeriği:
   - GitHub linki
   - HuggingFace linki
   - ZIP'i ek olarak
   - "Sistem hatası nedeniyle..." açıklaması

## Sınav Günü
Eğer sözlü savunma istenirse SUNUM.pdf'i göster + canlı demo'yu aç.

---

# 🎓 BAŞARILAR!

Bu kadar emek verilmiş bir projeye **iyi sonuç almaman çok zor**.

Sorularına hazırlık için: `SUNUM_YAPISI.md` dosyasındaki **10 olası soru ve cevabı** oku.

---

**Hazırlayan:** Claude (proje süreci boyunca AI asistanı)
**Tarih:** Haziran 2026
**Proje sahibi:** Betül Arslan, YTÜ Bilgisayar Mühendisliği
