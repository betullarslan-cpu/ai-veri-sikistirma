#  Sistem Mimarisi Diyagramları

## 1. Genel Sistem Akışı

```mermaid
flowchart TB
    A[Kullanıcı Metni] --> B[Özellik Çıkarımı]
    B --> |11 özellik| C[Sinir Ağı MLP 32→16→8]

    C -->|Tahmin: Huffman| D[Corpus Huffman]
    C -->|Tahmin: LZW| E[LZW + AI Sözlük]
    C -->|Tahmin: BWT| F[BWT + RLE + Huffman]

    D --> G[BWT Post-Check]
    E --> G
    F --> G

    G -->|En küçük olanı seç| H[Sıkıştırılmış Çıktı]
    H --> I[Binary .bin Dosyası]
    H --> J[Bit Dizisi Önizleme]
    H --> K[Hex Önizleme]
```

## 2. Sinir Ağı Mimarisi

```mermaid
graph LR
    subgraph "Girdi: 11 Özellik"
        I1[Entropi]
        I2[Benzersiz oran]
        I3[Top-3 yoğunluk]
        I4[Boşluk oranı]
        I5[Türkçe oranı]
        I6[Run-length]
        I7[Rakam oranı]
        I8[Büyük harf]
        I9[Bigram entropi]
        I10[Max koşu]
        I11[Alfabe log₂]
    end

    subgraph "Gizli Katmanlar"
        H1[32 nöron + ReLU]
        H2[16 nöron + ReLU]
        H3[8 nöron + ReLU]
    end

    subgraph "Çıktı: Softmax"
        O1[Huffman]
        O2[LZW]
        O3[BWT]
    end

    I1 & I2 & I3 & I4 & I5 & I6 & I7 & I8 & I9 & I10 & I11 --> H1
    H1 --> H2 --> H3
    H3 --> O1 & O2 & O3
```

## 3. BWT + RLE + Huffman Pipeline (bzip2 tekniği)

```mermaid
flowchart LR
    A[Orijinal Metin] -->|"Tüm permütasyonlar"| B[Suffix Array Sıralama]
    B --> C[BWT: son sütun + orig_idx]
    C -->|"Benzer karakterler kümelendi"| D[RLE: ch,count çiftleri]
    D --> E[Huffman Kodlama]
    E --> F[Sıkıştırılmış Bit Dizisi]

    F -.->|Decode için| G[BWT decode]
    G -->|LF Mapping| H[Orijinal Metin]
```

## 4. Modül Bağımlılıkları

```mermaid
graph TD
    A[app.py] --> B[core/hybrid.py]
    A --> C[core/bwt.py]
    A --> D[core/nn_selector.py]
    A --> E[core/benchmark.py]
    A --> F[core/ai_engine.py]
    A --> G[core/entropy.py]
    A --> H[core/huffman.py]
    A --> I[core/lzw.py]

    B --> H
    B --> I
    B --> C
    B --> D

    D --> C
    D --> H
    D --> I

    E --> H
    E --> I
    E --> C

    F -.->|"Groq API"| J[LLaMA 3.3 70B]
```

## 5. Veri Akışı (Encode/Decode)

```mermaid
sequenceDiagram
    participant U as Kullanıcı
    participant UI as Streamlit
    participant NN as Sinir Ağı
    participant ALG as Algoritma
    participant POS as BWT Post-Check

    U->>UI: Metin yapıştır
    UI->>NN: Özellikleri çıkar
    NN->>UI: Karar (Huffman/LZW/BWT)
    UI->>ALG: Seçilen algoritma çalıştır
    ALG->>UI: Bit dizisi + boyut
    UI->>POS: BWT ile karşılaştır
    POS->>UI: En küçük olanı döndür
    UI->>U: Sıkıştırılmış çıktı + grafik
```

---

**Mermaid kullanımı:** Bu diyagramlar GitHub'da otomatik render olur.
Yerel görmek için VS Code'da "Markdown Preview Mermaid Support" eklentisi kur.
