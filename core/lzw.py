"""
LZW Sıkıştırma — Lempel-Ziv-Welch (1984)
==========================================

ALGORİTMA MANTIĞI
-----------------
LZW, sözlük tabanlı bir kayıpsız sıkıştırma algoritmasıdır. Tekrarlayan
karakter dizilerini sözlükteki indekslerle değiştirerek çalışır.

Encode adımları:
1. Başlangıç sözlüğü: 0-255 ASCII karakterler (+ Türkçe karakterler)
2. Mevcut girdiyi (w) bir karaktere (c) ekle → wc
3. Eğer wc sözlükteyse, w'yu wc yap, devam et
4. Sözlükte değilse: w'nin kodunu yaz, wc'yi sözlüğe ekle, w'yu c yap
5. Sonda kalan w'nin kodunu yaz

Decode adımları:
- Aynı başlangıç sözlüğüyle başla
- Her kodu okuyup karşılık gelen string'i yaz
- Sözlüğü encoder ile **aynı sırada** büyüt (özellik!)

TEORİK ÖZELLİKLER
-----------------
- Sözlük dosya başında **gönderilmez** — decoder kendi sözlüğünü kurar
- Tekrarlı verilerde çok iyi (her pattern bir kod = log₂(N) bit)
- Karmaşıklık: O(n) encode, O(n) decode
- ASCII başlangıç + Türkçe karakter eklemesi (Unicode > 255 için zorunlu)

YENİLİKLER (Bu Projede)
-----------------------
- **AI Akıllı Sözlük:** Groq LLM'den metne özgü en sık 60 kelime alınır,
  başlangıç sözlüğüne eklenir. Bu sayede ilk pattern eşleşmeleri daha hızlı.
- **Unicode desteği:** Türkçe karakterler (ş, ğ, ü, ö, ç, ı) başlangıç
  sözlüğüne eklenir — KeyError'ı önler.

REFERANS
--------
Welch, T. A. (1984). "A Technique for High-Performance Data Compression."
IEEE Computer, 17(6), 8–19.
"""

from collections import Counter


# ─────────────────────────────────────────
# Standart LZW
# ─────────────────────────────────────────

def lzw_encode(text: str, initial_dict: dict = None) -> list[int]:
    # Tüm metindeki karakterleri sözlüğe ekle (Unicode desteği)
    all_chars = set(text)
    if initial_dict:
        dictionary = dict(initial_dict)
        next_code = max(initial_dict.values()) + 1
    else:
        dictionary = {chr(i): i for i in range(256)}
        next_code = 256
    # 256 üzeri Unicode karakterleri (ı, ş, ğ vb.) ekle
    for ch in all_chars:
        if ch not in dictionary:
            dictionary[ch] = next_code
            next_code += 1

    result = []
    w = ""
    for c in text:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            dictionary[wc] = next_code
            next_code += 1
            w = c
    if w:
        result.append(dictionary[w])
    return result, len(dictionary)


def lzw_decode(codes: list[int], initial_dict: dict = None) -> str:
    if initial_dict:
        rev = {v: k for k, v in initial_dict.items()}
        next_code = max(initial_dict.values()) + 1
    else:
        rev = {i: chr(i) for i in range(256)}
        next_code = 256

    result = []
    w = rev[codes[0]]
    result.append(w)
    for code in codes[1:]:
        if code in rev:
            entry = rev[code]
        elif code == next_code:
            entry = w + w[0]
        else:
            break
        result.append(entry)
        rev[next_code] = w + entry[0]
        next_code += 1
        w = entry
    return "".join(result)


# ─────────────────────────────────────────
# Sıkıştırma oranı hesabı
# ─────────────────────────────────────────

def compression_stats(text: str, codes: list[int], dict_size: int):
    import math
    bits_per_code = math.ceil(math.log2(max(dict_size, 2)))
    original_bits = len(text) * 8
    compressed_bits = len(codes) * bits_per_code
    ratio = compressed_bits / original_bits
    return {
        "original_bits": original_bits,
        "compressed_bits": compressed_bits,
        "ratio": ratio,
        "codes_count": len(codes),
        "dict_size": dict_size,
        "bits_per_code": bits_per_code,
    }


# ─────────────────────────────────────────
# AI sözlüğünü LZW formatına çevir
# ─────────────────────────────────────────

def build_ai_lzw_dict(ai_words: list[str]) -> dict:
    """
    AI'nın önerdiği kelime listesini LZW başlangıç sözlüğüne ekle.
    Tek karakterler 0-255, AI kelimeleri 256'dan başlar.
    """
    dictionary = {chr(i): i for i in range(256)}
    next_code = 256
    for word in ai_words:
        if word not in dictionary:
            dictionary[word] = next_code
            next_code += 1
    return dictionary


# ─────────────────────────────────────────
# Karşılaştırma
# ─────────────────────────────────────────

def compare(text: str, ai_words: list[str]) -> dict:
    # Standart LZW
    std_codes, std_dict_size = lzw_encode(text)
    std_stats = compression_stats(text, std_codes, std_dict_size)

    # AI-LZW
    ai_dict = build_ai_lzw_dict(ai_words)
    ai_codes, ai_dict_size = lzw_encode(text, ai_dict)
    ai_stats = compression_stats(text, ai_codes, ai_dict_size)

    return {
        "standard": std_stats,
        "ai_lzw": ai_stats,
        "ai_words_added": len(ai_words),
        "saved_bits": std_stats["compressed_bits"] - ai_stats["compressed_bits"],
    }
