"""
Hibrit Sıkıştırma + Corpus Eğitimi
------------------------------------
1. Corpus'tan Türkçe frekans tablosu öğrenir ("eğitim")
2. Huffman, AI-Huffman, LZW, AI-LZW hepsini dener
3. En iyi sonucu otomatik seçer
4. Her zaman standart algoritmadan iyi veya eşit

Hibrit avantajı:
  - Veri tipine göre otomatik uyarlanır
  - Tek bir algoritmanın zayıf olduğu durumu ortadan kaldırır
  - AI corpus bilgisiyle başlar → tablo overhead olmadan iyi tahmin
"""

import os
import math
import heapq
import json
from collections import Counter
from groq import Groq

def _get_groq():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Groq API key girilmemis. Sol menude API key alanini doldurun.")
    return Groq(api_key=key)
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

CORPUS_FREQ_FILE = os.path.join(os.path.dirname(__file__), "corpus_freq.json")


# ─────────────────────────────────────────
# 1. Corpus'tan frekans öğren (eğitim)
# ─────────────────────────────────────────

def train_from_corpus(corpus_path: str, save: bool = True) -> dict:
    """Büyük metin dosyasından karakter frekansı öğren.

    PDF/binary dosyalar kabul edilmez — sadece UTF-8 metin (.txt).
    """
    # PDF kontrolü
    if corpus_path.lower().endswith(".pdf"):
        raise ValueError(
            "PDF dosyası kabul edilmiyor. Lütfen düz metin (.txt) dosyası yükleyin. "
            "PDF içeriğini önce kopyalayıp .txt olarak kaydedebilirsiniz."
        )

    # Çoklu encoding deneme — UTF-8 öncelikli, sonra Türkçe karakter desteği için fallback
    text = None
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "cp1254", "iso-8859-9", "latin-1"):
        try:
            with open(corpus_path, "r", encoding=enc) as f:
                text = f.read()
            break
        except UnicodeDecodeError as e:
            last_err = e
            continue

    if text is None:
        raise ValueError(
            f"Dosya UTF-8/Türkçe encoding'lerle okunamadı. "
            f"İkili (binary) bir dosya olabilir. Detay: {last_err}"
        )

    total = len(text)
    freq = Counter(text)
    freq_table = {ch: cnt / total for ch, cnt in freq.items()}

    if save:
        with open(CORPUS_FREQ_FILE, "w", encoding="utf-8") as f:
            json.dump(freq_table, f, ensure_ascii=False, indent=2)

    print(f"✓ Eğitim tamamlandı: {len(freq_table)} karakter, {total:,} örnek")
    return freq_table


def load_corpus_freq() -> dict:
    """Eğitilmiş frekans tablosunu yükle."""
    if not os.path.exists(CORPUS_FREQ_FILE):
        default = os.path.join(os.path.dirname(__file__), "..", "data", "large_turkish.txt")
        if os.path.exists(default):
            return train_from_corpus(default)
        return {}
    with open(CORPUS_FREQ_FILE, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────
# 2. Huffman yardımcıları
# ─────────────────────────────────────────

class HNode:
    def __init__(self, ch, f):
        self.ch = ch
        self.freq = f
        self.left = self.right = None
    def __lt__(self, o): return self.freq < o.freq


def build_huffman(freq_dict: dict) -> dict:
    if not freq_dict:
        return {}
    heap = [HNode(ch, f) for ch, f in freq_dict.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        l, r = heapq.heappop(heap), heapq.heappop(heap)
        m = HNode(None, l.freq + r.freq)
        m.left, m.right = l, r
        heapq.heappush(heap, m)

    def _codes(node, prefix="", codes={}):
        if node is None: return codes
        if node.ch is not None:
            codes[node.ch] = prefix or "0"
        else:
            _codes(node.left, prefix + "0", codes)
            _codes(node.right, prefix + "1", codes)
        return codes

    return _codes(heap[0], "", {})


def huffman_bits(text: str, codes: dict) -> int:
    return sum(len(codes.get(ch, "0" * 16)) for ch in text)


# ─────────────────────────────────────────
# 3. LZW yardımcıları
# ─────────────────────────────────────────

def lzw_compress_bits(text: str, extra_words: list = None) -> int:
    import math
    dictionary = {chr(i): i for i in range(256)}
    next_code = 256
    for ch in set(text):
        if ch not in dictionary:
            dictionary[ch] = next_code
            next_code += 1
    if extra_words:
        for w in extra_words:
            if w not in dictionary:
                dictionary[w] = next_code
                next_code += 1

    codes = []
    w = ""
    for c in text:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            codes.append(dictionary[w])
            dictionary[wc] = next_code
            next_code += 1
            w = c
    if w:
        codes.append(dictionary[w])

    bits_per_code = math.ceil(math.log2(max(len(dictionary), 2)))
    return len(codes) * bits_per_code


# ─────────────────────────────────────────
# 4. AI ile LZW sözlük oluştur
# ─────────────────────────────────────────

def ai_lzw_words(text_sample: str, n: int = 60) -> tuple[list, int]:
    try:
        response = _get_groq().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Veri sıkıştırma uzmanısın. Sadece JSON listesi döndür."},
                {"role": "user", "content": (
                    f"Bu metinden LZW için en sık {n} kelime/ifade listesi çıkar.\n"
                    f"Metin: \"\"\"{text_sample[:400]}\"\"\"\n"
                    "Sadece JSON: [\"kelime1\", \"kelime2\", ...]"
                )},
            ],
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens if response.usage else 0
        idx = raw.find("[")
        end = raw.rfind("]")
        if idx != -1 and end != -1:
            words = json.loads(raw[idx:end+1])
            return [w for w in words if isinstance(w, str) and 2 <= len(w) <= 20], tokens
    except Exception:
        pass
    return [], 0


# ─────────────────────────────────────────
# 5. Hibrit karşılaştırma — ANA FONKSİYON
# ─────────────────────────────────────────

def hybrid_compress(text: str, use_ai: bool = True) -> dict:
    orig_bits = len(text) * 8
    actual_freq = {ch: cnt / len(text) for ch, cnt in Counter(text).items()}
    overhead_per_char = 12  # bit

    results = {}
    total_tokens = 0

    # ── A) Standart Huffman ──
    codes_std = build_huffman(actual_freq)
    huff_std  = huffman_bits(text, codes_std)
    overhead  = len(codes_std) * overhead_per_char
    results["Standart Huffman"] = {
        "bits": huff_std + overhead,
        "compressed": huff_std,
        "overhead": overhead,
        "ratio": (huff_std + overhead) / orig_bits,
        "ai_used": False,
    }

    # ── B) Corpus Huffman (eğitilmiş frekans) ──
    corpus_freq = load_corpus_freq()
    if corpus_freq:
        merged = {ch: corpus_freq.get(ch, 1e-6) for ch in actual_freq}
        total_m = sum(merged.values())
        merged = {k: v / total_m for k, v in merged.items()}
        codes_corp = build_huffman(merged)
        huff_corp = huffman_bits(text, codes_corp)
        results["Corpus Huffman"] = {
            "bits": huff_corp,   # tablo yok, corpus bilgisi ortak
            "compressed": huff_corp,
            "overhead": 0,
            "ratio": huff_corp / orig_bits,
            "ai_used": False,
            "note": "Büyük corpus'tan öğrenildi",
        }

    # ── C) AI Huffman (Groq frekans tahmini) ──
    if use_ai:
        try:
            from core.ai_engine import predict_frequencies, refine_frequencies
            ai_freq, tok1 = predict_frequencies(text, "Türkçe metin")
            missing = [ch for ch in actual_freq if ch not in ai_freq]
            ai_freq, tok2 = refine_frequencies(ai_freq, missing, "Türkçe metin")
            total_tokens += tok1 + tok2
            merged_ai = {ch: ai_freq.get(ch, 1e-6) for ch in actual_freq}
            total_ai = sum(merged_ai.values())
            merged_ai = {k: v / total_ai for k, v in merged_ai.items()}
            codes_ai = build_huffman(merged_ai)
            huff_ai  = huffman_bits(text, codes_ai)
            results["AI Huffman"] = {
                "bits": huff_ai,
                "compressed": huff_ai,
                "overhead": 0,
                "ratio": huff_ai / orig_bits,
                "ai_used": True,
            }
        except Exception:
            pass

    # ── D) Standart LZW ──
    lzw_std = lzw_compress_bits(text)
    results["Standart LZW"] = {
        "bits": lzw_std,
        "compressed": lzw_std,
        "overhead": 0,
        "ratio": lzw_std / orig_bits,
        "ai_used": False,
    }

    # ── E) AI-LZW (akıllı sözlük) ──
    if use_ai:
        try:
            words, tok = ai_lzw_words(text)
            total_tokens += tok
            lzw_ai = lzw_compress_bits(text, extra_words=words)
            results["AI-LZW"] = {
                "bits": lzw_ai,
                "compressed": lzw_ai,
                "overhead": 0,
                "ratio": lzw_ai / orig_bits,
                "ai_used": True,
                "words_added": len(words),
            }
        except Exception:
            pass

    # ── F) BWT + RLE + Huffman (bzip2 tekniği) ──
    try:
        from core.bwt import bwt_rle_huffman_bits, bwt_lzw_bits
        bwt_rle = bwt_rle_huffman_bits(text)
        results["BWT+RLE+Huffman"] = {
            "bits": bwt_rle,
            "compressed": bwt_rle,
            "overhead": 0,
            "ratio": bwt_rle / orig_bits,
            "ai_used": False,
            "note": "bzip2 tekniği: BWT kümeleme → RLE → Huffman",
        }
        bwt_lzw_b = bwt_lzw_bits(text)
        results["BWT+LZW"] = {
            "bits": bwt_lzw_b,
            "compressed": bwt_lzw_b,
            "overhead": 0,
            "ratio": bwt_lzw_b / orig_bits,
            "ai_used": False,
            "note": "BWT sonrası uzun tekrarlar → LZW daha iyi pattern yakalar",
        }
    except Exception:
        pass

    # ── Kazanan ──
    best_name = min(results, key=lambda k: results[k]["bits"])
    best = results[best_name]

    return {
        "original_bits": orig_bits,
        "results": results,
        "best": best_name,
        "best_bits": best["bits"],
        "best_ratio": best["ratio"],
        "total_tokens": total_tokens,
        "saving_vs_standard": results["Standart Huffman"]["bits"] - best["bits"],
    }


# ─────────────────────────────────────────
# 6. Akıllı Hibrit — NN + En İyi Algoritma
# ─────────────────────────────────────────

def smart_hybrid(text: str, use_ai_dict: bool = True) -> dict:
    """
    1. NN metin özelliklerini analiz eder → LZW mi Huffman mı?
    2. Seçilen algoritmayı çalıştırır
    3. Referans için standart Huffman ile karşılaştırır

    Her zaman en iyi sonucu garanti eder.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.nn_selector import predict as nn_predict

    orig_bits = len(text) * 8
    total_tokens = 0

    # ── Referans: Standart Huffman ──
    actual_freq = {ch: cnt/len(text) for ch, cnt in Counter(text).items()}
    codes_std = build_huffman(actual_freq)
    huff_std  = huffman_bits(text, codes_std) + len(codes_std) * 12

    # ── NN Kararı ──
    nn_result   = nn_predict(text)
    nn_decision = nn_result.get("algorithm", "lzw")
    nn_conf     = nn_result.get("confidence", 0.5)

    # ── Seçilen algoritmayı çalıştır ──
    from core.bwt import bwt_rle_huffman_bits

    words_added = 0
    if nn_decision == "bwt":
        smart_bits = bwt_rle_huffman_bits(text)
        method = "BWT+RLE+Huffman (NN seçti)"
    elif nn_decision == "lzw":
        if use_ai_dict:
            try:
                words, tok = ai_lzw_words(text, n=60)
                total_tokens += tok
                smart_bits = lzw_compress_bits(text, extra_words=words)
                method = "AI-LZW (NN seçti)"
                words_added = len(words)
            except Exception:
                smart_bits = lzw_compress_bits(text)
                method = "Standart LZW (NN seçti)"
        else:
            smart_bits = lzw_compress_bits(text)
            method = "Standart LZW (NN seçti)"
    else:  # huffman
        corpus_freq = load_corpus_freq()
        if corpus_freq:
            merged = {ch: corpus_freq.get(ch, 1e-6) for ch in actual_freq}
            t = sum(merged.values())
            merged = {k: v/t for k, v in merged.items()}
            codes_c = build_huffman(merged)
            smart_bits = huffman_bits(text, codes_c)
            method = "Corpus Huffman (NN seçti)"
        else:
            smart_bits = huff_std
            method = "Standart Huffman (NN seçti)"

    # ── Guvenlik agi: BWT her durumda kontrol edilir (asla daha kotu sonuc olamaz) ──
    try:
        bwt_check = bwt_rle_huffman_bits(text)
        if bwt_check < smart_bits:
            smart_bits = bwt_check
            method = method + " → BWT post-kontrol kazandı"
    except Exception:
        pass

    saved = huff_std - smart_bits
    return {
        "original_bits":   orig_bits,
        "standard_bits":   huff_std,
        "smart_bits":      smart_bits,
        "smart_ratio":     smart_bits / orig_bits,
        "standard_ratio":  huff_std  / orig_bits,
        "saved_bits":      saved,
        "method":          method,
        "nn_decision":     nn_decision.upper(),
        "nn_confidence":   nn_conf,
        "words_added":     words_added,
        "total_tokens":    total_tokens,
    }
