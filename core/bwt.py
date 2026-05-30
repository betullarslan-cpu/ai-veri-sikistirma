"""
BWT (Burrows-Wheeler Transform) + RLE + Huffman Pipeline
----------------------------------------------------------
bzip2'nin temel fikri:

1. BWT  → benzer karakterleri gruplar ("aaaa...bbbb..." gibi uzun koşular oluşturur)
2. RLE  → tekrarlı koşuları (ch, count) çiftine sıkıştırır
3. Huffman → sembolleri optimal bit uzunluğuyla kodlar

Neden Standart Huffman'dan iyi?
  - Standart Huffman her karakteri bağımsız işler → tekrarlı yapıdan faydalanmaz
  - BWT sonrası karakterler kümelenir → RLE çok kısa output üretir
  - Tekrarlı metinde 10x+, doğal dilde %10-20 daha iyi sıkıştırma
"""

import math
import heapq
from collections import Counter

# BWT sonu işaretçisi — metinde bulunmayan özel karakter
EOF_CHAR = "\x00"

MAX_BWT_LEN = 8000  # Performans için üst sınır (O(n²))


# ─────────────────────────────────────────────────────
# 1. BWT Dönüşümü
# ─────────────────────────────────────────────────────

def bwt_encode(text: str) -> tuple:
    """
    Burrows-Wheeler Transform uygula.

    Returns:
        (bwt_text, original_index)
        bwt_text  : dönüştürülmüş metin (karakter kümeleri oluşur)
        original_index : orijinal stringin sıralamadaki konumu
    """
    # Çok uzun metinleri kes — demo için yeterli
    if len(text) > MAX_BWT_LEN:
        text = text[:MAX_BWT_LEN]

    s = text + EOF_CHAR
    n = len(s)

    # Suffix array (basit sıralama — O(n log n))
    sa = sorted(range(n), key=lambda i: s[i:] + s[:i])

    bwt = "".join(s[(i - 1) % n] for i in sa)
    orig_idx = sa.index(0)

    return bwt, orig_idx


def bwt_decode(bwt: str, orig_idx: int) -> str:
    """BWT'yi tersine cevir -> orijinal metni geri al."""
    n = len(bwt)

    # Her karakterin BWT'deki sirasi (rank)
    seen = {}
    rank = []
    for ch in bwt:
        rank.append(seen.get(ch, 0))
        seen[ch] = seen.get(ch, 0) + 1

    # Sirali dizideki her karakterin baslangic konumu
    char_counts = {}
    for ch in bwt:
        char_counts[ch] = char_counts.get(ch, 0) + 1

    starts = {}
    pos = 0
    for ch in sorted(char_counts.keys()):
        starts[ch] = pos
        pos += char_counts[ch]

    # LF mapping: L'deki i. konumu F'deki konuma esler
    lf = [starts[bwt[i]] + rank[i] for i in range(n)]

    # Yeniden olustur: orig_idx'ten basla, bwt[idx] ekle, LF ile ilerle
    result = []
    idx = orig_idx
    for _ in range(n):
        result.append(bwt[idx])   # L[idx] -- BWT sutunundan oku
        idx = lf[idx]

    result.reverse()
    # EOF isaretsini cikar
    return "".join(ch for ch in result if ch != EOF_CHAR)


# ─────────────────────────────────────────────────────
# 2. RLE (Run-Length Encoding)
# ─────────────────────────────────────────────────────

def rle_compress(text: str) -> list:
    """
    Tekrarlı karakter dizilerini (char, count) çiftine çevir.
    BWT sonrası çok uzun koşular oluşur → RLE verimli olur.
    """
    if not text:
        return []
    result = []
    i = 0
    while i < len(text):
        ch = text[i]
        count = 0
        while i < len(text) and text[i] == ch:
            count += 1
            i += 1
        result.append((ch, count))
    return result


def rle_decompress(runs: list) -> str:
    """RLE'yi geri aç."""
    return "".join(ch * count for ch, count in runs)


# ─────────────────────────────────────────────────────
# 3. Huffman yardımcıları (BWT için bağımsız kopya)
# ─────────────────────────────────────────────────────

class _HNode:
    def __init__(self, ch, f):
        self.ch = ch
        self.freq = f
        self.left = self.right = None

    def __lt__(self, o):
        return self.freq < o.freq


def _build_huffman(freq_dict: dict) -> dict:
    if not freq_dict:
        return {}
    heap = [_HNode(ch, f) for ch, f in freq_dict.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        l, r = heapq.heappop(heap), heapq.heappop(heap)
        m = _HNode(None, l.freq + r.freq)
        m.left, m.right = l, r
        heapq.heappush(heap, m)
    root = heap[0]

    codes = {}

    def _walk(node, prefix=""):
        if node is None:
            return
        if node.ch is not None:
            codes[node.ch] = prefix or "0"
        else:
            _walk(node.left, prefix + "0")
            _walk(node.right, prefix + "1")

    _walk(root)
    return codes


# ─────────────────────────────────────────────────────
# 4. Bit Hesaplamaları
# ─────────────────────────────────────────────────────

def bwt_huffman_bits(text: str) -> int:
    """
    BWT + Huffman (RLE olmadan).
    BWT permüte ettiği için karakter frekansları aynı kalır
    → tek başına Huffman kazanımı minimal, ama LZW için temel sağlar.
    """
    bwt, _ = bwt_encode(text)
    total = len(bwt)
    if total == 0:
        return 0

    freq = {ch: c / total for ch, c in Counter(bwt).items()}
    codes = _build_huffman(freq)

    compressed = sum(len(codes.get(ch, "0" * 16)) for ch in bwt)
    overhead = len(codes) * 12  # tablo overhead
    return compressed + overhead


def bwt_rle_huffman_bits(text: str) -> int:
    """
    BWT + RLE + Huffman — bzip2 yaklaşımı.

    Teorik bit hesabı:
      - Her (char, count) çifti için Huffman char kodu + Elias-gamma count kodu
      - Uzun koşular → count büyük ama az tekrar → verimli
    """
    bwt, orig_idx = bwt_encode(text)
    if not bwt:
        return 0

    runs = rle_compress(bwt)
    if not runs:
        return 0

    # Karakter frekansları (RLE sonrası)
    char_freq = Counter(ch for ch, _ in runs)
    total_runs = len(runs)
    char_probs = {ch: c / total_runs for ch, c in char_freq.items()}
    codes = _build_huffman(char_probs)

    bits = 0
    for ch, count in runs:
        # Char kodu
        bits += len(codes.get(ch, "0" * 16))
        # Count kodu — Elias gamma: 2*floor(log2(count)) + 1 bit
        bits += 2 * int(math.log2(count)) + 1 if count > 0 else 1

    # BWT indeks → ceil(log2(n)) bit
    n = len(bwt)
    bits += math.ceil(math.log2(n + 1))

    # Huffman tablo overhead
    bits += len(codes) * 12

    return bits


def bwt_lzw_bits(text: str) -> int:
    """
    BWT + LZW.
    BWT sonrası uzun tekrarlı koşular → LZW daha uzun pattern bulur → daha az kod.
    """
    bwt, _ = bwt_encode(text)
    if not bwt:
        return 0

    # LZW sözlük başlat (Unicode dahil)
    dictionary = {chr(i): i for i in range(256)}
    for ch in set(bwt):
        if ch not in dictionary:
            dictionary[ch] = len(dictionary)

    next_code = len(dictionary)
    codes = []
    w = ""

    for c in bwt:
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


# ─────────────────────────────────────────────────────
# 5. Karşılaştırma (Ana Fonksiyon)
# ─────────────────────────────────────────────────────

def compare(text: str) -> dict:
    """
    Tüm BWT kombinasyonlarını standart algoritmalarla karşılaştır.

    Returns: {
        original_bits, results (dict), best, improvement_pct, ...
    }
    """
    # Metni kısalt (çok uzunsa yavaşlar)
    if len(text) > MAX_BWT_LEN:
        text = text[:MAX_BWT_LEN]

    orig_bits = len(text) * 8
    if orig_bits == 0:
        return {}

    # ── Referans: Standart Huffman ──
    total = len(text)
    freq = {ch: c / total for ch, c in Counter(text).items()}
    codes_std = _build_huffman(freq)
    std_huff = sum(len(codes_std.get(ch, "0"*16)) for ch in text) + len(codes_std) * 12

    # ── Standart LZW ──
    dictionary = {chr(i): i for i in range(256)}
    for ch in set(text):
        if ch not in dictionary:
            dictionary[ch] = len(dictionary)
    nxt = len(dictionary)
    lzw_codes = []
    w = ""
    for c in text:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            lzw_codes.append(dictionary[w])
            dictionary[wc] = nxt
            nxt += 1
            w = c
    if w:
        lzw_codes.append(dictionary[w])
    bpc = math.ceil(math.log2(max(len(dictionary), 2)))
    std_lzw = len(lzw_codes) * bpc

    # ── BWT + Huffman ──
    bwt_huff = bwt_huffman_bits(text)

    # ── BWT + RLE + Huffman ──
    bwt_rle_huff = bwt_rle_huffman_bits(text)

    # ── BWT + LZW ──
    bwt_lzw = bwt_lzw_bits(text)

    results = {
        "Standart Huffman": {
            "bits": std_huff, "ratio": std_huff / orig_bits,
            "bwt": False,
        },
        "Standart LZW": {
            "bits": std_lzw, "ratio": std_lzw / orig_bits,
            "bwt": False,
        },
        "BWT + Huffman": {
            "bits": bwt_huff, "ratio": bwt_huff / orig_bits,
            "bwt": True,
            "note": "BWT permütasyonu → Huffman (küçük iyileşme)",
        },
        "BWT + RLE + Huffman": {
            "bits": bwt_rle_huff, "ratio": bwt_rle_huff / orig_bits,
            "bwt": True,
            "note": "bzip2 tekniği: BWT kümeleme → RLE koşuları → Huffman",
        },
        "BWT + LZW": {
            "bits": bwt_lzw, "ratio": bwt_lzw / orig_bits,
            "bwt": True,
            "note": "BWT sonrası uzun tekrarlar → LZW daha fazla pattern yakalar",
        },
    }

    best_name = min(results, key=lambda k: results[k]["bits"])
    best = results[best_name]

    return {
        "original_bits": orig_bits,
        "results": results,
        "best": best_name,
        "best_bits": best["bits"],
        "best_ratio": best["ratio"],
        "std_huffman_bits": std_huff,
        "improvement_pct": (std_huff - best["bits"]) / std_huff * 100,
        "text_len": len(text),
    }


# ─────────────────────────────────────────────────────
# 6. BWT İstatistikleri (UI için)
# ─────────────────────────────────────────────────────

def bwt_stats(text: str) -> dict:
    """BWT'nin metin üzerindeki etkisini göster."""
    if len(text) > MAX_BWT_LEN:
        text = text[:MAX_BWT_LEN]

    bwt, orig_idx = bwt_encode(text)
    runs = rle_compress(bwt)

    orig_runs = rle_compress(text)

    return {
        "original_length":   len(text),
        "bwt_length":        len(bwt),
        "original_runs":     len(orig_runs),
        "bwt_runs":          len(runs),
        "run_reduction_pct": (1 - len(runs) / max(len(orig_runs), 1)) * 100,
        "longest_run":       max((c for _, c in runs), default=0),
        "avg_run_length":    len(bwt) / max(len(runs), 1),
        "bwt_preview":       bwt[:80] + ("..." if len(bwt) > 80 else ""),
        "orig_idx":          orig_idx,
    }
