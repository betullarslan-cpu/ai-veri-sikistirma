"""
Endüstri standartı sıkıştırma araçları (gzip, bzip2, zlib, lzma) ile karşılaştırma.

Bu modül bizim algoritmalarımızı gerçek dünya standartlarıyla karşılaştırır:
    - gzip (1992, RFC 1952) — en yaygın, hızlı
    - bzip2 (1996) — BWT tabanlı, yüksek oran
    - zlib (1995) — deflate tabanlı, web standardı
    - lzma (xz)   — LZMA2 tabanlı, en yüksek oran
"""

import time
import gzip
import bz2
import lzma
import zlib
from typing import Dict


def _olc_zaman(fn, *args, **kwargs):
    """Bir fonksiyonu çalıştır, sonuç + süre (ms) döndür."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return result, elapsed_ms


def endustri_benchmark(text: str) -> Dict:
    """
    Metni 4 farklı endüstri standardıyla sıkıştırıp ölçüm yapar.

    Returns:
        {
            "orijinal_byte": int,
            "gzip":  {"byte": int, "ratio": float, "sure_ms": float},
            "bzip2": {...},
            "zlib":  {...},
            "lzma":  {...},
        }
    """
    raw_bytes = text.encode("utf-8")
    orig_size = len(raw_bytes)

    sonuclar = {"orijinal_byte": orig_size}

    # gzip — en sıkı seviye 9
    gz_out, gz_ms = _olc_zaman(gzip.compress, raw_bytes, compresslevel=9)
    sonuclar["gzip"] = {
        "byte": len(gz_out),
        "ratio": len(gz_out) / orig_size,
        "sure_ms": gz_ms,
        "kucullme_pct": (1 - len(gz_out) / orig_size) * 100,
    }

    # bzip2 — BWT tabanlı (bizim BWT modülümüzün referansı)
    bz_out, bz_ms = _olc_zaman(bz2.compress, raw_bytes, compresslevel=9)
    sonuclar["bzip2"] = {
        "byte": len(bz_out),
        "ratio": len(bz_out) / orig_size,
        "sure_ms": bz_ms,
        "kucullme_pct": (1 - len(bz_out) / orig_size) * 100,
    }

    # zlib — deflate (web standardı)
    zl_out, zl_ms = _olc_zaman(zlib.compress, raw_bytes, 9)
    sonuclar["zlib"] = {
        "byte": len(zl_out),
        "ratio": len(zl_out) / orig_size,
        "sure_ms": zl_ms,
        "kucullme_pct": (1 - len(zl_out) / orig_size) * 100,
    }

    # lzma (xz) — en yüksek oran
    lz_out, lz_ms = _olc_zaman(lzma.compress, raw_bytes, preset=9)
    sonuclar["lzma"] = {
        "byte": len(lz_out),
        "ratio": len(lz_out) / orig_size,
        "sure_ms": lz_ms,
        "kucullme_pct": (1 - len(lz_out) / orig_size) * 100,
    }

    return sonuclar


def bizim_algoritmalar_benchmark(text: str) -> Dict:
    """
    Bizim 4 algoritmamızı çalıştırıp süre + boyut ölçer.
    Endüstri benchmark'ı ile yan yana karşılaştırma için kullanılır.
    """
    import math
    from collections import Counter
    from core.huffman import encode as huff_encode
    from core.bwt import bwt_rle_huffman_encode, huffman_encode_bytes

    raw_size = len(text.encode("utf-8"))
    sonuclar = {"orijinal_byte": raw_size}

    # Standart Huffman
    h_out, h_ms = _olc_zaman(huffman_encode_bytes, text)
    sonuclar["huffman"] = {
        "byte": len(h_out["byte_data"]),
        "ratio": len(h_out["byte_data"]) / raw_size,
        "sure_ms": h_ms,
        "kucullme_pct": (1 - len(h_out["byte_data"]) / raw_size) * 100,
    }

    # Standart LZW
    def _lzw_encode_bytes(t):
        dictionary = {chr(i): i for i in range(256)}
        for ch in set(t):
            if ch not in dictionary:
                dictionary[ch] = len(dictionary)
        nxt = len(dictionary)
        codes = []
        w = ""
        for c in t:
            wc = w + c
            if wc in dictionary:
                w = wc
            else:
                codes.append(dictionary[w])
                dictionary[wc] = nxt
                nxt += 1
                w = c
        if w:
            codes.append(dictionary[w])
        bpc = math.ceil(math.log2(max(len(dictionary), 2)))
        bit_str = "".join(format(c, f"0{bpc}b") for c in codes)
        padded = bit_str + "0" * ((8 - len(bit_str) % 8) % 8)
        return bytes(int(padded[i:i+8], 2) for i in range(0, len(padded), 8))

    lzw_bytes, lzw_ms = _olc_zaman(_lzw_encode_bytes, text)
    sonuclar["lzw"] = {
        "byte": len(lzw_bytes),
        "ratio": len(lzw_bytes) / raw_size,
        "sure_ms": lzw_ms,
        "kucullme_pct": (1 - len(lzw_bytes) / raw_size) * 100,
    }

    # BWT+RLE+Huffman (bizim en güçlü)
    b_out, b_ms = _olc_zaman(bwt_rle_huffman_encode, text)
    sonuclar["bwt_rle_huffman"] = {
        "byte": len(b_out["byte_data"]),
        "ratio": len(b_out["byte_data"]) / raw_size,
        "sure_ms": b_ms,
        "kucullme_pct": (1 - len(b_out["byte_data"]) / raw_size) * 100,
    }

    # Akıllı Hibrit (NN + post-check)
    from core.hybrid import smart_hybrid
    sh, sh_ms = _olc_zaman(smart_hybrid, text, False)  # use_ai_dict=False
    # Bit -> byte
    sh_byte = (sh["smart_bits"] + 7) // 8
    sonuclar["akilli_hibrit"] = {
        "byte": sh_byte,
        "ratio": sh_byte / raw_size,
        "sure_ms": sh_ms,
        "kucullme_pct": (1 - sh_byte / raw_size) * 100,
        "metod": sh["method"],
    }

    return sonuclar


def scaling_analizi(metin_temeli: str, katlar: list = None) -> Dict:
    """
    Metni farklı boyutlarda çoğaltıp her bir algoritma için süre ve oran ölç.

    Bu fonksiyon "O(n) mi O(n²) mi?" sorusunun ampirik cevabını verir.

    Args:
        metin_temeli: Çoğaltılacak temel metin (bir cümle/paragraf)
        katlar: Test edilecek çarpanlar [1, 5, 10, 25, 50, 100]

    Returns:
        {
            "boyutlar":  [N karakter],
            "huffman":   {"sureler_ms": [...], "boyut_byte": [...]},
            "bwt":       {"sureler_ms": [...], "boyut_byte": [...]},
            "gzip":      {"sureler_ms": [...], "boyut_byte": [...]},
        }
    """
    import gzip
    from core.bwt import bwt_rle_huffman_encode
    from core.huffman import encode as huff_encode

    if katlar is None:
        katlar = [1, 5, 10, 25, 50]

    sonuc = {
        "boyutlar": [],
        "huffman": {"sureler_ms": [], "boyut_byte": []},
        "bwt":     {"sureler_ms": [], "boyut_byte": []},
        "gzip":    {"sureler_ms": [], "boyut_byte": []},
    }

    for k in katlar:
        metin = metin_temeli * k
        # BWT için 8000 char sınırı
        bwt_metin = metin[:8000]
        sonuc["boyutlar"].append(len(metin))

        # Huffman
        (bits, codes), h_ms = _olc_zaman(huff_encode, metin)
        sonuc["huffman"]["sureler_ms"].append(h_ms)
        sonuc["huffman"]["boyut_byte"].append((len(bits) + 7) // 8)

        # BWT
        try:
            (out), b_ms = _olc_zaman(bwt_rle_huffman_encode, bwt_metin)
            sonuc["bwt"]["sureler_ms"].append(b_ms)
            sonuc["bwt"]["boyut_byte"].append(len(out["byte_data"]))
        except Exception:
            sonuc["bwt"]["sureler_ms"].append(None)
            sonuc["bwt"]["boyut_byte"].append(None)

        # gzip
        gz_out, gz_ms = _olc_zaman(gzip.compress, metin.encode("utf-8"), 9)
        sonuc["gzip"]["sureler_ms"].append(gz_ms)
        sonuc["gzip"]["boyut_byte"].append(len(gz_out))

    return sonuc


def tam_karsilastirma(text: str) -> Dict:
    """
    Hem endüstri standartlarını hem bizim algoritmalarımızı birlikte ölçer.

    Returns:
        {
            "orijinal_byte": int,
            "endustri": {gzip, bzip2, zlib, lzma},
            "bizim":   {huffman, lzw, bwt_rle_huffman, akilli_hibrit},
            "sonuc":   "BWT bzip2'ye %X uzakta",
        }
    """
    endustri = endustri_benchmark(text)
    bizim    = bizim_algoritmalar_benchmark(text)

    # Karşılaştırma metriği: Akıllı Hibrit vs bzip2
    bzip_byte = endustri["bzip2"]["byte"]
    bizim_en_iyi = min(
        bizim["bwt_rle_huffman"]["byte"],
        bizim["akilli_hibrit"]["byte"],
    )
    fark_pct = (bizim_en_iyi - bzip_byte) / bzip_byte * 100 if bzip_byte else 0

    return {
        "orijinal_byte": endustri["orijinal_byte"],
        "endustri": endustri,
        "bizim": bizim,
        "bizim_en_iyi_byte": bizim_en_iyi,
        "bzip2_byte": bzip_byte,
        "fark_yuzdesi": fark_pct,
        "yorum": (
            f"En iyi algoritmamız bzip2'den %{fark_pct:.1f} "
            f"{'daha büyük' if fark_pct > 0 else 'daha küçük'}."
        ),
    }


if __name__ == "__main__":
    # Hızlı test
    test_metni = (
        "Yapay zeka teknolojileri hizla gelisiyor. "
        "Veri sikistirma algoritmalari onemli bir konudur. " * 10
    )
    r = tam_karsilastirma(test_metni)
    print(f"Orijinal: {r['orijinal_byte']:,} byte\n")

    print("ENDÜSTRİ STANDARTLARI:")
    for alg, d in r["endustri"].items():
        if isinstance(d, dict):
            print(f"  {alg:8s}: {d['byte']:>6} byte | "
                  f"oran={d['ratio']:.3f} | süre={d['sure_ms']:6.2f} ms | "
                  f"küçülme=%{d['kucullme_pct']:.1f}")

    print("\nBİZİM ALGORİTMALARIMIZ:")
    for alg, d in r["bizim"].items():
        if isinstance(d, dict):
            print(f"  {alg:18s}: {d['byte']:>6} byte | "
                  f"oran={d['ratio']:.3f} | süre={d['sure_ms']:6.2f} ms | "
                  f"küçülme=%{d['kucullme_pct']:.1f}")

    print(f"\n{r['yorum']}")
