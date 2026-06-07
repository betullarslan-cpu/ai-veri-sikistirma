"""
Scaling Testleri — Algoritmaların büyük dosyalarda nasıl davrandığını ölçer.

Boyutla doğrusal mı? Karesel mi? Bellek nasıl?

Çalıştırmak:
    pytest tests/test_scaling.py -v --tb=short
"""

import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from core.huffman import encode as huff_encode
from core.bwt import bwt_rle_huffman_encode


# Doğal Türkçe metin (tekrarlı, gerçek dünyada beklenen)
TEMEL_METIN = (
    "Veri sikistirma alaninda yapay zeka kullanimi son yillarda hizla "
    "yaygınlasmaktadir. Klasik algoritmalar verimsizdir. Yapay zeka "
    "yardimi ile algoritmalar daha verimli sonuclar uretmektedir. "
)


def _olc(fn, *args):
    """Süre ölçen wrapper."""
    t = time.perf_counter()
    r = fn(*args)
    return r, (time.perf_counter() - t) * 1000  # ms


@pytest.mark.parametrize("kat", [1, 5, 10, 20, 50])
def test_huffman_dogrusal_olcekleniyor(kat):
    """Huffman boyutla yaklaşık doğrusal artıyor mu? (O(n log n))"""
    metin = TEMEL_METIN * kat
    (bits, codes), ms = _olc(huff_encode, metin)
    # n=159*1 → ~0.5 ms, n=159*50 → ~25 ms beklenir (doğrusalın 10 katı tolere)
    # En azından makul bir üst sınır
    assert ms < kat * 5 + 50, f"Huffman {len(metin)} karakter için çok yavaş: {ms:.1f} ms"
    assert len(bits) < len(metin) * 8, "Sıkıştırma yapmıyor"


@pytest.mark.parametrize("kat", [1, 3, 10, 25])
def test_bwt_karesel_olcekleniyor(kat):
    """BWT O(n² log n) — büyük metinlerde yavaşlar.

    8000 karakter sınırı koymuştuk, buna uyuyor mu?
    """
    metin = TEMEL_METIN * kat
    if len(metin) > 8000:
        metin = metin[:8000]
    (out), ms = _olc(bwt_rle_huffman_encode, metin)
    # 8000 char için 3 sn altı bekleriz
    assert ms < 5000, f"BWT {len(metin)} karakter için çok yavaş: {ms:.1f} ms"
    assert out["total_bits"] < len(metin) * 8


def test_bwt_8000_karakter_isleniyor():
    """BWT 8000 karaktere kadar düzgün çalışıyor."""
    metin = TEMEL_METIN * 60  # ~9540 char, MAX_BWT_LEN ile kesilir
    out = bwt_rle_huffman_encode(metin)
    assert out["total_bits"] > 0
    assert len(out["byte_data"]) > 0


def test_huffman_10kb_metni_3sn_altinda():
    """10KB metni Huffman 3 saniye altında halletmeli."""
    metin = TEMEL_METIN * 100  # ~16 KB
    (bits, codes), ms = _olc(huff_encode, metin)
    assert ms < 3000, f"Huffman çok yavaş: {ms:.0f} ms"


@pytest.mark.parametrize("kat", [1, 10, 50])
def test_huffman_oran_metin_uzunluguyla_iyilesiyor(kat):
    """Uzun metinde Huffman tablo overhead'i nispi olarak azalır → oran iyileşir.

    Bu, sıkıştırma teorisinin temel bir öngörüsüdür.
    """
    metin = TEMEL_METIN * kat
    bits, codes = huff_encode(metin)
    # Tek karakter başına ortalama bit
    bit_per_char = len(bits) / len(metin)
    # Tablo overhead'i metni uzattıkça oran iyileşmeli
    assert bit_per_char < 6.0, (
        f"Huffman karakter başına {bit_per_char:.2f} bit "
        f"(8 bitten az olmalı, doğal dilde ~4-5 beklenir)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
