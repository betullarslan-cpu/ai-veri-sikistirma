"""
Kayıpsızlık Testleri — Tüm algoritmalar için encode → decode garantili olmalı.

Çalıştırmak için:
    pip install pytest
    pytest tests/

Bu testler her commit'te otomatik çalıştırılabilir (CI/CD).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from core.huffman import encode as huff_encode, decode as huff_decode
from core.lzw import lzw_encode, lzw_decode
from core.bwt import bwt_encode, bwt_decode, bwt_rle_huffman_encode


# ─────────────────────────────────────────
# Test verileri
# ─────────────────────────────────────────

TEMEL_METINLER = [
    "merhaba dunya",
    "banana",
    "ABCABCABC",
    "abcdefghijklmnop",
    "Bugun hava cok guzel.",
    "The quick brown fox jumps over the lazy dog.",
]

TURKCE_METINLER = [
    "Türkiye Cumhuriyeti",
    "şğüöçıŞĞÜÖÇİ özel karakterler",
    "Yıldız Teknik Üniversitesi",
    "İstanbul'da yaşıyorum",
    "Çok güzel bir günaydın",
]

EDGE_CASE_METINLER = [
    "a",                              # tek karakter
    "ab",                             # iki karakter
    "aaaaa",                          # sadece tekrar
    "    ",                           # sadece boşluk
    "\n\n\n",                         # sadece newline
    "1234567890",                     # sadece rakam
    "!@#$%^&*()",                     # sadece sembol
    "a" * 100,                        # uzun tekrar
    "ab" * 50,                        # uzun 2-pattern
]

UZUN_METINLER = [
    "Veri sikistirma algoritmasi. " * 50,
    "ATCG" * 200,                     # DNA
    "X" * 500,                        # tek karakter uzun
]


# ─────────────────────────────────────────
# HUFFMAN TESTLERI
# ─────────────────────────────────────────

@pytest.mark.parametrize("metin", TEMEL_METINLER + TURKCE_METINLER)
def test_huffman_kayipsiz_temel(metin):
    """Huffman: temel metinler için encode → decode aynı olmalı."""
    bits, codes = huff_encode(metin)
    decoded = huff_decode(bits, codes)
    assert decoded == metin, f"Huffman bozuk: '{metin}' -> '{decoded}'"


@pytest.mark.parametrize("metin", EDGE_CASE_METINLER)
def test_huffman_edge_cases(metin):
    """Huffman: edge case'ler (tek karakter, sadece boşluk vb.)."""
    bits, codes = huff_encode(metin)
    decoded = huff_decode(bits, codes)
    assert decoded == metin


@pytest.mark.parametrize("metin", UZUN_METINLER)
def test_huffman_uzun_metin(metin):
    """Huffman: uzun ve karmaşık metinler."""
    bits, codes = huff_encode(metin)
    decoded = huff_decode(bits, codes)
    assert decoded == metin


# ─────────────────────────────────────────
# LZW TESTLERI
# ─────────────────────────────────────────

def _lzw_init_dict(text):
    """Test için başlangıç sözlüğü kur (Türkçe karakter dahil)."""
    d = {chr(i): i for i in range(256)}
    for ch in set(text):
        if ch not in d:
            d[ch] = len(d)
    return d


@pytest.mark.parametrize("metin", TEMEL_METINLER + TURKCE_METINLER)
def test_lzw_kayipsiz_temel(metin):
    """LZW: temel metinler kayıpsız."""
    init = _lzw_init_dict(metin)
    codes, _ = lzw_encode(metin, initial_dict=init)
    decoded = lzw_decode(codes, initial_dict=init)
    assert decoded == metin, f"LZW bozuk: '{metin}' -> '{decoded}'"


@pytest.mark.parametrize("metin", EDGE_CASE_METINLER)
def test_lzw_edge_cases(metin):
    """LZW: edge case'ler."""
    init = _lzw_init_dict(metin)
    codes, _ = lzw_encode(metin, initial_dict=init)
    decoded = lzw_decode(codes, initial_dict=init)
    assert decoded == metin


@pytest.mark.parametrize("metin", UZUN_METINLER)
def test_lzw_uzun_metin(metin):
    """LZW: uzun metinler."""
    init = _lzw_init_dict(metin)
    codes, _ = lzw_encode(metin, initial_dict=init)
    decoded = lzw_decode(codes, initial_dict=init)
    assert decoded == metin


# ─────────────────────────────────────────
# BWT TESTLERI
# ─────────────────────────────────────────

@pytest.mark.parametrize("metin", TEMEL_METINLER + TURKCE_METINLER)
def test_bwt_kayipsiz_temel(metin):
    """BWT: temel metinler için kayıpsız permütasyon + tersi."""
    bwt, idx = bwt_encode(metin)
    decoded = bwt_decode(bwt, idx)
    assert decoded == metin, f"BWT bozuk: '{metin}' -> '{decoded}'"


@pytest.mark.parametrize("metin", EDGE_CASE_METINLER)
def test_bwt_edge_cases(metin):
    """BWT: edge case'ler."""
    bwt, idx = bwt_encode(metin)
    decoded = bwt_decode(bwt, idx)
    assert decoded == metin


@pytest.mark.parametrize("metin", TEMEL_METINLER + TURKCE_METINLER)
def test_mtf_kayipsiz(metin):
    """MTF (Move-to-Front) encode → decode kayıpsız olmalı."""
    from core.bwt import mtf_encode, mtf_decode
    indices, alphabet = mtf_encode(metin)
    decoded = mtf_decode(indices, alphabet)
    assert decoded == metin, f"MTF bozuk: '{metin}' -> '{decoded}'"


def test_mtf_bwt_pipeline_kayipsiz():
    """BWT → MTF → MTF⁻¹ → BWT⁻¹ tüm pipeline kayıpsız."""
    from core.bwt import bwt_encode, bwt_decode, mtf_encode, mtf_decode
    metin = "merhaba dunya, bu klasik bzip2 testidir."
    # Forward: BWT → MTF
    bwt, idx = bwt_encode(metin)
    mtf_indices, alpha = mtf_encode(bwt)
    # Reverse: MTF⁻¹ → BWT⁻¹
    bwt_back = mtf_decode(mtf_indices, alpha)
    metin_back = bwt_decode(bwt_back, idx)
    assert metin_back == metin


def test_bwt_full_pipeline_dondurulebilir():
    """BWT+RLE+Huffman tam pipeline: encode'da bilgi kaybetmemeli."""
    metin = "merhaba dunya, bu bir test metnidir."
    out = bwt_rle_huffman_encode(metin)
    # En azından bit_string, byte_data, bwt, orig_idx alanları olmalı
    assert "bit_string" in out
    assert "byte_data" in out
    assert "bwt" in out
    assert "orig_idx" in out
    # BWT decode ile orijinali geri kur
    decoded = bwt_decode(out["bwt"], out["orig_idx"])
    assert decoded == metin


def test_bwt_uzun_metin_blokla_kayipsiz():
    """BWT 8000 karakter üstü metinleri BLOKLU işler, kayıpsız geri verir."""
    from core.bwt import bwt_chunked_encode, bwt_chunked_decode
    metin = "Veri sikistirma algoritmasi cok onemlidir. " * 200  # ~8600 char
    assert len(metin) > 8000, "Test metin yetersiz"

    chunks = bwt_chunked_encode(metin)
    assert len(chunks) >= 2, "Birden fazla blok olmalı"

    decoded = bwt_chunked_decode(chunks)
    assert decoded == metin, "Bloklu BWT decode kayıplı!"


def test_bwt_rle_huffman_uzun_metin_kayipsiz():
    """bwt_rle_huffman_encode uzun metinde otomatik bloklu çalışır."""
    from core.bwt import bwt_chunked_decode
    metin = "Yapay zeka ile veri sikistirma. " * 300  # ~9000 char
    assert len(metin) > 8000

    out = bwt_rle_huffman_encode(metin)
    assert out["n_chunks"] >= 2, "n_chunks bilgisi gelmeli"

    decoded = bwt_chunked_decode(out["chunks"])
    assert decoded == metin


# ─────────────────────────────────────────
# CROSS-ALGORITHM (Tüm algoritmalar aynı metni doğru sıkıştırmalı)
# ─────────────────────────────────────────

@pytest.mark.parametrize("metin", TEMEL_METINLER[:3])
def test_butun_algoritmalar_aynı_metni_isler(metin):
    """Aynı metin tüm algoritmalardan kayıpsız geçer."""
    # Huffman
    bits, codes = huff_encode(metin)
    assert huff_decode(bits, codes) == metin

    # LZW
    init = _lzw_init_dict(metin)
    codes, _ = lzw_encode(metin, initial_dict=init)
    assert lzw_decode(codes, initial_dict=init) == metin

    # BWT
    bwt, idx = bwt_encode(metin)
    assert bwt_decode(bwt, idx) == metin


# ─────────────────────────────────────────
# PERFORMANS GARANTILERI
# ─────────────────────────────────────────

def test_huffman_uzun_metni_kucultuyor():
    """Huffman tekrarlı metni gerçekten küçültmeli."""
    metin = "a" * 1000  # çok tekrarlı
    bits, codes = huff_encode(metin)
    # Tek karakter için Huffman 1 bit kullanır + tablo
    # 1000 karakter * 8 bit = 8000 bit orijinal
    # Bizim sonuç: ~1000 bit (her karakter 1 bit) + tablo
    assert len(bits) < 8000, "Huffman 'a'*1000'i küçültemiyorsa hata"


def test_bwt_tekrarli_veride_etkili():
    """BWT+RLE+Huffman tekrarlı veride çok kazanmalı."""
    metin = "ABCABCABCABCABC" * 30
    out = bwt_rle_huffman_encode(metin)
    orig_bits = len(metin) * 8
    assert out["total_bits"] < orig_bits * 0.2, (
        f"BWT tekrarlı metinde %80'den çok küçültmeli, "
        f"şu an {out['total_bits']/orig_bits*100:.1f}%."
    )


if __name__ == "__main__":
    # Pytest olmadan da çalıştırılabilir
    pytest.main([__file__, "-v"])
