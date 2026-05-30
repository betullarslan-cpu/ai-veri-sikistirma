"""
OCR + Sıkıştırma Pipeline
--------------------------
Görüntüdeki metin → AI (Groq vision) ile oku → Huffman + LZW ile sıkıştır.

Gerçek kullanım senaryosu:
  Taranmış belgeler, faturalar, kitap sayfaları gibi görüntüleri
  önce metne çevir, sonra verimli sıkıştır.
"""

import os
import io
import base64
from groq import Groq
from PIL import Image

def _get_groq():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Groq API key girilmemis. Sol menude API key alanini doldurun.")
    return Groq(api_key=key)


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ocr_extract(img: Image.Image) -> dict:
    """Groq vision modeli ile görüntüden metin çıkar."""
    b64 = image_to_base64(img)

    response = _get_groq().chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Bu görüntüdeki tüm metni aynen oku ve yaz. "
                            "Sadece görüntüdeki metni ver, başka hiçbir şey ekleme. "
                            "Metin yoksa 'Metin bulunamadı' yaz."
                        ),
                    },
                ],
            }
        ],
        max_tokens=2048,
    )

    extracted = response.choices[0].message.content.strip()
    tokens = response.usage.total_tokens if response.usage else 0
    return {"text": extracted, "tokens": tokens}


def compress_extracted(text: str) -> dict:
    """Çıkarılan metni Huffman + LZW ile sıkıştır, karşılaştır."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.huffman import encode as huffman_encode
    from core.lzw import lzw_encode, compression_stats

    orig_bits = len(text) * 8

    # Huffman
    huff_bits, codes = huffman_encode(text)
    huff_compressed = len(huff_bits)
    huff_overhead = len(codes) * 12

    # LZW
    lzw_codes, dict_size = lzw_encode(text)
    lzw_stats = compression_stats(text, lzw_codes, dict_size)

    return {
        "original_bits": orig_bits,
        "huffman": {
            "bits": huff_compressed,
            "overhead": huff_overhead,
            "total": huff_compressed + huff_overhead,
            "ratio": (huff_compressed + huff_overhead) / orig_bits,
        },
        "lzw": {
            "bits": lzw_stats["compressed_bits"],
            "ratio": lzw_stats["ratio"],
        },
        "best": "huffman" if (huff_compressed + huff_overhead) < lzw_stats["compressed_bits"] else "lzw",
    }
