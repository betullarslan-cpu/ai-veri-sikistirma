"""
AI Tabanlı Görüntü Sıkıştırma
------------------------------
Fikir (PDF'den):
  Görüntüyü ızgara bloklarına böl.
  AI, görüntü türüne göre hangi bölgelerin önemli
  (merkez, yüzler, odak noktası) hangilerinin önemsiz
  (arka plan, köşeler) olduğunu belirler.
  Önemli bölgeler yüksek kalite, önemsizler düşük kalite JPEG.

Standart JPEG (tek kalite) vs AI Bölgeli JPEG karşılaştırması.
"""

import os
import io
import json
from PIL import Image
from groq import Groq

def _get_groq():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Groq API key girilmemis. Sol menude API key alanini doldurun.")
    return Groq(api_key=key)
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


# ─────────────────────────────────────────
# AI bölge önem haritası
# ─────────────────────────────────────────

def get_importance_map(image_desc: str, grid_rows: int = 3, grid_cols: int = 3) -> dict:
    """
    AI'dan görüntü türüne göre bölge önem skoru al.
    Dönen dict: {"grid": [[90,90,90],[90,95,90],[50,40,50]], "reasoning": "..."}
    grid[row][col] = o bloğun JPEG kalitesi (20-95)
    """
    prompt = f"""Bir görüntü sıkıştırma sistemi için karar veriyorsun.

Görüntü türü: "{image_desc}"

Bu görüntüyü {grid_rows}x{grid_cols}'luk ızgaraya böldük.
Her hücre için JPEG kalite skoru belirle (20=çok düşük kalite, 95=çok yüksek kalite).

Kural:
- Önemli bölgeler (merkez, yüzler, ana nesne): 85-95 kalite
- Orta bölgeler: 60-80 kalite
- Önemsiz bölgeler (arka plan, köşeler): 20-45 kalite

Sadece JSON döndür:
{{
  "grid": [[sol_üst, orta_üst, sağ_üst], [sol_orta, merkez, sağ_orta], [sol_alt, orta_alt, sağ_alt]],
  "reasoning": "Kısa gerekçe Türkçe",
  "important_regions": "hangi bölgeler önemli"
}}"""

    response = _get_groq().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Görüntü sıkıştırma uzmanısın. Sadece JSON döndür."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
    )
    raw = response.choices[0].message.content.strip()
    tokens = response.usage.total_tokens if response.usage else 0

    # JSON çıkar
    idx = raw.find("{")
    end_idx = raw.rfind("}")
    if idx != -1 and end_idx != -1:
        raw = raw[idx:end_idx+1]

    result = json.loads(raw)
    result["tokens"] = tokens
    return result


# ─────────────────────────────────────────
# Standart JPEG sıkıştırma
# ─────────────────────────────────────────

def compress_standard(img: Image.Image, quality: int = 50) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ─────────────────────────────────────────
# AI bölgeli sıkıştırma
# ─────────────────────────────────────────

def compress_ai_regions(img: Image.Image, quality_grid: list) -> bytes:
    """
    Görüntüyü grid'e göre bloklara böl, her bloka farklı kalite uygula,
    sonra birleştir.
    """
    img_rgb = img.convert("RGB")
    w, h = img_rgb.size
    rows = len(quality_grid)
    cols = len(quality_grid[0])
    result = img_rgb.copy()

    block_w = w // cols
    block_h = h // rows

    for r in range(rows):
        for c in range(cols):
            quality = quality_grid[r][c]
            x0 = c * block_w
            y0 = r * block_h
            x1 = x0 + block_w if c < cols - 1 else w
            y1 = y0 + block_h if r < rows - 1 else h

            block = img_rgb.crop((x0, y0, x1, y1))

            # Bloğu JPEG ile sıkıştır ve geri aç
            buf = io.BytesIO()
            block.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            compressed_block = Image.open(buf).copy()

            result.paste(compressed_block, (x0, y0))

    # Sonucu byte'a çevir
    out_buf = io.BytesIO()
    result.save(out_buf, format="JPEG", quality=85)
    return out_buf.getvalue(), result


# ─────────────────────────────────────────
# Karşılaştırma
# ─────────────────────────────────────────

def compare(img: Image.Image, image_desc: str, std_quality: int = 50):
    original_buf = io.BytesIO()
    img.convert("RGB").save(original_buf, format="PNG")
    original_size = len(original_buf.getvalue())

    # Standart JPEG
    std_bytes = compress_standard(img, quality=std_quality)

    # AI bölgeli
    importance = get_importance_map(image_desc)
    grid = importance.get("grid", [[50]*3]*3)
    ai_bytes, ai_img = compress_ai_regions(img, grid)

    return {
        "original_bytes": original_size,
        "standard_bytes": len(std_bytes),
        "ai_bytes": len(ai_bytes),
        "standard_ratio": len(std_bytes) / original_size,
        "ai_ratio": len(ai_bytes) / original_size,
        "saved_bytes": len(std_bytes) - len(ai_bytes),
        "importance_map": importance,
        "ai_image": ai_img,
        "std_image": Image.open(io.BytesIO(std_bytes)),
        "quality_grid": grid,
    }
