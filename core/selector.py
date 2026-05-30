"""
AI Otomatik Algoritma Seçici
----------------------------
AI, metni analiz ederek hangi sıkıştırma algoritmasının
daha uygun olduğuna karar verir ve gerekçesini açıklar.

PDF'deki "Hibrit Modeller" önerisinin implementasyonu.
"""

import os
import json
from groq import Groq

def _get_groq():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Groq API key girilmemis. Sol menude API key alanini doldurun.")
    return Groq(api_key=key)
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def analyze_and_select(text: str) -> dict:
    """
    AI'ya metni analiz ettirip en iyi algoritmayı seçtir.
    Dönen dict: algorithm, confidence, reasoning, characteristics
    """
    sample = text[:500]

    prompt = f"""Sana bir metin veriyorum. Veri sıkıştırma uzmanı olarak analiz et.

Metin örneği:
\"\"\"{sample}\"\"\"

Şu soruları yanıtla ve JSON formatında döndür:

1. Bu metin için hangi algoritma daha uygundur? (huffman / lzw / hybrid)
2. Güven skoru 0-100 arası
3. Kısa gerekçe (1-2 cümle Türkçe)
4. Metnin özellikleri: tekrar oranı yüksek mi, kelime çeşitliliği nasıl, karakter dağılımı nasıl?

Sadece bu JSON formatında yanıt ver:
{{
  "algorithm": "huffman veya lzw veya hybrid",
  "confidence": 85,
  "reasoning": "Gerekçe buraya",
  "characteristics": {{
    "repetition": "yüksek/orta/düşük",
    "vocabulary": "zengin/orta/sınırlı",
    "best_for": "kısa açıklama"
  }}
}}"""

    response = _get_groq().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": (
                "Sen bir veri sıkıştırma uzmanısın. "
                "Metin özelliklerine bakarak hangi algoritmanın daha iyi sıkıştırma "
                "sağlayacağına karar verirsin. Sadece JSON döndür."
            )},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
    )

    raw = response.choices[0].message.content.strip()
    tokens = response.usage.total_tokens if response.usage else 0

    # JSON çıkar
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            p = part.strip().lstrip("json").strip()
            if p.startswith("{"):
                raw = p
                break

    idx = raw.find("{")
    end_idx = raw.rfind("}")
    if idx != -1 and end_idx != -1:
        raw = raw[idx:end_idx+1]

    result = json.loads(raw)
    result["tokens"] = tokens
    result["raw_response"] = response.choices[0].message.content
    return result


def run_selected(text: str, algorithm: str) -> dict:
    """Seçilen algoritmayı çalıştır, sonuç döndür."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.huffman import encode as huffman_encode
    from core.lzw import lzw_encode, compression_stats

    orig_bits = len(text) * 8
    results = {}

    if algorithm in ("huffman", "hybrid"):
        bits, codes = huffman_encode(text)
        results["huffman"] = {
            "bits": len(bits),
            "ratio": len(bits) / orig_bits,
            "overhead": len(codes) * 12,
        }

    if algorithm in ("lzw", "hybrid"):
        codes, dict_size = lzw_encode(text)
        stats = compression_stats(text, codes, dict_size)
        results["lzw"] = stats

    return results
