"""
Aritmetik Kodlama + AI Olasılık Tahmini
-----------------------------------------
Aritmetik kodlama, Huffman'dan farklı olarak teorik Shannon sınırına
çok daha yakın sıkıştırma yapabilir. Her sembolü tam bit sayısıyla değil,
kesirli bit cinsinden ifade eder.

Standart : sembol frekanslarını dosyadan hesapla
AI       : Groq'tan olasılık al → daha iyi model, tablo gerektirmez
"""

import os
import math
from collections import Counter
from groq import Groq

def _get_groq():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Groq API key girilmemis. Sol menude API key alanini doldurun.")
    return Groq(api_key=key)
MODEL  = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

PRECISION = 32  # bit hassasiyeti


# ─────────────────────────────────────────
# Temel aritmetik kodlama (integer tabanlı)
# ─────────────────────────────────────────

def _build_cum_freq(freq: dict) -> tuple[dict, dict, int]:
    """Kümülatif frekans tablosu oluştur."""
    symbols = sorted(freq.keys())
    total   = sum(freq.values())

    cum_low  = {}
    cum_high = {}
    cum = 0
    SCALE = 10000  # tam sayı hassasiyeti

    for s in symbols:
        p = freq[s] / total
        cum_low[s]  = cum
        cum += max(1, int(p * SCALE))
        cum_high[s] = cum

    total_scale = cum
    return cum_low, cum_high, total_scale


def arithmetic_encode_bits(text: str, freq: dict) -> int:
    """
    Gerçek aritmetik kodlama yerine teorik bit sayısı hesapla.
    Bit sayısı = -Σ log2(p(x_i)) (her sembol için optimal uzunluk)
    Bu, aritmetik kodlamanın asimptotik bit sayısına eşittir.
    """
    total = len(text)
    total_freq = sum(freq.values())
    bits = 0.0
    for ch in text:
        p = freq.get(ch, 1e-9) / total_freq
        bits += -math.log2(p)
    return math.ceil(bits)


def arithmetic_overhead(freq: dict) -> int:
    """Frekans tablosunu saklamak için bit: her giriş ~20 bit."""
    return len(freq) * 20


# ─────────────────────────────────────────
# AI olasılık tahmini (Groq)
# ─────────────────────────────────────────

def ai_probability_model(text_sample: str, text_type: str = "Türkçe metin") -> tuple[dict, int]:
    """
    AI'dan olasılık dağılımı al.
    Aritmetik kodlamada bu olasılıklar sembol aralıklarını belirler.
    """
    from collections import Counter as C
    hint = dict(C(text_sample[:300]).most_common(8))
    total_h = sum(hint.values())
    hint_pct = {ch: f"{cnt/total_h:.3f}" for ch, cnt in hint.items()}

    import json as _json
    response = _get_groq().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": (
                "Aritmetik kodlama için olasılık modeli oluşturuyorsun. "
                "Her karakterin görünme olasılığını tahmin et. Sadece JSON döndür."
            )},
            {"role": "user", "content": (
                f"Metin türü: {text_type}\n"
                f"Örnek frekans ipucu: {hint_pct}\n"
                f"Metin: \"\"\"{text_sample[:300]}\"\"\"\n\n"
                "Tüm karakterler için 0-1 arası olasılık ver. "
                "Türkçe özel karakterleri (ş,ğ,ü,ö,ç,ı) dahil et. "
                "Toplam 1.0 olsun. Sadece JSON."
            )},
        ],
        max_tokens=1024,
    )
    raw  = response.choices[0].message.content.strip()
    toks = response.usage.total_tokens if response.usage else 0

    # JSON çıkar
    idx = raw.find("{"); end = raw.rfind("}")
    if idx != -1 and end != -1:
        raw = raw[idx:end+1]
    try:
        import ast as _ast
        probs = _json.loads(raw)
    except Exception:
        probs = _ast.literal_eval(raw.replace("'", '"'))

    total_p = sum(probs.values())
    probs = {k: v/total_p for k, v in probs.items()}
    return probs, toks


# ─────────────────────────────────────────
# Karşılaştırma
# ─────────────────────────────────────────

def compare(text: str, text_type: str = "Türkçe metin") -> dict:
    orig_bits = len(text) * 8
    total     = len(text)
    actual_freq = dict(Counter(text))

    # A) Standart aritmetik kodlama
    std_bits     = arithmetic_encode_bits(text, actual_freq)
    std_overhead = arithmetic_overhead(actual_freq)
    std_total    = std_bits + std_overhead

    # B) AI aritmetik kodlama
    ai_probs, tokens = ai_probability_model(text, text_type)
    # Eksik karakterler için küçük olasılık ekle
    for ch in actual_freq:
        if ch not in ai_probs:
            ai_probs[ch] = 1e-5
    total_ap = sum(ai_probs.values())
    ai_probs = {k: v/total_ap for k, v in ai_probs.items()}

    ai_bits  = arithmetic_encode_bits(text, {ch: p for ch, p in ai_probs.items()})
    ai_total = ai_bits  # tablo gönderilmez

    # C) Teorik minimum (Shannon)
    theory_bits = math.ceil(
        -sum((c/total) * math.log2(c/total) for c in actual_freq.values()) * total
    )

    return {
        "original_bits": orig_bits,
        "standard": {
            "compressed": std_bits,
            "overhead": std_overhead,
            "total": std_total,
            "ratio": std_total / orig_bits,
        },
        "ai_arithmetic": {
            "compressed": ai_bits,
            "overhead": 0,
            "total": ai_total,
            "ratio": ai_total / orig_bits,
        },
        "theoretical_min": theory_bits,
        "tokens": tokens,
        "saved_vs_standard": std_total - ai_total,
    }
