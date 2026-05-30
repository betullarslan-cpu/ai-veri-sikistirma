"""
Shannon Entropi Analizi
-----------------------
H(X) = -Σ p(x) * log2(p(x))

Teorik minimum bit/karakter değerini verir.
Hiçbir kayıpsız algoritma bu sınırın altına inemez.
"""

import math
from collections import Counter


def shannon_entropy(text: str) -> float:
    """Bit/karakter cinsinden Shannon entropisi."""
    if not text:
        return 0.0
    total = len(text)
    freq = Counter(text)
    return -sum((cnt / total) * math.log2(cnt / total)
                for cnt in freq.values())


def theoretical_min_bits(text: str) -> int:
    """Teorik olarak mümkün en düşük bit sayısı."""
    return math.ceil(shannon_entropy(text) * len(text))


def compression_efficiency(text: str, compressed_bits: int) -> dict:
    """Bir algoritmanın teorik sınıra ne kadar yaklaştığını ölçer."""
    entropy     = shannon_entropy(text)
    theory_bits = theoretical_min_bits(text)
    original    = len(text) * 8
    efficiency  = (theory_bits / compressed_bits * 100) if compressed_bits > 0 else 0

    return {
        "entropy_per_char": round(entropy, 4),
        "theoretical_min_bits": theory_bits,
        "original_bits": original,
        "compressed_bits": compressed_bits,
        "theory_ratio": round(theory_bits / original, 4),
        "actual_ratio": round(compressed_bits / original, 4),
        "efficiency_pct": round(efficiency, 1),   # 100% = mükemmel
        "gap_bits": compressed_bits - theory_bits,
    }


def per_char_analysis(text: str) -> list[dict]:
    """Her karakterin entropi katkısını gösterir (grafik için)."""
    total = len(text)
    freq = Counter(text)
    rows = []
    for ch, cnt in sorted(freq.items(), key=lambda x: -x[1]):
        p = cnt / total
        contrib = -p * math.log2(p)
        rows.append({
            "char": repr(ch),
            "count": cnt,
            "probability": round(p, 4),
            "bits_needed": round(math.log2(1 / p), 2),
            "entropy_contribution": round(contrib, 4),
        })
    return rows
