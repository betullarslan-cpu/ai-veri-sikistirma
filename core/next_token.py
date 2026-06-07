"""
Next-Token Tahmin Tabanlı Sıkıştırma Analizi
=============================================

Shannon (1951) "Prediction and Entropy of Printed English" makalesinin
modern bir yorumu: karakterleri tek tek tahmin ederek kontekstli entropi
tahmini yapar.

İki seviye sunar:
    1. **Unigram (iid)** — P(c) — Klasik Shannon entropisi
    2. **Bigram (1. derece Markov)** — P(c | önceki_c)
    3. **Trigram (2. derece Markov)** — P(c | önceki_2c)

Her seviyede karakterin bilgi içeriği:
    I(c) = -log₂ P(c | context)

Tüm metnin "ideal" sıkıştırma boyutu:
    bits = Σ -log₂ P(c | context)

KAYNAK
------
Shannon, C. E. (1951). "Prediction and Entropy of Printed English,"
Bell System Technical Journal, 30(1), 50-64.

Bu yöntemle Shannon İngilizce için ~1.3 bit/karakter ölçtü.
Modern LLM'ler aynı yaklaşımı kullanarak ~0.8 bit/karakter elde eder.
"""

import os
import math
from collections import Counter, defaultdict
from typing import Dict, Tuple


def _load_corpus(corpus_path: str = None) -> str:
    """Türkçe corpus'u yükle (n-gram tablosu için)."""
    if corpus_path is None:
        # Varsayılan: en büyük Türkçe corpus
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        corpus_path = os.path.join(base, "data", "large_turkish.txt")
    if not os.path.exists(corpus_path):
        return ""
    with open(corpus_path, "r", encoding="utf-8") as f:
        return f.read()


def unigram_table(corpus: str) -> Dict[str, float]:
    """Tek karakter olasılıkları P(c) — Shannon (1948) iid entropisi."""
    if not corpus:
        return {}
    total = len(corpus)
    counts = Counter(corpus)
    return {ch: c / total for ch, c in counts.items()}


def bigram_table(corpus: str) -> Dict[str, Dict[str, float]]:
    """
    Bigram olasılıkları P(c_i | c_{i-1}).

    Returns:
        {prev_char: {next_char: probability}}
    """
    if not corpus or len(corpus) < 2:
        return {}
    counts = defaultdict(Counter)
    for i in range(len(corpus) - 1):
        counts[corpus[i]][corpus[i + 1]] += 1
    return {
        prev: {nxt: c / sum(d.values()) for nxt, c in d.items()}
        for prev, d in counts.items()
    }


def trigram_table(corpus: str) -> Dict[str, Dict[str, float]]:
    """
    Trigram olasılıkları P(c_i | c_{i-2}, c_{i-1}).

    Returns:
        {prev_2chars: {next_char: probability}}
    """
    if not corpus or len(corpus) < 3:
        return {}
    counts = defaultdict(Counter)
    for i in range(len(corpus) - 2):
        ctx = corpus[i:i + 2]
        counts[ctx][corpus[i + 2]] += 1
    return {
        ctx: {nxt: c / sum(d.values()) for nxt, c in d.items()}
        for ctx, d in counts.items()
    }


def entropy_bits(text: str,
                 unigram: Dict[str, float],
                 bigram: Dict[str, Dict[str, float]] = None,
                 trigram: Dict[str, Dict[str, float]] = None,
                 smoothing: float = 1e-6) -> Tuple[float, list]:
    """
    Metnin "ideal" sıkıştırma bit sayısı (next-token tahmin tabanlı).

    Trigram → Bigram → Unigram → smoothing sırasıyla backoff yapar
    (Markov modellerinde klasik back-off, MacKay §6.2).

    Args:
        text:      Test metni
        unigram:   P(c) tablosu
        bigram:    P(c | önceki) tablosu (opsiyonel)
        trigram:   P(c | önceki_2) tablosu (opsiyonel)
        smoothing: Tablodan eksik karakterler için minimum olasılık

    Returns:
        (toplam_bit, karakter_basina_listesi)
    """
    total_bits = 0.0
    per_char = []

    for i, c in enumerate(text):
        # Trigram (en spesifik)
        prob = None
        if trigram and i >= 2:
            ctx = text[i - 2:i]
            if ctx in trigram and c in trigram[ctx]:
                prob = trigram[ctx][c]

        # Bigram (fallback)
        if prob is None and bigram and i >= 1:
            prev = text[i - 1]
            if prev in bigram and c in bigram[prev]:
                prob = bigram[prev][c]

        # Unigram (fallback)
        if prob is None and unigram and c in unigram:
            prob = unigram[c]

        # Smoothing (görülmemiş karakter)
        if prob is None or prob <= 0:
            prob = smoothing

        bits_c = -math.log2(prob)
        total_bits += bits_c
        per_char.append((c, prob, bits_c))

    return total_bits, per_char


def karsilastir(text: str, corpus_path: str = None) -> Dict:
    """
    Metni 3 farklı modelle karşılaştır: unigram / bigram / trigram.

    Returns:
        {
            "n_chars": int,
            "orig_bits": int (8 * n_chars),
            "unigram": {"bits": int, "bpc": float, "label": "Shannon iid"},
            "bigram":  {"bits": int, "bpc": float, "label": "1. derece Markov"},
            "trigram": {"bits": int, "bpc": float, "label": "2. derece Markov"},
        }
    """
    corpus = _load_corpus(corpus_path)
    if not corpus:
        return {"error": "Corpus yüklenemedi"}

    uni = unigram_table(corpus)
    bi  = bigram_table(corpus)
    tri = trigram_table(corpus)

    n = len(text)
    orig = n * 8

    u_bits, _ = entropy_bits(text, uni)
    b_bits, _ = entropy_bits(text, uni, bigram=bi)
    t_bits, _ = entropy_bits(text, uni, bigram=bi, trigram=tri)

    return {
        "n_chars":   n,
        "orig_bits": orig,
        "corpus_size": len(corpus),
        "unigram": {
            "bits": int(u_bits),
            "bpc":  u_bits / n if n else 0,
            "ratio": u_bits / orig if orig else 0,
            "label": "Unigram (Shannon iid)",
            "aciklama": "Karakter olasılıkları bağımsız. Shannon 1948."
        },
        "bigram": {
            "bits": int(b_bits),
            "bpc":  b_bits / n if n else 0,
            "ratio": b_bits / orig if orig else 0,
            "label": "Bigram (1. derece Markov)",
            "aciklama": "P(c | önceki). Karakterler birinci derece bağımlı."
        },
        "trigram": {
            "bits": int(t_bits),
            "bpc":  t_bits / n if n else 0,
            "ratio": t_bits / orig if orig else 0,
            "label": "Trigram (2. derece Markov)",
            "aciklama": "P(c | önceki, önceki). Shannon (1951) bu yaklaşımı kullandı."
        },
    }


if __name__ == "__main__":
    # Hızlı test
    test = "Yapay zeka teknolojileri hizla gelisiyor. Veri sikistirma onemlidir."
    r = karsilastir(test)
    if "error" in r:
        print(r["error"])
    else:
        print(f"Test metni: {r['n_chars']} karakter, orjinal {r['orig_bits']} bit")
        print(f"Corpus boyutu: {r['corpus_size']:,} karakter\n")
        for key in ["unigram", "bigram", "trigram"]:
            d = r[key]
            print(f"  {d['label']:<28} {d['bits']:>5} bit  ({d['bpc']:.2f} bit/karakter)")
