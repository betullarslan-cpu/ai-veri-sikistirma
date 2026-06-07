"""
AI Engine — Groq ile Huffman frekans tahmini + LZW sözlük oluşturma
"""

import os
import json
import math
from typing import Tuple, List, Dict, Union
from collections import Counter
from groq import Groq

def _get_client() -> Groq:
    """Her cagrida guncel API key ile yeni client uret.
    Streamlit sidebar'dan girilen key dinamik olarak yansisin diye."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Groq API key bulunamadi. Streamlit sol menusunde 'Groq API Key' "
            "kutusuna anahtarinizi girin. (https://console.groq.com/keys)"
        )
    return Groq(api_key=key)


def _get_model() -> str:
    return os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


# Geriye uyumluluk (eski referanslar icin)
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def _chat(system: str, user: str, max_tokens: int = 1024) -> Tuple[str, int]:
    """
    Groq API'ye bir chat çağrısı yapar.

    GÜVENLİK NOTU: Çağrı tamamlandıktan sonra GROQ_API_KEY env değişkeni
    SİLİNİR. Bu sayede her API çağrısı için kullanıcının key'i yeniden
    girmesi gerekir → sızma riski minimize edilir.
    """
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=_get_model(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
        )
    except Exception as e:
        # Anlamli Turkce hata
        msg = str(e).lower()
        if "invalid api key" in msg or "401" in msg:
            raise RuntimeError(
                "❌ Groq API key gecersiz. Sol menudeki kutuya gecerli bir key girin.\n"
                "Yeni key icin: https://console.groq.com/keys"
            )
        if "rate limit" in msg or "429" in msg:
            raise RuntimeError("⏳ Groq rate limit asildi. Bir dakika bekleyip tekrar deneyin.")
        if "connection" in msg or "network" in msg:
            raise RuntimeError("🌐 Internet baglantisi yok veya Groq sunucularına ulasilamiyor.")
        raise RuntimeError(f"Groq API hatasi: {e}")
    finally:
        # Güvenlik: Her çağrıdan sonra key env'den silinir
        # → kullanıcı bir sonraki çağrı için yeniden girmek zorunda
        os.environ.pop("GROQ_API_KEY", None)
    raw = response.choices[0].message.content.strip()
    tokens = response.usage.total_tokens if response.usage else 0
    return raw, tokens


def _parse_json(raw: str):
    import ast

    # ``` bloklarını temizle
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            p = part.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") or p.startswith("["):
                raw = p
                break

    raw = raw.strip()

    # { } veya [ ] bloğunu çıkar
    for sc, ec in [("{", "}"), ("[", "]")]:
        idx = raw.find(sc)
        if idx != -1:
            end_idx = raw.rfind(ec)
            if end_idx != -1:
                raw = raw[idx:end_idx+1]
                break

    # Önce standart JSON dene
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Tek tırnaklı Python dict/list ise ast ile dene
    try:
        result = ast.literal_eval(raw)
        if isinstance(result, (dict, list)):
            return result
    except Exception:
        pass

    # Son çare: tek tırnakları çift tırnağa çevir
    try:
        fixed = raw.replace("'", '"')
        return json.loads(fixed)
    except Exception:
        pass

    raise ValueError(f"JSON parse edilemedi. AI cevabı:\n{raw[:300]}")


# ─────────────────────────────────────────
# 1. Huffman: frekans tahmini
# ─────────────────────────────────────────

def predict_frequencies(text_sample: str, text_type: str = "Türkçe metin") -> Tuple[dict, int]:
    from collections import Counter
    # Gerçek frekansların ilk 10'unu ipucu olarak ver → AI daha doğru tahmin eder
    sample_counter = Counter(text_sample[:600])
    total_sample = sum(sample_counter.values())
    top10_hint = {ch: round(cnt/total_sample, 4)
                  for ch, cnt in sample_counter.most_common(10)}

    raw, tokens = _chat(
        system="Veri sıkıştırma uzmanısın. Her zaman geçerli JSON döndür, başka hiçbir şey yazma.",
        user=f"""Sana bir {text_type} örneği veriyorum.
Tüm metnin karakter frekans dağılımını tahmin et.

Metin örneği:
\"\"\"{text_sample[:600]}\"\"\"

İpucu — örnekteki ilk 10 karakter frekansı: {top10_hint}

Bu ipucunu kullanarak TÜM karakterler için (Türkçe özel dahil: ş,ğ,ü,ö,ç,ı) tam dağılımı çıkar.
Sadece JSON: karakter → 0-1 arası olasılık. Toplam 1.0 olsun."""
    )
    freqs = _parse_json(raw)
    total = sum(freqs.values())
    freqs = {k: v / total for k, v in freqs.items()}
    return freqs, tokens


def refine_frequencies(predicted: dict, missing: List[str], text_type: str) -> Tuple[dict, int]:
    if not missing:
        return predicted, 0
    missing_str = ", ".join(repr(c) for c in missing[:10])
    raw, tokens = _chat(
        system="Frekans tahmini uzmanısın. Sadece JSON döndür.",
        user=f"{text_type} için şu karakterler eksik kaldı: {missing_str}\n"
             "Bu karakterler için 0-1 arası olasılık ver. Sadece JSON.",
        max_tokens=256,
    )
    extra = _parse_json(raw)
    merged = {**predicted, **extra}
    total = sum(merged.values())
    return {k: v / total for k, v in merged.items()}, tokens


def kl_divergence(actual: dict, predicted: dict) -> float:
    eps = 1e-9
    return sum(p * math.log(p / max(predicted.get(ch, eps), eps))
               for ch, p in actual.items())


# ─────────────────────────────────────────
# 2. LZW: AI akıllı sözlük oluştur
# ─────────────────────────────────────────

def generate_lzw_dictionary(text_sample: str, text_type: str = "Türkçe metin",
                             n_words: int = 100) -> Tuple[List[str], int, str]:
    raw, tokens = _chat(
        system="Veri sıkıştırma uzmanısın. LZW algoritması için sözlük oluşturursun. Sadece JSON döndür.",
        user=f"""Sana bir {text_type} örneği veriyorum.
LZW sıkıştırma için en sık geçen {n_words} kelime/kelime çifti/yaygın ifadeyi listele.
Bunları sözlüğe önceden ekleyerek sıkıştırmayı iyileştireceğiz.

Metin örneği:
\"\"\"{text_sample[:400]}\"\"\"

Sadece JSON listesi döndür: ["kelime1", "kelime2", ...]
Kısa (2-15 karakter) ve sık geçen ifadeleri tercih et.""",
        max_tokens=1024,
    )
    words = _parse_json(raw)
    if not isinstance(words, list):
        words = list(words.values()) if isinstance(words, dict) else []
    # Sadece string olanları al, çok uzunları at
    words = [w for w in words if isinstance(w, str) and 2 <= len(w) <= 20]
    return words, tokens, raw
