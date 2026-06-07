"""
UI Yardımcı Fonksiyonları
==========================

Streamlit arayüzünde tekrar kullanılan görsel bileşenler.
app.py'dan ayrı tutuldu: arayüz mantığı modüler.
"""

import math
import streamlit as st


def randomness_test(bit_string: str) -> dict:
    """
    "Perfect compression looks like random noise" testi (Shannon).

    Sıkıştırılmış bit dizisinin rastgele görünüp görünmediğini ölçer.
    Mükemmel sıkıştırma → 0/1 dağılımı %50/%50'ye yakın olmalı.

    Returns:
        {
            "n_bits": int,
            "zeros_pct": float, "ones_pct": float,
            "balance": float,        # 50'den ne kadar uzak (0 = mükemmel)
            "chi_square": float,     # Ki-kare istatistiği
            "is_random_like": bool,  # Kabaca rastgele görünüyor mu
            "info_density": float,   # Bit başına bilgi yoğunluğu (0-1)
        }
    """
    if not bit_string:
        return {"n_bits": 0, "zeros_pct": 0, "ones_pct": 0,
                "balance": 0, "chi_square": 0,
                "is_random_like": False, "info_density": 0}

    n = len(bit_string)
    zeros = bit_string.count("0")
    ones = n - zeros
    z_pct = zeros / n * 100
    o_pct = ones / n * 100
    balance = abs(z_pct - 50.0)

    # Ki-kare testi (beklenen: n/2 sıfır, n/2 bir)
    expected = n / 2
    chi_sq = ((zeros - expected) ** 2 + (ones - expected) ** 2) / expected if expected else 0

    # Shannon entropisi (bit başına bilgi)
    if zeros > 0 and ones > 0:
        p0 = zeros / n
        p1 = ones / n
        info_dens = -(p0 * math.log2(p0) + p1 * math.log2(p1))
    else:
        info_dens = 0.0  # tüm 0 veya tüm 1 → bilgi yok

    return {
        "n_bits": n,
        "zeros_pct": z_pct,
        "ones_pct": o_pct,
        "balance": balance,
        "chi_square": chi_sq,
        "is_random_like": balance < 5 and chi_sq < 10,
        "info_density": info_dens,
    }


def goster_randomness_testi(bit_string: str, algoritma_adi: str) -> None:
    """
    Shannon'un 'perfect compression = random noise' testini ekrana getir.
    """
    if not bit_string:
        return

    r = randomness_test(bit_string)
    st.markdown(f"### 🎲 Rastgelelik Testi — *{algoritma_adi}*")
    st.caption(
        "Shannon (1948): **Mükemmel sıkıştırma rastgele gürültüden ayırt edilemez.** "
        "Bit dağılımı %50/%50'ye ne kadar yakınsa o kadar yüksek bilgi yoğunluğu."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("0 oranı", f"%{r['zeros_pct']:.1f}",
              delta=f"{-r['balance']:+.1f}" if r['balance'] < 5 else f"+{r['balance']:.1f}",
              delta_color="inverse" if r['balance'] > 5 else "normal")
    c2.metric("1 oranı", f"%{r['ones_pct']:.1f}")
    c3.metric("Bilgi yoğunluğu", f"{r['info_density']:.4f} bit/bit",
              help="1.0'a yakın = mükemmel sıkıştırma. Düşük = boşa giden bitler var.")
    c4.metric("Ki-kare χ²", f"{r['chi_square']:.2f}",
              help="Rastgele dağılımdan sapma. <10 = rastgele görünür.")

    if r["is_random_like"]:
        st.success(
            f"✅ **Bit dizisi rastgele görünüyor** — Shannon'un kriterine uygun. "
            f"Bilgi yoğunluğu: {r['info_density']:.4f} (1.0'a yakın = ideal). "
            f"Bu, sıkıştırmanın optimale yakın olduğunun **dolaylı kanıtıdır**."
        )
    else:
        kayip = (1 - r['info_density']) * r['n_bits']
        st.info(
            f"ℹ️ Bit dağılımı %{r['zeros_pct']:.1f}/%{r['ones_pct']:.1f} — "
            f"hâlâ {kayip:.0f} bit kazanım potansiyeli var "
            f"(rastgeleliğe %{abs(50-r['zeros_pct']):.1f} uzakta). "
            f"Arithmetic coding gibi daha sıkı algoritmalar veya LLM tabanlı "
            f"sıkıştırıcılar bu farkı kapatabilir."
        )


def goster_sikistirma_ciktisi(algoritma_adi: str, bit_string: str,
                               byte_data: bytes, orig_byte: int,
                               key_suffix: str = "") -> None:
    """
    Sıkıştırılmış çıktıyı 4 farklı görünümle ekranda gösterir.

    Args:
        algoritma_adi: "Standart Huffman", "LZW", "BWT+RLE+Huffman" gibi
        bit_string:    "01001011..." biçiminde sıkıştırılmış bit dizisi
        byte_data:     8-bit paketli binary
        orig_byte:     Orijinal metnin byte cinsinden boyutu (karşılaştırma için)
        key_suffix:    Streamlit widget'larının benzersiz key'i ("huff_std" gibi)

    Görüntülenenler:
        1. 4 metric (orijinal, sıkışmış, toplam bit, karakter başı bit)
        2. Binary önizleme (ilk 320 bit, 8'erli gruplar)
        3. Hex önizleme (ilk 80 byte)
        4. Tam binary expander (320 bit üstünde)
        5. Download butonu (.bin)
    """
    if not bit_string:
        return

    total_bits = len(bit_string)
    bin_size = len(byte_data)
    kucullme = (1 - bin_size / orig_byte) * 100 if orig_byte else 0

    st.markdown(f"### 💾 Sıkıştırılmış Çıktı — *{algoritma_adi}*")

    # ── 4 Boyut metriği ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orijinal (text)", f"{orig_byte:,} byte")
    c2.metric("Sıkışmış (binary)", f"{bin_size:,} byte",
              delta=f"-%{kucullme:.1f}", delta_color="off")
    c3.metric("Toplam bit", f"{total_bits:,}")
    c4.metric("Karakter başı",
              f"{total_bits/orig_byte if orig_byte else 0:.2f} bit",
              help="Standart 8 bit/karaktere göre kıyas")

    # ── 2 sütun: Binary + Hex ──
    cb, ch = st.columns(2)
    with cb:
        st.markdown("**🔢 Binary (ilk 320 bit)**")
        preview = bit_string[:320]
        formatted = " ".join(preview[i:i+8] for i in range(0, len(preview), 8))
        st.code(formatted + ("..." if total_bits > 320 else ""), language=None)

    with ch:
        st.markdown("**🔣 Hexadecimal (ilk 80 byte)**")
        hb = byte_data[:80].hex().upper()
        lines = []
        for i in range(0, len(hb), 32):
            chunk = hb[i:i+32]
            line = " ".join(chunk[j:j+2] for j in range(0, len(chunk), 2))
            lines.append(line)
        st.code("\n".join(lines) + ("\n..." if bin_size > 80 else ""), language=None)

    # ── Tam binary (opsiyonel expander) ──
    if total_bits > 320:
        with st.expander(f"📜 Tam binary dizisini göster ({total_bits:,} bit)"):
            blocks = [bit_string[i:i+64] for i in range(0, len(bit_string), 64)]
            st.code("\n".join(blocks[:50])
                    + ("\n... (kalanı indir)" if len(blocks) > 50 else ""))

    # ── Download butonu ──
    st.download_button(
        f"⬇ {algoritma_adi} sıkıştırılmış (.bin) — {bin_size:,} byte",
        data=byte_data,
        file_name=f"sikistirilmis_{key_suffix}.bin",
        mime="application/octet-stream",
        key=f"dl_{key_suffix}",
    )
