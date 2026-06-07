"""
UI Yardımcı Fonksiyonları
==========================

Streamlit arayüzünde tekrar kullanılan görsel bileşenler.
app.py'dan ayrı tutuldu: arayüz mantığı modüler.
"""

import streamlit as st


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
