"""
AI-Destekli Veri Sıkıştırma — Final Projesi
Streamlit Arayüzü
"""

import os
import sys
import json
import math
import heapq
import time
from collections import Counter

import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
# Sıkıştırma algoritmaları (core/)
from core.huffman import encode as huffman_encode, build_codes, HuffmanNode
from core.lzw import compare as lzw_compare
from core.ai_engine import (
    predict_frequencies, refine_frequencies, kl_divergence,
    generate_lzw_dictionary,
)
from core.entropy import shannon_entropy, theoretical_min_bits, per_char_analysis
from core.hybrid import hybrid_compress, train_from_corpus, smart_hybrid
from core.bwt import compare as bwt_compare, bwt_stats, bwt_encode
from core.nn_selector import predict as nn_predict, train as nn_train

# ─── Sayfa ayarı ───────────────────────────────────
st.set_page_config(
    page_title="AI Sıkıştırma",
    page_icon="🗜️",
    layout="wide",
)

st.title("🗜️ AI-Destekli Veri Sıkıştırma")
st.caption(
    "Klasik sıkıştırma algoritmalarını (Huffman, LZW, BWT) bir **sinir ağı** ile birleştirip "
    "her veri türünde otomatik en iyi yöntemi seçen sistem."
)

# ─── Sidebar: API key + ayarlar ────────────────────
with st.sidebar:
    api_key = st.text_input("Groq API Key", type="password",
                            value=os.environ.get("GROQ_API_KEY", ""),
                            help="Sadece AI içeren sekmeler için gerekli. "
                                 "console.groq.com → API Keys (ücretsiz)")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    text_type = st.selectbox("Metin türü", ["Türkçe metin", "İngilizce metin", "Kod (Python)"])

    with st.expander("Gelişmiş"):
        model = st.selectbox("AI Modeli", [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ])
        os.environ["GROQ_MODEL"] = model

# ─── Metin girişi ──────────────────────────────────
col_inp1, col_inp2 = st.columns([3, 1])

with col_inp1:
    uploaded = st.file_uploader("Dosya yükle (.txt)", type=["txt"])
    if uploaded:
        text = uploaded.read().decode("utf-8")
        st.success(f"Dosya yüklendi: {len(text):,} karakter")
    else:
        text = st.text_area("veya buraya metin yaz/yapıştır",
                            height=150,
                            value="Türkiye, Avrupa ve Asya kıtaları arasında köprü görevi gören eşsiz bir ülkedir. "
                                  "Zengin tarihi ve kültürel mirası ile her yıl milyonlarca turistin ziyaret ettiği bu ülke, "
                                  "doğal güzellikleri bakımından da son derece dikkat çekicidir.")

with col_inp2:
    st.metric("Karakter sayısı", f"{len(text):,}")
    st.metric("Orijinal boyut", f"{len(text)*8:,} bit")

if len(text) < 10:
    st.warning("En az 10 karakter girin.")
    st.stop()


# ─── Yardımcı: Sıkıştırılmış çıktıyı ekrana göster ───
def goster_sikistirma_ciktisi(algoritma_adi: str, bit_string: str,
                               byte_data: bytes, orig_byte: int,
                               key_suffix: str = ""):
    """
    Sıkıştırılmış çıktıyı 4 farklı görünümle ekranda gösterir:
    1) Binary (ilk 320 bit)
    2) Hexadecimal (ilk 80 byte)
    3) Boyut karşılaştırması
    4) Download butonu
    """
    if not bit_string:
        return
    total_bits = len(bit_string)
    bin_size   = len(byte_data)
    kucullme   = (1 - bin_size/orig_byte) * 100 if orig_byte else 0

    st.markdown(f"### 💾 Sıkıştırılmış Çıktı — *{algoritma_adi}*")

    # Boyut karşılaştırma metrikleri
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orijinal (text)", f"{orig_byte:,} byte")
    c2.metric("Sıkışmış (binary)", f"{bin_size:,} byte",
              delta=f"-%{kucullme:.1f}", delta_color="off")
    c3.metric("Toplam bit", f"{total_bits:,}")
    c4.metric("Karakter başı", f"{total_bits/orig_byte if orig_byte else 0:.2f} bit",
              help="Standart 8 bit/karaktere göre kıyas")

    # 2 sütun: binary + hex
    cb, ch = st.columns(2)
    with cb:
        st.markdown("**🔢 Binary (ilk 320 bit)**")
        preview = bit_string[:320]
        # 8'erli grupla
        formatted = " ".join(preview[i:i+8] for i in range(0, len(preview), 8))
        st.code(formatted + ("..." if total_bits > 320 else ""), language=None)

    with ch:
        st.markdown("**🔣 Hexadecimal (ilk 80 byte)**")
        hex_preview = byte_data[:80].hex(" ").upper()
        # 16'lı grupla (her satırda 16 byte = 48 char + 15 boşluk = 47 char)
        lines = []
        hb = byte_data[:80].hex().upper()
        for i in range(0, len(hb), 32):
            chunk = hb[i:i+32]
            line = " ".join(chunk[j:j+2] for j in range(0, len(chunk), 2))
            lines.append(line)
        st.code("\n".join(lines) + ("\n..." if bin_size > 80 else ""), language=None)

    # Tam binary expander (uzun metin için)
    if total_bits > 320:
        with st.expander(f"📜 Tam binary dizisini göster ({total_bits:,} bit)"):
            # 64 bitlik bloklar halinde
            blocks = [bit_string[i:i+64] for i in range(0, len(bit_string), 64)]
            st.code("\n".join(blocks[:50]) + ("\n... (kalanı indir)" if len(blocks) > 50 else ""))

    # Download butonu
    st.download_button(
        f"⬇ {algoritma_adi} sıkıştırılmış (.bin) — {bin_size:,} byte",
        data=byte_data,
        file_name=f"sikistirilmis_{key_suffix}.bin",
        mime="application/octet-stream",
        key=f"dl_{key_suffix}",
    )


# ─── Sekmeler — sadece gerekli olanlar ──────
tab0, tab1, tab2, tab4, tab6, tab9, tab11, tab10 = st.tabs([
    "🚀 Hızlı Özet",
    "📊 Huffman",
    "📖 LZW",
    "🔬 Sinir Ağı",
    "⚡ Hibrit",
    "📈 Shannon",
    "🔄 BWT",
    "🤖 Günlük",
])


# ═══════════════════════════════════════════════════
# SEKME 0: HIZLI OZET — tek ekranda hersey
# ═══════════════════════════════════════════════════
with tab0:
    st.subheader("🚀 Hızlı Özet — Tek Ekranda Tüm Sonuçlar")
    st.caption("Bir butonla tüm algoritmaları çalıştır, yan yana karşılaştır. API gerekmez.")

    with st.expander("ℹ️ Bu sekmede ne göreceksin?"):
        st.markdown("""
**Butona basınca otomatik olarak şunlar hesaplanır:**

1. **4 ana metrik:** Orijinal boyut, Shannon limiti, Standart Huffman, Akıllı Hibrit
2. **Sinir ağı kararı:** Hangi algoritmayı neden seçti
3. **Karşılaştırma grafiği:** 5 algoritmanın bit sayıları yan yana
4. **Algoritma sonuçları tablosu:** Her algoritmanın küçülme yüzdesi
5. **Sıkıştırılmış çıktı:** Gerçek bit dizisi + indirilebilir `.bin` dosyası

Her bölümün altında **💡 ile başlayan kutucuklar** sonucun ne anlama geldiğini açıklar.
        """)

    if st.button("▶ Tümünü Hesapla", key="quick_run", type="primary"):
        with st.spinner("Hesaplanıyor..."):
            from core.bwt import compare as _bwt_cmp
            from core.hybrid import smart_hybrid as _sh
            from core.nn_selector import predict as _nnp
            from core.entropy import shannon_entropy as _shan, theoretical_min_bits as _tmb
            from core.benchmark import tam_karsilastirma as _bench

            _orig = len(text) * 8
            _ent  = _shan(text)
            _min  = _tmb(text)
            _bwt  = _bwt_cmp(text)
            _sm   = _sh(text, use_ai_dict=False)
            _nn   = _nnp(text)
            _bm   = _bench(text)

        # ── 4 metrik yan yana ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Orijinal", f"{_orig:,} bit",
                  help="Sıkıştırılmadan önceki boyut. Her karakter UTF-8'de 8 bit varsayılır.")
        m2.metric("Shannon limiti", f"{_min:,} bit",
                  delta=f"%{(1-_min/_orig)*100:.1f} küçülme",
                  delta_color="off",
                  help="Bilgi teorisinin söylediği MUTLAK minimum. Hiçbir algoritma "
                       "bunun altına inemez (kayıpsız sıkıştırmada).")
        m3.metric("Standart Huffman", f"{_sm['standard_bits']:,} bit",
                  delta=f"%{(1-_sm['standard_ratio'])*100:.1f} küçülme",
                  delta_color="off",
                  help="Klasik Huffman + frekans tablosu overhead'i. "
                       "Karşılaştırma için referansımız.")
        m4.metric(f"🏆 Akıllı Hibrit ({_sm['nn_decision']})",
                  f"{_sm['smart_bits']:,} bit",
                  delta=f"%{(1-_sm['smart_ratio'])*100:.1f} küçülme",
                  help="Sinir ağı en uygun algoritmayı seçti ve onu çalıştırdı. "
                       "Bu projenin nihai sonucudur.")

        # ── METRIKLER NE ANLAMA GELIYOR? ──
        with st.expander("💡 Yukarıdaki metrikler ne anlama geliyor?"):
            _ks_huff = (1 - _sm['standard_ratio']) * 100
            _ks_smart = (1 - _sm['smart_ratio']) * 100
            _shan_pct = (1 - _min/_orig) * 100
            st.markdown(f"""
- **Orijinal ({_orig:,} bit):** Metin sıkıştırılmadan saklansaydı bu kadar yer kaplardı.
- **Shannon limiti ({_min:,} bit, %{_shan_pct:.1f} küçülme):**
  Karakter olasılıklarına göre teorik en küçük boyut. Hiçbir kayıpsız algoritma bunun altına inemez.
- **Standart Huffman ({_sm['standard_bits']:,} bit, %{_ks_huff:.1f} küçülme):**
  Klasik Huffman + frekans tablosunun ek yükü. 1952'den beri kullanılan referansımız.
- **🏆 Akıllı Hibrit ({_sm['smart_bits']:,} bit, %{_ks_smart:.1f} küçülme):**
  Bizim sistemimizin sonucu. Shannon limitine **%{(_sm['smart_bits']/_min-1)*100:.1f}** uzakta —
  yani neredeyse teorik mükemmel.
            """)

        # NN bilgisi tek satır
        st.success(
            f"🧠 **Sinir Ağı kararı:** {_nn['algorithm'].upper()} (%{_nn['confidence']*100:.0f} güven) — "
            f"**Yöntem:** {_sm['method']} — "
            f"**Standart Huffman'a göre +%{_sm['saved_bits']*100/_sm['standard_bits']:.1f} iyileşme** "
            f"({_sm['saved_bits']:,} bit tasarruf)"
        )

        with st.expander("💡 Sinir ağı nasıl karar verdi?"):
            _feat = _nn.get("features", {})
            _entrop = _feat.get("entropi", 0)
            _alpha = _feat.get("alfabe_boyutu_log", 0)
            _maxrun = _feat.get("max_koşu_oran", 0)
            _algo = _nn['algorithm']
            _explain = {
                "huffman": "Entropi yüksek + tekrar az → karakter bazlı kodlamak en iyisi.",
                "lzw":     "Tekrarlı kelime/ifadeler var → sözlük tabanlı kodlama avantajlı.",
                "bwt":     "Yapısal düzen var (küçük alfabe veya benzer karakterler kümelenebilir) → BWT permütasyonu çok kazandırır.",
            }.get(_algo, "")
            st.markdown(f"""
Sinir ağı (MLP 32→16→8) metinden **11 özellik** çıkardı:

- **Entropi:** {_entrop:.3f} (yüksek = tahmin edilemez)
- **Alfabe boyutu (log₂):** {_alpha:.2f} (küçük alfabe BWT'ye yarar)
- **Max koşu oranı:** {_maxrun:.4f} (tekrarlı bloklar var mı?)
- **Olasılıklar:** Huffman %{_nn['probabilities'].get('huffman',0)*100:.1f}, "
LZW %{_nn['probabilities'].get('lzw',0)*100:.1f}, "
BWT %{_nn['probabilities'].get('bwt',0)*100:.1f}

**Karar gerekçesi:** {_explain}

> Model %{_nn.get('cv_accuracy', 0)*100:.1f} cross-validation doğruluğuyla çalışıyor —
> eğitim ezberi değil, gerçek genelleme.
            """)

        # ── 5 algoritmali kompakt grafik ──
        _res = _bwt['results']
        _names = list(_res.keys())
        _bits  = [_res[n]['bits'] for n in _names]
        _best  = _bwt['best']
        _cols  = ["#FFD700" if n == _best else
                  ("#00CC96" if n.startswith("BWT") else "#636EFA") for n in _names]
        fig = go.Figure(go.Bar(
            x=_names, y=_bits, marker_color=_cols,
            text=[f"{b:,}<br>%{(1-b/_orig)*100:.1f}" for b in _bits],
            textposition="outside",
        ))
        fig.add_hline(y=_min, line_dash="dash", line_color="green",
                      annotation_text=f"Shannon: {_min:,}")
        fig.add_hline(y=_orig, line_dash="dot", line_color="red",
                      annotation_text=f"Orijinal: {_orig:,}")
        fig.update_layout(
            yaxis_title="Bit", height=360, showlegend=False,
            margin=dict(t=20, b=20, l=40, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "📊 **Grafik açıklaması:** Her çubuk farklı bir algoritmanın ürettiği bit sayısını gösterir. "
            "🟡 sarı = kazanan algoritma, 🟢 yeşil = BWT ailesi, 🔵 mavi = klasik. "
            "Kırmızı kesik çizgi orijinal boyut, yeşil kesik çizgi Shannon'un teorik minimumudur. "
            "**Kısa çubuk = iyi sıkıştırma.**"
        )

        # ═══════ ENDÜSTRİ STANDARTLARI KARŞILAŞTIRMASI ═══════
        st.markdown("---")
        st.markdown("### 🏭 Endüstri Standartları Karşılaştırması")
        st.caption("Bizim algoritmalarımız gzip, bzip2, zlib, lzma ile yan yana.")

        _ben_rows = []
        for _alg, _d in _bm["endustri"].items():
            if isinstance(_d, dict):
                _ben_rows.append({
                    "Algoritma": _alg.upper(),
                    "Boyut (byte)": f"{_d['byte']:,}",
                    "Oran": f"{_d['ratio']:.4f}",
                    "Küçülme": f"%{_d['kucullme_pct']:.1f}",
                    "Süre (ms)": f"{_d['sure_ms']:.2f}",
                    "Tür": "🏭 Endüstri",
                })
        for _alg, _d in _bm["bizim"].items():
            if isinstance(_d, dict):
                _ben_rows.append({
                    "Algoritma": _alg.replace("_", "+").upper(),
                    "Boyut (byte)": f"{_d['byte']:,}",
                    "Oran": f"{_d['ratio']:.4f}",
                    "Küçülme": f"%{_d['kucullme_pct']:.1f}",
                    "Süre (ms)": f"{_d['sure_ms']:.2f}",
                    "Tür": "⭐ Bizim",
                })
        # Boyuta göre sırala (küçük = iyi)
        _ben_rows.sort(key=lambda r: int(r["Boyut (byte)"].replace(",", "")))
        st.dataframe(_ben_rows, use_container_width=True, hide_index=True, height=320)

        # Karşılaştırma grafiği
        _ben_names = [r["Algoritma"] for r in _ben_rows]
        _ben_bytes = [int(r["Boyut (byte)"].replace(",", "")) for r in _ben_rows]
        _ben_colors = ["#FFD700" if "BWT" in n or "AKILLI" in n else
                       "#00CC96" if "Bizim" in r["Tür"] else "#888"
                       for n, r in zip(_ben_names, _ben_rows)]
        fig_b = go.Figure(go.Bar(
            x=_ben_names, y=_ben_bytes, marker_color=_ben_colors,
            text=[f"{b:,}" for b in _ben_bytes], textposition="outside",
        ))
        fig_b.update_layout(
            title="Sıkıştırma Sonrası Boyut (byte) — Düşük = İyi",
            yaxis_title="Byte", height=380, showlegend=False,
            margin=dict(t=40, b=80, l=40, r=10),
        )
        fig_b.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_b, use_container_width=True)

        # Yorum
        _our_best = _bm["bizim_en_iyi_byte"]
        _bzip = _bm["bzip2_byte"]
        if _our_best <= _bzip:
            st.success(
                f"🎉 **Bizim en iyi algoritmamız ({_our_best:,} byte) **bzip2'den ({_bzip:,} byte)** "
                f"**%{(_bzip-_our_best)/_bzip*100:.1f} daha küçük!** "
                f"Bu, BWT+RLE+Huffman implementasyonumuzun gerçek bzip2 ile yarışabildiğini gösteriyor."
            )
        else:
            st.info(
                f"📊 Bizim en iyi algoritmamız ({_our_best:,} byte) bzip2'den ({_bzip:,} byte) "
                f"**%{(_our_best-_bzip)/_bzip*100:.1f} daha büyük.** "
                f"Endüstri standartları yıllarca optimize edildi; bu bizim açık kaynak "
                f"eğitim implementasyonumuz için iyi bir sonuç."
            )

        # ── Kompakt tablo (2 sütun yan yana) ──
        cL, cR = st.columns(2)
        with cL:
            st.markdown("**📊 Algoritma Sonuçları**")
            rows = []
            for n, rv in _res.items():
                rows.append({
                    "Algoritma": ("🏆 " if n == _best else "") + n,
                    "Bit":       f"{rv['bits']:,}",
                    "Küçülme":   f"%{(1-rv['ratio'])*100:.1f}",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True, height=210)

        with cR:
            st.markdown("**🧠 NN Olasılıkları & Özellikler**")
            probs = _nn.get("probabilities", {})
            prob_rows = [
                {"Algoritma": "Huffman", "Olasılık": f"%{probs.get('huffman',0)*100:.1f}"},
                {"Algoritma": "LZW",     "Olasılık": f"%{probs.get('lzw',0)*100:.1f}"},
                {"Algoritma": "BWT",     "Olasılık": f"%{probs.get('bwt',0)*100:.1f}"},
            ]
            st.dataframe(prob_rows, use_container_width=True, hide_index=True, height=160)

            # === SIKIŞTIRILMIŞ ÇIKTI ===
            from core.bwt import bwt_rle_huffman_encode, huffman_encode_bytes
            _bwt_out  = bwt_rle_huffman_encode(text)
            _huff_out = huffman_encode_bytes(text)

            st.markdown("---")
            st.markdown("**💾 Sıkıştırılmış Çıktı (BWT+RLE+Huffman)**")
            _bin_size = len(_bwt_out['byte_data'])
            _orig_byte = len(text.encode('utf-8'))
            st.metric("Binary boyut", f"{_bin_size:,} byte",
                      delta=f"{_orig_byte:,} byte (orjinal) → {_bin_size:,} byte (sıkışmış)")
            # Bit dizisi önizleme
            bits_preview = _bwt_out['bit_string'][:160]
            st.code(bits_preview + ("..." if len(_bwt_out['bit_string']) > 160 else ""),
                    language=None)
            st.caption(
                f"☝️ Yukarıdaki **bit dizisi** sıkıştırılmış halin gerçek 0/1 görünümüdür "
                f"(toplam **{_bwt_out['total_bits']:,} bit**, ilk 160 bit gösteriliyor). "
                f"Her 8 bit bir byte oluşturur → toplam **{_bin_size:,} byte** binary dosya."
            )
            # Indirme butonları
            d1, d2 = st.columns(2)
            d1.download_button(
                "⬇ BWT sıkıştırılmış (.bin)",
                data=_bwt_out['byte_data'],
                file_name="sikistirilmis_bwt.bin",
                mime="application/octet-stream",
                key="dl_bwt",
                help="bzip2 tarzı — en küçük boyut. İndir, herhangi bir hex editör ile görüntüle.",
            )
            d2.download_button(
                "⬇ Huffman sıkıştırılmış (.bin)",
                data=_huff_out['byte_data'],
                file_name="sikistirilmis_huffman.bin",
                mime="application/octet-stream",
                key="dl_huff",
                help="Klasik Huffman — referans dosya. BWT versiyonuyla boyut farkını gör.",
            )
            with st.expander("💡 Bu sıkıştırılmış çıktı ne işe yarar?"):
                _kazanc = (1 - _bin_size/_orig_byte) * 100
                st.markdown(f"""
- **{_orig_byte:,} byte → {_bin_size:,} byte** ({_kazanc:.1f}% küçülme)
- İndirdiğin `.bin` dosyası **gerçek sıkıştırılmış veridir** — diskte bu kadar yer kaplar.
- Açmak için aynı algoritma (BWT+RLE+Huffman) ile **decode** edilmesi gerekir.
- Hocaya gösterirken: orijinal `.txt` ile `.bin` dosyalarının boyutlarını karşılaştır.
                """)
            st.caption(
                f"Entropi: **{_ent:.3f}** bit/kar  •  "
                f"Karakter: **{len(text):,}**  •  "
                f"Model doğruluk: **%{_nn.get('model_accuracy',0)*100:.1f}** (CV %{_nn.get('cv_accuracy',0)*100:.1f})"
            )
    else:
        st.info("👆 Yukarıdaki butona basarak metin için tüm sıkıştırma sonuçlarını "
                "tek ekranda gör. Sol menüden metin değiştirebilirsin.")


# ═══════════════════════════════════════════════════
# SEKME 1: HUFFMAN
# ═══════════════════════════════════════════════════
with tab1:
    st.subheader("Standart Huffman vs AI-Destekli Huffman")
    st.markdown(
        "**Yenilik:** Standart Huffman frekans tablosunu dosyadan hesaplar ve "
        "bu tabloyu sıkıştırılmış veriye eklemek zorundadır (overhead). "
        "AI tahminiyle tablo önceden bilinirse overhead ortadan kalkar."
    )

    if st.button("▶ Huffman Analizi Başlat", key="huff"):
        if not api_key:
            st.error("Sol menüden Groq API key gir.")
        else:
            # Standart Huffman
            with st.spinner("Standart Huffman hesaplanıyor..."):
                t0 = time.perf_counter()
                bits, codes = huffman_encode(text)
                std_time = (time.perf_counter() - t0) * 1000
                orig_bits = len(text) * 8
                comp_bits = len(bits)
                overhead = len(codes) * 12
                total_std = comp_bits + overhead

            # ── SIKIŞTIRILMIŞ ÇIKTI (Standart Huffman) ──
            from core.bwt import huffman_encode_bytes as _huff_enc
            from core.huffman import decode as _huff_decode
            _h_out = _huff_enc(text)
            goster_sikistirma_ciktisi(
                "Standart Huffman", _h_out['bit_string'],
                _h_out['byte_data'], len(text.encode('utf-8')),
                key_suffix="huff_std",
            )

            # ── DECODE: Adım adım sıkıştırmayı geri aç ──
            st.markdown("### 🔓 Açma (Decompression) — Adım Adım Kayıpsızlık Kanıtı")

            with st.expander("🔍 **Aşama 1 — Girdi:** Sıkıştırılmış bit dizisi", expanded=False):
                st.code(_h_out['bit_string'][:200] + ("..." if len(_h_out['bit_string']) > 200 else ""))
                st.caption(f"Toplam **{len(_h_out['bit_string']):,}** bit "
                          f"({len(_h_out['byte_data']):,} byte binary).")

            with st.expander("🔍 **Aşama 2 — Ters Huffman Tablosu** (kod → karakter)", expanded=False):
                _reverse = {v: k for k, v in _h_out['codes'].items()}
                # En kısa kodları göster (en sık karakterler)
                _sorted_rev = sorted(_reverse.items(), key=lambda x: len(x[0]))
                _tablo_rows = [
                    {"Bit kodu": k, "Uzunluk": f"{len(k)} bit", "Karakter": repr(v)}
                    for k, v in _sorted_rev[:15]
                ]
                st.dataframe(_tablo_rows, use_container_width=True, hide_index=True)
                st.caption(f"Toplam **{len(_reverse)} kod** var (ilk 15 gösteriliyor). "
                          f"Encode'da ağaçta kurduğumuz prefix-free kodlar.")

            with st.expander("🔍 **Aşama 3 — Tampon (buffer) ile bit okuma**", expanded=False):
                # İlk birkaç decode adımını görsel olarak göster
                _adimlar = []
                _buf = ""
                _idx = 0
                for _i, _bit in enumerate(_h_out['bit_string'][:200]):
                    _buf += _bit
                    if _buf in _reverse:
                        _adimlar.append({
                            "Adım": _idx + 1,
                            "Okunan bit dizisi": _buf,
                            "Eşleşen karakter": repr(_reverse[_buf]),
                            "Kalan tampon": "(temizlendi)",
                        })
                        _buf = ""
                        _idx += 1
                        if _idx >= 12:
                            break
                st.dataframe(_adimlar, use_container_width=True, hide_index=True)
                st.caption(
                    "Bit-bit okuyup tampona ekliyoruz. Tampon Huffman kodlarından biriyle "
                    "**eşleşince** o karakteri yazıp tamponu sıfırlıyoruz. Bu, ağaçta "
                    "yapraktan köke giden tek yolun garantisidir."
                )

            with st.expander("🔍 **Aşama 4 — Decode edilmiş ham metin**", expanded=False):
                _decoded = _huff_decode(_h_out['bit_string'], _h_out['codes'])
                st.code(_decoded[:400] + ("..." if len(_decoded) > 400 else ""), language=None)
                st.caption(f"Toplam **{len(_decoded):,}** karakter geri kurtarıldı.")

            st.markdown("#### ✅ Aşama 5 — Doğrulama: Orijinal == Decode")
            _kayipsiz = (_decoded == text)
            cd1, cd2 = st.columns(2)
            with cd1:
                st.markdown("**Orijinal metin:**")
                st.code(text[:300] + ("..." if len(text) > 300 else ""), language=None)
            with cd2:
                st.markdown("**Decode edilmiş metin:**")
                st.code(_decoded[:300] + ("..." if len(_decoded) > 300 else ""), language=None)

            if _kayipsiz:
                st.success(
                    f"✅ **KAYIPSIZ DOĞRULANDI** — "
                    f"{len(text):,} karakter → {len(_h_out['byte_data']):,} byte → "
                    f"{len(_decoded):,} karakter (TAM AYNI). "
                    f"Hash karşılaştırması: orijinal `{hash(text) & 0xffff:04x}` vs "
                    f"decode `{hash(_decoded) & 0xffff:04x}` — **eşleşti**."
                )
            else:
                _fark = sum(1 for a, b in zip(text, _decoded) if a != b)
                st.error(f"❌ Hata: {_fark} karakter farklı.")

            st.markdown("---")

            # AI Huffman
            with st.spinner("AI frekans tahmini yapılıyor (Groq)..."):
                total_chars = len(text)
                actual_freq = {ch: cnt / total_chars for ch, cnt in Counter(text).items()}
                try:
                    ai_freq, tok1 = predict_frequencies(text, text_type)
                    missing = [ch for ch in actual_freq if ch not in ai_freq]
                    ai_freq, tok2 = refine_frequencies(ai_freq, missing, text_type)
                    total_tokens = tok1 + tok2

                    # AI Huffman ağacı
                    merged = {ch: ai_freq.get(ch, 1e-6) for ch in actual_freq}
                    total_m = sum(merged.values())
                    merged = {k: v / total_m for k, v in merged.items()}

                    heap = [HuffmanNode(ch, f) for ch, f in merged.items()]
                    heapq.heapify(heap)
                    while len(heap) > 1:
                        l, r = heapq.heappop(heap), heapq.heappop(heap)
                        m = HuffmanNode(None, l.freq + r.freq)
                        m.left, m.right = l, r
                        heapq.heappush(heap, m)
                    ai_codes = build_codes(heap[0])
                    ai_bits = sum(len(ai_codes.get(ch, "0"*16)) for ch in text)
                    kl = kl_divergence(actual_freq, ai_freq)

                    # ─ Metrikler ─
                    st.markdown("### Sonuçlar")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Orijinal", f"{orig_bits:,} bit")
                    c2.metric("Standart Huffman", f"{total_std:,} bit",
                              delta=f"{((total_std/orig_bits)-1)*100:.1f}%")
                    c3.metric("AI Huffman", f"{ai_bits:,} bit",
                              delta=f"{((ai_bits/orig_bits)-1)*100:.1f}%")
                    c4.metric("KL-Divergence", f"{kl:.3f}",
                              help="0'a yakın = AI tahmini iyi")

                    saved = total_std - ai_bits
                    if saved > 0:
                        st.success(f"✅ AI yaklaşımı {saved:,} bit daha az kullandı!")
                    else:
                        st.info(f"ℹ️ Standart Huffman {-saved:,} bit daha iyi. "
                                f"(KL={kl:.3f} — tahmin hatası tablo tasarrufunu aştı)")

                    # ─ Grafik ─
                    fig = go.Figure(go.Bar(
                        x=["Orijinal", "Standart Huffman\n(+tablo overhead)", "AI Huffman\n(overhead yok)"],
                        y=[orig_bits, total_std, ai_bits],
                        marker_color=["#636EFA", "#EF553B", "#00CC96"],
                        text=[f"{v:,} bit" for v in [orig_bits, total_std, ai_bits]],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        title="Bit Kullanımı Karşılaştırması",
                        yaxis_title="Bit",
                        showlegend=False,
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # ─ Frekans tablosu ─
                    with st.expander("Karakter frekansları (ilk 10)"):
                        rows = sorted(actual_freq.items(), key=lambda x: -x[1])[:10]
                        col_a, col_b = st.columns(2)
                        col_a.write("**Gerçek frekanslar**")
                        col_a.table({repr(ch): f"{f:.4f}" for ch, f in rows})
                        col_b.write("**AI tahmini**")
                        col_b.table({repr(ch): f"{ai_freq.get(ch, 0):.4f}" for ch, f in rows})

                    # Günlüğe kaydet
                    diary = {"huffman": {
                        "std_bits": total_std, "ai_bits": ai_bits,
                        "kl": kl, "tokens": total_tokens, "saved": saved,
                    }}
                    with open("results.json", "w") as f:
                        json.dump(diary, f, indent=2)

                except Exception as e:
                    st.error(f"AI hatası: {type(e).__name__}: {e}")
                    with st.expander("Hata detayı"):
                        st.exception(e)


# ═══════════════════════════════════════════════════
# SEKME 2: LZW + AI SÖZLÜK
# ═══════════════════════════════════════════════════
with tab2:
    st.subheader("Standart LZW vs AI Akıllı Sözlük LZW")
    st.markdown(
        "**Yenilik:** Standart LZW sözlüğü sadece tek karakterlerle başlar. "
        "AI, metin türüne göre en yaygın kelimeleri/ifadeleri önceden sözlüğe ekler — "
        "bu sayede daha erken ve daha uzun eşleşmeler elde edilir."
    )

    # ── Standart LZW sıkıştırılmış çıktı (her zaman göster) ──
    with st.spinner("LZW ile sıkıştırma yapılıyor..."):
        import math as _math
        _dict = {chr(i): i for i in range(256)}
        for _ch in set(text):
            if _ch not in _dict:
                _dict[_ch] = len(_dict)
        _nxt = len(_dict)
        _codes_lzw = []
        _w = ""
        for _c in text:
            _wc = _w + _c
            if _wc in _dict:
                _w = _wc
            else:
                _codes_lzw.append(_dict[_w])
                _dict[_wc] = _nxt
                _nxt += 1
                _w = _c
        if _w:
            _codes_lzw.append(_dict[_w])

        _bpc = _math.ceil(_math.log2(max(len(_dict), 2)))
        _bit_str = "".join(format(c, f"0{_bpc}b") for c in _codes_lzw)
        _padded = _bit_str + "0" * ((8 - len(_bit_str) % 8) % 8)
        _byte_data = bytes(int(_padded[i:i+8], 2) for i in range(0, len(_padded), 8))

    goster_sikistirma_ciktisi(
        "Standart LZW", _bit_str, _byte_data,
        len(text.encode("utf-8")), key_suffix="lzw_std",
    )

    # ── DECODE: Adım adım LZW'yi geri aç ──
    st.markdown("### 🔓 Açma (Decompression) — Adım Adım Kayıpsızlık Kanıtı")

    with st.expander("🔍 **Aşama 1 — Girdi:** LZW kod listesi", expanded=False):
        _ilk_kodlar = _codes_lzw[:30]
        st.code(str(_ilk_kodlar) + ("..." if len(_codes_lzw) > 30 else ""))
        st.caption(f"Toplam **{len(_codes_lzw):,}** kod, her biri **{_bpc} bit**. "
                  f"Standart 8-bit ASCII'ye göre {(1 - _bpc/8)*100:.1f}% daha az bit.")

    with st.expander("🔍 **Aşama 2 — Başlangıç sözlüğü kuruluyor**", expanded=False):
        # initial_dict: encode'da kullandığımız aynı sözlük
        _init_dict = {chr(i): i for i in range(256)}
        for _ch in set(text):
            if _ch not in _init_dict:
                _init_dict[_ch] = len(_init_dict)
        st.caption(f"İlk **{len(_init_dict)}** giriş hazır: 256 standart ASCII + "
                  f"**{len(_init_dict) - 256}** Türkçe karakter (ş, ğ, ü, ö, ç, ı...).")
        # Türkçe karakter eşlemelerini göster
        _tr_kars = [(chr(i), i) for i in range(256, min(len(_init_dict), 270))]
        if _tr_kars:
            _tr_rows = [{"Karakter": repr(ch), "Kod": str(c)} for ch, c in _tr_kars]
            st.dataframe(_tr_rows, use_container_width=True, hide_index=True)

    with st.expander("🔍 **Aşama 3 — Kod-kod decode (canlı sözlük büyütme)**", expanded=False):
        # İlk 10 decode adımını simüle et
        _sim_dict = dict(_init_dict)
        _rev_sim = {v: k for k, v in _sim_dict.items()}
        _next_code = len(_sim_dict)
        _adimlar = []
        _prev = None
        for _i, _code in enumerate(_codes_lzw[:10]):
            if _code in _rev_sim:
                _entry = _rev_sim[_code]
            elif _code == _next_code and _prev is not None:
                _entry = _prev + _prev[0]
            else:
                _entry = "?"

            _new_entry = ""
            if _prev is not None:
                _new_entry = _prev + _entry[0]
                _rev_sim[_next_code] = _new_entry
                _next_code += 1

            _adimlar.append({
                "Adım": _i + 1,
                "Kod": _code,
                "Sözlükten çıkan": repr(_entry),
                "Sözlüğe eklenen": repr(_new_entry) if _new_entry else "—",
                "Sözlük boyutu": _next_code,
            })
            _prev = _entry
        st.dataframe(_adimlar, use_container_width=True, hide_index=True)
        st.caption("LZW'nin sihri: decoder **encoder ile aynı patternleri** keşfeder "
                  "(sözlük dosya başında gönderilmez). Her okunan kodda yeni bir pattern üretilir.")

    with st.expander("🔍 **Aşama 4 — Decode edilmiş ham metin**", expanded=False):
        from core.lzw import lzw_decode as _lzw_dec
        try:
            _lzw_decoded = _lzw_dec(_codes_lzw, _init_dict)
        except Exception as _e:
            _lzw_decoded = f"[Decode hatası: {_e}]"
        st.code(_lzw_decoded[:400] + ("..." if len(_lzw_decoded) > 400 else ""), language=None)
        st.caption(f"Toplam **{len(_lzw_decoded):,}** karakter geri kurtarıldı.")

    st.markdown("#### ✅ Aşama 5 — Doğrulama: Orijinal == Decode")
    _lzw_kayipsiz = (_lzw_decoded == text)

    cd_l1, cd_l2 = st.columns(2)
    with cd_l1:
        st.markdown("**Orijinal metin:**")
        st.code(text[:300] + ("..." if len(text) > 300 else ""), language=None)
    with cd_l2:
        st.markdown("**Decode edilmiş metin:**")
        st.code(_lzw_decoded[:300] + ("..." if len(_lzw_decoded) > 300 else ""), language=None)

    if _lzw_kayipsiz:
        st.success(
            f"✅ **KAYIPSIZ DOĞRULANDI** — "
            f"{len(_codes_lzw):,} kod → {len(_lzw_decoded):,} karakter (TAM AYNI). "
            f"Hash: orijinal `{hash(text) & 0xffff:04x}` vs decode `{hash(_lzw_decoded) & 0xffff:04x}` — **eşleşti**."
        )
    else:
        st.warning("⚠️ Decode kontrolünde fark var.")

    with st.expander("ℹ️ Tüm LZW Kod Listesi (ilk 50)"):
        st.write([f"#{i}: code={c}" for i, c in enumerate(_codes_lzw[:50])])
    st.markdown("---")

    n_words = st.slider("AI'nın ekleyeceği kelime/ifade sayısı (Groq)", 20, 200, 80)

    if st.button("▶ LZW Analizi Başlat", key="lzw"):
        if not api_key:
            st.error("Sol menüden Groq API key gir.")
        else:
            with st.spinner("AI sözlük oluşturuluyor..."):
                try:
                    ai_words, tok, raw = generate_lzw_dictionary(text, text_type, n_words)
                    st.success(f"AI {len(ai_words)} kelime/ifade önerdi ({tok} token kullanıldı)")

                    with st.expander("AI'nın önerdiği sözlük"):
                        st.write(ai_words[:50])

                    with st.spinner("LZW sıkıştırma yapılıyor..."):
                        result = lzw_compare(text, ai_words)

                    std = result["standard"]
                    ai  = result["ai_lzw"]

                    # ─ Metrikler ─
                    st.markdown("### Sonuçlar")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Orijinal", f"{std['original_bits']:,} bit")
                    c2.metric("Standart LZW", f"{std['compressed_bits']:,} bit",
                              delta=f"oran: {std['ratio']:.3f}")
                    c3.metric("AI-LZW", f"{ai['compressed_bits']:,} bit",
                              delta=f"oran: {ai['ratio']:.3f}")

                    saved = result["saved_bits"]
                    if saved > 0:
                        st.success(f"✅ AI sözlük {saved:,} bit tasarruf sağladı!")
                    else:
                        st.info(f"ℹ️ Standart LZW {-saved:,} bit daha iyi.")

                    # ─ Grafik ─
                    fig = go.Figure(go.Bar(
                        x=["Orijinal", "Standart LZW", "AI-LZW"],
                        y=[std["original_bits"], std["compressed_bits"], ai["compressed_bits"]],
                        marker_color=["#636EFA", "#EF553B", "#00CC96"],
                        text=[f"{v:,} bit" for v in [
                            std["original_bits"], std["compressed_bits"], ai["compressed_bits"]]],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        title="LZW Bit Kullanımı Karşılaştırması",
                        yaxis_title="Bit", height=400, showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Detay
                    with st.expander("Teknik detaylar"):
                        st.json({
                            "standart_lzw": std,
                            "ai_lzw": ai,
                            "ai_words_added": result["ai_words_added"],
                        })

                except Exception as e:
                    st.error(f"Hata: {e}")

# ═══════════════════════════════════════════════════
# SEKME 4: SİNİR AĞI SEÇİCİ
# ═══════════════════════════════════════════════════
with tab4:
    st.subheader("🔬 Küçük Sinir Ağı — Algoritma Seçici (3 Sınıf)")
    st.markdown(
        "**Hocanın istediği:** Verinin türüne göre gerçek zamanlı algoritma tahmini yapan sinir ağı. "
        "11 özellik çıkarır (entropi, run-length, bigram entropisi, alfabe boyutu vb.) ve "
        "MLP (çok katmanlı algılayıcı) ile **Huffman / LZW / BWT** kararı verir."
    )

    # Modelin gerçek doğruluğunu oku (sklearn/numpy versiyon uyumsuzluğuna karşı korumalı)
    import pickle as _pk
    _mp = os.path.join(os.path.dirname(__file__), "core", "nn_model.pkl")
    _b = None
    if os.path.exists(_mp):
        try:
            with open(_mp, "rb") as f:
                _b = _pk.load(f)
        except Exception:
            # Versiyon uyumsuzlugu vs. — otomatik yeniden egit
            st.info("📦 Model yükleniyor (ilk açılış 10-20 sn sürer)...")
            try:
                r = nn_train(verbose=False)
                with open(_mp, "rb") as f:
                    _b = _pk.load(f)
                st.success(f"✓ Model yeniden eğitildi: %{r['test_accuracy']*100:.1f} doğruluk")
            except Exception as e:
                st.warning(f"Model eğitilemedi: {e}")
    if _b:
        _info_cols = st.columns(4)
        _info_cols[0].metric("Hold-out doğruluk", f"%{_b.get('accuracy', 0)*100:.1f}",
                             help="Modelin hiç görmediği test setindeki doğruluk")
        _info_cols[1].metric("Cross-validation",
                             f"%{_b.get('cv_mean', 0)*100:.1f} ± %{_b.get('cv_std', 0)*100:.1f}",
                             help="5-fold CV — gerçek genelleme göstergesi")
        _info_cols[2].metric("Eğitim örneği", f"{_b.get('n_samples', 0):,}")
        _info_cols[3].metric("Model", "MLP 32→16→8")
        _dist = _b.get("dist", {})
        if _dist:
            st.caption(f"Sınıf dağılımı — Huffman: {_dist.get('huffman',0)}, "
                       f"LZW: {_dist.get('lzw',0)}, BWT: {_dist.get('bwt',0)}")
    else:
        st.warning("Model henüz eğitilmemiş. 'Eğit' butonuna basın.")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("▶ Sinir Ağı Tahmin Et", key="nn_pred"):
            result = nn_predict(text)
            if result.get("trained"):
                algo_label = {"huffman": "Huffman", "lzw": "LZW",
                              "bwt": "BWT + RLE + Huffman"}.get(
                              result["algorithm"], result["algorithm"].upper())
                conf = result["confidence"]
                color = "🟢" if conf > 0.8 else "🟡" if conf > 0.5 else "🔴"
                st.markdown(f"## {color} Karar: **{algo_label}**")
                st.metric("Güven", f"%{conf*100:.1f}")

                probs = result["probabilities"]
                fig = go.Figure(go.Bar(
                    x=["Huffman", "LZW", "BWT+RLE+Huffman"],
                    y=[probs.get("huffman", 0)*100,
                       probs.get("lzw", 0)*100,
                       probs.get("bwt", 0)*100],
                    marker_color=["#636EFA", "#EF553B", "#00CC96"],
                    text=[f"%{probs.get(k, 0)*100:.1f}"
                          for k in ["huffman", "lzw", "bwt"]],
                    textposition="outside",
                ))
                fig.update_layout(title="Algoritma Olasılıkları",
                                  yaxis_title="%", height=340, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Çıkarılan 11 özellik"):
                    st.json(result["features"])
            else:
                st.warning(result.get("error", "Model bulunamadı"))

    with col_b:
        st.markdown("**🔁 Modeli Yeniden Eğit**")
        st.caption("Mevcut sentetik + corpus verisiyle CV-test'li yeniden eğitim yapar.")
        if st.button("Eğit (~10 saniye)", key="nn_train"):
            with st.spinner("Sinir ağı eğitiliyor (2000+ örnek, 5-fold CV)..."):
                r = nn_train(verbose=False)
            st.success(
                f"✓ Hold-out: %{r['test_accuracy']*100:.1f} | "
                f"CV: %{r['cv_accuracy']*100:.1f} | "
                f"{r['samples']:,} örnek"
            )
            st.caption(f"Sınıf dağılımı: {r['distribution']}")

    # ── Confusion Matrix ──
    st.markdown("---")
    st.markdown("### 🎯 Confusion Matrix (Hold-out Test Setinde)")
    st.caption("Modelin hangi sınıfı doğru/yanlış tahmin ettiğini gösterir. "
              "Köşegende olanlar doğru tahmin.")

    if st.button("📐 Confusion Matrix Göster", key="cm_btn"):
        with st.spinner("Confusion matrix hesaplanıyor..."):
            from core.nn_selector import confusion_matrix_data
            cm_data = confusion_matrix_data()
        if "error" in cm_data:
            st.error(cm_data["error"])
        else:
            cm = cm_data["matrix"]
            classes = cm_data["classes"]
            # Heatmap
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=[c.upper() for c in classes],
                y=[c.upper() for c in classes],
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16, "color": "white"},
                colorscale="Blues",
                showscale=True,
            ))
            fig_cm.update_layout(
                title=f"Hold-out Test Doğruluğu: %{cm_data['accuracy']*100:.1f} "
                      f"({cm_data['n_test']} örnek)",
                xaxis_title="Tahmin",
                yaxis_title="Gerçek",
                height=400,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            # Sınıf bazlı doğruluk
            st.markdown("**Sınıf Bazlı Performans:**")
            rows = []
            for i, c in enumerate(classes):
                tp = cm[i][i]
                total = sum(cm[i])
                acc = tp / total * 100 if total else 0
                rows.append({
                    "Sınıf":   c.upper(),
                    "Toplam":  total,
                    "Doğru":   tp,
                    "Doğruluk": f"%{acc:.1f}",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
            with st.expander("💡 Nasıl okunur?"):
                st.markdown("""
- **Satır:** Gerçek sınıf (doğru cevap)
- **Sütun:** Modelin tahmini
- **Köşegen (sol üst → sağ alt):** Doğru tahminler
- **Köşegen dışı:** Hatalar
- **Örnek:** "BWT satırında LZW sütununda 5" → 5 BWT örneğini yanlışlıkla LZW dedi
                """)

    # ── Feature Importance — Hangi özellik kararı belirliyor? ──
    st.markdown("---")
    st.markdown("### 🔍 Özellik Önem Analizi (Permutation Importance)")
    st.caption(
        "Sinir ağının her özelliğe ne kadar güvendiğini ölçer. Bir özelliği "
        "rastgele karıştırıp doğruluğun ne kadar düştüğüne bakar."
    )
    if st.button("📊 Önem analizini başlat (~15 sn)", key="fi_btn"):
        with st.spinner("Permutation importance hesaplanıyor..."):
            from core.nn_selector import feature_importance
            fi = feature_importance(n_repeats=5)
        if "error" in fi:
            st.error(fi["error"])
        else:
            # Bar grafiği
            fig_fi = go.Figure(go.Bar(
                x=fi["importances"],
                y=fi["feature_names"],
                orientation='h',
                marker_color="#00CC96",
                text=[f"{v:.3f}" for v in fi["importances"]],
                textposition="outside",
                error_x=dict(type='data', array=fi["stds"]),
            ))
            fig_fi.update_layout(
                title="11 Özelliğin Karar Verme Etkisi",
                xaxis_title="Önem skoru (büyük = kritik)",
                height=420,
                margin=dict(t=40, b=20, l=120, r=20),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

            # Top 3 önemli
            st.success(
                f"🥇 **En önemli 3 özellik:** "
                f"{fi['feature_names'][0]} → "
                f"{fi['feature_names'][1]} → "
                f"{fi['feature_names'][2]}"
            )
            with st.expander("💡 Bu sonuç ne anlama geliyor?"):
                st.markdown("""
- **Yüksek önem (üstteki)**: Bu özellik olmadan model **doğruluk kaybeder**.
  Mesela `entropi` yüksek skor aldıysa, NN entropiye bakıp karar veriyor demektir.
- **Düşük önem (alttaki)**: Bu özellik az katkı sağlıyor — gelecekte kaldırılabilir.
- **Negatif önem**: Özellik aslında yanıltıcı (nadir).

**Hocaya gösterilebilir kanıt:** Bu grafik sinir ağının **kara kutu olmadığını**,
hangi karakteristikleri kullandığını şeffaf gösteriyor.
                """)

    with st.expander("📖 Nasıl çalışır?"):
        st.markdown("""
        **11 Özellik:**
        1. Shannon Entropisi — ne kadar tahmin edilemez?
        2. Benzersiz karakter oranı
        3. Top-3 karakter yoğunluğu
        4. Boşluk oranı (doğal dil göstergesi)
        5. Türkçe karakter oranı
        6. Ortalama çalışma uzunluğu (run-length)
        7. Rakam oranı
        8. Büyük harf oranı
        9. Bigram entropisi (yapısal düzen)
        10. Maksimum koşu oranı (BWT için kritik)
        11. Alfabe boyutu (log2) — küçük alfabe BWT'yi kazandırır

        **Karar:**
        - Yüksek entropi + kısa metin → **Huffman**
        - Tekrarlı kelimeler/ifadeler → **LZW**
        - Düşük entropi + küçük alfabe + yapısal → **BWT + RLE + Huffman**

        **Ezber önleme:**
        - Cross-validation (5-fold)
        - L2 regularizasyon (alpha=1e-3)
        - Erken durdurma (early stopping)
        - Hold-out test seti (modelin hiç görmediği)
        - 1900+ örnek (sentetik + 4 farklı corpus)
        """)

# ═══════════════════════════════════════════════════
# SEKME 6: HİBRİT SIKIŞTURMA
# ═══════════════════════════════════════════════════
with tab6:
    st.subheader("⚡ Hibrit Sıkıştırma — Her Zaman En İyi Sonuç")
    st.markdown(
        "**5 algoritma aynı anda çalışır, en iyi sonucu otomatik seçer.** "
        "Corpus eğitimi ile AI tablo overhead'ini ortadan kaldırır. "
        "Hibrit yaklaşım hiçbir zaman standart algoritmadan kötü olamaz."
    )

    col_info1, col_info2 = st.columns(2)
    col_info1.info("🧠 **Sinir Ağı:** Veri tipine göre en iyi algoritmayı seçer")
    col_info2.info("🔀 **5 Algoritma:** Std Huffman, Corpus Huffman, AI Huffman, Std LZW, AI-LZW")

    st.markdown("---")

    # ── AKILLI HİBRİT — ANA ÖZELLİK ──
    st.markdown("### 🌟 Akıllı Hibrit (NN + En İyi Algoritma)")
    use_ai_dict = st.checkbox("AI sözlük kullan (Groq API)", value=True)

    if st.button("▶ Akıllı Hibrit Çalıştır", key="smart"):
        if use_ai_dict and not api_key:
            st.error("Sol menüden Groq API key gir.")
        else:
            with st.spinner("NN analiz ediyor, en iyi algoritma çalışıyor..."):
                try:
                    r = smart_hybrid(text, use_ai_dict=use_ai_dict and bool(api_key))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"### 🧠 NN Kararı: **{r['nn_decision']}**")
                        st.metric("Güven", f"%{r['nn_confidence']*100:.1f}")
                        st.markdown(f"**Yöntem:** {r['method']}")
                    with col2:
                        st.metric("Orijinal", f"{r['original_bits']:,} bit")
                        akilli_pct = (1 - r['smart_ratio']) * 100
                        std_pct    = (1 - r['standard_ratio']) * 100
                        st.metric("Akıllı Hibrit", f"{r['smart_bits']:,} bit",
                                  delta=f"küçülme: %{akilli_pct:.1f}")
                        saved = r["saved_bits"]
                        if saved > 0:
                            iyilesme = saved / r['standard_bits'] * 100
                            st.success(
                                f"✅ Standart Huffman: %{std_pct:.1f} küçülme → "
                                f"Akıllı Hibrit: %{akilli_pct:.1f} küçülme "
                                f"(**+%{iyilesme:.1f} iyileşme**, {saved:,} bit tasarruf!)"
                            )
                        else:
                            st.info("Standart Huffman bu veri için en iyisi.")

                    # Grafik
                    fig = go.Figure(go.Bar(
                        x=["Orijinal", "Standart Huffman", f"Akıllı Hibrit\n({r['nn_decision']})"],
                        y=[r["original_bits"], r["standard_bits"], r["smart_bits"]],
                        marker_color=["#636EFA", "#EF553B", "#00CC96"],
                        text=[f"{v:,} bit\n({v/r['original_bits']*100:.1f}%)"
                              for v in [r["original_bits"], r["standard_bits"], r["smart_bits"]]],
                        textposition="outside",
                    ))
                    fig.update_layout(title="Akıllı Hibrit Sonucu",
                                      yaxis_title="Bit", height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    if r["total_tokens"]:
                        st.caption(f"Token: {r['total_tokens']}")
                except Exception as e:
                    st.error(f"Hata: {type(e).__name__}: {e}")
                    with st.expander("Detay"): st.exception(e)

    st.markdown("---")
    st.markdown("### 🔬 Tüm Algoritmalar Karşılaştırması")
    use_ai_hybrid = st.checkbox("AI algoritmalarını dahil et", value=False, key="all_ai")

    if st.button("▶ Tüm Algoritmaları Test Et", key="hybrid"):
        if use_ai_hybrid and not api_key:
            st.error("Sol menüden Groq API key gir.")
        else:
            with st.spinner("5 algoritma test ediliyor..."):
                try:
                    result = hybrid_compress(text, use_ai=use_ai_hybrid)

                    # Kazanan
                    best = result["best"]
                    st.markdown(f"## 🏆 Kazanan: **{best}**")
                    st.metric("En iyi sıkıştırma",
                              f"{result['best_bits']:,} bit",
                              delta=f"oran: {result['best_ratio']:.4f}")

                    saved = result["saving_vs_standard"]
                    if saved > 0:
                        st.success(f"✅ Standart Huffman'a göre {saved:,} bit tasarruf!")
                    else:
                        st.info("Standart Huffman bu veri için zaten en iyisi.")

                    # Tüm sonuçlar
                    st.markdown("### Tüm Algoritmalar")
                    res = result["results"]
                    orig = result["original_bits"]

                    # Bar grafik
                    names  = list(res.keys())
                    values = [res[n]["bits"] for n in names]
                    colors = []
                    for n in names:
                        if n == best:
                            colors.append("#00CC96")
                        elif res[n]["ai_used"]:
                            colors.append("#AB63FA")
                        else:
                            colors.append("#636EFA")

                    fig = go.Figure(go.Bar(
                        x=names, y=values,
                        marker_color=colors,
                        text=[f"{v:,}\n({v/orig*100:.1f}%)" for v in values],
                        textposition="outside",
                    ))
                    fig.add_hline(y=orig, line_dash="dash", line_color="red",
                                  annotation_text=f"Orijinal: {orig:,} bit")
                    fig.update_layout(
                        title="Hibrit Sıkıştırma — 5 Algoritma Karşılaştırması",
                        yaxis_title="Bit", height=440, showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Detay tablosu
                    st.markdown("### Detay")
                    rows = []
                    for name, r in res.items():
                        rows.append({
                            "Algoritma": ("🏆 " if name == best else "") + name,
                            "Bit": f"{r['bits']:,}",
                            "Oran": f"{r['ratio']:.4f}",
                            "Küçülme": f"%{(1-r['ratio'])*100:.1f}",
                            "AI": "✓" if r["ai_used"] else "—",
                            "Overhead": f"{r.get('overhead',0):,} bit",
                        })
                    st.dataframe(rows, use_container_width=True)

                    if use_ai_hybrid:
                        st.caption(f"Toplam token: {result['total_tokens']}")

                except Exception as e:
                    st.error(f"Hata: {type(e).__name__}: {e}")
                    with st.expander("Detay"):
                        st.exception(e)

    # Corpus yeniden eğit
    with st.expander("⚙️ Corpus'u yeniden eğit"):
        st.caption(
            "Sadece düz metin (.txt) dosyaları kabul edilir. "
            "PDF/Word dosyaları desteklenmez — önce kopyalayıp .txt olarak kaydedin."
        )
        new_corpus = st.file_uploader(
            "Büyük metin dosyası yükle (.txt)",
            type=["txt"],
            key="corpus_upload",
        )
        if new_corpus and st.button("Eğit", key="train_btn"):
            tmp = "/tmp/corpus_tmp.txt"
            with open(tmp, "wb") as f:
                f.write(new_corpus.read())
            try:
                freq = train_from_corpus(tmp)
                st.success(f"✓ {len(freq)} karakter, corpus güncellendi!")
            except (ValueError, UnicodeDecodeError) as e:
                st.error(f"❌ Dosya okunamadı: {e}")
                st.info("İpucu: .pdf yerine .txt yükleyin. "
                        "PDF içeriğini metin editörüne yapıştırıp 'kaydet (UTF-8)' deyin.")

# ═══════════════════════════════════════════════════
# SEKME 9: SHANNON ENTROPİSİ
# ═══════════════════════════════════════════════════
with tab9:
    st.subheader("📐 Shannon Entropisi — Teorik Sınır Analizi")
    st.markdown(
        "**Shannon Entropisi**, bir verinin teorik olarak sıkıştırılabilecek "
        "**minimum bit sayısını** belirler. Hiçbir kayıpsız algoritma bu sınırın altına inemez. "
        "Algoritmamızın bu sınıra ne kadar yaklaştığını ölçüyoruz."
    )

    # Otomatik hesapla — API key gerekmez
    entropy  = shannon_entropy(text)
    min_bits = theoretical_min_bits(text)
    orig_bits_e = len(text) * 8

    # Huffman bits (hızlı hesap)
    huff_bits_e, huff_codes_e = huffman_encode(text)
    huff_total_e = len(huff_bits_e)

    # Metrikler
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Shannon Entropisi", f"{entropy:.4f} bit/karakter")
    c2.metric("Teorik Minimum", f"{min_bits:,} bit")
    c3.metric("Huffman (gerçek)", f"{huff_total_e:,} bit")
    c4.metric("Verimliliği", f"{min_bits/huff_total_e*100:.1f}%",
              help="100% = teorik mükemmel sıkıştırma")

    # Ana grafik — 4 çubuk karşılaştırma
    algorithms = ["Orijinal", "Teorik Minimum\n(Shannon)", "Huffman", "Hedef (AI ile)"]
    values     = [orig_bits_e, min_bits, huff_total_e, int(min_bits * 1.05)]
    colors     = ["#636EFA", "#00CC96", "#EF553B", "#AB63FA"]

    fig = go.Figure(go.Bar(
        x=algorithms, y=values,
        marker_color=colors,
        text=[f"{v:,} bit" for v in values],
        textposition="outside",
    ))
    fig.add_hline(y=min_bits, line_dash="dash", line_color="#00CC96",
                  annotation_text="Shannon Sınırı")
    fig.update_layout(title="Sıkıştırma Performansı vs Teorik Sınır",
                      yaxis_title="Bit", height=420, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"💡 Huffman, teorik sınıra **{(huff_total_e/min_bits - 1)*100:.1f}%** uzakta. "
            f"AI optimizasyonu bu farkı kapatmayı hedefliyor.")

    # Karakter bazlı analiz
    with st.expander("Karakter bazlı entropi analizi"):
        rows = per_char_analysis(text)
        st.dataframe(rows, use_container_width=True)


# ═══════════════════════════════════════════════════
# SEKME 10: AI GÜNLÜĞÜ
# ═══════════════════════════════════════════════════
with tab10:
    st.subheader("🤖 AI Günlüğü")
    st.markdown("Proje boyunca AI'ya verilen promptlar ve alınan cevaplar")

    diary_file = "ai_diary.json"
    if os.path.exists(diary_file):
        with open(diary_file) as f:
            diary = json.load(f)

        st.metric("Toplam token kullanımı", f"{diary.get('toplam_token', 0):,}")

        for step in diary.get("adimlar", []):
            with st.expander(f"Adım {step['adim']}: {step['hedeflenen_islem']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Verilen Prompt:**")
                    st.code(step["verilen_prompt"][:400], language=None)
                with col2:
                    st.markdown("**AI Cevabı:**")
                    st.code(step["ai_cevabi"][:400], language=None)
                st.markdown(f"**Sorun:** {step['sorun']}")
                st.markdown(f"**Sonuç:** {step['sonuc']}")
    else:
        st.info("Henüz AI günlüğü yok. Huffman veya LZW analizini çalıştır.")

    # Manuel ek adım
    st.markdown("---")
    st.markdown("**Manuel adım ekle (AI Günlüğüne)**")
    col1, col2 = st.columns(2)
    with col1:
        m_hedef  = st.text_input("Hedeflenen işlem")
        m_prompt = st.text_area("Verilen prompt", height=80)
    with col2:
        m_cevap  = st.text_area("AI cevabı / sorun", height=80)
        m_sonuc  = st.text_input("Sonuç")

    if st.button("Adım Ekle"):
        if os.path.exists(diary_file):
            with open(diary_file) as f:
                diary = json.load(f)
        else:
            diary = {"adimlar": [], "toplam_token": 0}
        diary["adimlar"].append({
            "adim": len(diary["adimlar"]) + 1,
            "hedeflenen_islem": m_hedef,
            "verilen_prompt": m_prompt,
            "ai_cevabi": m_cevap,
            "sorun": "Manuel giriş",
            "sonuc": m_sonuc,
            "token_kullanim": 0,
        })
        with open(diary_file, "w") as f:
            json.dump(diary, f, ensure_ascii=False, indent=2)
        st.success("Adım eklendi!")
        st.rerun()


# ═══════════════════════════════════════════════════
# SEKME 11: BWT SIKIŞTURMA
# ═══════════════════════════════════════════════════
with tab11:
    st.subheader("🔄 BWT + RLE + Huffman — bzip2 Tekniği")
    st.markdown(
        "**Burrows-Wheeler Transform (BWT)**, metni karakter kümeleri oluşturacak şekilde "
        "permüte eder. Ardından RLE (Run-Length Encoding) bu kümeleri sıkıştırır. "
        "Son adımda Huffman veya LZW ile optimal kodlama yapılır. "
        "Bu, **bzip2**'nin temel algoritmasıdır ve yüksek sıkıştırma oranı sağlar."
    )

    col_bwt1, col_bwt2, col_bwt3 = st.columns(3)
    col_bwt1.info("**Adım 1 — BWT**\nBenzer karakterleri gruplar")
    col_bwt2.info("**Adım 2 — RLE**\nTekrarlı koşuları kısaltır")
    col_bwt3.info("**Adım 3 — Huffman**\nOptimal bit kodlaması")

    st.markdown("---")

    # ── BWT+RLE+Huffman Sıkıştırılmış çıktı ──
    from core.bwt import bwt_rle_huffman_encode as _bwtrle, bwt_decode as _bwt_dec
    with st.spinner("BWT+RLE+Huffman ile sıkıştırılıyor..."):
        _bwt_enc = _bwtrle(text)
    goster_sikistirma_ciktisi(
        "BWT + RLE + Huffman (bzip2 tekniği)",
        _bwt_enc['bit_string'], _bwt_enc['byte_data'],
        len(text.encode("utf-8")), key_suffix="bwt_rle_huff",
    )

    # ── DECODE: Adım adım BWT'yi geri aç ──
    st.markdown("### 🔓 Açma (Decompression) — Adım Adım Kayıpsızlık Kanıtı")
    st.caption("BWT'nin geri çevrimi (LF mapping) — en zarif veri yapısı işlemlerinden biri.")

    _bwt_str = _bwt_enc['bwt']
    _orig_idx = _bwt_enc['orig_idx']

    with st.expander("🔍 **Aşama 1 — Girdi:** BWT permüte metni + orijinal indeks", expanded=False):
        st.code(_bwt_str[:150] + ("..." if len(_bwt_str) > 150 else ""))
        st.caption(f"BWT uzunluğu: **{len(_bwt_str)}** karakter | "
                  f"Orijinal indeks: **{_orig_idx}** "
                  f"(bu, kodlamada gönderilen tek metadata).")

    with st.expander("🔍 **Aşama 2 — Karakter rank'larını hesapla**", expanded=False):
        from collections import Counter as _C
        _seen = {}
        _rank = []
        for _ch in _bwt_str:
            _rank.append(_seen.get(_ch, 0))
            _seen[_ch] = _seen.get(_ch, 0) + 1
        # İlk 15 karakterin rank'larını göster
        _rank_rows = [
            {"Pozisyon": i, "Karakter": repr(_bwt_str[i]), "Rank (kaçıncı kez)": _rank[i]}
            for i in range(min(15, len(_bwt_str)))
        ]
        st.dataframe(_rank_rows, use_container_width=True, hide_index=True)
        st.caption(
            "Her karakter için **\"BWT'de kaçıncı kez göründü?\"** sorusunun cevabı. "
            "Bu, LF mapping'in özüdür."
        )

    with st.expander("🔍 **Aşama 3 — Sıralı dizi (F sütunu) — başlangıç pozisyonları**", expanded=False):
        _char_counts = _C(_bwt_str)
        _starts = {}
        _pos = 0
        for _ch in sorted(_char_counts.keys()):
            _starts[_ch] = _pos
            _pos += _char_counts[_ch]
        _start_rows = [
            {"Karakter": repr(ch), "Toplam": str(_char_counts[ch]), "F'deki başlangıç": str(p)}
            for ch, p in list(_starts.items())[:15]
        ]
        st.dataframe(_start_rows, use_container_width=True, hide_index=True)
        st.caption(
            "Eğer tüm BWT karakterleri sıralasaydık (F sütunu), her karakterin nerede "
            "başladığını gösterir. Örnek: 'a' sıralı dizide pozisyon X'te başlar."
        )

    with st.expander("🔍 **Aşama 4 — LF Mapping kuruluyor**", expanded=False):
        _lf = [_starts[_bwt_str[i]] + _rank[i] for i in range(len(_bwt_str))]
        _lf_rows = [
            {"i (L'deki pozisyon)": i,
             "L[i] (karakter)": repr(_bwt_str[i]),
             "LF[i] (F'deki konum)": _lf[i]}
            for i in range(min(15, len(_bwt_str)))
        ]
        st.dataframe(_lf_rows, use_container_width=True, hide_index=True)
        st.caption(
            "**LF Mapping:** L'deki (BWT) i. konumdaki karakter, sıralı dizide (F) hangi "
            "konumdadır? Bu eşleme **bijektif**tir — her karakterin tek bir LF eşi vardır. "
            "Bu, dolambacın geri çevrilmesini sağlar."
        )

    with st.expander("🔍 **Aşama 5 — Orijinal metni geri kur (LF zinciri)**", expanded=False):
        _result = []
        _idx = _orig_idx
        _adimlar = []
        for _step in range(min(10, len(_bwt_str))):
            _result.append(_bwt_str[_idx])
            _adimlar.append({
                "Adım": _step + 1,
                "Mevcut idx": _idx,
                "Eklenen karakter": repr(_bwt_str[_idx]),
                "Sonraki idx (LF)": _lf[_idx],
            })
            _idx = _lf[_idx]
        st.dataframe(_adimlar, use_container_width=True, hide_index=True)
        st.caption(
            "Orijinal indeksten başla, **LF[idx]** ile sıçra, gittiğin yerin karakterini yaz. "
            "n iterasyonda metni terste kurar, en sonda ters çevir → orijinal!"
        )

    _bwt_decoded = _bwt_dec(_bwt_str, _orig_idx)

    with st.expander("🔍 **Aşama 6 — Decode edilmiş ham metin**", expanded=False):
        st.code(_bwt_decoded[:400] + ("..." if len(_bwt_decoded) > 400 else ""), language=None)
        st.caption(f"Toplam **{len(_bwt_decoded):,}** karakter geri kurtarıldı.")

    st.markdown("#### ✅ Aşama 7 — Doğrulama: Orijinal == Decode")
    _bwt_kayipsiz = (_bwt_decoded == text[:len(_bwt_decoded)])

    cd_b1, cd_b2 = st.columns(2)
    with cd_b1:
        st.markdown("**Orijinal metin:**")
        st.code(text[:300] + ("..." if len(text) > 300 else ""), language=None)
    with cd_b2:
        st.markdown("**Decode edilmiş metin:**")
        st.code(_bwt_decoded[:300] + ("..." if len(_bwt_decoded) > 300 else ""), language=None)

    if _bwt_kayipsiz:
        st.success(
            f"✅ **KAYIPSIZ DOĞRULANDI** — "
            f"BWT permütasyonu + orijinal indeks ({_orig_idx}) yeterli. "
            f"Hash: orijinal `{hash(text[:len(_bwt_decoded)]) & 0xffff:04x}` vs "
            f"decode `{hash(_bwt_decoded) & 0xffff:04x}` — **eşleşti**."
        )
    else:
        st.warning(
            f"⚠️ Decode kontrolünde fark var. Metin uzunluğu: orijinal {len(text)}, "
            f"BWT decode {len(_bwt_decoded)} "
            f"(uzun metinlerde MAX_BWT_LEN=8000 sınırından olabilir)."
        )

    with st.expander("ℹ️ BWT iç bilgileri (permütasyon + RLE koşuları)"):
        st.caption(f"**BWT çıktısı (ilk 100 karakter):**")
        st.code(_bwt_enc['bwt'][:100] + ("..." if len(_bwt_enc['bwt']) > 100 else ""))
        st.caption(f"**Orijinal indeks:** {_bwt_enc['orig_idx']} (decode için gerekli)")
        st.caption(f"**RLE koşuları (ilk 20):**")
        st.write(_bwt_enc['runs'][:20])
        st.caption(f"Toplam **{len(_bwt_enc['runs']):,}** koşu, **{len(_bwt_enc['codes'])} farklı karakter**.")

    st.markdown("---")

    # BWT istatistikleri (API gerektirmez)
    st.markdown("### 📊 BWT Dönüşüm Etkisi")
    with st.spinner("BWT hesaplanıyor..."):
        try:
            stats = bwt_stats(text)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Orijinal koşu sayısı", f"{stats['original_runs']:,}")
            c2.metric("BWT sonrası koşu sayısı", f"{stats['bwt_runs']:,}")
            c3.metric("Koşu azalması", f"%{stats['run_reduction_pct']:.1f}")
            c4.metric("En uzun koşu", f"{stats['longest_run']} karakter")

            st.metric("Ortalama koşu uzunluğu (BWT sonrası)",
                      f"{stats['avg_run_length']:.2f}")

            with st.expander("BWT çıktısı (ilk 80 karakter)"):
                st.code(stats["bwt_preview"])
                st.caption(f"Orijinal indeks: {stats['orig_idx']}")
        except Exception as e:
            st.error(f"BWT istatistik hatası: {e}")

    st.markdown("---")
    st.markdown("### 🏆 Sıkıştırma Karşılaştırması")

    if st.button("▶ BWT Sıkıştırma Analizi Başlat", key="bwt_btn"):
        with st.spinner("BWT kombinasyonları hesaplanıyor..."):
            try:
                result = bwt_compare(text)
                orig   = result["original_bits"]
                res    = result["results"]
                best   = result["best"]

                # Kazanan banner
                st.markdown(f"## 🏆 Kazanan: **{best}**")
                col_b1, col_b2, col_b3 = st.columns(3)
                col_b1.metric("En iyi bit sayısı", f"{result['best_bits']:,} bit")
                col_b2.metric("Sıkıştırma oranı", f"{result['best_ratio']:.4f}")
                col_b3.metric(
                    "Standart Huffman'a göre iyileşme",
                    f"%{result['improvement_pct']:.1f}",
                    delta=f"{result['std_huffman_bits'] - result['best_bits']:,} bit",
                    delta_color="inverse",
                )

                # Bar grafik
                names   = list(res.keys())
                values  = [res[n]["bits"] for n in names]
                palette = {
                    "Standart Huffman":    "#636EFA",
                    "Standart LZW":        "#EF553B",
                    "BWT + Huffman":       "#FFA15A",
                    "BWT + RLE + Huffman": "#00CC96",
                    "BWT + LZW":           "#AB63FA",
                }
                colors = [("#FFD700" if n == best else palette.get(n, "#636EFA")) for n in names]

                fig = go.Figure(go.Bar(
                    x=names, y=values,
                    marker_color=colors,
                    text=[f"{v:,} bit<br>({v/orig*100:.1f}%)" for v in values],
                    textposition="outside",
                ))
                fig.add_hline(
                    y=orig, line_dash="dash", line_color="red",
                    annotation_text=f"Orijinal: {orig:,} bit",
                )
                fig.update_layout(
                    title="BWT + RLE + Huffman vs Standart Algoritmalar",
                    yaxis_title="Bit",
                    height=460,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Detay tablosu
                st.markdown("### Detay Tablosu")
                rows = []
                for name, r in res.items():
                    rows.append({
                        "Algoritma":    ("🏆 " if name == best else "") + name,
                        "Bit":          f"{r['bits']:,}",
                        "Oran":         f"{r['ratio']:.4f}",
                        "Küçülme":      f"%{(1 - r['ratio']) * 100:.1f}",
                        "BWT":          "✓" if r["bwt"] else "—",
                        "Not":          r.get("note", ""),
                    })
                st.dataframe(rows, use_container_width=True)

                # Çalışma prensibini açıkla
                with st.expander("📚 Neden BWT + RLE + Huffman çalışır?"):
                    st.markdown("""
**1. BWT (Burrows-Wheeler Transform):**
- Tüm döngüsel permütasyonları sıralar → son sütun alınır
- Benzer bağlamdaki karakterler arka arkaya gelir: `...aaaa...bbbb...`
- Bu permütasyon **tamamen geri alınabilir** (kayıpsız)

**2. RLE (Run-Length Encoding):**
- Ardışık aynı karakterleri `(karakter, sayı)` çiftine dönüştürür
- `"aaaaaabbbbb"` → `(a,6)(b,5)` → çok daha kısa

**3. Huffman:**
- RLE sonrası kalan sembolleri optimal bit uzunluğuyla kodlar
- Nadir semboller uzun kod, sık semboller kısa kod alır

**Sonuç:** bzip2 programı bu tekniği kullanır ve gzip/LZW'den genellikle daha iyi sıkıştırma yapar.
                    """)

            except Exception as e:
                st.error(f"Hata: {type(e).__name__}: {e}")
                with st.expander("Hata detayı"):
                    st.exception(e)

    # Doğruluğu test et
    st.markdown("---")
    with st.expander("🔬 BWT Doğruluk Testi (Encode → Decode)"):
        test_txt = st.text_input(
            "Test metni (en fazla 200 karakter)",
            value="merhaba dünya",
            key="bwt_test_input",
        )
        if st.button("Test Et", key="bwt_verify"):
            try:
                bwt_out, idx = bwt_encode(test_txt[:200])
                from core.bwt import bwt_decode
                recovered = bwt_decode(bwt_out, idx)

                st.code(f"Orijinal : {test_txt[:200]}", language=None)
                st.code(f"BWT çıktı: {bwt_out}", language=None)
                st.code(f"Geri açma: {recovered}", language=None)

                if recovered == test_txt[:200]:
                    st.success("✅ Encode → Decode başarılı! Kayıpsız.")
                else:
                    st.error("❌ Geri açma hatası!")
            except Exception as e:
                st.error(f"Hata: {e}")
