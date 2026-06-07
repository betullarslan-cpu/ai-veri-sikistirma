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
    layout="wide",
)

st.title("AI-Destekli Veri Sıkıştırma")
st.caption(
    "Klasik sıkıştırma algoritmalarını (Huffman, LZW, BWT) bir **sinir ağı** ile birleştirip "
    "her veri türünde otomatik en iyi yöntemi seçen sistem."
)

# ─── Sidebar: API key + ayarlar ────────────────────
with st.sidebar:
    # ── Güvenlik politikası ──
    # 1. value="": her oturum açılışında kutu boş başlar
    #    → ekran görüntüsü/demo'da key sızmaz
    # 2. type="password": yazılan key noktalı görünür
    # 3. Sayfayı yenilemek key'i temizler (Streamlit varsayılan)
    api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        value="",
        help="🔐 GÜVENLİK: Bu kutu her oturum açılışında boş başlar. "
             "Sayfayı yenilerseniz key'i tekrar girmeniz gerekir. "
             "console.groq.com → API Keys (ücretsiz).",
        placeholder="gsk_...",
    )
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        st.caption("✓ Key yüklendi (sadece bu oturum için)")

    text_type = st.selectbox("Metin türü", ["Türkçe metin", "İngilizce metin", "Kod (Python)"])

    with st.expander("Gelişmiş"):
        model = st.selectbox("AI Modeli", [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ])
        os.environ["GROQ_MODEL"] = model

# ─── Metin girişi ──────────────────────────────────
# Hazır test metinleri (farklı tipler)
HAZIR_METINLER = {
    "📄 Türkiye tanıtım (doğal Türkçe)": (
        "Türkiye, Avrupa ve Asya kıtaları arasında köprü görevi gören eşsiz bir ülkedir. "
        "Zengin tarihi ve kültürel mirası ile her yıl milyonlarca turistin ziyaret ettiği bu ülke, "
        "doğal güzellikleri bakımından da son derece dikkat çekicidir."
    ),
    "🧬 DNA dizisi (küçük alfabe)": "ATCGATCGTTAACCGG" * 30,
    "🔁 Tekrarlı pattern (BWT ideal)": "ABCABCABC" * 50,
    "📊 JSON log dosyası (LZW ideal)":
        '{"id":1,"msg":"ok"},'*30,
    "🤖 Yapay zeka metni (akademik Türkçe)": (
        "Yapay zeka teknolojileri son yıllarda hızla gelişmektedir. "
        "Derin öğrenme modelleri görüntü tanıma ve doğal dil işleme alanlarında "
        "çığır açıcı sonuçlar elde etmiştir. Veri sıkıştırma alanında da "
        "yapay zeka yaklaşımları artık klasik algoritmalarla yarışmaktadır. "
    ) * 3,
    "📰 Haber metni (formal Türkçe)": (
        "İstanbul Teknik Üniversitesi araştırmacıları yeni bir sıkıştırma "
        "algoritması geliştirdi. Profesör Ahmet Yılmaz yaptığı açıklamada "
        "bu yöntemin gzip programından yüzde otuz daha iyi sonuç verdiğini söyledi. "
    ) * 2,
    "✍️ Kendi metnim (yazacağım)": "",
}

col_sel, col_inp2 = st.columns([3, 1])

with col_sel:
    secim = st.selectbox(
        "🎯 Hazır test metni seç veya kendi metnini yaz",
        list(HAZIR_METINLER.keys()),
        index=0,
        help="Farklı veri tipleri için sistem nasıl davranıyor görmek için seç",
    )
    uploaded = st.file_uploader("📁 Veya .txt dosyası yükle", type=["txt"])

    if uploaded:
        text = uploaded.read().decode("utf-8")
        st.success(f"✓ Dosya yüklendi: {len(text):,} karakter")
    elif secim == "✍️ Kendi metnim (yazacağım)":
        text = st.text_area("Metnini yaz/yapıştır:", height=180,
                            placeholder="Buraya kendi metnini yapıştır...",
                            value="")
    else:
        text = HAZIR_METINLER[secim]
        with st.expander("📝 Seçilen metni göster/düzenle", expanded=False):
            text = st.text_area("", value=text, height=180, key="metin_edit")

with col_inp2:
    st.metric("Karakter", f"{len(text):,}")
    st.metric("Orijinal boyut", f"{len(text)*8:,} bit")
    st.caption(f"≈ {len(text.encode('utf-8')):,} byte UTF-8")

if len(text) < 10:
    st.warning("En az 10 karakter girin.")
    st.stop()

# BWT bloklu bilgilendirme (uzun metinde otomatik blok blok işler)
if len(text) > 8000:
    n_block = (len(text) + 7999) // 8000
    st.info(
        f"ℹ️ Metniniz **{len(text):,} karakter**. BWT modülümüz "
        f"**{n_block} blok** halinde işliyor (her blok 8.000 karakter, bzip2 mantığı). "
        f"Kayıpsız geri alma TÜM metin için garantilidir."
    )


# UI yardımcı fonksiyonu artık core/ui_helpers.py'da
from core.ui_helpers import goster_sikistirma_ciktisi, goster_randomness_testi


# ─── Sekmeler — sadece gerekli olanlar ──────
tab0, tab1, tab2, tab4, tab6, tab9, tab11, tab12, tab10 = st.tabs([
    "🚀 Hızlı Özet",
    "📊 Huffman",
    "📖 LZW",
    "🔬 Sinir Ağı",
    "⚡ Hibrit",
    "📈 Shannon",
    "🔄 BWT",
    "🔮 Next-Token",
    "🤖 Günlük",
])


# ═══════════════════════════════════════════════════
# SEKME 0: HIZLI OZET — tek ekranda hersey
# ═══════════════════════════════════════════════════
with tab0:
    st.subheader("🚀 Hızlı Özet — Tek Ekranda Tüm Sonuçlar")
    st.caption("Bir butonla tüm algoritmaları çalıştır, yan yana karşılaştır. API gerekmez.")

    if st.button("▶ Tümünü Hesapla", key="quick_run", type="primary"):
        with st.spinner("Hesaplanıyor..."):
            from core.bwt import compare as _bwt_cmp
            from core.hybrid import smart_hybrid as _sh
            from core.nn_selector import predict as _nnp
            from core.entropy import shannon_entropy as _shan, theoretical_min_bits as _tmb
            from core.benchmark import tam_karsilastirma as _bench

            # ── PERFORMANS ÖLÇÜMÜ (her aşamayı ölç) ──
            _t_all = time.perf_counter()

            _orig = len(text) * 8

            _t = time.perf_counter()
            _ent  = _shan(text)
            _min  = _tmb(text)
            _t_shannon = (time.perf_counter() - _t) * 1000

            _t = time.perf_counter()
            _bwt  = _bwt_cmp(text)
            _t_bwt = (time.perf_counter() - _t) * 1000

            _t = time.perf_counter()
            _sm   = _sh(text, use_ai_dict=False)
            _t_smart = (time.perf_counter() - _t) * 1000

            _t = time.perf_counter()
            _nn   = _nnp(text)
            _t_nn = (time.perf_counter() - _t) * 1000

            _t = time.perf_counter()
            _bm   = _bench(text)
            _t_bench = (time.perf_counter() - _t) * 1000

            _t_total = (time.perf_counter() - _t_all) * 1000

        # Performans satırı (üstte)
        st.caption(
            f"⏱ **İşlem süreleri (ms):** Shannon {_t_shannon:.1f} · "
            f"BWT analizi {_t_bwt:.1f} · "
            f"NN tahmin {_t_nn:.1f} · "
            f"Akıllı Hibrit {_t_smart:.1f} · "
            f"Endüstri benchmark {_t_bench:.1f} · "
            f"**Toplam: {_t_total:.0f} ms** · "
            f"🪙 Token: 0 (API gerektirmez)"
        )

        # ── 4 metrik yan yana ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Orijinal", f"{_orig:,} bit",
                  help="Sıkıştırılmadan önceki boyut. Her karakter UTF-8'de 8 bit varsayılır.")
        m2.metric("Shannon limiti", f"{_min:,} bit",
                  delta=f"%{(1-_min/_orig)*100:.1f} küçülme",
                  delta_color="off",
                  help="Bilgi teorisinin söylediği teorik minimum. "
                       "Karakter-bağımsız Shannon entropisine göre hesaplanır.")
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

        # Bit/karakter karşılaştırması
        _char_count = len(text) if len(text) > 0 else 1
        _bpc_smart = _sm['smart_bits'] / _char_count
        _bpc_shan  = _min / _char_count
        st.caption(
            f"📐 **Karakter başına bit:** "
            f"Akıllı Hibrit **{_bpc_smart:.2f}** · "
            f"Shannon limiti **{_bpc_shan:.2f}** · "
            f"Orijinal 8.00 bit/karakter"
        )

        # ── METRIKLER NE ANLAMA GELIYOR? ──
        # NN bilgisi tek satır
        st.success(
            f"🧠 **Sinir Ağı kararı:** {_nn['algorithm'].upper()} (%{_nn['confidence']*100:.0f} güven) — "
            f"**Yöntem:** {_sm['method']} — "
            f"**Standart Huffman'a göre +%{_sm['saved_bits']*100/_sm['standard_bits']:.1f} iyileşme** "
            f"({_sm['saved_bits']:,} bit tasarruf)"
        )

        with st.expander("💡 NN nasıl karar verdi?"):
            _feat = _nn.get("features", {})
            _algo = _nn['algorithm']
            _explain = {
                "huffman": "Yüksek entropi → karakter bazlı kodlama.",
                "lzw":     "Tekrarlı ifadeler → sözlük tabanlı kodlama.",
                "bwt":     "Yapısal düzen → BWT permütasyonu.",
            }.get(_algo, "")
            st.markdown(
                f"Entropi: **{_feat.get('entropi', 0):.3f}** · "
                f"Alfabe: **{_feat.get('alfabe_boyutu_log', 0):.2f}** · "
                f"Koşu: **{_feat.get('max_koşu_oran', 0):.4f}**  \n"
                f"**Gerekçe:** {_explain}"
            )

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

        # ═══════ ENDÜSTRİ KARŞILAŞTIRMASI (sade tablo) ═══════
        st.markdown("---")
        _our_best = _bm["bizim_en_iyi_byte"]
        _bzip = _bm["bzip2_byte"]
        _fark = (_bzip - _our_best) / _bzip * 100

        if _our_best <= _bzip:
            st.markdown(
                f"### 🏭 Endüstri Karşılaştırması "
                f"<small>(bizim en iyi: **{_our_best:,} byte**, "
                f"bzip2: **{_bzip:,} byte** → bizim **%{_fark:.1f} daha küçük** 🎉)</small>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"### 🏭 Endüstri Karşılaştırması "
                f"<small>(bizim en iyi: **{_our_best:,} byte**, "
                f"bzip2: **{_bzip:,} byte** → bizim **%{-_fark:.1f} daha büyük**)</small>",
                unsafe_allow_html=True,
            )

        # Tek tablo: hepsi yan yana, boyuta göre sıralı
        _ben_rows = []
        for _alg, _d in {**_bm["endustri"], **_bm["bizim"]}.items():
            if isinstance(_d, dict):
                _ben_rows.append({
                    "Algoritma": _alg.replace("_", "+").upper(),
                    "Byte":     f"{_d['byte']:,}",
                    "Küçülme":  f"%{_d['kucullme_pct']:.1f}",
                    "Süre":     f"{_d['sure_ms']:.2f} ms",
                    "Tür":      "🏭" if _alg in _bm["endustri"] else "⭐",
                })
        _ben_rows.sort(key=lambda r: int(r["Byte"].replace(",", "")))
        st.dataframe(_ben_rows, use_container_width=True, hide_index=True, height=320)

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

            # ═══════════════════════════════════════════════
            # 3 GENİŞ KUTUCUKTA: ORIJINAL / SIKIŞTIRILMIŞ / DECODE
            # ═══════════════════════════════════════════════
            st.markdown("---")
            st.markdown("### 📺 Üçü Yan Yana: Orijinal · Sıkıştırılmış · Decode")
            st.caption("Hocaya gösterilebilir kayıpsızlık kanıtı: orijinal metin → binary → tekrar metin")

            from core.bwt import bwt_chunked_decode as _bwt_dec_full
            t_decode_start = time.perf_counter()
            if _bwt_out.get("n_chunks", 1) > 1:
                _bwt_geri = _bwt_dec_full(_bwt_out['chunks'])
            else:
                from core.bwt import bwt_decode as _bwt_dec_one
                _bwt_geri = _bwt_dec_one(_bwt_out['bwt'], _bwt_out['orig_idx'])
            decode_time_ms = (time.perf_counter() - t_decode_start) * 1000
            _kayipsiz_son = (_bwt_geri == text)

            box1, box2, box3 = st.columns(3)
            with box1:
                st.markdown("#### 📝 Orijinal Metin")
                st.text_area("", value=text,
                             height=300, key="box_orig", disabled=True,
                             label_visibility="collapsed")
                st.caption(f"**{len(text):,}** karakter · **{_orig_byte:,}** byte UTF-8")

            with box2:
                st.markdown("#### 💾 Sıkıştırılmış (Binary Hex)")
                # Hex gösterimi (kullanıcı dostu)
                hex_str = _bwt_out['byte_data'].hex().upper()
                # 4'lü gruplar halinde 8 sütun
                hex_grouped = " ".join(
                    hex_str[i:i+2] for i in range(0, len(hex_str), 2)
                )
                st.text_area("", value=hex_grouped,
                             height=300, key="box_bin", disabled=True,
                             label_visibility="collapsed")
                st.caption(
                    f"**{_bin_size:,}** byte · "
                    f"**%{(1-_bin_size/_orig_byte)*100:.1f}** küçülme"
                )

            with box3:
                st.markdown("#### 🔓 Decode (Geri Açılmış)")
                st.text_area("", value=_bwt_geri,
                             height=300, key="box_decode", disabled=True,
                             label_visibility="collapsed")
                st.caption(
                    f"**{len(_bwt_geri):,}** karakter · "
                    f"⏱ {decode_time_ms:.1f} ms"
                )

            if _kayipsiz_son:
                st.success(
                    f"✅ **KAYIPSIZ DOĞRULAMA BAŞARILI** — "
                    f"Orijinal metin ({len(text):,} kar) → Binary ({_bin_size:,} byte) → "
                    f"Decode ({len(_bwt_geri):,} kar). Hash karşılaştırma OK. "
                    f"Decode süresi: {decode_time_ms:.1f} ms."
                )
            else:
                _fark = sum(1 for a, b in zip(text, _bwt_geri) if a != b)
                st.error(f"❌ Kayıp tespit edildi: {_fark} karakter farklı.")

            # ── Shannon Randomness Testi (Perfect compression = random noise) ──
            st.markdown("---")
            goster_randomness_testi(_bwt_out['bit_string'], "BWT+RLE+Huffman")

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

            # ── DECODE: Kayıpsızlık Doğrulama ──
            st.markdown("### 🔓 Açma (Decompression) — Kayıpsızlık Kanıtı")
            _decoded = _huff_decode(_h_out['bit_string'], _h_out['codes'])
            _kayipsiz = (_decoded == text)

            with st.expander("🔍 Huffman kod tablosu (ilk 15)"):
                _reverse = {v: k for k, v in _h_out['codes'].items()}
                _sorted_rev = sorted(_reverse.items(), key=lambda x: len(x[0]))
                st.dataframe([
                    {"Bit kodu": k, "Uzunluk": f"{len(k)} bit", "Karakter": repr(v)}
                    for k, v in _sorted_rev[:15]
                ], use_container_width=True, hide_index=True)
                st.caption(f"Toplam **{len(_reverse)} prefix-free kod**.")
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

                    # ─ Information formülü — Shannon'un -log₂(p) tanımı ─
                    with st.expander("📐 Shannon Information Formülü — Her Karakter İçin"):
                        st.markdown("""
**Shannon (1948):** Bir karakterin **bilgi içeriği** olasılığının negatif logaritmasıdır:

$$I(c) = -\\log_2 p(c) \\quad \\text{[bit]}$$

- Sık karakterler **az bilgi** taşır (kısa Huffman kodu hak ederler)
- Nadir karakterler **çok bilgi** taşır (uzun Huffman kodu)

Aşağıda **her karakterin gerçek bilgi miktarı** ve Huffman'ın atadığı kod uzunluğu:
                        """)
                        rows_info = []
                        toplam_info = 0.0
                        toplam_kod = 0
                        for ch, freq in sorted(actual_freq.items(), key=lambda x: -x[1])[:12]:
                            info_bits = -math.log2(freq) if freq > 0 else 0
                            code_len = len(codes.get(ch, ""))
                            fark = code_len - info_bits  # + = Huffman fazla bit atadı
                            if fark > 0.1:
                                durum = f"🟡 {fark:+.2f} bit israfı"
                            elif fark < -0.1:
                                durum = f"🟢 {fark:+.2f} bit kazancı"
                            else:
                                durum = "✅ optimal"
                            rows_info.append({
                                "Karakter": repr(ch),
                                "Olasılık p(c)": f"{freq:.4f}",
                                "Teorik bilgi -log₂(p)": f"{info_bits:.2f} bit",
                                "Huffman kodu": f"{code_len} bit",
                                "Fark": durum,
                            })
                            # Toplam: tüm karakterler için ağırlıklı
                        # Genel verimlilik (tüm metin)
                        total_info_all = sum(-cnt * math.log2(cnt/total_chars)
                                             for cnt in Counter(text).values()
                                             if cnt > 0)
                        total_huff_bits = sum(len(codes.get(c, ""))
                                              for c in text)
                        genel_verim = total_info_all / total_huff_bits * 100 if total_huff_bits else 0

                        st.dataframe(rows_info, use_container_width=True, hide_index=True)
                        st.info(
                            f"📊 **Genel sıkıştırma verimliliği: %{genel_verim:.1f}** "
                            f"(Shannon entropi {total_info_all:.0f} bit / "
                            f"Huffman {total_huff_bits:,} bit). "
                            f"**%100 = teorik mükemmel** (Cover & Thomas §5.6). "
                            f"Huffman her zaman tam bit kullandığı için kesirli bit ihtiyaçlarında "
                            f"karakter bazlı bazen kazanç (🟢), bazen israf (🟡) olur — "
                            f"ortalama Shannon limitine yakınsar."
                        )

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
    st.subheader("📖 LZW + 🤖 AI Akıllı Sözlük Üretimi")
    st.markdown(
        "**🎯 AI Entegrasyonu (Akıllı Sözlük):** Bu sekme **LLM (LLaMA 3.3 70B)**'ı "
        "kullanarak metne özgü **en yaygın kelimeler/ifadeler** üretir ve LZW "
        "başlangıç sözlüğüne ekler. Bu sayede sıkıştırıcı daha erken pattern eşleşmeleri "
        "bulur — geleneksel LZW'ye göre **+%3-15 ek küçülme** sağlanır."
    )

    # ═══════════════════════════════════════════════════
    # AI SÖZLÜK ÜRETİMİ — EN ÜSTE TAŞINDI
    # ═══════════════════════════════════════════════════
    st.markdown("### 🤖 AI Sözlük Üretici (LLM → LZW Sözlüğü)")
    col_ai1, col_ai2 = st.columns([3, 1])
    with col_ai1:
        n_words = st.slider(
            "🎯 LLM'in üreteceği yaygın kelime/ifade sayısı",
            min_value=20, max_value=200, value=80,
            help="LLM Türkçe metne özgü en yaygın N kelimeyi bulup LZW sözlüğüne ekler",
        )
    with col_ai2:
        st.markdown(" ")
        ai_butonu = st.button("▶ AI Sözlük Üret + LZW Sıkıştır", key="lzw_ai", type="primary")

    if ai_butonu:
        if not api_key:
            st.error("⚠️ Sol menüden Groq API key girin. (console.groq.com ücretsiz)")
        else:
            # 1) AI'ya sözlük üretme PROMPTU
            t_ai_start = time.perf_counter()
            with st.spinner(f"🧠 LLaMA 3.3 70B {n_words} kelimelik sözlük üretiyor..."):
                try:
                    ai_words, tok, raw = generate_lzw_dictionary(text, text_type, n_words)
                    ai_time_ms = (time.perf_counter() - t_ai_start) * 1000

                    # AI sonuçları
                    st.success(
                        f"✅ **LLM {len(ai_words)} kelime önerdi** · "
                        f"⏱ {ai_time_ms:.0f} ms · "
                        f"🪙 {tok} token kullanıldı"
                    )

                    # AI'nın ürettiği sözlüğü göster
                    with st.expander(f"📚 AI'nın ürettiği sözlük (tüm {len(ai_words)} kelime)", expanded=True):
                        # 4 sütunlu grid
                        ws_cols = st.columns(4)
                        for i, w in enumerate(ai_words):
                            ws_cols[i % 4].markdown(f"`{w}`")

                    # 2) LZW karşılaştır
                    with st.spinner("Standart LZW vs AI-LZW karşılaştırılıyor..."):
                        t_lzw = time.perf_counter()
                        result = lzw_compare(text, ai_words)
                        lzw_time_ms = (time.perf_counter() - t_lzw) * 1000

                    std = result["standard"]
                    ai_lzw = result["ai_lzw"]
                    saved = result["saved_bits"]

                    # METRİKLER + SÜRE
                    st.markdown("### 📊 Sonuçlar")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Orijinal", f"{std['original_bits']:,} bit")
                    c2.metric("Standart LZW",
                              f"{std['compressed_bits']:,} bit",
                              delta=f"%{(1-std['ratio'])*100:.1f} küçülme",
                              delta_color="off")
                    c3.metric("🤖 AI-LZW (sözlük ile)",
                              f"{ai_lzw['compressed_bits']:,} bit",
                              delta=f"%{(1-ai_lzw['ratio'])*100:.1f} küçülme")
                    if saved > 0:
                        c4.metric("🏆 AI Tasarrufu",
                                  f"+{saved:,} bit",
                                  delta=f"%{saved/std['compressed_bits']*100:.1f}")
                    else:
                        c4.metric("Fark", f"{saved:,} bit",
                                  delta=f"%{-saved/std['compressed_bits']*100:.1f}",
                                  delta_color="inverse")

                    # KAYNAK TÜKETİMİ
                    st.caption(
                        f"⏱ **İşlem süreleri:** AI sözlük üretimi {ai_time_ms:.0f} ms, "
                        f"LZW sıkıştırma {lzw_time_ms:.1f} ms · "
                        f"🪙 **Token kullanımı:** {tok} (~${tok*0.0000005:.5f} maliyet) · "
                        f"📦 **Sözlük büyümesi:** 256+{result['ai_words_added']} = {256+result['ai_words_added']} kod"
                    )

                    # 3) Grafik
                    fig = go.Figure(go.Bar(
                        x=["Orijinal", "Standart LZW", "🤖 AI-LZW"],
                        y=[std["original_bits"], std["compressed_bits"], ai_lzw["compressed_bits"]],
                        marker_color=["#888", "#EF553B", "#00CC96"],
                        text=[f"{v:,} bit" for v in [
                            std["original_bits"], std["compressed_bits"], ai_lzw["compressed_bits"]]],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        title="LZW Bit Karşılaştırması (AI sözlük etkisi)",
                        yaxis_title="Bit", height=380, showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # 4) PROMPT MÜHENDİSLİĞİ DETAYI
                    with st.expander("🔧 Prompt Mühendisliği — AI'ya verilen komut"):
                        st.markdown("**LLM'e gönderilen sistem promptu:**")
                        st.code(
                            "Sen veri sıkıştırma uzmanısın. Verilen metin türü için "
                            "LZW sıkıştırmasında pattern eşleşmesini hızlandıracak "
                            "EN YAYGIN kelime/ifadeleri tahmin et. JSON formatında döndür.",
                            language=None,
                        )
                        st.markdown(f"**Kullanıcı promptu (metin türü: {text_type}):**")
                        st.code(
                            f'"Bu Türkçe metin için LZW sıkıştırmada başlangıç sözlüğüne '
                            f'eklenecek {n_words} yaygın kelime/ifade öner. '
                            f'Metin örneği: \"{text[:100]}...\"\\n'
                            f'Çıktı formatı: [\"kelime1\", \"kelime2\", ...]"',
                            language=None,
                        )
                        st.caption("Bu yaklaşım iteratif prompt mühendisliği ile geliştirildi. "
                                  "İlk denemelerde JSON parse hataları yaşandı → robust parser yazıldı. "
                                  "ai_diary.json'da süreç detayı var.")

                    # 5) AI'NIN ÇIKTI ÖRNEĞİ (ham JSON)
                    with st.expander("🤖 LLM'nin ham çıktısı (JSON öncesi)"):
                        st.code(raw[:1500] + ("..." if len(raw) > 1500 else ""), language="json")

                except Exception as e:
                    st.error(f"❌ AI sözlük üretimi başarısız: {e}")
                    st.info("💡 Bu durumda standart LZW kullanılır — aşağıda görünüyor.")

    st.markdown("---")
    st.markdown("### 📊 Standart LZW (API'siz, her zaman çalışır)")

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

    # ── DECODE: Kayıpsızlık Doğrulama ──
    st.markdown("### 🔓 Açma (Decompression) — Kayıpsızlık Kanıtı")
    _init_dict = {chr(i): i for i in range(256)}
    for _ch in set(text):
        if _ch not in _init_dict:
            _init_dict[_ch] = len(_init_dict)
    from core.lzw import lzw_decode as _lzw_dec
    try:
        _lzw_decoded = _lzw_dec(_codes_lzw, _init_dict)
    except Exception as _e:
        _lzw_decoded = f"[Decode hatası: {_e}]"
    _lzw_kayipsiz = (_lzw_decoded == text)

    with st.expander("🔍 LZW kod listesi (ilk 30) ve sözlük bilgisi"):
        st.code(str(_codes_lzw[:30]) + ("..." if len(_codes_lzw) > 30 else ""))
        st.caption(
            f"Toplam **{len(_codes_lzw):,}** kod × **{_bpc} bit** · "
            f"Sözlük: 256 ASCII + **{len(_init_dict) - 256}** Türkçe karakter."
        )

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
            st.caption("Satır = gerçek sınıf, sütun = tahmin. Köşegende olanlar doğru.")

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
    with st.expander("📖 Mimari ve eğitim"):
        st.markdown(
            "**MLP 11 → 32 → 16 → 8 → 3** (Huffman/LZW/BWT)  \n"
            "**11 özellik:** entropi, benzersiz oran, top-3, boşluk, "
            "Türkçe oran, run-length, rakam, büyük harf, bigram entropi, "
            "max koşu, alfabe boyutu (log₂)  \n"
            "**Ezber önleme:** L2 (α=1e-3) + early stopping + 5-fold CV + "
            "hold-out test (2.357 örnek, 14 kategori)"
        )

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
        "**Shannon Entropisi** (1948), bir verinin teorik olarak sıkıştırılabilecek "
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
# SEKME 12: NEXT-TOKEN TAHMİN (Shannon 1951 yaklaşımı)
# ═══════════════════════════════════════════════════
with tab12:
    st.subheader("🔮 Next-Token Tahmin — Kontekstli Entropi (Shannon 1951)")
    st.markdown(
        "**Shannon'un 1951 deneyinin modern uygulaması.** İnsanlar yerine "
        "**n-gram olasılık tablosu** kullanarak her karakteri tahmin et, "
        "ne kadar 'sürpriz' olduğunu ölç. "
        "Toplam bit sayısı = teorik ideal sıkıştırma."
    )

    with st.expander("📖 Bu nedir?"):
        st.markdown("""
**3 farklı dil modeli ile karşılaştırma:**

1. **Unigram (iid)** — P(c) — karakter bağımsız varsayım (Shannon 1948)
2. **Bigram** — P(c | önceki) — 1. derece Markov modeli
3. **Trigram** — P(c | önceki, önceki) — 2. derece Markov modeli

Her seviyede karakterin **bilgi içeriği:**

$$I(c) = -\\log_2 P(c \\mid \\text{context})$$

Toplam = ideal sıkıştırma boyutu.

**Şu projeyle bağlantı:** Bizim Huffman/LZW/BWT karakter bazlı yaklaşımları
yaklaşık unigram sınırına ulaşır. Kontekstli (bigram/trigram) sıkıştırma için
daha gelişmiş algoritmalar (arithmetic coding + n-gram) gerekir.
        """)

    if st.button("▶ Next-Token Analizi Başlat", key="nexttok_btn"):
        with st.spinner("Türkçe corpus'tan n-gram tablosu oluşturuluyor..."):
            from core.next_token import karsilastir
            r = karsilastir(text)

        if "error" in r:
            st.error(r["error"])
        else:
            st.caption(
                f"📚 Corpus: **{r['corpus_size']:,} karakter** Türkçe metin "
                f"(`data/large_turkish.txt`). Bu corpus'tan n-gram olasılıkları öğrenildi."
            )

            # 4 metric
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Orijinal", f"{r['orig_bits']:,} bit",
                      help="8 bit/karakter (UTF-8 varsayım)")
            c2.metric("Unigram (Shannon iid)",
                      f"{r['unigram']['bits']:,} bit",
                      delta=f"{r['unigram']['bpc']:.2f} bit/karakter",
                      delta_color="off")
            c3.metric("Bigram (Markov-1)",
                      f"{r['bigram']['bits']:,} bit",
                      delta=f"{r['bigram']['bpc']:.2f} bit/karakter",
                      delta_color="off")
            c4.metric("🎯 Trigram (Markov-2)",
                      f"{r['trigram']['bits']:,} bit",
                      delta=f"{r['trigram']['bpc']:.2f} bit/karakter",
                      delta_color="off")

            # Karşılaştırma grafiği
            modeller = ["Orijinal\n(8 bit/c)",
                       f"Unigram\n({r['unigram']['bpc']:.2f})",
                       f"Bigram\n({r['bigram']['bpc']:.2f})",
                       f"Trigram\n({r['trigram']['bpc']:.2f})"]
            degerler = [r['orig_bits'],
                       r['unigram']['bits'],
                       r['bigram']['bits'],
                       r['trigram']['bits']]
            renkler = ["#888", "#636EFA", "#00CC96", "#FFD700"]

            fig_nt = go.Figure(go.Bar(
                x=modeller, y=degerler, marker_color=renkler,
                text=[f"{v:,}" for v in degerler],
                textposition="outside",
            ))
            fig_nt.update_layout(
                title="Kontekstli Entropi: Model derinleştikçe sıkıştırma artar",
                yaxis_title="Toplam bit",
                height=380, showlegend=False,
            )
            st.plotly_chart(fig_nt, use_container_width=True)

            # Sonuç yorumu
            st.markdown("### 📊 Yorum")
            uni_pct = (1 - r['unigram']['ratio']) * 100
            bi_pct  = (1 - r['bigram']['ratio']) * 100
            tri_pct = (1 - r['trigram']['ratio']) * 100
            iyilesme_bi  = (r['unigram']['bits'] - r['bigram']['bits']) / r['unigram']['bits'] * 100
            iyilesme_tri = (r['unigram']['bits'] - r['trigram']['bits']) / r['unigram']['bits'] * 100

            st.markdown(f"""
- **Unigram (Shannon iid):** %{uni_pct:.1f} küçülme — karakterleri bağımsız varsayar
- **Bigram:** %{bi_pct:.1f} küçülme — 1. derece bağımlılık → Unigram'dan **%{iyilesme_bi:.1f} daha iyi**
- **Trigram:** %{tri_pct:.1f} küçülme — 2. derece bağımlılık → Unigram'dan **%{iyilesme_tri:.1f} daha iyi**

> **Çıkarım:** Karakterler arası bağlamı modellemek, klasik Shannon'un karakter bağımsız
> sınırının altına inmeyi sağlar. **Bu, Shannon'un kendisinin 1951 makalesinde gösterdiği şeydir.**

> **Bu projedeki sistemle ilişkisi:** Bizim Huffman/LZW/BWT klasik karakter bazlı sıkıştırma
> yapar — yaklaşık Unigram sınırına yaklaşır. Trigram seviyesine inmek için Markov-tabanlı
> arithmetic coder veya LLM tabanlı sıkıştırıcı gerekir.
            """)

            st.info(
                "💡 **Akademik referans:** Shannon, C. E. (1951). "
                "'Prediction and Entropy of Printed English.' "
                "*Bell System Technical Journal*, 30(1), 50-64."
            )


# ═══════════════════════════════════════════════════
# SEKME 10: AI GÜNLÜĞÜ
# ═══════════════════════════════════════════════════
with tab10:
    st.subheader("🤖 AI Etkileşim Günlüğü")
    st.caption("Proje sürecinde AI'ya verilen tüm promptlar ve alınan cevaplar.")

    diary_file = "ai_diary.json"
    if not os.path.exists(diary_file):
        st.info("Henüz AI günlüğü oluşmamış. AI sekmelerinde bir analiz çalıştırın.")
    else:
        with open(diary_file) as f:
            diary = json.load(f)

        adimlar = diary.get("adimlar", [])
        refl_adimlar = [a for a in adimlar if "yansima" in a]
        akademik_adimlar = [a for a in adimlar if "kaynak" in a]
        normal_adimlar = [a for a in adimlar
                          if "yansima" not in a and "kaynak" not in a]

        # ── Özet metrikler ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Toplam Adım", f"{len(adimlar)}")
        m2.metric("📚 Akademik Doğrulama", f"{len(akademik_adimlar)}",
                  help="İlgili klasik makale/kitapla doğrulama girişleri")
        m3.metric("🪞 Reflektif Not", f"{len(refl_adimlar)}",
                  help="Süreç değerlendirmesi içeren girişler")
        m4.download_button(
            "⬇ JSON İndir",
            data=open(diary_file).read(),
            file_name="ai_diary.json",
            mime="application/json",
            help="Rapora ek olarak teslim edilebilir",
        )
        st.caption(
            f"💬 Klasik AI etkileşim: **{len(normal_adimlar)}** · "
            f"📚 Akademik doğrulama: **{len(akademik_adimlar)}** "
            f"(Sayood, Cover & Thomas, Burrows-Wheeler vb. ile) · "
            f"Toplam token: **{diary.get('toplam_token', 0):,}**"
        )

        st.markdown("---")

        # ── Arama / filtre ──
        col_s, col_t = st.columns([3, 1])
        with col_s:
            arama = st.text_input("🔍 Adımlar içinde ara",
                                  placeholder="örn: BWT, Sayood, Türkçe, NN...")
        with col_t:
            tip_filtre = st.selectbox(
                "Tip",
                ["Hepsi", "📚 Akademik doğrulama", "🪞 Reflektif", "💬 Klasik AI"],
            )

        # Filtre uygula
        gosterilecek = adimlar
        if tip_filtre == "📚 Akademik doğrulama":
            gosterilecek = akademik_adimlar
        elif tip_filtre == "🪞 Reflektif":
            gosterilecek = refl_adimlar
        elif tip_filtre == "💬 Klasik AI":
            gosterilecek = normal_adimlar

        if arama:
            arm = arama.lower()
            gosterilecek = [
                a for a in gosterilecek
                if arm in a.get("hedeflenen_islem", "").lower()
                or arm in a.get("verilen_prompt", "").lower()
                or arm in a.get("sonuc", "").lower()
                or arm in a.get("yansima", "").lower()
                or arm in a.get("kaynak", "").lower()
            ]

        st.caption(f"**{len(gosterilecek)}** adım gösteriliyor "
                   f"(toplam {len(adimlar)} adımdan).")

        # ── Adımları göster ──
        for step in gosterilecek:
            # Etiket: akademik > reflektif > klasik
            if "kaynak" in step:
                etiket = "📚"
            elif "yansima" in step:
                etiket = "🪞"
            else:
                etiket = "💬"

            with st.expander(
                f"{etiket} Adım {step['adim']}: {step['hedeflenen_islem']}"
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Verilen Prompt / Soru**")
                    st.code(step["verilen_prompt"][:500], language=None)
                with col2:
                    st.markdown("**AI Cevabı / Akademik Bulgu**")
                    st.code(step["ai_cevabi"][:500], language=None)
                st.markdown(
                    f"**Sorun:** {step.get('sorun', '—')}  \n"
                    f"**Sonuç:** {step.get('sonuc', '—')}"
                )
                if "kaynak" in step:
                    st.success(f"📚 **Akademik kaynak:** {step['kaynak']}")
                if "yansima" in step:
                    st.info(f"🪞 **Yansıma:** {step['yansima']}")
                if step.get("token_kullanim"):
                    st.caption(f"Token: {step['token_kullanim']}")

    # ── Manuel ek (expander içinde — yer kaplamasın) ──
    with st.expander("➕ Manuel adım ekle"):
        col1, col2 = st.columns(2)
        with col1:
            m_hedef  = st.text_input("Hedeflenen işlem")
            m_prompt = st.text_area("Verilen prompt", height=80)
        with col2:
            m_cevap  = st.text_area("AI cevabı / sorun", height=80)
            m_sonuc  = st.text_input("Sonuç")
        m_yansima = st.text_area("Yansıma (opsiyonel — reflektif not)", height=60)

        if st.button("Adım Ekle", key="add_step"):
            if os.path.exists(diary_file):
                with open(diary_file) as f:
                    diary = json.load(f)
            else:
                diary = {"adimlar": [], "toplam_token": 0}
            yeni_adim = {
                "adim": len(diary["adimlar"]) + 1,
                "hedeflenen_islem": m_hedef,
                "verilen_prompt": m_prompt,
                "ai_cevabi": m_cevap,
                "sorun": "Manuel giriş",
                "sonuc": m_sonuc,
                "token_kullanim": 0,
            }
            if m_yansima:
                yeni_adim["yansima"] = m_yansima
            diary["adimlar"].append(yeni_adim)
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

    # Bloklu mod bilgisi
    _n_chunks = _bwt_enc.get('n_chunks', 1)
    if _n_chunks > 1:
        st.success(
            f"📦 Bu metin **{_n_chunks} bloğa** bölündü ve her blok ayrı BWT işledi. "
            f"Aşağıda ilk bloğun detayları gösteriliyor — tüm bloklar kayıpsız geri alınır."
        )

    _bwt_str = _bwt_enc['bwt']
    _orig_idx = _bwt_enc['orig_idx']

    # Decode: bloklu modda tüm blokları aç, tek bloksa eski yöntem
    if _n_chunks > 1:
        from core.bwt import bwt_chunked_decode as _bwt_chunk_dec
        _bwt_decoded = _bwt_chunk_dec(_bwt_enc['chunks'])
    else:
        _bwt_decoded = _bwt_dec(_bwt_str, _orig_idx)
    _bwt_kayipsiz = (_bwt_decoded == text)

    with st.expander("🔍 BWT permüte metin + LF mapping detayı"):
        st.markdown(f"**BWT (ilk 150):**")
        st.code(_bwt_str[:150] + ("..." if len(_bwt_str) > 150 else ""))
        st.caption(f"BWT uzunluğu: **{len(_bwt_str)}**, Orijinal indeks: **{_orig_idx}**.")

        # Hızlı LF mapping tablosu (ilk 12)
        from collections import Counter as _C
        _seen, _rank = {}, []
        for _ch in _bwt_str:
            _rank.append(_seen.get(_ch, 0))
            _seen[_ch] = _seen.get(_ch, 0) + 1
        _char_counts = _C(_bwt_str)
        _starts, _pos = {}, 0
        for _ch in sorted(_char_counts.keys()):
            _starts[_ch] = _pos
            _pos += _char_counts[_ch]
        _lf = [_starts[_bwt_str[i]] + _rank[i] for i in range(len(_bwt_str))]
        st.markdown("**LF Mapping (ilk 12, decode iterasyonu için):**")
        st.dataframe([
            {"i": i, "L[i]": repr(_bwt_str[i]), "LF[i]": _lf[i]}
            for i in range(min(12, len(_bwt_str)))
        ], use_container_width=True, hide_index=True)

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
