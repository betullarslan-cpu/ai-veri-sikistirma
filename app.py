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
from core.huffman import encode as huffman_encode, build_codes, HuffmanNode
from core.lzw import compare as lzw_compare, build_ai_lzw_dict, lzw_encode
from core.ai_engine import (
    predict_frequencies, refine_frequencies, kl_divergence,
    generate_lzw_dictionary,
)
from core.entropy import shannon_entropy, theoretical_min_bits, compression_efficiency, per_char_analysis
from core.selector import analyze_and_select, run_selected
from core.image_compress import compare as image_compare
from core.ocr_compress import ocr_extract, compress_extracted
from core.hybrid import hybrid_compress, train_from_corpus, smart_hybrid
from core.bwt import compare as bwt_compare, bwt_stats, bwt_encode, rle_compress
from core.nn_selector import predict as nn_predict, train as nn_train, extract_features
from core.arithmetic import compare as arith_compare
from PIL import Image

# ─── Sayfa ayarı ───────────────────────────────────
st.set_page_config(
    page_title="AI Sıkıştırma",
    page_icon="🗜️",
    layout="wide",
)

st.title("🗜️ AI-Destekli Veri Sıkıştırma")

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

# ─── Sekmeler — Hızlı Özet en başta, geri kalanlar kısa isimle ──────
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "🚀 Hızlı Özet",
    "📊 Huffman",
    "📖 LZW",
    "🧠 AI Seçici",
    "🔬 Sinir Ağı",
    "📐 Aritmetik",
    "⚡ Hibrit",
    "🖼️ Görüntü",
    "🔍 OCR",
    "📈 Shannon",
    "🤖 Günlük",
    "🔄 BWT",
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

            _orig = len(text) * 8
            _ent  = _shan(text)
            _min  = _tmb(text)
            _bwt  = _bwt_cmp(text)
            _sm   = _sh(text, use_ai_dict=False)
            _nn   = _nnp(text)

        # ── 4 metrik yan yana ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Orijinal", f"{_orig:,} bit",
                  help="Her karakter 8 bit varsayımı")
        m2.metric("Shannon limiti", f"{_min:,} bit",
                  delta=f"%{(1-_min/_orig)*100:.1f} küçülme",
                  delta_color="off",
                  help="Teorik en küçük boyut")
        m3.metric("Standart Huffman", f"{_sm['standard_bits']:,} bit",
                  delta=f"%{(1-_sm['standard_ratio'])*100:.1f} küçülme",
                  delta_color="off")
        m4.metric(f"🏆 Akıllı Hibrit ({_sm['nn_decision']})",
                  f"{_sm['smart_bits']:,} bit",
                  delta=f"%{(1-_sm['smart_ratio'])*100:.1f} küçülme")

        # NN bilgisi tek satır
        st.success(
            f"🧠 **Sinir Ağı kararı:** {_nn['algorithm'].upper()} (%{_nn['confidence']*100:.0f} güven) — "
            f"**Yöntem:** {_sm['method']} — "
            f"**Standart Huffman'a göre +%{_sm['saved_bits']*100/_sm['standard_bits']:.1f} iyileşme** "
            f"({_sm['saved_bits']:,} bit tasarruf)"
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

    n_words = st.slider("AI'nın ekleyeceği kelime/ifade sayısı", 20, 200, 80)

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
# SEKME 3: AI ALGORİTMA SEÇİCİ
# ═══════════════════════════════════════════════════
with tab3:
    st.subheader("🧠 AI Otomatik Algoritma Seçici")
    st.markdown(
        "**AI, metni analiz ederek Huffman mı yoksa LZW mi daha uygun olduğuna karar verir.** "
        "Gerekçesini açıklar, ardından seçtiği algoritmayı çalıştırır."
    )

    if st.button("▶ AI Analiz Etsin", key="selector"):
        if not api_key:
            st.error("Sol menüden Groq API key gir.")
        else:
            with st.spinner("AI metni analiz ediyor..."):
                try:
                    result = analyze_and_select(text)

                    algo = result.get("algorithm", "huffman")
                    conf = result.get("confidence", 0)
                    reason = result.get("reasoning", "")
                    chars = result.get("characteristics", {})

                    # AI kararı
                    algo_label = {"huffman": "Huffman", "lzw": "LZW", "hybrid": "Hibrit (Her ikisi)"}.get(algo, algo)
                    color = "🟢" if conf >= 70 else "🟡"
                    st.markdown(f"## {color} AI Kararı: **{algo_label}**")
                    st.markdown(f"**Güven:** {conf}/100")
                    st.info(f"💬 **Gerekçe:** {reason}")

                    # Özellikler
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Tekrar Oranı", chars.get("repetition", "-"))
                    c2.metric("Kelime Çeşitliliği", chars.get("vocabulary", "-"))
                    c3.metric("En İyi Kullanım", chars.get("best_for", "-"))

                    # Algoritmayı çalıştır
                    with st.spinner(f"{algo_label} çalıştırılıyor..."):
                        run_results = run_selected(text, algo)

                    orig = len(text) * 8
                    st.markdown("### Sonuçlar")
                    cols = st.columns(len(run_results) + 1)
                    cols[0].metric("Orijinal", f"{orig:,} bit")
                    for i, (name, r) in enumerate(run_results.items()):
                        bits = r.get("bits") or r.get("compressed_bits", 0)
                        ratio = r.get("ratio", bits/orig)
                        cols[i+1].metric(
                            name.upper(),
                            f"{bits:,} bit",
                            delta=f"%{(1-ratio)*100:.1f} küçüldü"
                        )

                    # Grafik
                    names = ["Orijinal"] + [n.upper() for n in run_results]
                    vals  = [orig] + [
                        r.get("bits") or r.get("compressed_bits", 0)
                        for r in run_results.values()
                    ]
                    fig = go.Figure(go.Bar(
                        x=names, y=vals,
                        marker_color=["#636EFA", "#00CC96", "#EF553B"][:len(vals)],
                        text=[f"{v:,} bit" for v in vals],
                        textposition="outside",
                    ))
                    fig.update_layout(title=f"AI Seçimi: {algo_label}",
                                      yaxis_title="Bit", height=380, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                    st.metric("Token kullanımı", result.get("tokens", 0))

                except Exception as e:
                    st.error(f"Hata: {type(e).__name__}: {e}")
                    with st.expander("Detay"):
                        st.exception(e)


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

    # Modelin gerçek doğruluğunu oku
    import pickle as _pk
    _mp = "core/nn_model.pkl"
    if os.path.exists(_mp):
        with open(_mp, "rb") as f:
            _b = _pk.load(f)
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
        st.warning("Model henüz eğitilmemiş.")

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
            with st.spinner("Sinir ağı eğitiliyor (1900+ örnek, 5-fold CV)..."):
                r = nn_train(verbose=False)
            st.success(
                f"✓ Hold-out: %{r['test_accuracy']*100:.1f} | "
                f"CV: %{r['cv_accuracy']*100:.1f} | "
                f"{r['samples']:,} örnek"
            )
            st.caption(f"Sınıf dağılımı: {r['distribution']}")

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
# SEKME 5: ARİTMETİK KODLAMA + AI
# ═══════════════════════════════════════════════════
with tab5:
    st.subheader("📐 Aritmetik Kodlama + AI Olasılık Modeli")
    st.markdown(
        "**Aritmetik kodlama**, Huffman'ın aksine sembol başına kesirli bit kullanır — "
        "Shannon sınırına çok daha yakın sıkıştırma yapar. "
        "AI olasılık tahmini ile tablo overhead'i ortadan kalkar."
    )

    if st.button("▶ Aritmetik Analiz Başlat", key="arith"):
        if not api_key:
            st.error("Sol menüden Groq API key gir.")
        else:
            with st.spinner("AI olasılık modeli oluşturuluyor..."):
                try:
                    result = arith_compare(text, text_type)
                    orig = result["original_bits"]
                    std  = result["standard"]
                    ai   = result["ai_arithmetic"]
                    theo = result["theoretical_min"]

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Orijinal", f"{orig:,} bit")
                    c2.metric("Standart Aritmetik", f"{std['total']:,} bit",
                              delta=f"oran: {std['ratio']:.4f}")
                    c3.metric("AI Aritmetik", f"{ai['total']:,} bit",
                              delta=f"oran: {ai['ratio']:.4f}")
                    c4.metric("Shannon Sınırı", f"{theo:,} bit",
                              delta=f"oran: {theo/orig:.4f}")

                    saved = result["saved_vs_standard"]
                    if saved > 0:
                        st.success(f"✅ AI aritmetik kodlama {saved:,} bit tasarruf sağladı!")
                    else:
                        st.info(f"ℹ️ Standart {-saved:,} bit daha iyi. AI olasılık tahmini iyileştirilebilir.")

                    fig = go.Figure(go.Bar(
                        x=["Orijinal", "Standart\nAritmetik", "AI\nAritmetik", "Shannon\nSınırı"],
                        y=[orig, std["total"], ai["total"], theo],
                        marker_color=["#636EFA", "#EF553B", "#AB63FA", "#00CC96"],
                        text=[f"{v:,}" for v in [orig, std["total"], ai["total"], theo]],
                        textposition="outside",
                    ))
                    fig.add_hline(y=theo, line_dash="dash", line_color="#00CC96",
                                  annotation_text="Teorik Sınır")
                    fig.update_layout(title="Aritmetik Kodlama vs Shannon Sınırı",
                                      yaxis_title="Bit", height=420, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Token kullanımı: {result['tokens']}")

                except Exception as e:
                    st.error(f"Hata: {type(e).__name__}: {e}")
                    with st.expander("Detay"): st.exception(e)


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
# SEKME 7: GÖRÜNTÜ SIKIŞTURMA
# ═══════════════════════════════════════════════════
with tab7:
    st.subheader("🖼️ AI Tabanlı Görüntü Sıkıştırma")
    st.markdown(
        "**AI, görüntünün hangi bölgelerinin önemli olduğuna karar verir.** "
        "Merkez/odak bölgesi yüksek kalitede, arka plan/köşeler düşük kalitede sıkıştırılır. "
        "Böylece görsel kalite korunurken dosya boyutu küçülür."
    )

    img_file = st.file_uploader("Görüntü yükle (PNG / JPG)", type=["png", "jpg", "jpeg"])
    img_desc = st.text_input("Görüntüyü kısaca tanımla",
                             placeholder="örn: portre fotoğrafı, şehir manzarası, teknik diyagram")
    std_q = st.slider("Standart JPEG kalitesi (karşılaştırma için)", 10, 90, 50)

    if img_file and img_desc:
        img = Image.open(img_file)

        col_orig, col_info = st.columns([2, 1])
        with col_orig:
            st.image(img, caption="Orijinal", use_column_width=True)
        with col_info:
            st.metric("Boyut", f"{img.size[0]}×{img.size[1]} px")
            st.metric("Mod", img.mode)

        if st.button("▶ AI Analiz Et ve Sıkıştır", key="img_compress"):
            if not api_key:
                st.error("Sol menüden Groq API key gir.")
            else:
                with st.spinner("AI bölge haritası oluşturuyor..."):
                    try:
                        result = image_compare(img, img_desc, std_quality=std_q)
                        imp = result["importance_map"]

                        # AI kararı
                        st.info(f"💬 **AI Gerekçesi:** {imp.get('reasoning', '')}")
                        st.caption(f"Önemli bölgeler: {imp.get('important_regions', '')}")

                        # Kalite ızgarası
                        grid = imp.get("grid", [[50]*3]*3)
                        st.markdown("**Kalite Haritası (3×3 ızgara):**")
                        grid_cols = st.columns(3)
                        labels = ["Sol", "Orta", "Sağ"]
                        row_labels = ["Üst", "Orta", "Alt"]
                        for r in range(3):
                            for c in range(3):
                                q = grid[r][c] if r < len(grid) and c < len(grid[r]) else 50
                                color = "🟢" if q >= 80 else "🟡" if q >= 50 else "🔴"
                                grid_cols[c].metric(
                                    f"{row_labels[r]}-{labels[c]}",
                                    f"{color} Q{q}"
                                )

                        # Metrikler
                        st.markdown("### Boyut Karşılaştırması")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Orijinal PNG", f"{result['original_bytes']:,} byte")
                        m2.metric(f"Standart JPEG (Q{std_q})",
                                  f"{result['standard_bytes']:,} byte",
                                  delta=f"%{(1-result['standard_ratio'])*100:.1f} küçüldü")
                        m3.metric("AI Bölgeli JPEG",
                                  f"{result['ai_bytes']:,} byte",
                                  delta=f"%{(1-result['ai_ratio'])*100:.1f} küçüldü")

                        saved = result["saved_bytes"]
                        if saved > 0:
                            st.success(f"✅ AI yaklaşımı standart JPEG'den {saved:,} byte daha az!")
                        else:
                            st.info(f"ℹ️ Standart JPEG {-saved:,} byte daha küçük. "
                                    "AI bölge birleştirme maliyeti var.")

                        # Grafik
                        fig = go.Figure(go.Bar(
                            x=["Orijinal PNG", f"Standart JPEG\n(Q{std_q})", "AI Bölgeli JPEG"],
                            y=[result["original_bytes"], result["standard_bytes"], result["ai_bytes"]],
                            marker_color=["#636EFA", "#EF553B", "#00CC96"],
                            text=[f"{v:,} B" for v in [
                                result["original_bytes"], result["standard_bytes"], result["ai_bytes"]]],
                            textposition="outside",
                        ))
                        fig.update_layout(title="Görüntü Boyutu Karşılaştırması",
                                          yaxis_title="Byte", height=380, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                        # Sonuç görüntü
                        c1, c2 = st.columns(2)
                        c1.image(result["std_image"],
                                 caption=f"Standart JPEG (Q{std_q})", use_column_width=True)
                        c2.image(result["ai_image"],
                                 caption="AI Bölgeli JPEG", use_column_width=True)

                        st.metric("Token kullanımı", imp.get("tokens", 0))

                    except Exception as e:
                        st.error(f"Hata: {type(e).__name__}: {e}")
                        with st.expander("Detay"):
                            st.exception(e)
    else:
        st.info("Görüntü yükle ve tanımla, ardından AI analiz etsin.")


# ═══════════════════════════════════════════════════
# SEKME 8: OCR + SIKIŞTURMA
# ═══════════════════════════════════════════════════
with tab8:
    st.subheader("🔍 OCR + Sıkıştırma Pipeline")
    st.markdown(
        "**Görüntüdeki metni AI ile oku → Huffman + LZW ile sıkıştır.** "
        "Taranmış belgeler, faturalar, kitap sayfaları için ideal. "
        "AI (vision modeli) OCR yapar, ardından en iyi algoritma seçilir."
    )

    st.info("💡 Akış: 📷 Görüntü → 🤖 AI OCR → 📝 Metin → 🗜️ Huffman / LZW")

    ocr_file = st.file_uploader("Metin içeren görüntü yükle", type=["png", "jpg", "jpeg"],
                                key="ocr_upload")

    if ocr_file:
        ocr_img = Image.open(ocr_file)
        st.image(ocr_img, caption="Yüklenen görüntü", width=400)

        if st.button("▶ OCR Oku ve Sıkıştır", key="ocr_btn"):
            if not api_key:
                st.error("Sol menüden Groq API key gir.")
            else:
                # Adım 1: OCR
                with st.spinner("AI görüntüdeki metni okuyor..."):
                    try:
                        ocr_result = ocr_extract(ocr_img)
                        extracted_text = ocr_result["text"]
                        ocr_tokens = ocr_result["tokens"]

                        st.markdown("### 📝 Çıkarılan Metin")
                        st.text_area("OCR Sonucu", extracted_text, height=150)
                        st.caption(f"Karakter sayısı: {len(extracted_text):,} | Token: {ocr_tokens}")

                        if len(extracted_text) < 5 or "bulunamadı" in extracted_text.lower():
                            st.warning("Görüntüde okunabilir metin bulunamadı.")
                        else:
                            # Adım 2: Sıkıştırma
                            with st.spinner("Metin sıkıştırılıyor..."):
                                comp = compress_extracted(extracted_text)

                            st.markdown("### 🗜️ Sıkıştırma Sonuçları")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Orijinal Metin", f"{comp['original_bits']:,} bit")
                            c2.metric("Huffman",
                                      f"{comp['huffman']['total']:,} bit",
                                      delta=f"%{(1-comp['huffman']['ratio'])*100:.1f} küçüldü")
                            c3.metric("LZW",
                                      f"{comp['lzw']['bits']:,} bit",
                                      delta=f"%{(1-comp['lzw']['ratio'])*100:.1f} küçüldü")

                            best = "Huffman" if comp["best"] == "huffman" else "LZW"
                            st.success(f"✅ Bu metin için en iyi algoritma: **{best}**")

                            # Grafik
                            fig = go.Figure(go.Bar(
                                x=["Orijinal", "Huffman", "LZW"],
                                y=[comp["original_bits"],
                                   comp["huffman"]["total"],
                                   comp["lzw"]["bits"]],
                                marker_color=["#636EFA", "#EF553B", "#00CC96"],
                                text=[f"{v:,} bit" for v in [
                                    comp["original_bits"],
                                    comp["huffman"]["total"],
                                    comp["lzw"]["bits"]]],
                                textposition="outside",
                            ))
                            fig.update_layout(
                                title="OCR Metni Sıkıştırma Karşılaştırması",
                                yaxis_title="Bit", height=380, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Hata: {type(e).__name__}: {e}")
                        with st.expander("Detay"):
                            st.exception(e)
    else:
        st.info("Metin içeren bir görüntü yükle (taranmış belge, fatura, kitap sayfası vb.)")


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
