"""Yeni modeli farkli orneklerle test et + yuzde karsilastirma."""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter
from core.nn_selector import predict
from core.bwt import bwt_rle_huffman_bits
from core.hybrid import lzw_compress_bits, build_huffman, huffman_bits

print("=" * 70)
print("1. MODEL BILGISI")
print("=" * 70)
with open("core/nn_model.pkl", "rb") as f:
    b = pickle.load(f)
print(f"Hold-out dogruluk : %{b.get('accuracy', 0)*100:.1f}")
print(f"CV dogruluk       : %{b.get('cv_mean', 0)*100:.1f} ± %{b.get('cv_std', 0)*100:.1f}")
print(f"Egitim ornek sayisi: {b.get('n_samples', 0)}")
print(f"Sinif dagilimi    : {b.get('dist', {})}")
print(f"Siniflar          : {b.get('classes')}")


def real_bits(text):
    """3 algoritmanin gercek bit sayilari."""
    if len(text) < 10:
        return None
    total = len(text)
    freq = {ch: c / total for ch, c in Counter(text).items()}
    codes = build_huffman(freq)
    return {
        "huffman": huffman_bits(text, codes) + len(codes) * 12,
        "lzw":     lzw_compress_bits(text),
        "bwt":     bwt_rle_huffman_bits(text),
    }


# ─────────────────────────────────────────────
# Cesitli test ornekleri (hocanin verecegi tipte)
# ─────────────────────────────────────────────
tests = {
    "Kisa Turkce cumle":  "Bugun hava cok guzel, disari cikip yuruyus yapacagim.",
    "Uzun Turkce yazi":   ("Yapay zeka, son yillarda hayatimizin her alanina giren bir teknolojidir. "
                           "Saglik, egitim, ulasim ve eglence sektorlerinde devrim yaratmaktadir. ") * 6,
    "Tekrarli ABC":       "ABCABCABC" * 50,
    "DNA dizilimi":       "ATCGATCGTTAACCGG" * 30,
    "Tek karakter":       "A" * 400,
    "Sayilar dizisi":     "1234567890" * 40,
    "JSON benzeri":       '{"id":1,"v":"x"},' * 50,
    "Log satirlari":      ("[INFO 2026-01-01] islem basarili\n" * 30),
    "Sozluk kelime":      "merhaba dunya selam evet hayir " * 40,
    "Karisik Ingilizce":  "The quick brown fox jumps over the lazy dog. " * 20,
    "Rastgele 100":       "x7#k2@9mPq!zL4&vR8nWa3Z*bC6dE5fH" * 5,
}
# Dosyalardan
for fn in ["turkce_dogal.txt", "sample.txt", "test_metni.txt", "large_turkish.txt"]:
    p = os.path.join("data", fn)
    if os.path.exists(p):
        with open(p) as f:
            tests[fn[:20]] = f.read()[:2500]

print()
print("=" * 70)
print("2. GENELLEME TESTI (gercek metinlerde tahmin vs gercek en iyi)")
print("=" * 70)
print(f"{'Ornek':<22} {'GERCEK':<8} {'NN':<8} {'OK':<5} {'Guven':<8} {'NN ratio':<10} {'En iyi ratio'}")
print("-" * 80)

correct = 0
total_n = 0
total_loss_pct = 0.0
for name, text in tests.items():
    if len(text) < 30:
        continue
    bits = real_bits(text)
    if bits is None:
        continue
    gercek = min(bits, key=bits.get)
    pred = predict(text)
    nn_algo = pred.get("algorithm", "?")

    orig_bits = len(text) * 8
    best_ratio = bits[gercek] / orig_bits
    nn_ratio = bits.get(nn_algo, bits["huffman"]) / orig_bits

    # NN secimiyle kac yuzde kaybediyoruz?
    loss_pct = (nn_ratio - best_ratio) / best_ratio * 100
    total_loss_pct += loss_pct

    ok = "[OK]" if gercek == nn_algo else "[X]"
    if gercek == nn_algo:
        correct += 1
    total_n += 1
    print(f"{name:<22} {gercek:<8} {nn_algo:<8} {ok:<5} %{pred.get('confidence',0)*100:<6.1f} "
          f"{nn_ratio:<10.4f} {best_ratio:.4f}")

print("-" * 80)
print(f"Genelleme dogrulugu: {correct}/{total_n} = %{correct/total_n*100:.1f}")
print(f"Ortalama kayip     : %{total_loss_pct/total_n:.2f} (NN secince en iyiden ne kadar uzakta)")
print()

# ─────────────────────────────────────────────
# Bit-bazli karsilastirma (yuzde gosterimi)
# ─────────────────────────────────────────────
print("=" * 70)
print("3. ALGORITMA BAZINDA SIKIŞTIRMA YUZDELERI")
print("=" * 70)
print(f"{'Ornek':<22} {'Orig (bit)':<12} {'Huffman':<14} {'LZW':<14} {'BWT':<14}")
print("-" * 80)

bwt_wins = 0
huff_wins = 0
lzw_wins = 0
for name, text in list(tests.items())[:15]:
    if len(text) < 30:
        continue
    bits = real_bits(text)
    if bits is None:
        continue
    orig = len(text) * 8
    best = min(bits, key=bits.get)
    if best == "bwt": bwt_wins += 1
    elif best == "lzw": lzw_wins += 1
    else: huff_wins += 1

    def cell(algo):
        b = bits[algo]
        pct = (1 - b / orig) * 100   # ne kadar kuculdu
        star = " <" if algo == best else "  "
        return f"%{pct:>5.1f}{star}"

    print(f"{name:<22} {orig:<12,} {cell('huffman'):<14} {cell('lzw'):<14} {cell('bwt'):<14}")

print("-" * 80)
print(f"Kazanma sayilari: BWT={bwt_wins}, Huffman={huff_wins}, LZW={lzw_wins}")
print("(% = orjinale gore kucuilme orani, yuksek = daha iyi)")
