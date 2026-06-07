"""
Kucuk Sinir Agi — Algoritma Secici (3 Sinifli: Huffman / LZW / BWT)
--------------------------------------------------------------------
Hocanin istedigi: "verinin turune gore gercek zamanli tahmin eden kucuk sinir agi"

Metin ozelliklerini (entropi, tekrar orani, karakter cesitliligi vb.) cikarir,
egitilmis MLP ile en iyi algoritmayi tahmin eder.

Egitim stratejisi (ezber yerine genelleme icin):
  1. Cesitli kaynaklardan veri (dogal dil, tekrarli, DNA, kod, log, rastgele)
  2. Veri augmentasyonu: ayni icerigi farkli uzunluklarda parcala
  3. Cross-validation ile gercek dogrulugu olc
  4. StandardScaler ile ozellik normalizasyonu
  5. Daha derin model: (32, 16, 8)
  6. Erken durdurma + L2 regularizasyon ile overfitting onleme
"""

import os
import math
import json
import pickle
import random
import numpy as np
from collections import Counter

MODEL_FILE  = os.path.join(os.path.dirname(__file__), "nn_model.pkl")
CLASS_NAMES = ["huffman", "lzw", "bwt"]

# ─────────────────────────────────────────
# Ozellik cikarma (10 ozellik)
# ─────────────────────────────────────────

def extract_features(text: str) -> list:
    """Metinden 11 ozellik cikar."""
    if not text or len(text) < 2:
        return [0.0] * 11

    total = len(text)
    freq = Counter(text)
    n_unique = len(freq)

    # 1. Shannon entropisi
    entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())

    # 2. Benzersiz karakter orani (norm)
    unique_ratio = n_unique / total

    # 3. En sik 3 karakterin toplam orani
    top3_ratio = sum(c for _, c in freq.most_common(3)) / total

    # 4. Bosluk orani (dogal dil gostergesi)
    space_ratio = freq.get(' ', 0) / total

    # 5. Turkce ozel karakter orani
    tr_chars = set('sgucoiSGUCOI' + 'şğüöçıŞĞÜÖÇİ')
    tr_ratio = sum(freq.get(c, 0) for c in tr_chars) / total

    # 6. Ortalama calisma uzunlugu (run-length) — BWT/RLE icin kritik
    runs = 1
    for i in range(1, total):
        if text[i] != text[i-1]:
            runs += 1
    avg_run = total / runs

    # 7. Rakam orani
    digit_ratio = sum(1 for c in text if c.isdigit()) / total

    # 8. Buyuk harf orani
    upper_ratio = sum(1 for c in text if c.isupper()) / total

    # 9. Bigram entropisi (yapısal düzen göstergesi — LZW için kritik)
    bigram_freq = Counter()
    for i in range(total - 1):
        bigram_freq[text[i:i+2]] += 1
    bg_total = sum(bigram_freq.values()) or 1
    bg_entropy = -sum((c / bg_total) * math.log2(c / bg_total)
                      for c in bigram_freq.values())

    # 10. Maks koşu uzunluğu (BWT'nin kazanç göstergesi)
    max_run = 1
    cur = 1
    for i in range(1, total):
        if text[i] == text[i-1]:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 1
    max_run_norm = max_run / total

    # 11. Alfabe boyutu (log-normalize) — kucuk alfabe BWT'yi kazandirir
    alphabet_size_log = math.log2(max(n_unique, 2))

    return [entropy, unique_ratio, top3_ratio, space_ratio,
            tr_ratio, avg_run, digit_ratio, upper_ratio,
            bg_entropy, max_run_norm, alphabet_size_log]


FEATURE_NAMES = [
    "entropi", "benzersiz_oran", "top3_oran", "boşluk_oran",
    "türkçe_oran", "ort_çalışma", "rakam_oran", "büyük_oran",
    "bigram_entropi", "max_koşu_oran", "alfabe_boyutu_log",
]


# ─────────────────────────────────────────
# Sentetik veri uretici (cesitlilik icin)
# ─────────────────────────────────────────

def _synthetic_samples(n_per_type: int = 80, seed: int = 42) -> list:
    """Farkli veri tiplerinden sentetik ornek uret (cesitlilik icin bol)."""
    rng = random.Random(seed)
    samples = []

    # 1. Tekrarli desenler — BWT/RLE icin ideal (genis aralik)
    for _ in range(n_per_type * 2):  # daha fazla
        n = rng.randint(2, 10)
        pat = "".join(rng.choice("abcdefghijkABC123") for _ in range(n))
        reps = rng.randint(15, 200)  # 30 ile 2000 char arasi
        samples.append(pat * reps)

    # 2. Tek karakter uzun blok — BWT icin ideal
    for _ in range(n_per_type // 2):
        ch = rng.choice("abc123 ")
        samples.append(ch * rng.randint(80, 500))

    # 3. KUCUK alfabe (DNA / ikili / 3-4 sembol) — BWT'nin gizli kazandigi yer
    #    Burada uzun kosu yok ama alfabe kucuk -> BWT cok kazanir
    for _ in range(n_per_type * 3):  # cok bol
        n = rng.randint(80, 2000)   # kisa-orta-uzun karisik
        alpha_size = rng.choice([2, 2, 3, 3, 4, 4, 4, 5])
        if alpha_size == 2:
            alphabet = rng.choice(["01", "AT", "ab", "CG"])
        elif alpha_size <= 4:
            alphabet = "ATCG"[:alpha_size]
        else:
            alphabet = "ACGTU"
        samples.append("".join(rng.choice(alphabet) for _ in range(n)))

    # 3b. KUCUK alfabeli kisa tekrarli motifler (ATCGATCG, ABCABC tipi)
    for _ in range(n_per_type * 2):
        motif_len = rng.randint(3, 12)
        alpha = rng.choice(["ATCG", "ABC", "abcd", "01", "abcde", "AB12"])
        motif = "".join(rng.choice(alpha) for _ in range(motif_len))
        reps = rng.randint(10, 200)
        samples.append(motif * reps)

    # 4. Rastgele yuksek-entropi — Huffman icin ideal (BWT/LZW kotu)
    for _ in range(n_per_type):
        n = rng.randint(100, 800)
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?@#$%"
        samples.append("".join(rng.choice(alphabet) for _ in range(n)))

    # 4b. KISA rastgele metin (Huffman'in en cok kazandigi yer — overhead az)
    for _ in range(n_per_type):
        n = rng.randint(30, 120)
        alphabet = "abcdefghijklmnopqrstuvwxyz "
        samples.append("".join(rng.choice(alphabet) for _ in range(n)))

    # 5. Tekrarli kelimeler — LZW'nin guclu yani
    for _ in range(n_per_type * 2):
        words = ["merhaba", "dunya", "selam", "evet", "hayir", "the", "and", "for",
                 "veri", "sikistirma", "algoritma", "model", "yapay", "zeka",
                 "bilgisayar", "muhendislik", "proje", "donem", "okul"]
        chosen = [rng.choice(words) for _ in range(rng.randint(20, 80))]
        samples.append(" ".join(chosen))

    # 6. JSON benzeri yapilandirilmis
    for _ in range(n_per_type):
        items = []
        for _ in range(rng.randint(5, 25)):
            items.append(f'{{"id":{rng.randint(1,99)},"v":"x"}}')
        samples.append(",".join(items))

    # 7. Log benzeri
    for _ in range(n_per_type):
        lines = []
        for _ in range(rng.randint(5, 30)):
            t = rng.randint(0, 99)
            lines.append(f"[INFO 2026-01-{t:02d}] ok")
        samples.append("\n".join(lines))

    # 8. KISA Turkce/Ingilizce cumleler (overhead-hassas — Huffman icin)
    short_phrases = [
        "Bugun hava cok guzel.", "Yarin sinav var.",
        "The quick brown fox.", "Hello world this is a test.",
        "Veri sikistirma onemli bir konudur.",
        "Algoritma performansi olcer.",
        "Merhaba arkadaslar nasilsiniz",
        "Bu cumle kisa bir test icin yazildi.",
    ]
    for _ in range(n_per_type):
        n_rep = rng.randint(1, 6)
        phrase = rng.choice(short_phrases)
        samples.append(" ".join([phrase] * n_rep))

    # 9. Karisik Turkce paragraflar (cesitli uzunluklarda) — BWT icin
    turkce_paragraf = (
        "Yapay zeka teknolojileri hizla gelisiyor. Makine ogrenmesi modelleri "
        "verilerden ornek alir ve tahmin yapar. Sinir aglari karmasik desenleri "
        "yakalamada cok basarilidir. Sikistirma algoritmalari bilgi teorisinin "
        "temel uygulamalarindandir. Huffman ve LZW klasik yontemlerdendir. "
    )
    for _ in range(n_per_type):
        n_rep = rng.randint(1, 8)
        samples.append(turkce_paragraf * n_rep)

    # 10. AKADEMIK metin (hocanin verecegi tipte)
    akademik = (
        "Veri sikistirma, bilgisayar bilimleri alanindaki temel arastirma "
        "konularindan biridir. Kayipsiz sikistirma algoritmalari orjinal "
        "verinin tam olarak geri kazanilmasini saglar. Bu durum ozellikle "
        "metin dosyalari, programlar ve veritabanlari icin onem tasir. "
        "Huffman kodlamasi 1952 yilinda Davide Huffman tarafindan "
        "gelistirilmistir. LZW algoritmasi ise 1984 yilinda Welch tarafindan "
        "yayimlanmistir. Burrows-Wheeler donusumu 1994 yilinda gelistirilen "
        "modern bir yaklasimdir ve bzip2 programinin temelini olusturur. "
    )
    for _ in range(n_per_type):
        n_rep = rng.randint(1, 6)
        samples.append(akademik * n_rep)

    # 11. HABER tarzi (formal Turkce)
    haber = (
        "Istanbul Teknik Universitesi arastirmacilari yeni bir sikistirma "
        "algoritmasi gelistirdi. Calisma, uluslararasi bir konferansta sunuldu. "
        "Profesor Ahmet Yilmaz konuya iliskin aciklamalar yaptı. "
        "Yapilan testler, yeni yontemin gzip programindan yuzde otuz daha iyi "
        "performans gosterdigini ortaya koydu. Sonuçlar yakinda dergide yayimlanacak. "
    )
    for _ in range(n_per_type):
        n_rep = rng.randint(1, 5)
        samples.append(haber * n_rep)

    # 12. KOD parcalari (Python/Java tarzi)
    kod = (
        "def encode(text):\n    result = []\n    for ch in text:\n"
        "        result.append(codes[ch])\n    return result\n\n"
        "def decode(bits):\n    output = ''\n    node = root\n"
        "    for bit in bits:\n        node = node.left if bit == '0' else node.right\n"
        "    return output\n"
    )
    for _ in range(n_per_type):
        n_rep = rng.randint(1, 4)
        samples.append(kod * n_rep)

    # 13. SIIR/EDEBI metin (farkli noktalama)
    siir = (
        "Sonbahar geldi yine, yapraklar dokuldu,\n"
        "Ruzgarda salinan dallar uzuldu.\n"
        "Mavi gokyuzunde bulutlar yarista,\n"
        "Kuslar uzak yerlere gocmek istiyor.\n"
    )
    for _ in range(n_per_type):
        n_rep = rng.randint(1, 6)
        samples.append(siir * n_rep)

    # 14. ORTA UZUNLUKTA cumle dizileri (en sik karsilasilan tip)
    cumleler = [
        "Bu bir ornek cumle.", "Veri yapilari onemlidir.",
        "Algoritmalar bilgisayar biliminin temelidir.",
        "Sinir aglari karmasik problemleri cozer.",
        "Sikistirma orani performansi belirler.",
        "Bilgi teorisi Shannon tarafindan gelistirilmistir.",
        "Huffman agaci frekanslara gore olusur.",
        "LZW sozluk tabanli calisir.",
        "BWT permutasyon yapar.",
        "RLE tekrarli karakterleri sayar.",
    ]
    for _ in range(n_per_type * 2):
        n_c = rng.randint(3, 30)
        text = " ".join(rng.choice(cumleler) for _ in range(n_c))
        samples.append(text)

    return samples


def _file_chunks(path: str, chunk_sizes: list = None) -> list:
    """Bir corpus dosyasini farkli boyutlarda parcala (augmentasyon)."""
    if chunk_sizes is None:
        chunk_sizes = [300, 600, 1000, 1500]
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = []
    for cs in chunk_sizes:
        # %50 atlamali kayan pencere — daha cok ornek
        step = max(cs // 2, 50)
        for i in range(0, max(len(text) - cs, 0), step):
            chunks.append(text[i:i + cs])
    return chunks


# ─────────────────────────────────────────
# Etiket (gercek en iyi algoritma)
# ─────────────────────────────────────────

def _true_label(chunk: str) -> int:
    """3 algoritmayi calistirip kazanani dondurur. 0=Huffman, 1=LZW, 2=BWT."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.hybrid import lzw_compress_bits, build_huffman, huffman_bits
    from core.bwt import bwt_rle_huffman_bits

    if len(chunk) < 20:
        return 0

    total = len(chunk)
    freq = {ch: c / total for ch, c in Counter(chunk).items()}
    codes = build_huffman(freq)

    huff_bits = huffman_bits(chunk, codes) + len(codes) * 12
    lzw_bits  = lzw_compress_bits(chunk)
    try:
        bwt_bits = bwt_rle_huffman_bits(chunk)
    except Exception:
        bwt_bits = float("inf")

    opts = [huff_bits, lzw_bits, bwt_bits]
    return int(opts.index(min(opts)))


# ─────────────────────────────────────────
# Egitim verisi olusturma (cok kaynakli)
# ─────────────────────────────────────────

def build_dataset(verbose: bool = True) -> tuple:
    """Cok kaynakli, dengeli egitim seti olustur."""
    chunks = []

    # 1. Sentetik orneklerden
    chunks.extend(_synthetic_samples(n_per_type=40))

    # 2. Corpus dosyalarindan parcalar
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    for fn in ["large_turkish.txt", "diverse_corpus.txt",
               "turkce_dogal.txt", "sample.txt"]:
        chunks.extend(_file_chunks(os.path.join(data_dir, fn)))

    if verbose:
        print(f"  Toplam aday: {len(chunks)} parça")

    X, y = [], []
    for ch in chunks:
        if len(ch) < 30:
            continue
        try:
            feats = extract_features(ch)
            label = _true_label(ch)
            X.append(feats)
            y.append(label)
        except Exception:
            continue

    return np.array(X), np.array(y)


# ─────────────────────────────────────────
# Model egit (cross-validation ile gercek dogruluk)
# ─────────────────────────────────────────

def train(corpus_path: str = None, verbose: bool = True) -> dict:
    """3-sinifli, cross-validation'lu egitim. corpus_path parametresi geriye uyumluluk icin."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    if verbose:
        print("📦 Egitim verisi hazirlaniyor...")
    X, y = build_dataset(verbose=verbose)

    if len(X) < 30:
        raise RuntimeError(f"Yetersiz veri: {len(X)} ornek")

    cls_dist = {CLASS_NAMES[i]: int((y == i).sum()) for i in range(3)}
    if verbose:
        print(f"  Toplam: {len(X)} ornek | Dagilim: {cls_dist}")

    # Hold-out test seti (modelin hic gormedigi)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if min(cls_dist.values()) >= 2 else None,
    )

    # ── Cross-validation (gercek dogrulugu olc) ──
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    if verbose:
        print("\n🔁 5-fold cross-validation...")
    for fold, (tr, va) in enumerate(skf.split(X_tv, y_tv), 1):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_tv[tr])
        Xva = scaler.transform(X_tv[va])
        m = MLPClassifier(
            hidden_layer_sizes=(32, 16, 8),
            activation="relu",
            alpha=1e-3,                       # L2 regularizasyon
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        )
        m.fit(Xtr, y_tv[tr])
        sc = accuracy_score(y_tv[va], m.predict(Xva))
        cv_scores.append(sc)
        if verbose:
            print(f"  Fold {fold}: %{sc*100:.1f}")

    cv_mean = float(np.mean(cv_scores))
    cv_std  = float(np.std(cv_scores))
    if verbose:
        print(f"\n  CV ortalama: %{cv_mean*100:.1f} ± %{cv_std*100:.1f}")

    # ── Tum train+val verisiyle final modeli egit ──
    scaler = StandardScaler()
    X_tv_s = scaler.fit_transform(X_tv)
    X_test_s = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(32, 16, 8),
        activation="relu",
        alpha=1e-3,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
    )
    model.fit(X_tv_s, y_tv)

    # Hold-out test (modelin hic gormedigi)
    y_pred = model.predict(X_test_s)
    test_acc = accuracy_score(y_test, y_pred)
    if verbose:
        print(f"\n🎯 Hold-out test dogrulugu: %{test_acc*100:.1f}")
        print("\nSınıf bazli rapor:")
        print(classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, zero_division=0,
        ))
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        print("Karisiklik matrisi (satir=gercek, sutun=tahmin):")
        print(f"          {'Huff':>6} {'LZW':>6} {'BWT':>6}")
        for i, n in enumerate(CLASS_NAMES):
            print(f"  {n:<8} {cm[i][0]:>6} {cm[i][1]:>6} {cm[i][2]:>6}")

    # ── Kaydet ──
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({
            "model":     model,
            "scaler":    scaler,
            "accuracy":  test_acc,
            "cv_mean":   cv_mean,
            "cv_std":    cv_std,
            "classes":   CLASS_NAMES,
            "n_samples": len(X),
            "dist":      cls_dist,
        }, f)

    return {
        "test_accuracy": test_acc,
        "cv_accuracy":   cv_mean,
        "cv_std":        cv_std,
        "samples":       len(X),
        "distribution":  cls_dist,
    }


# ─────────────────────────────────────────
# Tahmin
# ─────────────────────────────────────────

def confusion_matrix_data() -> dict:
    """
    Hold-out test setindeki confusion matrix'i döndürür.

    Returns:
        {"matrix": 3x3 list, "classes": [...], "accuracy": float}
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, accuracy_score

    if not os.path.exists(MODEL_FILE):
        return {"error": "Model henuz egitilmemis"}

    with open(MODEL_FILE, "rb") as f:
        bundle = pickle.load(f)

    X, y = build_dataset(verbose=False)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if min(np.bincount(y)) >= 2 else None,
    )
    X_test_s = bundle["scaler"].transform(X_test)
    y_pred = bundle["model"].predict(X_test_s)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    acc = accuracy_score(y_test, y_pred)

    return {
        "matrix":   cm.tolist(),
        "classes":  CLASS_NAMES,
        "accuracy": float(acc),
        "n_test":   len(y_test),
    }


def feature_importance(n_repeats: int = 5) -> dict:
    """
    Permutation importance: Her özelliği rastgele karıştırıp doğruluk düşüşünü ölç.

    Düşüş büyükse → özellik kararı belirleyici.
    Düşüş küçükse → özellik az katkı sağlıyor.

    Returns:
        {"feature_names": [...], "importances": [...], "stds": [...]}
    """
    import numpy as np
    from sklearn.inspection import permutation_importance

    if not os.path.exists(MODEL_FILE):
        return {"error": "Model henuz egitilmemis"}

    with open(MODEL_FILE, "rb") as f:
        bundle = pickle.load(f)

    # Test verisi hazırla
    X, y = build_dataset(verbose=False)
    if len(X) == 0:
        return {"error": "Veri olusturulamadi"}

    Xs = bundle["scaler"].transform(X)
    r = permutation_importance(
        bundle["model"], Xs, y,
        n_repeats=n_repeats, random_state=42, n_jobs=1,
    )

    # Sırala — büyükten küçüğe
    idx = np.argsort(r.importances_mean)[::-1]
    return {
        "feature_names": [FEATURE_NAMES[i] for i in idx],
        "importances":   [float(r.importances_mean[i]) for i in idx],
        "stds":          [float(r.importances_std[i])  for i in idx],
    }


def predict(text: str) -> dict:
    # Model yoksa veya bozuksa (sklearn/numpy versiyon farkı) otomatik egit
    if not os.path.exists(MODEL_FILE):
        try:
            train(verbose=False)
        except Exception as e:
            return {"algorithm": "huffman", "confidence": 0.5,
                    "error": f"Model egitilemedi: {e}", "trained": False}

    try:
        with open(MODEL_FILE, "rb") as f:
            bundle = pickle.load(f)
    except Exception:
        # Versiyon uyumsuzlugu — sil, yeniden egit
        try:
            os.remove(MODEL_FILE)
        except Exception:
            pass
        try:
            train(verbose=False)
            with open(MODEL_FILE, "rb") as f:
                bundle = pickle.load(f)
        except Exception as e:
            return {"algorithm": "huffman", "confidence": 0.5,
                    "error": f"Model yuklenemedi: {e}", "trained": False}

    model   = bundle["model"]
    scaler  = bundle["scaler"]
    classes = bundle.get("classes", ["huffman", "lzw"])

    feats = extract_features(text)
    feats_s = scaler.transform(np.array([feats]))

    pred  = model.predict(feats_s)[0]
    proba = model.predict_proba(feats_s)[0]
    algo  = classes[int(pred)] if int(pred) < len(classes) else "huffman"

    probs = {classes[i]: float(proba[i]) for i in range(len(classes))}

    return {
        "algorithm":      algo,
        "confidence":     float(max(proba)),
        "probabilities":  probs,
        "features":       dict(zip(FEATURE_NAMES, [round(f, 4) for f in feats])),
        "trained":        True,
        "model_accuracy": bundle.get("accuracy", 0),
        "cv_accuracy":    bundle.get("cv_mean", 0),
        "n_samples":      bundle.get("n_samples", 0),
    }


# ─────────────────────────────────────────
# CLI: python -m core.nn_selector
# ─────────────────────────────────────────

if __name__ == "__main__":
    train(verbose=True)
