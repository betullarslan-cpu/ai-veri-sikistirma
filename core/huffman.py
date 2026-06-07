"""
Huffman Kodlaması — David Huffman (1952)
==========================================

Huffman kodlaması, **prefix-free** (önek-serbest) ikili kod tablosu üreten
optimal bir kayıpsız sıkıştırma algoritmasıdır.

ALGORİTMA MANTIĞI
-----------------
1. Karakter frekansları hesaplanır
2. Her karakter bir yaprak düğüm haline gelir (öncelik kuyruğuna eklenir)
3. En düşük frekanslı iki düğüm birleştirilip yeni bir iç düğüm oluşturulur
4. Bu işlem 1 ağaç kalana kadar tekrarlanır
5. Ağaçtaki yola göre 0/1 kodları atanır

TEORİK ÖZELLİKLER
-----------------
- **Optimal:** Sembol bazlı kodlamalar arasında en az bit kullanan
- **Prefix-free:** Hiçbir kod başka kodun öneki değildir → kodlar ayrıştırılabilir
- **Sıkıştırma oranı:** Shannon entropisine yakın (kesirli bit olmadığı için tam değil)
- **Karmaşıklık:** O(n log n) — heap işlemleri

REFERANS
--------
Huffman, D. A. (1952). "A Method for the Construction of Minimum-Redundancy
Codes." Proceedings of the IRE, 40(9), 1098–1101.
"""

import heapq
from collections import Counter


class HuffmanNode:
    """Huffman ağacı için bir düğüm.

    Attributes:
        char (str|None):  Yaprak düğümde karakter, iç düğümde None
        freq (int|float): Karakterin frekansı (yaprak) veya alt ağaçların toplam frekansı
        left, right:      Sol ve sağ alt düğümler (iç düğümde dolu, yaprakta None)
    """
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        """Heapq için karşılaştırma — frekans bazlı sıralama."""
        return self.freq < other.freq


def build_tree(text: str) -> HuffmanNode:
    """Metinden Huffman ağacı kur.

    Adımlar:
        1. Karakter frekanslarını say
        2. Her karakter için yaprak düğüm yarat, öncelik kuyruğuna ekle
        3. En küçük 2 düğümü birleştir → yeni iç düğüm
        4. Tek düğüm kalana kadar tekrarla (köke ulaşana kadar)

    Args:
        text: Sıkıştırılacak ham metin

    Returns:
        Huffman ağacının kök düğümü
    """
    freq = Counter(text)
    heap = [HuffmanNode(ch, f) for ch, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        # En düşük 2 frekans
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        # Birleştir
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]


def build_codes(node: HuffmanNode, prefix: str = "", codes: dict = None) -> dict:
    """Ağaçtan {karakter: bit dizisi} sözlüğü çıkar.

    Sol dala '0', sağ dala '1' ekleyerek köke kadar gider.
    Yapraklarda biriken bit dizisi, o karakterin kodudur.

    Args:
        node:   Mevcut düğüm (başlangıçta kök)
        prefix: Şu ana kadar biriken bit dizisi
        codes:  Sonuçların eklendiği sözlük (recursive paylaşım için)

    Returns:
        {karakter: "0101..." bit dizisi} formatında sözlük
    """
    if codes is None:
        codes = {}
    if node.char is not None:
        # Yaprak: karaktere atanmış kod
        # Tek karakterlik metin için "0" varsayılan (boş prefix yerine)
        codes[node.char] = prefix or "0"
    else:
        build_codes(node.left, prefix + "0", codes)
        build_codes(node.right, prefix + "1", codes)
    return codes


def encode(text: str) -> tuple:
    """Metni Huffman ile sıkıştır.

    Args:
        text: Sıkıştırılacak metin

    Returns:
        (bit_dizisi: str, kod_tablosu: dict)
        - bit_dizisi: "01001011..." biçiminde 0/1 stringi
        - kod_tablosu: Decode için gerekli {karakter: kod} sözlüğü
    """
    root = build_tree(text)
    codes = build_codes(root)
    encoded_bits = "".join(codes[ch] for ch in text)
    return encoded_bits, codes


def decode(encoded_bits: str, codes: dict) -> str:
    """Huffman ile sıkıştırılmış bit dizisini orijinal metne çevir.

    KAYIPSIZLIK GARANTİSİ:
    Huffman kodları prefix-free olduğu için, bit dizisi tek bir şekilde
    çözümlenebilir. Bu yüzden decode tamamen deterministiktir.

    Args:
        encoded_bits: Encode'dan çıkan 0/1 stringi
        codes:        Encode'dan çıkan {karakter: kod} sözlüğü

    Returns:
        Orijinal metin (kayıpsız)
    """
    # Ters tablo: kod → karakter (decode için)
    reverse_codes = {v: k for k, v in codes.items()}
    result = []
    buf = ""
    # Bit-bit oku, tampona ekle, tablo eşleşmesi olursa karaktere çevir
    for bit in encoded_bits:
        buf += bit
        if buf in reverse_codes:
            result.append(reverse_codes[buf])
            buf = ""  # Tamponu temizle, sonraki karaktere geç
    return "".join(result)


def compression_ratio(original_text: str, encoded_bits: str) -> tuple:
    """Sıkıştırma oranını hesapla.

    Args:
        original_text: Orijinal metin
        encoded_bits:  Encode edilmiş bit dizisi

    Returns:
        (ratio, original_bits, compressed_bits)
        ratio < 1 → sıkıştırma başarılı
    """
    original_bits = len(original_text) * 8  # UTF-8 8-bit varsayımı
    compressed_bits = len(encoded_bits)
    ratio = compressed_bits / original_bits
    return ratio, original_bits, compressed_bits


if __name__ == "__main__":
    sample = "bu bir veri sikistirma projesidir ve huffman algoritmasini test ediyoruz"
    bits, codes = encode(sample)
    ratio, orig, comp = compression_ratio(sample, bits)

    print(f"Orijinal: {orig} bit")
    print(f"Sıkıştırılmış: {comp} bit")
    print(f"Sıkıştırma oranı: {ratio:.3f}  ({(1-ratio)*100:.1f}% küçüldü)")
    print(f"\nKarakter kodları örneği:")
    for ch, code in sorted(codes.items(), key=lambda x: len(x[1])):
        print(f"  '{ch}': {code}")
