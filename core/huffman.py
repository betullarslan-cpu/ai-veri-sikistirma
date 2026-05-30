"""
Huffman Coding - Temel implementasyon
Proje: Veri Sıkıştırma Dönem Projesi
"""

import heapq
from collections import Counter


class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_tree(text):
    freq = Counter(text)
    heap = [HuffmanNode(ch, f) for ch, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]


def build_codes(node, prefix="", codes=None):
    if codes is None:
        codes = {}
    if node.char is not None:
        codes[node.char] = prefix or "0"
    else:
        build_codes(node.left, prefix + "0", codes)
        build_codes(node.right, prefix + "1", codes)
    return codes


def encode(text):
    root = build_tree(text)
    codes = build_codes(root)
    encoded_bits = "".join(codes[ch] for ch in text)
    return encoded_bits, codes


def compression_ratio(original_text, encoded_bits):
    original_bits = len(original_text) * 8
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
