"""
ai_diary.json dosyasını okunabilir tablo formatına çevirir.
Öğretmene teslim edilecek AI Günlüğü tablosunu üretir.
"""

import json
import os


def export(diary_file="ai_diary.json", out_file="AI_Gunlugu.txt"):
    if not os.path.exists(diary_file):
        print("Önce ai_optimizer.py çalıştır, günlük oluşsun.")
        return

    with open(diary_file, "r", encoding="utf-8") as f:
        diary = json.load(f)

    lines = []
    lines.append("=" * 90)
    lines.append("  AI GÜNLÜĞÜ — Context-Aware Huffman Projesi")
    lines.append("=" * 90)
    lines.append("")

    header = f"{'Adım':<5} {'Hedeflenen İşlem':<30} {'Verilen Prompt (özet)':<35} {'AI Cevabı / Sorun':<25} {'Sonuç'}"
    lines.append(header)
    lines.append("-" * 130)

    for s in diary["adimlar"]:
        prompt_short = s["verilen_prompt"][:32].replace("\n", " ") + "..."
        cevap_short  = s["ai_cevabi"][:22].replace("\n", " ") + "..."
        sorun_short  = s["sorun"][:22] if s["sorun"] else "Yok"
        sonuc_short  = s["sonuc"][:30].replace("\n", " ")

        lines.append(
            f"{s['adim']:<5} {s['hedeflenen_islem'][:28]:<30} "
            f"{prompt_short:<35} {cevap_short + ' / ' + sorun_short:<25} {sonuc_short}"
        )

    lines.append("")
    lines.append(f"Toplam token kullanımı: {diary['toplam_token']}")
    lines.append("=" * 90)

    output = "\n".join(lines)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\n→ {out_file} dosyasına kaydedildi.")


if __name__ == "__main__":
    export()
