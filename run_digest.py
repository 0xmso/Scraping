#!/usr/bin/env python3
"""Daily Strategic Tech Digest — entry point.

Usage:
    python3 run_digest.py               # full run → Notion
    python3 run_digest.py --dry-run     # score & print, skip Notion write
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from fetcher import fetch_raw_articles
from analyzer import analyze_articles
from notion_writer import create_digest_page


def main():
    dry_run = "--dry-run" in sys.argv

    # ── 1. Fetch ──────────────────────────────────────────────────────────────
    print("📡 RSS feed'leri taranıyor...")
    raw_articles, total_scanned = fetch_raw_articles(max_candidates=40)
    print(f"   {total_scanned} haber tarandı · {len(raw_articles)} aday pre-filter'dan geçti\n")

    if not raw_articles:
        print("Hiç aday haber bulunamadı.")
        return

    # ── 2. Analyze ────────────────────────────────────────────────────────────
    print("🧠 Claude ile analiz ediliyor...")
    scored = analyze_articles(raw_articles, min_total=70, max_results=10)

    # ── 3. Report ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"✅ {len(scored)} haber seçildi (toplam skor ≥ 70):\n")
    for i, art in enumerate(scored, 1):
        bonus = " ⭐" if art.has_bonus else ""
        print(f"  {i:2}. {art.signal_level}{bonus} | {art.total_score:.0f}pts")
        print(f"      A:{art.score_a} B:{art.score_b} C:{art.score_c} "
              f"D:{art.score_d} E:{art.score_e}")
        print(f"      {art.title[:75]}")
        print(f"      {art.url}\n")

    if dry_run:
        print("[DRY RUN] Notion'a yazılmadı.")
        return

    if not scored:
        print("Eşik üzerinde haber yok, Notion sayfası oluşturulmadı.")
        return

    # ── 4. Write to Notion ────────────────────────────────────────────────────
    print("\n📝 Notion'a yazılıyor...")
    page_url = create_digest_page(scored, total_scanned, len(raw_articles))
    print(f"\n🎉 Tamamlandı! → {page_url}")


if __name__ == "__main__":
    main()
