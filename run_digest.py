#!/usr/bin/env python3
"""Daily Tech Digest — entry point.

Usage:
    python3 run_digest.py               # normal run
    python3 run_digest.py --dry-run     # print articles, skip Notion write
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the same directory as this script
load_dotenv(Path(__file__).parent / ".env")

from fetcher import fetch_articles
from notion_writer import create_digest_page


def main():
    dry_run = "--dry-run" in sys.argv

    print("📡 RSS feed'leri taranıyor...")
    articles, total_scanned = fetch_articles(min_score=7, max_articles=10)

    print(f"\n🔍 Toplam {total_scanned} haber tarandı.")
    print(f"✅ {len(articles)} haber seçildi (puan ≥ 7):\n")

    for i, art in enumerate(articles, 1):
        print(f"  {i:2}. [{art.score}/10] {art.category} | {art.title[:80]}")
        print(f"      {art.url}\n")

    if dry_run:
        print("[DRY RUN] Notion'a yazılmadı.")
        return

    if not articles:
        print("Seçilen haber yok, Notion sayfası oluşturulmadı.")
        return

    print("\n📝 Notion'a yazılıyor...")
    page_url = create_digest_page(articles, total_scanned)
    print(f"\n🎉 Tamamlandı! Sayfa: {page_url}")


if __name__ == "__main__":
    main()
