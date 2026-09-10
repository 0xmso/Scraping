#!/usr/bin/env python3
"""Daily Strategic Tech Digest — entry point.

Usage:
    python3 run_digest.py               # full run → Notion
    python3 run_digest.py --dry-run     # score & print, skip Notion write

Exits non-zero when the run finished but produced a suspiciously thin digest,
or when a feed went dead — those are silent-degradation modes that otherwise
look like success, so the workflow surfaces them as a failure issue.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import llm
from fetcher import fetch_raw_articles
from dedup import filter_new_articles
from analyzer import analyze_articles
from notion_writer import create_digest_page

MIN_TOTAL = 55
MAX_RESULTS = 10
# Below this the digest isn't worth reading — usually a miscalibrated prompt or
# threshold rather than a genuinely quiet news day.
MIN_HEALTHY_RESULTS = 3


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    warnings: list[str] = []

    # ── 1. Fetch ──────────────────────────────────────────────────────────────
    print("📡 RSS feed'leri taranıyor...")
    raw_articles, total_scanned, dead_feeds = fetch_raw_articles(max_candidates=40)
    print(f"   {total_scanned} haber tarandı · {len(raw_articles)} aday pre-filter'dan geçti")
    if dead_feeds:
        warnings.append(f"{len(dead_feeds)} feed hiç haber döndürmedi: {', '.join(dead_feeds)}")
        print(f"   ⚠️  {len(dead_feeds)} feed ölü görünüyor")
    print()

    if not raw_articles:
        print("Hiç aday haber bulunamadı.")
        return 1

    # ── 1.5. Tekrar kontrolü — geçmişte gönderilmiş haberleri ele ─────────────
    print("🔁 Geçmişte gönderilen haberler kontrol ediliyor...")
    raw_articles, skipped = filter_new_articles(raw_articles)
    print(f"   {skipped} tekrar elendi · {len(raw_articles)} yeni aday kaldı\n")

    if not raw_articles:
        print("Tüm adaylar daha önce gönderilmiş — yeni haber yok.")
        return 0

    # ── 2. Analyze ────────────────────────────────────────────────────────────
    print("🧠 Claude ile analiz ediliyor...")
    scored = analyze_articles(
        raw_articles, min_total=MIN_TOTAL, max_results=MAX_RESULTS
    )

    # ── 3. Report ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"✅ {len(scored)} haber seçildi (toplam skor ≥ {MIN_TOTAL}):\n")
    for i, art in enumerate(scored, 1):
        bonus = " ⭐" if art.has_bonus else ""
        print(f"  {i:2}. {art.signal_level}{bonus} | {art.total_score:.0f}pts")
        print(f"      A:{art.score_a} B:{art.score_b} C:{art.score_c} "
              f"D:{art.score_d} E:{art.score_e}")
        print(f"      {art.title[:75]}")
        print(f"      {art.url}\n")

    print(f"💰 {llm.usage_summary()}")

    if len(scored) < MIN_HEALTHY_RESULTS:
        warnings.append(
            f"Yalnızca {len(scored)} haber seçildi (beklenen ≥ {MIN_HEALTHY_RESULTS}) "
            f"— prompt veya eşik kalibrasyonu bozulmuş olabilir"
        )

    if dry_run:
        print("[DRY RUN] Notion'a yazılmadı.")
        return _finish(warnings)

    if not scored:
        print("Eşik üzerinde haber yok, Notion sayfası oluşturulmadı.")
        return _finish(warnings)

    # ── 4. Write to Notion ────────────────────────────────────────────────────
    print("\n📝 Notion'a yazılıyor...")
    page_url = create_digest_page(scored, total_scanned, len(raw_articles))
    print(f"\n🎉 Tamamlandı! → {page_url}")

    return _finish(warnings)


def _finish(warnings: list[str]) -> int:
    """Report soft failures. The digest itself is already written — these are
    degradation signals worth a failure issue, not reasons to discard the run.
    """
    if not warnings:
        return 0
    print(f"\n{'─'*70}")
    print("⚠️  Dikkat gerektiren durumlar:")
    for w in warnings:
        print(f"   • {w}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
