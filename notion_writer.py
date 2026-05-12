"""Writes the daily strategic digest to Notion.

Two writes per run:
1. Articles database — one row per article (for feedback)
2. Daily Tech Digest database — one formatted summary page (for reading)
"""

import os
from datetime import datetime, timezone
from notion_client import Client
from analyzer import ScoredArticle


def _get_client() -> Client:
    token = os.environ.get("NOTION_TOKEN")
    if not token:
        raise RuntimeError("NOTION_TOKEN environment variable is not set.")
    return Client(auth=token)


def _get_digest_db_id() -> str:
    db_id = os.environ.get("NOTION_DATABASE_ID")
    if not db_id:
        raise RuntimeError("NOTION_DATABASE_ID environment variable is not set.")
    return db_id


def _get_articles_db_id() -> str:
    db_id = os.environ.get("NOTION_ARTICLES_DB_ID")
    if not db_id:
        raise RuntimeError("NOTION_ARTICLES_DB_ID environment variable is not set.")
    return db_id


# ── Block helpers ──────────────────────────────────────────────────────────────

def _p(text: str) -> dict:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": [{"type": "text", "text": {"content": text[:2000]}}]},
    }


def _p_rich(*segments) -> dict:
    rich = []
    for seg in segments:
        text, bold, code = seg if len(seg) == 3 else (*seg, False)
        rich.append({
            "type": "text",
            "text": {"content": text[:2000]},
            "annotations": {"bold": bold, "code": code},
        })
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": rich}}


def _h3(text: str) -> dict:
    return {
        "object": "block",
        "type": "heading_3",
        "heading_3": {"rich_text": [{"type": "text", "text": {"content": text[:2000]}}]},
    }


def _callout(text: str, emoji: str = "📌") -> dict:
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "icon": {"type": "emoji", "emoji": emoji},
            "rich_text": [{"type": "text", "text": {"content": text[:2000]}}],
        },
    }


def _divider() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


# ── Articles database writer ───────────────────────────────────────────────────

def _write_article_row(notion: Client, articles_db_id: str, art: ScoredArticle, today_iso: str):
    """Create one row in the Articles database for a single article."""
    notion.pages.create(
        parent={"database_id": articles_db_id},
        properties={
            "Name":              {"title": [{"type": "text", "text": {"content": art.title}}]},
            "URL":               {"url": art.url},
            "Date":              {"date": {"start": today_iso}},
            "Source":            {"rich_text": [{"type": "text", "text": {"content": art.source}}]},
            "Signal":            {"select": {"name": art.signal_level}},
            "Total Score":       {"number": round(art.total_score)},
            "Score A (AI)":      {"number": art.score_a},
            "Score B (Fintech)": {"number": art.score_b},
            "Score C (Funding)": {"number": art.score_c},
            "Score D (Crypto)":  {"number": art.score_d},
            "Score E (Tech)":    {"number": art.score_e},
            "Kesişim Bonusu":    {"checkbox": art.has_bonus},
            "Özet":              {"rich_text": [{"type": "text", "text": {"content": art.ozet[:2000]}}]},
            "Sektörel":          {"rich_text": [{"type": "text", "text": {"content": art.neden_onemli_sektorel[:2000]}}]},
            "Bankacılık":        {"rich_text": [{"type": "text", "text": {"content": art.neden_onemli_bankacilik[:2000]}}]},
            "Stratejik Çıkarım": {"rich_text": [{"type": "text", "text": {"content": art.stratejik_cikarim[:2000]}}]},
        },
    )


# ── Digest page writer ─────────────────────────────────────────────────────────

def _article_blocks(art: ScoredArticle) -> list[dict]:
    blocks = []
    blocks.append(_h3(art.title))

    pub = art.published.strftime("%d %b %Y") if art.published else "—"
    blocks.append(_p(f"Kaynak: {art.source}  |  Tarih: {pub}  |  🔗 {art.url}"))

    score_line = f"A:{art.score_a}  B:{art.score_b}  C:{art.score_c}  D:{art.score_d}  E:{art.score_e}"
    bonus_tag  = "  ⭐ Kesişim Bonusu" if art.has_bonus else ""
    total_line = f"Toplam: {art.total_score:.0f}  {art.signal_level}{bonus_tag}"
    blocks.append(_p_rich(
        (score_line, False, True),
        ("  →  ", False, False),
        (total_line, True, False),
    ))

    blocks.append(_p_rich(("Özet:", True, False)))
    blocks.append(_p(art.ozet or "(özet üretilemedi)"))

    blocks.append(_p_rich(("⚡ Neden Önemli", True, False)))
    blocks.append(_p_rich(("Sektörel: ", True, False), (art.neden_onemli_sektorel or "—", False, False)))
    blocks.append(_p_rich(("Bankacılık açısından: ", True, False), (art.neden_onemli_bankacilik or "—", False, False)))
    blocks.append(_callout(art.stratejik_cikarim or "—", "🎯"))
    blocks.append(_divider())
    return blocks


def _write_digest_page(notion: Client, digest_db_id: str, articles: list[ScoredArticle],
                       total_scanned: int, candidates: int, today: str) -> str:
    title = f"📰 Tech Digest — {today}"
    kritik = sum(1 for a in articles if "KRİTİK" in a.signal_level)
    yuksek = sum(1 for a in articles if "YÜKSEK" in a.signal_level)
    header_text = (
        f"🤖 Bugün {total_scanned} haber tarandı · {candidates} aday Claude'a gönderildi · "
        f"{len(articles)} haber seçildi   [🔴 {kritik} Kritik · 🟠 {yuksek} Yüksek · diğerleri Orta]"
    )

    all_blocks: list[dict] = [
        {"object": "block", "type": "quote",
         "quote": {"rich_text": [{"type": "text", "text": {"content": header_text}}]}},
        _divider(),
    ]
    for art in articles:
        all_blocks.extend(_article_blocks(art))

    MAX_BLOCKS = 98
    response = notion.pages.create(
        parent={"database_id": digest_db_id},
        properties={"title": {"title": [{"type": "text", "text": {"content": title}}]}},
        children=all_blocks[:MAX_BLOCKS],
    )
    page_id  = response["id"]
    page_url = response.get("url", "")

    remaining = all_blocks[MAX_BLOCKS:]
    for i in range(0, len(remaining), MAX_BLOCKS):
        notion.blocks.children.append(page_id, children=remaining[i:i + MAX_BLOCKS])

    return page_url


# ── Public entry point ─────────────────────────────────────────────────────────

def create_digest_page(articles: list[ScoredArticle], total_scanned: int, candidates: int) -> str:
    notion        = _get_client()
    digest_db_id  = _get_digest_db_id()
    articles_db_id = _get_articles_db_id()

    today     = datetime.now(timezone.utc).strftime("%d %B %Y")
    today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 1. Write each article as a row in Articles database
    print(f"   Articles tablosuna {len(articles)} satır yazılıyor...")
    for art in articles:
        try:
            _write_article_row(notion, articles_db_id, art, today_iso)
        except Exception as exc:
            print(f"   [WARN] Article row yazılamadı: {art.title[:50]} — {exc}")

    # 2. Write formatted digest page
    print("   Digest sayfası oluşturuluyor...")
    page_url = _write_digest_page(notion, digest_db_id, articles, total_scanned, candidates, today)

    print(f"[OK] Notion sayfası oluşturuldu: {page_url}")
    return page_url
