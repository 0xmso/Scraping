"""Writes the daily strategic digest to a Notion database."""

import os
from datetime import datetime, timezone
from notion_client import Client
from analyzer import ScoredArticle


def _get_client() -> Client:
    token = os.environ.get("NOTION_TOKEN")
    if not token:
        raise RuntimeError("NOTION_TOKEN environment variable is not set.")
    return Client(auth=token)


def _get_database_id() -> str:
    db_id = os.environ.get("NOTION_DATABASE_ID")
    if not db_id:
        raise RuntimeError("NOTION_DATABASE_ID environment variable is not set.")
    return db_id


def _p(text: str) -> dict:
    """Plain paragraph block."""
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]},
    }


def _p_rich(*segments) -> dict:
    """Paragraph block with multiple rich-text segments.
    Each segment: (text, bold=False, code=False)
    """
    rich = []
    for seg in segments:
        text, bold, code = seg if len(seg) == 3 else (*seg, False)
        rich.append({
            "type": "text",
            "text": {"content": text},
            "annotations": {"bold": bold, "code": code},
        })
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": rich},
    }


def _h3(text: str) -> dict:
    return {
        "object": "block",
        "type": "heading_3",
        "heading_3": {"rich_text": [{"type": "text", "text": {"content": text}}]},
    }


def _callout(text: str, emoji: str = "📌") -> dict:
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "icon": {"type": "emoji", "emoji": emoji},
            "rich_text": [{"type": "text", "text": {"content": text}}],
        },
    }


def _divider() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


def _article_blocks(art: ScoredArticle) -> list[dict]:
    blocks = []

    # Title as H3
    blocks.append(_h3(art.title))

    # Source / date / link
    pub = art.published.strftime("%d %b %Y") if art.published else "—"
    blocks.append(_p(f"Kaynak: {art.source}  |  Tarih: {pub}  |  🔗 {art.url}"))

    # Score table line
    score_line = (
        f"A:{art.score_a}  B:{art.score_b}  C:{art.score_c}  "
        f"D:{art.score_d}  E:{art.score_e}"
    )
    bonus_tag = "  ⭐ Kesişim Bonusu" if art.has_bonus else ""
    total_line = f"Toplam: {art.total_score:.0f}  {art.signal_level}{bonus_tag}"
    blocks.append(_p_rich(
        (score_line, False, True),
        ("  →  ", False, False),
        (total_line, True, False),
    ))

    # Özet
    blocks.append(_p_rich(("Özet:", True, False)))
    blocks.append(_p(art.ozet or "(özet üretilemedi)"))

    # Neden Önemli
    blocks.append(_p_rich(("⚡ Neden Önemli", True, False)))
    blocks.append(_p_rich(
        ("Sektörel: ", True, False),
        (art.neden_onemli_sektorel or "—", False, False),
    ))
    blocks.append(_p_rich(
        ("Bankacılık açısından: ", True, False),
        (art.neden_onemli_bankacilik or "—", False, False),
    ))

    # Stratejik Çıkarım callout
    blocks.append(_callout(art.stratejik_cikarim or "—", "🎯"))

    blocks.append(_divider())
    return blocks


def create_digest_page(articles: list[ScoredArticle], total_scanned: int, candidates: int) -> str:
    notion = _get_client()
    db_id = _get_database_id()

    today = datetime.now(timezone.utc).strftime("%d %B %Y")
    title = f"📰 Tech Digest — {today}"
    n_selected = len(articles)

    # Header summary block
    kritik = sum(1 for a in articles if "KRİTİK" in a.signal_level)
    yuksek = sum(1 for a in articles if "YÜKSEK" in a.signal_level)
    header_text = (
        f"🤖 Bugün {total_scanned} haber tarandı · {candidates} aday Claude'a gönderildi · "
        f"{n_selected} haber seçildi   "
        f"[🔴 {kritik} Kritik · 🟠 {yuksek} Yüksek · diğerleri Orta]"
    )

    all_blocks: list[dict] = [
        {
            "object": "block",
            "type": "quote",
            "quote": {"rich_text": [{"type": "text", "text": {"content": header_text}}]},
        },
        _divider(),
    ]

    for art in articles:
        all_blocks.extend(_article_blocks(art))

    # Notion API limit: 100 blocks per request — chunk if needed
    MAX_BLOCKS = 98
    response = notion.pages.create(
        parent={"database_id": db_id},
        properties={"title": {"title": [{"type": "text", "text": {"content": title}}]}},
        children=all_blocks[:MAX_BLOCKS],
    )
    page_id = response["id"]
    page_url = response.get("url", "")

    # Append remaining blocks if any
    remaining = all_blocks[MAX_BLOCKS:]
    if remaining:
        for i in range(0, len(remaining), MAX_BLOCKS):
            notion.blocks.children.append(
                page_id, children=remaining[i:i + MAX_BLOCKS]
            )

    print(f"[OK] Notion sayfası oluşturuldu: {page_url}")
    return page_url
