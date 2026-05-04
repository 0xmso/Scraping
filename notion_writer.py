"""Writes the daily digest to a Notion database."""

import os
from datetime import datetime, timezone
from notion_client import Client
from fetcher import Article


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


def _article_blocks(articles: list[Article]) -> list[dict]:
    blocks = []
    for art in articles:
        # Heading
        blocks.append({
            "object": "block",
            "type": "heading_3",
            "heading_3": {
                "rich_text": [{"type": "text", "text": {"content": art.title}}]
            },
        })
        # URL + meta line
        meta = f"🔗 {art.url}\n⭐ Puan: {art.score}/10 | 🏷 Kategori: {art.category}"
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": meta}}]
            },
        })
        # Summary label (bold)
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {"type": "text", "text": {"content": "Özet (2-3 cümle Türkçe):"}, "annotations": {"bold": True}}
                ]
            },
        })
        # Summary text
        summary_text = art.summary if art.summary else "(Özet mevcut değil)"
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": summary_text}}]
            },
        })
        # "Neden önemli" placeholder
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {"type": "text", "text": {"content": "Neden önemli: "}, "annotations": {"bold": True}},
                    {"type": "text", "text": {"content": f"[{art.category} alanında dikkat çeken gelişme — kaynak: {art.source}]"}},
                ]
            },
        })
        # Divider
        blocks.append({"object": "block", "type": "divider", "divider": {}})
    return blocks


def create_digest_page(articles: list[Article], total_scanned: int) -> str:
    notion = _get_client()
    db_id = _get_database_id()

    today = datetime.now(timezone.utc).strftime("%d %B %Y")
    title = f"📰 Tech Digest — {today}"
    n_selected = len(articles)

    header_text = f"🤖 Bugün {total_scanned} haber tarandı, {n_selected} tanesi seçildi."

    header_block = {
        "object": "block",
        "type": "quote",
        "quote": {
            "rich_text": [{"type": "text", "text": {"content": header_text}}]
        },
    }

    article_blocks = _article_blocks(articles)

    response = notion.pages.create(
        parent={"database_id": db_id},
        properties={
            "title": {
                "title": [{"type": "text", "text": {"content": title}}]
            },
        },
        children=[header_block] + article_blocks,
    )

    page_url = response.get("url", "")
    print(f"[OK] Notion sayfası oluşturuldu: {page_url}")
    return page_url
