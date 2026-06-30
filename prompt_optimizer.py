"""Reads feedback from Notion and uses Claude to refine system_prompt.txt.

Run schedule: Monday and Thursday at 00:00 UTC (03:00 TSİ).
Reads pages with Feedback property set from the last 30 days,
asks Claude to analyze patterns and rewrite the scoring prompt.
"""

import os
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import anthropic
import httpx

PROMPT_FILE = Path(__file__).parent / "system_prompt.txt"
LOOKBACK_DAYS = 30
NOTION_VERSION = "2022-06-28"

OPTIMIZER_SYSTEM = """Sen bir prompt mühendisisin. Görevin: bir haber kürasyon sisteminin
puanlama promptunu, kullanıcı feedback'lerine dayanarak iyileştirmek.

Mevcut promptu ve feedback örneklerini alacaksın. Şunlara dikkat et:
- "❌ Yanlış seçim" → bu tür haberler neden seçildi? Puanlama kriterleri nasıl daraltılmalı?
- "⚠️ Skor yanlış" → hangi kategori ağırlıkları veya kural açıklamaları güncellenmeli?
- "✅ Doğru seçim" → bu iyi örnekleri few-shot olarak prompta ekle

ÇIKTI: Sadece güncellenmiş prompt metnini döndür. Başka açıklama ekleme.
Formatı koru: JSON çıktı talimatı ve tüm kategoriler eksiksiz kalsın."""


def _get_articles_db_id() -> str:
    db_id = os.environ.get("NOTION_ARTICLES_DB_ID")
    if not db_id:
        raise RuntimeError("NOTION_ARTICLES_DB_ID environment variable is not set.")
    return db_id


def _get_notion_feedback(token: str) -> list[dict]:
    """Fetch article rows with feedback from the Articles database (last LOOKBACK_DAYS).

    Uses the Notion REST API directly (httpx) since notion-client 3.x removed
    databases.query.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).isoformat()
    articles_db_id = _get_articles_db_id()
    url = f"https://api.notion.com/v1/databases/{articles_db_id}/query"
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    results = []
    cursor = None

    with httpx.Client(timeout=30) as client:
        while True:
            body = {
                "filter": {
                    "and": [
                        {"property": "Feedback", "select": {"is_not_empty": True}},
                        {"timestamp": "created_time", "created_time": {"after": since}},
                    ]
                },
                "page_size": 100,
            }
            if cursor:
                body["start_cursor"] = cursor

            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            results.extend(data.get("results", []))

            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")

    return results


def _extract_feedback_items(pages: list[dict]) -> list[dict]:
    """Parse Articles database rows into structured feedback dicts."""
    items = []
    for page in pages:
        props = page.get("properties", {})

        # Title is in "Name" property in Articles DB
        name_prop = props.get("Name", {}).get("title", [])
        title = name_prop[0]["plain_text"] if name_prop else "(başlık yok)"

        feedback_prop = props.get("Feedback", {}).get("select")
        feedback = feedback_prop["name"] if feedback_prop else None

        note_prop = props.get("Feedback Notu", {}).get("rich_text", [])
        note = note_prop[0]["plain_text"] if note_prop else ""

        # Also grab scores and signal for richer context
        signal = props.get("Signal", {}).get("select", {})
        signal_name = signal.get("name", "") if signal else ""
        total_score = props.get("Total Score", {}).get("number", 0)

        if feedback:
            items.append({
                "title": title,
                "feedback": feedback,
                "note": note,
                "signal": signal_name,
                "total_score": total_score,
            })

    return items


def _build_optimizer_prompt(current_prompt: str, feedback_items: list[dict]) -> str:
    feedback_block = "\n".join(
        f"- [{item['feedback']}] \"{item['title']}\" (Skor: {item['total_score']}, {item['signal']})"
        + (f"\n  Not: {item['note']}" if item["note"] else "")
        for item in feedback_items
    )

    return f"""MEVCUT PROMPT:
---
{current_prompt}
---

SON {LOOKBACK_DAYS} GÜNDEKİ FEEDBACK ({len(feedback_items)} kayıt):
{feedback_block}

Bu feedback'lere dayanarak promptu güncelle. Özellikle:
- Yanlış seçilen haberlerin ortak özelliklerini analiz et
- Puanlama kurallarını daha isabetli hale getir
- Başarılı seçimleri few-shot örnek olarak ekle (maksimum 3 örnek)
- Sektörel/bankacılık bakış açısı talimatlarını güçlendir"""


def run_optimization() -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    notion_token = os.environ.get("NOTION_TOKEN")
    db_id = os.environ.get("NOTION_DATABASE_ID")

    if not all([api_key, notion_token, db_id]):
        raise RuntimeError("ANTHROPIC_API_KEY, NOTION_TOKEN veya NOTION_DATABASE_ID eksik.")

    client = anthropic.Anthropic(api_key=api_key)

    # 1. Read current prompt
    current_prompt = PROMPT_FILE.read_text(encoding="utf-8")
    print(f"📄 Mevcut prompt yüklendi ({len(current_prompt)} karakter)")

    # 2. Fetch feedback from Articles database
    print("📥 Articles tablosundan feedback'ler okunuyor...")
    pages = _get_notion_feedback(notion_token)
    items = _extract_feedback_items(pages)
    print(f"   {len(items)} feedback kaydı bulundu.")

    if not items:
        print("⚠️  Hiç feedback yok — prompt güncellenmedi.")
        return current_prompt

    # Break down by type
    for label in ["✅ Doğru seçim", "❌ Yanlış seçim", "⚠️ Skor yanlış"]:
        count = sum(1 for i in items if i["feedback"] == label)
        print(f"   {label}: {count}")

    # 3. Ask Claude to optimize
    print("\n🧠 Claude ile prompt optimize ediliyor...")
    user_msg = _build_optimizer_prompt(current_prompt, items)

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        system=OPTIMIZER_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    new_prompt = response.content[0].text.strip()

    # 4. Save updated prompt
    PROMPT_FILE.write_text(new_prompt, encoding="utf-8")
    print(f"\n✅ system_prompt.txt güncellendi ({len(new_prompt)} karakter)")
    print(f"   Değişiklik: {len(new_prompt) - len(current_prompt):+d} karakter")

    return new_prompt


if __name__ == "__main__":
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
    run_optimization()
