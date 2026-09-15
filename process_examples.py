"""Processes pending rows in the "Örnek Haberler" Notion database into
scored training examples.

Flow: Kübra pastes a URL she already knows is a good example into Notion.
This script fetches the article, extracts its real text (trafilatura),
and runs it through the same Stage-2 scoring call (Opus 5) the daily
digest uses — so "meaning extraction" (title, Turkish summary, category
scores, strategic takeaway) is entirely the model's job, not manual entry.
Results are written back to the Notion row and appended to
golden_articles.json's must_pass bucket for the optimizer's quality gate.

Run schedule: before the prompt optimizer (Mon/Thu 03:00 TSİ), so anything
added during the week is folded in before that run's prompt rewrite.
"""

import os
import json
from pathlib import Path

import httpx

import article_text

import llm
from analyzer import _deep_analyze, _compute_total, _scores
from fetcher import RawArticle

NOTION_VERSION = "2022-06-28"
GOLDEN_FILE = Path(__file__).parent / "golden_articles.json"
SYSTEM_PROMPT_FILE = Path(__file__).parent / "system_prompt.txt"


# What gets sent to Opus 5 as the "source excerpt" — generous since this
# pipeline fetches full article text, not a ≤600-char RSS blurb.
MAX_ARTICLE_CHARS = 8000
# What gets stored in golden_articles.json — short, like the other entries.
GOLDEN_SUMMARY_CHARS = 500


def _get_examples_db_id() -> str:
    db_id = os.environ.get("NOTION_EXAMPLES_DB_ID")
    if not db_id:
        raise RuntimeError("NOTION_EXAMPLES_DB_ID environment variable is not set.")
    return db_id


def _notion_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def fetch_pending_rows(token: str) -> list[dict]:
    """Rows with no Durum set, or Durum = Beklemede."""
    db_id = _get_examples_db_id()
    url = f"https://api.notion.com/v1/databases/{db_id}/query"
    headers = _notion_headers(token)
    results, cursor = [], None

    with httpx.Client(timeout=30) as client:
        while True:
            body = {"page_size": 100}
            if cursor:
                body["start_cursor"] = cursor
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            for page in data.get("results", []):
                status = page["properties"].get("Durum", {}).get("select")
                status_name = status["name"] if status else None
                if status_name in (None, "Beklemede"):
                    results.append(page)
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")

    return results


def update_notion_row(token: str, page_id: str, properties: dict) -> None:
    url = f"https://api.notion.com/v1/pages/{page_id}"
    headers = _notion_headers(token)
    with httpx.Client(timeout=30) as client:
        resp = client.patch(url, headers=headers, json={"properties": properties})
        resp.raise_for_status()


def _mark_error(token: str, page_id: str, message: str) -> None:
    update_notion_row(token, page_id, {
        "Durum": {"select": {"name": "Hata"}},
        "Skor Notu": {"rich_text": [{"type": "text", "text": {"content": message[:2000]}}]},
    })


def extract_article(url: str) -> tuple[str, str]:
    """Fetch a URL and return (title, article_text). Raises on failure."""
    try:
        art = article_text.fetch(url)
    except article_text.FetchBlocked:
        raise ValueError(
            "site bot koruması (Cloudflare) gösteriyor — bu kaynaktan otomatik "
            "çekilemiyor, makale metnini elle 'Özet' alanına yapıştırman gerekir"
        ) from None
    if not art.title:
        raise ValueError("sayfadan başlık çıkarılamadı")
    return art.title, art.text


def _truncate_to_sentence(text: str, max_chars: int) -> str:
    """Cut at the last sentence boundary before max_chars, not mid-word."""
    if len(text) <= max_chars:
        return text.strip()
    cut = text[:max_chars]
    last_stop = max(cut.rfind(". "), cut.rfind(".\n"))
    if last_stop > max_chars * 0.4:  # keep at least a substantial chunk
        cut = cut[: last_stop + 1]
    return cut.strip()


def append_to_golden_set(title: str, summary: str) -> bool:
    """Append to golden_articles.json's must_pass bucket.

    Returns False (no-op) if this title is already present, so re-running
    the script on an already-processed backlog doesn't duplicate entries.
    """
    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    existing_titles = {item["title"] for item in golden.get("must_pass", [])}
    if title in existing_titles:
        return False
    golden.setdefault("must_pass", []).append({"title": title, "summary": summary})
    GOLDEN_FILE.write_text(
        json.dumps(golden, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return True


def process_examples() -> int:
    """Process every pending row. Returns the count successfully processed.

    Per-row failures (fetch error, paywall, scoring error) are caught and
    written back as Durum=Hata rather than raised — one bad link shouldn't
    stop the rest of the batch. Only missing top-level config raises.
    """
    notion_token = os.environ.get("NOTION_TOKEN")
    if not notion_token:
        raise RuntimeError("NOTION_TOKEN environment variable is not set.")
    if not llm.has_credentials():
        raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK veya ANTHROPIC_API_KEY eksik.")

    print(f"🔌 Backend: {llm.backend_name()}")

    pending = fetch_pending_rows(notion_token)
    print(f"📥 {len(pending)} bekleyen örnek bulundu.")
    if not pending:
        return 0

    client = llm.get_client()
    system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
    processed = 0

    for page in pending:
        page_id = page["id"]
        props = page["properties"]
        url_prop = props.get("URL", {}).get("url")

        # Manual override: if Kübra already pasted a title + full text herself
        # (the fallback for sites the fetcher can't reach, e.g. Finextra's
        # Cloudflare challenge), skip fetching and score that directly.
        manual_summary_blocks = props.get("Özet", {}).get("rich_text", [])
        manual_summary = manual_summary_blocks[0]["plain_text"] if manual_summary_blocks else ""
        manual_title_blocks = props.get("Name", {}).get("title", [])
        manual_title = manual_title_blocks[0]["plain_text"] if manual_title_blocks else ""

        print(f"\n→ {url_prop or '(URL yok — manuel giriş bekleniyor)'}")
        try:
            if manual_summary:
                if not manual_title:
                    raise ValueError("Özet elle girilmiş ama Name (başlık) boş — başlık da gerekli")
                title, article_text = manual_title, manual_summary
            elif url_prop:
                title, article_text = extract_article(url_prop)
            else:
                raise ValueError("URL boş ve manuel Özet de girilmemiş")

            excerpt = article_text[:MAX_ARTICLE_CHARS]

            art = RawArticle(
                title=title, url=url_prop, summary=excerpt,
                published=None, source="örnek-haberler",
            )
            data = _deep_analyze(client, art, system_prompt)
            a, b, c, d, e = _scores(data)
            total, has_bonus = _compute_total(a, b, c, d, e)

            skor_notu = (
                f"A:{a} B:{b} C:{c} D:{d} E:{e} → {total:.0f}pt{' ⭐' if has_bonus else ''}\n\n"
                f"Sektörel: {data.get('neden_onemli_sektorel', '')}\n"
                f"Stratejik: {data.get('stratejik_cikarim', '')}"
            )
            update_notion_row(notion_token, page_id, {
                "Name": {"title": [{"type": "text", "text": {"content": title[:2000]}}]},
                "Özet": {"rich_text": [{"type": "text", "text": {"content": data.get("ozet", "")[:2000]}}]},
                "Skor Notu": {"rich_text": [{"type": "text", "text": {"content": skor_notu[:2000]}}]},
                "Toplam Skor": {"number": round(total)},
                "Durum": {"select": {"name": "İşlendi"}},
            })

            golden_summary = _truncate_to_sentence(article_text, GOLDEN_SUMMARY_CHARS)
            added = append_to_golden_set(title, golden_summary)
            print(f"   ✅ {total:.0f}pt{' ⭐' if has_bonus else ''} · "
                  f"golden sete {'eklendi' if added else 'zaten vardı, atlandı'}")
            processed += 1

        except Exception as exc:
            print(f"   ❌ Hata: {exc}")
            _mark_error(notion_token, page_id, str(exc))

    print(f"\n{llm.usage_summary()}")
    return processed


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
    n = process_examples()
    print(f"\n{'='*60}\n{n} örnek başarıyla işlendi.")
