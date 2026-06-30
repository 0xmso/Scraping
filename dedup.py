"""Cross-run deduplication — prevents re-posting articles already sent.

Uses the Articles Notion database as the source of truth: every article ever
posted has a row there with its URL. Before analysis we fetch the URLs posted
in the last LOOKBACK_DAYS and drop any candidate that matches. This both avoids
duplicate digest entries and saves Claude API tokens (we never re-analyze a
story we've already covered).

URLs are normalized (scheme/host lowercased, tracking query params and trailing
slashes stripped) so the same article arriving with different UTM tags is still
recognized as a duplicate.
"""

import os
from datetime import datetime, timezone, timedelta
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import httpx

from fetcher import RawArticle

LOOKBACK_DAYS = 30
NOTION_VERSION = "2022-06-28"

# Query params that never identify the article itself — strip before comparing.
_TRACKING_PREFIXES = ("utm_", "fbclid", "gclid", "mc_", "ref", "ref_", "source")


def normalize_url(url: str) -> str:
    """Canonicalize a URL so trivially-different variants compare equal."""
    if not url:
        return ""
    try:
        parts = urlsplit(url.strip())
        # Treat http and https as the same article — always canonicalize to https.
        scheme = "https"
        netloc = parts.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parts.path.rstrip("/")
        # Drop tracking query params; keep meaningful ones (e.g. ?id=123)
        kept = [
            (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False)
            if not any(k.lower().startswith(p) for p in _TRACKING_PREFIXES)
        ]
        query = urlencode(sorted(kept))
        return urlunsplit((scheme, netloc, path, query, ""))
    except Exception:
        return url.strip().lower()


def _get_articles_db_id() -> str:
    db_id = os.environ.get("NOTION_ARTICLES_DB_ID")
    if not db_id:
        raise RuntimeError("NOTION_ARTICLES_DB_ID environment variable is not set.")
    return db_id


def fetch_seen_urls(token: str) -> set[str]:
    """Return the normalized URLs of articles posted in the last LOOKBACK_DAYS.

    Uses the Notion REST API directly (httpx) rather than the SDK, since
    notion-client 3.x removed databases.query.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).isoformat()
    articles_db_id = _get_articles_db_id()
    url = f"https://api.notion.com/v1/databases/{articles_db_id}/query"
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    seen: set[str] = set()
    cursor = None

    with httpx.Client(timeout=30) as client:
        while True:
            body = {
                "filter": {"timestamp": "created_time", "created_time": {"after": since}},
                "page_size": 100,
            }
            if cursor:
                body["start_cursor"] = cursor

            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

            for page in data.get("results", []):
                art_url = page.get("properties", {}).get("URL", {}).get("url")
                if art_url:
                    seen.add(normalize_url(art_url))

            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")

    return seen


def filter_new_articles(
    raw_articles: list[RawArticle],
) -> tuple[list[RawArticle], int]:
    """Drop candidates already posted in the last LOOKBACK_DAYS.

    Returns (fresh_articles, skipped_count). Fails open — if Notion is
    unreachable, returns all articles rather than blocking the digest.
    """
    token = os.environ.get("NOTION_TOKEN")
    if not token:
        print("   [WARN] NOTION_TOKEN yok — tekrar kontrolü atlandı.")
        return raw_articles, 0

    try:
        seen = fetch_seen_urls(token)
    except Exception as exc:
        print(f"   [WARN] Tekrar kontrolü başarısız ({exc}) — tüm adaylar geçiyor.")
        return raw_articles, 0

    fresh: list[RawArticle] = []
    skipped = 0
    for art in raw_articles:
        if normalize_url(art.url) in seen:
            skipped += 1
        else:
            fresh.append(art)

    return fresh, skipped
