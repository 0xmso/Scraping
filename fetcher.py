"""RSS feed fetcher — returns raw articles for downstream Claude analysis."""

import feedparser
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

FEEDS = [
    "https://techcrunch.com/feed/",
    "https://feeds.feedburner.com/TechCrunchIT",
    "https://hnrss.org/frontpage",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://feeds.feedburner.com/venturebeat/SZYF",
    "https://www.theinformation.com/feed",
]

# Light pre-filter — keeps articles with at least one of these signals
# to avoid sending 100% noise to Claude API
PRE_FILTER_KEYWORDS = [
    "ai", "llm", "gpt", "claude", "gemini", "openai", "anthropic",
    "machine learning", "neural", "artificial intelligence", "agent",
    "mistral", "llama", "generative", "model",
    "fintech", "payment", "banking", "neobank", "stripe", "visa",
    "funding", "raises", "acquisition", "ipo", "series", "unicorn",
    "startup", "invest", "merger", "valuation",
    "crypto", "bitcoin", "ethereum", "blockchain", "web3", "stablecoin",
    "cbdc", "token", "defi",
    "apple", "google", "meta", "microsoft", "amazon", "samsung",
    "launch", "release", "platform",
]


@dataclass
class RawArticle:
    title: str
    url: str
    summary: str          # raw RSS excerpt (≤600 chars)
    published: Optional[datetime]
    source: str


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_date(entry) -> Optional[datetime]:
    try:
        t = entry.get("published_parsed") or entry.get("updated_parsed")
        if t:
            return datetime(*t[:6], tzinfo=timezone.utc)
    except Exception:
        pass
    return None


def _passes_prefilter(title: str, summary: str) -> bool:
    text = (title + " " + summary).lower()
    for kw in PRE_FILTER_KEYWORDS:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text):
            return True
    return False


def fetch_raw_articles(max_candidates: int = 40) -> tuple[list[RawArticle], int]:
    """Fetch and lightly pre-filter RSS articles.

    Returns (candidate_articles, total_scanned).
    candidate_articles is capped at max_candidates to control API cost.
    """
    seen_urls: set[str] = set()
    candidates: list[RawArticle] = []
    total_scanned = 0

    for feed_url in FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            source = feed.feed.get("title", feed_url)
            for entry in feed.entries:
                total_scanned += 1
                url = entry.get("link", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                title = _clean(entry.get("title", ""))
                raw = entry.get("summary", entry.get("description", ""))
                summary = _clean(raw)[:600]

                if not _passes_prefilter(title, summary):
                    continue

                candidates.append(
                    RawArticle(
                        title=title,
                        url=url,
                        summary=summary,
                        published=_parse_date(entry),
                        source=source,
                    )
                )

                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break
        except Exception as exc:
            print(f"[WARN] Feed error ({feed_url}): {exc}")

    return candidates, total_scanned
