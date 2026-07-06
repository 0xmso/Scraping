"""RSS feed fetcher — returns raw articles for downstream Claude analysis."""

import feedparser
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

FEEDS = [
    "https://techcrunch.com/feed/",
    "https://feeds.feedburner.com/TechCrunchIT",
    "https://www.finextra.com/rss/headlines.aspx",
    "https://hnrss.org/frontpage",
    "https://www.coindesk.com/arc/outboundfeeds/rss",
    "https://feeds.feedburner.com/venturebeat/SZYF",
    "https://www.theinformation.com/feed",
    "https://upcorn.co/feed/",
]

# Max candidates a single feed may contribute per run, so no single source
# floods the digest. CoinDesk is capped tighter to keep crypto's share low.
DEFAULT_FEED_CAP = 12
FEED_CAPS = {
    "coindesk.com": 3,
}

# Light pre-filter — keeps articles with at least one of these signals
# to avoid sending 100% noise to Claude API
PRE_FILTER_KEYWORDS = [
    "ai", "llm", "gpt", "claude", "gemini", "openai", "anthropic",
    "machine learning", "neural", "artificial intelligence", "agent",
    "mistral", "llama", "generative", "model",
    "fintech", "payment", "payments", "banking", "bank", "banks",
    "neobank", "stripe", "visa", "mastercard", "paypal", "klarna",
    "revolut", "wise", "fraud", "lending", "wallet", "regtech",
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
        feed_cap = next(
            (cap for domain, cap in FEED_CAPS.items() if domain in feed_url),
            DEFAULT_FEED_CAP,
        )
        feed_count = 0
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
                feed_count += 1

                if feed_count >= feed_cap or len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break
        except Exception as exc:
            print(f"[WARN] Feed error ({feed_url}): {exc}")

    return candidates, total_scanned
