"""RSS feed fetcher — returns raw articles for downstream Claude analysis."""

import feedparser
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

FEEDS = [
    "https://techcrunch.com/feed/",
    "https://www.finextra.com/rss/headlines.aspx",
    "https://hnrss.org/frontpage",
    "https://www.coindesk.com/arc/outboundfeeds/rss",
    "https://feeds.feedburner.com/venturebeat/SZYF",
    "https://upcorn.co/feed/",
]

# Removed — both had been contributing zero articles:
#   feeds.feedburner.com/TechCrunchIT — the FeedBurner address was abandoned and
#     now answers 200 with an unrelated Japanese WordPress page, not a feed.
#     TechCrunch's main feed above already covers this ground.
#   www.theinformation.com/feed — bot protection answers Python 403 (a plain
#     curl gets 200, so it is fingerprinting the client, not the User-Agent).
#     Unusable from the Actions runner; left in, it would trip the dead-feed
#     warning on every single run.

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


def fetch_raw_articles(
    max_candidates: int = 40,
) -> tuple[list[RawArticle], int, list[str]]:
    """Fetch and lightly pre-filter RSS articles.

    Returns (candidate_articles, total_scanned, dead_feeds).
    candidate_articles is capped at max_candidates to control API cost.

    dead_feeds lists feeds that returned no entries at all. A feed can break
    silently — CoinDesk once changed its URL and served a redirect feedparser
    doesn't follow, so it returned nothing for an unknown stretch before anyone
    noticed. Callers should surface this rather than treat it as normal.

    Feeds are collected separately and then interleaved round-robin, so the
    max_candidates budget is shared rather than consumed front-to-back. Reading
    feeds in order starved whichever ones sat at the end of the list — Upcorn
    was reaching the digest on almost no days.
    """
    seen_urls: set[str] = set()
    total_scanned = 0
    dead_feeds: list[str] = []
    per_feed: list[list[RawArticle]] = []

    for feed_url in FEEDS:
        feed_cap = next(
            (cap for domain, cap in FEED_CAPS.items() if domain in feed_url),
            DEFAULT_FEED_CAP,
        )
        collected: list[RawArticle] = []
        try:
            feed = feedparser.parse(feed_url)
            source = feed.feed.get("title", feed_url)
            if not feed.entries:
                dead_feeds.append(feed_url)
                print(f"[WARN] Feed hiç haber döndürmedi: {feed_url}")
                continue
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

                collected.append(
                    RawArticle(
                        title=title,
                        url=url,
                        summary=summary,
                        published=_parse_date(entry),
                        source=source,
                    )
                )
                if len(collected) >= feed_cap:
                    break
        except Exception as exc:
            dead_feeds.append(feed_url)
            print(f"[WARN] Feed error ({feed_url}): {exc}")

        if collected:
            per_feed.append(collected)

    # Interleave: one from each feed per pass, until the budget is spent.
    candidates: list[RawArticle] = []
    for i in range(max(len(f) for f in per_feed) if per_feed else 0):
        for feed_articles in per_feed:
            if i < len(feed_articles):
                candidates.append(feed_articles[i])
                if len(candidates) >= max_candidates:
                    return candidates, total_scanned, dead_feeds

    return candidates, total_scanned, dead_feeds
