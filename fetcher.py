"""RSS feed fetcher and article scorer."""

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

# (keywords, weight, category_label)
SCORING_RULES = [
    (
        ["ai", "llm", "gpt", "claude", "gemini", "openai", "anthropic",
         "machine learning", "deep learning", "neural", "artificial intelligence",
         "large language", "foundation model", "generative", "chatbot", "agent",
         "mistral", "llama", "diffusion", "transformer"],
        10,
        "Yapay Zeka / LLM",
    ),
    (
        ["fintech", "payment", "stripe", "visa", "mastercard", "banking",
         "neobank", "open banking", "paytech", "ödeme", "financial technology"],
        9,
        "Fintech & Ödeme",
    ),
    (
        ["funding", "raises", "acquisition", "acquires", "ipo", "series a",
         "series b", "series c", "venture", "seed round", "valuation",
         "unicorn", "startup", "invested", "merger"],
        9,
        "Startup / Funding",
    ),
    (
        ["crypto", "bitcoin", "ethereum", "blockchain", "web3", "defi",
         "nft", "solana", "binance", "coinbase", "token", "wallet",
         "stablecoin", "dao"],
        8,
        "Kripto & Web3",
    ),
    (
        ["apple", "google", "meta", "microsoft", "amazon", "samsung",
         "android", "ios", "iphone", "pixel", "product launch", "released",
         "update", "feature"],
        6,
        "Genel Tech",
    ),
]


@dataclass
class Article:
    title: str
    url: str
    summary: str
    published: Optional[datetime]
    source: str
    score: int = 0
    category: str = ""


def _score(title: str, summary: str) -> tuple[int, str]:
    text = (title + " " + summary).lower()
    best_weight = 0
    best_cat = ""
    for keywords, weight, cat in SCORING_RULES:
        for kw in keywords:
            # Require word boundaries so "neural" doesn't match "neuroscience" etc.
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, text):
                if weight > best_weight:
                    best_weight = weight
                    best_cat = cat
                break
    return best_weight, best_cat


def _clean(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _parse_date(entry) -> Optional[datetime]:
    try:
        t = entry.get("published_parsed") or entry.get("updated_parsed")
        if t:
            return datetime(*t[:6], tzinfo=timezone.utc)
    except Exception:
        pass
    return None


def fetch_articles(min_score: int = 7, max_articles: int = 10) -> tuple[list[Article], int]:
    seen_urls: set[str] = set()
    all_articles: list[Article] = []
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
                raw_summary = entry.get("summary", entry.get("description", ""))
                summary = _clean(raw_summary)[:500]

                score, category = _score(title, summary)
                if score < min_score:
                    continue

                all_articles.append(
                    Article(
                        title=title,
                        url=url,
                        summary=summary,
                        published=_parse_date(entry),
                        source=source,
                        score=score,
                        category=category,
                    )
                )
        except Exception as exc:
            print(f"[WARN] Feed error ({feed_url}): {exc}")

    # Sort by score desc, then by date desc
    all_articles.sort(key=lambda a: (a.score, a.published or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
    return all_articles[:max_articles], total_scanned
