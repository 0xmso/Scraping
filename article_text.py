"""Full article fetch: main text, title and lead image from an article URL.

The daily digest scores from RSS excerpts (≤600 chars). The grounding check
showed the model filling that gap with background knowledge, so the articles
that reach deep analysis get their full text instead. The lead image comes
from the same request and becomes the slide image in the monthly deck.
"""

import time
from dataclasses import dataclass
from typing import Optional

import httpx
import trafilatura

# A plain requests-style UA gets blocked by some sites; a browser UA does not.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
MIN_TEXT_CHARS = 200


class FetchBlocked(ValueError):
    """The site refused a plain HTTP client (e.g. a Cloudflare JS challenge)."""


@dataclass
class Article:
    title: str
    text: str
    image_url: Optional[str]


def fetch(url: str, timeout: float = 20) -> Article:
    """Fetch and extract an article. Raises on failure.

    Retries once on 429/503. A Cloudflare JS challenge can't be solved by
    retrying — plain HTTP has no JS engine — so it raises FetchBlocked.
    """
    resp = None
    for attempt in range(2):
        resp = httpx.get(url, headers={"User-Agent": BROWSER_UA}, timeout=timeout, follow_redirects=True)
        if resp.status_code in (429, 503) and attempt == 0:
            time.sleep(4)
            continue
        break

    head = resp.text[:2000]
    if "Just a moment" in head or "challenges.cloudflare.com" in head:
        raise FetchBlocked("site bot koruması (Cloudflare) gösteriyor")
    resp.raise_for_status()

    text = trafilatura.extract(resp.text)
    if not text or len(text) < MIN_TEXT_CHARS:
        raise ValueError("makale metni çıkarılamadı veya çok kısa (paywall/JS olabilir)")

    meta = trafilatura.extract_metadata(resp.text)
    title = (meta.title if meta and meta.title else "").strip()
    image = (getattr(meta, "image", None) or "").strip() if meta else ""
    return Article(title=title, text=text, image_url=image or None)
