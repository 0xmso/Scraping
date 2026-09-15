"""Cross-run deduplication — prevents re-posting articles already sent.

Two layers, both running BEFORE Claude analysis (so duplicates cost no
Opus tokens):

1. URL layer — uses the Articles Notion database as the source of truth:
   every article ever posted has a row there with its URL. Candidates whose
   normalized URL was posted in the last LOOKBACK_DAYS are dropped.
   URLs are normalized (host lowercased, www/utm/http-https variants
   collapsed) so trivially-different links still match.

2. Semantic layer — one cheap model call compares candidate titles against
   recently posted titles (and against each other) to catch the same story
   arriving from a different source with a different URL, e.g. TechCrunch
   and VentureBeat both covering the same announcement.

Both layers fail open: if Notion or the Anthropic API is unreachable, all
candidates pass rather than blocking the digest.
"""

import os
import re
from datetime import datetime, timezone, timedelta
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import httpx

import llm
from fetcher import RawArticle

LOOKBACK_DAYS = 30            # URL-level memory window
SEMANTIC_LOOKBACK_DAYS = 10   # title window for the semantic check (keeps tokens low)
NOTION_VERSION = "2022-06-28"
SEMANTIC_TIER = "fast"        # cheap model; resolved per backend in llm.py

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


def fetch_seen(token: str) -> tuple[set[str], list[str]]:
    """Return (normalized URLs of last LOOKBACK_DAYS, titles of last SEMANTIC_LOOKBACK_DAYS).

    Uses the Notion REST API directly (httpx) rather than the SDK, since
    notion-client 3.x removed databases.query.
    """
    now = datetime.now(timezone.utc)
    since_urls = (now - timedelta(days=LOOKBACK_DAYS)).isoformat()
    since_titles = now - timedelta(days=SEMANTIC_LOOKBACK_DAYS)
    articles_db_id = _get_articles_db_id()
    url = f"https://api.notion.com/v1/databases/{articles_db_id}/query"
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    seen_urls: set[str] = set()
    seen_titles: list[str] = []
    cursor = None

    with httpx.Client(timeout=30) as client:
        while True:
            body = {
                "filter": {"timestamp": "created_time", "created_time": {"after": since_urls}},
                "page_size": 100,
            }
            if cursor:
                body["start_cursor"] = cursor

            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

            for page in data.get("results", []):
                props = page.get("properties", {})
                art_url = props.get("URL", {}).get("url")
                if art_url:
                    seen_urls.add(normalize_url(art_url))

                created = page.get("created_time", "")
                try:
                    created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                except ValueError:
                    created_dt = None
                if created_dt and created_dt >= since_titles:
                    title_prop = props.get("Name", {}).get("title", [])
                    if title_prop:
                        seen_titles.append(title_prop[0]["plain_text"])

            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")

    return seen_urls, seen_titles


# Backwards-compatible alias (older callers/tests used this name)
def fetch_seen_urls(token: str) -> set[str]:
    return fetch_seen(token)[0]


_SEMANTIC_SYSTEM = """Sen bir haber tekrar dedektörüsün. Sana iki liste verilecek:
1. Son günlerde ZATEN YAYINLANMIŞ haber başlıkları
2. Numaralı ADAY haber başlıkları

Bir aday şu iki durumdan birine uyuyorsa TEKRAR sayılır:
- Zaten yayınlanmış bir haberle AYNI OLAYI anlatıyor (farklı kaynak/farklı ifade olsa bile)
- Listede kendinden önce gelen başka bir adayla aynı olayı anlatıyor

DİKKAT: Aynı konudaki YENİ GELİŞME tekrar değildir. Örneğin "X şirketi yatırım görüşmelerinde"
yayınlandıysa, "X şirketi yatırımı tamamladı" yeni gelişmedir, elenmez. Sadece aynı olayın
farklı kaynaktan/farklı ifadeyle tekrarını ele.

Sadece JSON döndür: tekrar olan adayların numaraları."""

_SEMANTIC_SCHEMA = {
    "type": "object",
    "properties": {
        "duplicate_indices": {
            "type": "array",
            "items": {"type": "integer"},
        },
    },
    "required": ["duplicate_indices"],
    "additionalProperties": False,
}


# Shortlisting. Embeddings alone can't decide "same event": on live data, pairs
# the model judged duplicates scored 0.55–0.74 while distinct follow-ups reached
# 0.70. But nothing below 0.50 was a duplicate, so embeddings pick each
# candidate's few nearest stories and the model judges only those pairs — a
# short, specific comparison instead of scanning one long list of titles.
SHORTLIST_SIM = 0.50
SHORTLIST_SEEN = 3
SHORTLIST_PEERS = 2

_PAIR_SYSTEM = """Sen bir haber tekrar dedektörüsün. Her ADAY haberin altında, ona en benzer
bulunan birkaç haber listelenmiş (daha önce yayınlanmış olanlar ve aynı partide ondan önce
gelen adaylar).

Bir aday, listelenen haberlerden biriyle AYNI OLAYI anlatıyorsa TEKRAR sayılır (farklı
kaynak, farklı dil veya farklı ifade olsa bile).

TEKRAR DEĞİLDİR:
- Aynı konudaki YENİ GELİŞME ("X yatırım görüşmesinde" → "X yatırımı tamamladı",
  "Y açıklama yaptı" → "Z, Y'nin açıklamasına yanıt verdi")
- Aynı şirketin FARKLI bir haberi ("OpenAI halka arz olmayacak" ≠ "OpenAI Pro aboneliği durdurdu")
- Aynı temadaki farklı olay

HER aday için ayrı bir karar yaz: aday kimliği (ör. "A5"), tekrar (true/false) ve tekrarsa
aynı olayı anlatan listedeki haberin başlığı (ayni_olay). Emin değilsen tekrar=false."""

_PAIR_SCHEMA = {
    "type": "object",
    "properties": {"kararlar": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "aday": {"type": "string"},
            "tekrar": {"type": "boolean"},
            "ayni_olay": {"type": "string"},
        },
        "required": ["aday", "tekrar", "ayni_olay"],
        "additionalProperties": False,
    }}},
    "required": ["kararlar"],
    "additionalProperties": False,
}


def _shortlists(articles: list[RawArticle], seen_titles: list[str]) -> dict[int, list[str]]:
    """1-based candidate index -> nearby stories worth a same-event check."""
    seen_vecs = llm.embed(seen_titles, "search_document") if seen_titles else []
    cand_vecs = llm.embed([a.title for a in articles], "search_query")
    dot = lambda x, y: sum(p * q for p, q in zip(x, y))
    out = {}
    for i, vec in enumerate(cand_vecs):
        seen = sorted(((dot(vec, sv), t) for sv, t in zip(seen_vecs, seen_titles)), reverse=True)
        peers = sorted(((dot(vec, cand_vecs[j]), j) for j in range(i)), reverse=True)
        # No numbers on the nearby items: numbering them invited the model to
        # return a nearby item's number instead of the candidate's.
        near = [f"(daha önce yayınlandı) {t}" for s, t in seen[:SHORTLIST_SEEN] if s >= SHORTLIST_SIM]
        near += [f"(bu partide önceki aday) {articles[j].title}"
                 for s, j in peers[:SHORTLIST_PEERS] if s >= SHORTLIST_SIM]
        if near:
            out[i + 1] = near
    return out


def _duplicate_indices_by_pairs(client, articles, seen_titles) -> set[int]:
    shortlists = _shortlists(articles, seen_titles)
    if not shortlists:
        return set()
    blocks = []
    for idx, near in shortlists.items():
        lines = "\n".join(f"   - {n}" for n in near)
        blocks.append(f"[A{idx}] {articles[idx - 1].title}\n{lines}")
    data = llm.call_structured(
        client,
        model=llm.model(SEMANTIC_TIER),
        max_tokens=4000,
        system=_PAIR_SYSTEM,
        user_content="\n\n".join(blocks),
        schema=_PAIR_SCHEMA,
        tool_name="tekrar_kararlari",
        tool_description="Her aday için ayrı tekrar kararı.",
    )
    flagged = set()
    for verdict in llm.coerce_list(data.get("kararlar")):
        if not isinstance(verdict, dict):
            continue
        match = re.fullmatch(r"\[?A(\d+)\]?", str(verdict.get("aday", "")).strip())
        if match and str(verdict.get("tekrar", "")).strip().lower() == "true" and verdict.get("ayni_olay"):
            flagged.add(int(match.group(1)))
    # Only shortlisted candidates can be duplicates; ignore anything else.
    return flagged & set(shortlists)


def semantic_filter(
    articles: list[RawArticle], seen_titles: list[str]
) -> tuple[list[RawArticle], int]:
    """Drop candidates that cover the same story as an already-posted article
    (or as an earlier candidate in the same batch).

    Embedding shortlists + one model call over those pairs; if embeddings are
    unavailable, falls back to a single call over the full title lists.
    Returns (fresh_articles, skipped_count). Fails open on any error.
    """
    if not articles:
        return articles, 0

    if not llm.has_credentials():
        print("   [WARN] Model kimlik bilgisi yok — semantik tekrar kontrolü atlandı.")
        return articles, 0

    client = llm.get_client()
    try:
        dup_indices = _duplicate_indices_by_pairs(client, articles, seen_titles)
        return _apply(articles, dup_indices)
    except Exception as exc:
        print(f"   [WARN] Embedding tabanlı tekrar kontrolü yapılamadı ({exc}) — tam liste yöntemine geçiliyor.")

    try:
        seen_block = "\n".join(f"- {t}" for t in seen_titles) or "(yok)"
        cand_block = "\n".join(f"{i}. {a.title}" for i, a in enumerate(articles, 1))
        user_msg = (
            f"ZATEN YAYINLANMIŞ HABERLER (son {SEMANTIC_LOOKBACK_DAYS} gün):\n{seen_block}\n\n"
            f"ADAY HABERLER:\n{cand_block}"
        )

        data = llm.call_structured(
            client,
            model=llm.model(SEMANTIC_TIER),
            max_tokens=2000,
            system=_SEMANTIC_SYSTEM,
            user_content=user_msg,
            schema=_SEMANTIC_SCHEMA,
            tool_name="tekrar_bildir",
            tool_description="Tekrar olan adayların numaralarını döndür.",
        )
        dup_indices = {
            int(x) for x in llm.coerce_list(data.get("duplicate_indices"))
            if str(x).strip().lstrip("-").isdigit()
        }
    except Exception as exc:
        print(f"   [WARN] Semantik tekrar kontrolü başarısız ({exc}) — tüm adaylar geçiyor.")
        return articles, 0
    return _apply(articles, dup_indices)


def _apply(articles: list[RawArticle], dup_indices: set[int]) -> tuple[list[RawArticle], int]:
    fresh: list[RawArticle] = []
    skipped = 0
    for i, art in enumerate(articles, 1):
        if i in dup_indices:
            skipped += 1
            print(f"   ↻ semantik tekrar elendi: {art.title[:65]}")
        else:
            fresh.append(art)

    return fresh, skipped


def filter_new_articles(
    raw_articles: list[RawArticle],
) -> tuple[list[RawArticle], int]:
    """Drop candidates already posted (by URL) or already covered (by story).

    Returns (fresh_articles, skipped_count). Fails open — if Notion is
    unreachable, returns all articles rather than blocking the digest.
    """
    token = os.environ.get("NOTION_TOKEN")
    if not token:
        print("   [WARN] NOTION_TOKEN yok — tekrar kontrolü atlandı.")
        return raw_articles, 0

    try:
        seen_urls, seen_titles = fetch_seen(token)
    except Exception as exc:
        print(f"   [WARN] Tekrar kontrolü başarısız ({exc}) — tüm adaylar geçiyor.")
        return raw_articles, 0

    # Layer 1 — exact URL match
    fresh: list[RawArticle] = []
    url_skipped = 0
    for art in raw_articles:
        if normalize_url(art.url) in seen_urls:
            url_skipped += 1
        else:
            fresh.append(art)
    if url_skipped:
        print(f"   {url_skipped} aday URL eşleşmesiyle elendi")

    # Layer 2 — same story, different source/URL
    fresh, semantic_skipped = semantic_filter(fresh, seen_titles)

    return fresh, url_skipped + semantic_skipped
