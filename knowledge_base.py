"""Labelled-example memory: every article Kübra has judged, retrieved by similarity.

The optimizer rewrites the prompt from the last 30 days of feedback, so anything
older survives only if a rewrite happened to encode it. This module keeps the
whole history instead: all Articles rows Kübra labelled (Etiket or Feedback) plus
every processed row from Örnek Haberler. For each article being scored, the most
similar labelled examples are shown to the model as calibration.

That makes curated good examples teach (not just test), and removes forgetting.
Notion stays the single source of truth; embeddings are recomputed each run,
which is cheap at hundreds of rows. Revisit with a cache if it reaches many
thousands.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import httpx

import llm
from dedup import normalize_url
from fetcher import RawArticle

NOTION_VERSION = "2022-06-28"

TOP_K = 6
CANDIDATE_POOL = 30     # MMR re-ranks within the top-N most similar
MMR_LAMBDA = 0.7        # 1.0 = pure similarity, lower = more diversity
MIN_SIMILARITY = 0.35   # below this an "example" is noise, not a neighbour

_FEEDBACK_AS_LABEL = {
    "✅ Doğru seçim": "Doğru seçim (kitle belirtilmemiş)",
    "❌ Yanlış seçim": "Alakasız",
    "⚠️ Skor yanlış": "Skor yanlış",
}


@dataclass
class LabeledExample:
    title: str
    text: str
    url: str
    label: str
    depth: str = ""
    score: Optional[float] = None
    note: str = ""
    origin: str = ""   # "Articles" or "Örnek Haberler"
    vector: list = field(default_factory=list, repr=False)


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ['NOTION_TOKEN']}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def _query_all(db_id: str, notion_filter: dict) -> list[dict]:
    url = f"https://api.notion.com/v1/databases/{db_id}/query"
    pages, cursor = [], None
    with httpx.Client(timeout=30) as client:
        while True:
            body = {"filter": notion_filter, "page_size": 100}
            if cursor:
                body["start_cursor"] = cursor
            resp = client.post(url, headers=_headers(), json=body)
            resp.raise_for_status()
            data = resp.json()
            pages.extend(data.get("results", []))
            if not data.get("has_more"):
                return pages
            cursor = data.get("next_cursor")


def _text(props: dict, name: str) -> str:
    prop = props.get(name, {})
    blocks = prop.get("title") or prop.get("rich_text") or []
    return "".join(b.get("plain_text", "") for b in blocks)


def _select(props: dict, name: str) -> str:
    value = props.get(name, {}).get("select")
    return value["name"] if value else ""


def _from_articles(pages: list[dict]) -> list[LabeledExample]:
    out = []
    for page in pages:
        p = page["properties"]
        label = _select(p, "Etiket") or _FEEDBACK_AS_LABEL.get(_select(p, "Feedback"), "")
        title = _text(p, "Name")
        if not (label and title):
            continue
        out.append(LabeledExample(
            title=title,
            text=f"{title}\n\n{_text(p, 'Özet')}",
            url=p.get("URL", {}).get("url") or "",
            label=label,
            depth=_select(p, "Derinlik"),
            score=p.get("Total Score", {}).get("number"),
            note=_text(p, "Feedback Notu"),
            origin="Articles",
        ))
    return out


def _from_examples(pages: list[dict]) -> list[LabeledExample]:
    out = []
    for page in pages:
        p = page["properties"]
        title = _text(p, "Name")
        if not title:
            continue
        out.append(LabeledExample(
            title=title,
            text=f"{title}\n\n{_text(p, 'Özet')}",
            url=p.get("URL", {}).get("url") or "",
            label=_select(p, "Etiket") or "İyi örnek (kitle belirtilmemiş)",
            depth=_select(p, "Derinlik"),
            score=p.get("Toplam Skor", {}).get("number"),
            origin="Örnek Haberler",
        ))
    return out


class KnowledgeBase:
    def __init__(self, examples: list[LabeledExample]):
        self.examples = examples

    @classmethod
    def load(cls) -> "KnowledgeBase":
        examples = _from_articles(_query_all(
            os.environ["NOTION_ARTICLES_DB_ID"],
            {"or": [
                {"property": "Etiket", "select": {"is_not_empty": True}},
                {"property": "Feedback", "select": {"is_not_empty": True}},
            ]},
        ))
        examples_db = os.environ.get("NOTION_EXAMPLES_DB_ID")
        if examples_db:
            examples += _from_examples(_query_all(
                examples_db, {"property": "Durum", "select": {"equals": "İşlendi"}}
            ))
        if examples:
            for ex, vec in zip(examples, llm.embed([e.text for e in examples], "search_document")):
                ex.vector = vec
        return cls(examples)

    def neighbours(self, articles: list[RawArticle]) -> list[list[tuple[LabeledExample, float]]]:
        """For each article, up to TOP_K similar-but-diverse labelled examples."""
        if not self.examples or not articles:
            return [[] for _ in articles]
        queries = llm.embed([f"{a.title}\n\n{a.summary}" for a in articles], "search_query")
        return [self._mmr(art, q) for art, q in zip(articles, queries)]

    def _mmr(self, article: RawArticle, query: list) -> list[tuple[LabeledExample, float]]:
        own_url, own_title = normalize_url(article.url), article.title.strip().lower()
        scored = []
        for ex in self.examples:
            # Never show an article its own label — that would leak the answer
            # (matters most when the golden check or a re-run scores known items).
            if (own_url and normalize_url(ex.url) == own_url) or ex.title.strip().lower() == own_title:
                continue
            sim = _dot(query, ex.vector)
            if sim >= MIN_SIMILARITY:
                scored.append((ex, sim))
        pool = sorted(scored, key=lambda t: t[1], reverse=True)[:CANDIDATE_POOL]

        chosen: list[tuple[LabeledExample, float]] = []
        while pool and len(chosen) < TOP_K:
            best = max(pool, key=lambda t: MMR_LAMBDA * t[1] - (1 - MMR_LAMBDA) * max(
                (_dot(t[0].vector, c[0].vector) for c in chosen), default=0.0))
            chosen.append(best)
            pool.remove(best)
        return chosen


def _dot(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))


def load_safely() -> Optional[KnowledgeBase]:
    """Load, or return None so scoring proceeds without examples (fail open)."""
    if not llm.use_bedrock() or not os.environ.get("NOTION_ARTICLES_DB_ID"):
        print("   [WARN] Bilgi tabanı atlandı (Bedrock veya Notion ayarı yok).")
        return None
    try:
        kb = KnowledgeBase.load()
        print(f"   📚 Bilgi tabanı: {len(kb.examples)} etiketli örnek")
        return kb
    except Exception as exc:
        print(f"   [WARN] Bilgi tabanı yüklenemedi ({exc}) — örneksiz devam ediliyor.")
        return None


def format_block(neighbours: list[tuple[LabeledExample, float]], detailed: bool) -> str:
    """Calibration block prepended to the user message (never the cached system prompt)."""
    if not neighbours:
        return ""
    lines = [
        "REFERANS — KÜBRA'NIN DAHA ÖNCE DEĞERLENDİRDİĞİ BENZER HABERLER:",
        "Bunlar yalnızca puan eşiğini ve hedef kitle ayrımını kalibre etmek içindir. "
        "İçeriklerini analizine TAŞIMA: özet ve çıkarımlar sadece aşağıdaki "
        "DEĞERLENDİRİLECEK HABER'e dayanmalı.",
        "Etiket anlamı: Alakasız = seçilmemeliydi; Dijital Ekipler / Üst Yönetim / "
        "İkisi De = doğru seçim ve hedef kitle. Skor, o haber için modelin verdiği puandır.",
        "",
    ]
    for i, (ex, _) in enumerate(neighbours, 1):
        parts = [f"Etiket: {ex.label}"]
        if detailed and ex.depth:
            parts.append(f"Derinlik: {ex.depth}")
        if ex.score is not None:
            parts.append(f"Skor: {ex.score:.0f}")
        lines.append(f"{i}. \"{ex.title}\" → {' · '.join(parts)}")
        if detailed and ex.note:
            lines.append(f"   Kübra'nın notu: {ex.note[:300]}")
    lines += ["", "DEĞERLENDİRİLECEK HABER:", ""]
    return "\n".join(lines)
