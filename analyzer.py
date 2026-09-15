"""Claude-powered multi-dimensional article scorer and Turkish analyst.

Two-stage architecture for token efficiency:
  Stage 1 — Sonnet 5 (fast): score every candidate in 5 categories (scores only)
  Stage 2 — Opus 5 (deep):   full Turkish analysis for articles passing threshold

Scoring system:
  A — Yapay Zeka & LLM        ×10
  B — Fintech & Ödeme         ×10
  C — Startup Funding/M&A/IPO ×8
  D — Kripto & Web3            ×3   (deliberately low — crypto was dominating)
  E — Genel Tech & Big Tech    ×5

Thresholds:
  200+  → 🔴 KRİTİK
  120-199 → 🟠 YÜKSEK
  70-119  → 🟡 ORTA
  <55   → elenir (run_digest passes min_total)

Cross-category bonus: 2+ categories with score ≥ 6 → ×1.25
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import llm
from fetcher import RawArticle

# ── Models ────────────────────────────────────────────────────────────────────
# Resolved per backend (Bedrock vs first-party) — see llm.py.
STAGE1_TIER = "fast"   # cheap pre-filter
STAGE2_TIER = "deep"   # deep analysis only on passing articles

# ── Scoring constants ─────────────────────────────────────────────────────────
WEIGHTS = {"A": 10, "B": 10, "C": 8, "D": 3, "E": 5}
BONUS_MULTIPLIER = 1.25
BONUS_MIN_SCORE = 6
BONUS_MIN_CATEGORIES = 2
THRESHOLD_KRITIK = 200
THRESHOLD_YUKSEK = 120
THRESHOLD_ORTA = 55   # matches run_digest's min_total — anything selected is ≥ ORTA

# Stage-1 safety margin: pass to stage 2 if the pre-filter total >= min_total * 0.7.
# Keeps borderline cases alive so Stage 2 can re-score with full analysis.
STAGE1_SAFETY_MARGIN = 0.7

_PROMPT_FILE = Path(__file__).parent / "system_prompt.txt"


# ── Stage 1 prompt (compact, scoring only) ────────────────────────────────────
# Inlined and stable so the pre-filter call is fast and predictable. The auto-tuned
# system_prompt.txt only applies to the deeper Stage 2 analysis.
STAGE1_SCORING_PROMPT = """Sen bir teknoloji haber puanlayıcısısın. Verilen haber başlığı ve özetini okuyup 5 kategoride 0-10 arası puanla.

KATEGORİLER:
A = Yapay Zeka & LLM (modeller, agentic AI, AI altyapısı)
B = Fintech & Ödeme Sistemleri (embedded finance, anlık ödeme, neobank, açık bankacılık)
C = Startup Funding / M&A / IPO ($10M+ yatırım, satın alma, halka arz, unicorn)
D = Kripto & Web3 (SADECE kurumsal düzey: stablecoin altyapısı, CBDC, tokenize
    mevduat, düzenleyici çerçeve, kurumsal saklama)
E = Genel Tech & Big Tech (Apple/Google/Meta/Amazon/Microsoft ürün lansmanları)

KURALLAR:
- 0 = hiç alakası yok, 10 = o kategorinin tam kalbinde
- Spekülatif/söylenti haberler için tüm puanları 3'ün altında tut
- Sayılarla (miktar, kullanıcı sayısı, metrik) desteklenen haberlere daha yüksek puan ver
- Token fiyat hareketleri, DeFi protokol güncellemeleri, cüzdan/borsa duyuruları,
  kripto hazine alım-satımları piyasa gürültüsüdür: D'yi 3'ün üstüne çıkarma

Sadece JSON döndür, başka metin ekleme."""


# ── Schemas for the forced tool call that shapes the output ───────────────────
# Numerical constraints (minimum/maximum) are NOT accepted — the 0-10 range is
# prompt-enforced and clamped in _scores().
STAGE1_SCHEMA = {
    "type": "object",
    "properties": {
        "score_a": {"type": "integer"},
        "score_b": {"type": "integer"},
        "score_c": {"type": "integer"},
        "score_d": {"type": "integer"},
        "score_e": {"type": "integer"},
    },
    "required": ["score_a", "score_b", "score_c", "score_d", "score_e"],
    "additionalProperties": False,
}

# Must match the "Etiket" / "Model Tahmini" select options in Notion exactly.
AUDIENCE_LABELS = ("Dijital Ekipler", "Üst Yönetim", "İkisi De", "Alakasız")

STAGE2_SCHEMA = {
    "type": "object",
    "properties": {
        "score_a": {"type": "integer"},
        "score_b": {"type": "integer"},
        "score_c": {"type": "integer"},
        "score_d": {"type": "integer"},
        "score_e": {"type": "integer"},
        "ozet": {"type": "string"},
        "neden_onemli_sektorel": {"type": "string"},
        "stratejik_cikarim": {"type": "string"},
        "hedef_kitle_tahmini": {"type": "string", "enum": list(AUDIENCE_LABELS)},
    },
    "required": [
        "score_a", "score_b", "score_c", "score_d", "score_e",
        "ozet", "neden_onemli_sektorel", "stratejik_cikarim", "hedef_kitle_tahmini",
    ],
    "additionalProperties": False,
}


def _load_system_prompt() -> str:
    if _PROMPT_FILE.exists():
        return _PROMPT_FILE.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"system_prompt.txt bulunamadı: {_PROMPT_FILE}")


@dataclass
class ScoredArticle:
    # Identity
    title: str
    url: str
    raw_summary: str        # the text the analysis was written from (full article when fetched)
    published: Optional[datetime]
    source: str
    # Scores
    score_a: int = 0
    score_b: int = 0
    score_c: int = 0
    score_d: int = 0
    score_e: int = 0
    total_score: float = 0.0
    has_bonus: bool = False
    signal_level: str = ""
    # Turkish analysis
    ozet: str = ""
    neden_onemli_sektorel: str = ""
    stratejik_cikarim: str = ""
    # Written to a Notion column hidden from Kübra's view, so her label stays
    # independent and model-vs-human agreement measures real learning.
    hedef_kitle_tahmini: str = ""
    # None = check not run (e.g. it errored); "" = clean; otherwise the flagged claims.
    supheli_iddialar: Optional[str] = None
    # "Seçildi" goes in the digest; "Sınırda"/"Rastgele" are rejected articles
    # written to Notion only, so Kübra's labels can show what the model missed.
    secim_tipi: str = "Seçildi"
    image_url: Optional[str] = None
    full_text: bool = False   # True when analysed from the fetched article, not the RSS excerpt


def _audience(data: dict) -> str:
    """The model's audience guess, or "" if it came back outside the enum."""
    value = data.get("hedef_kitle_tahmini", "")
    return value if value in AUDIENCE_LABELS else ""


def _compute_total(a: int, b: int, c: int, d: int, e: int) -> tuple[float, bool]:
    raw = (
        a * WEIGHTS["A"] + b * WEIGHTS["B"] + c * WEIGHTS["C"]
        + d * WEIGHTS["D"] + e * WEIGHTS["E"]
    )
    scores = [a, b, c, d, e]
    high_count = sum(1 for s in scores if s >= BONUS_MIN_SCORE)
    has_bonus = high_count >= BONUS_MIN_CATEGORIES
    total = raw * BONUS_MULTIPLIER if has_bonus else float(raw)
    return total, has_bonus


def _signal_level(total: float) -> str:
    if total >= THRESHOLD_KRITIK:
        return "🔴 KRİTİK"
    elif total >= THRESHOLD_YUKSEK:
        return "🟠 YÜKSEK"
    elif total >= THRESHOLD_ORTA:
        return "🟡 ORTA"
    return "⚫ DÜŞÜK"


def knowledge_base_block(found, detailed: bool) -> str:
    """Calibration examples for one article, or "" when there are none."""
    if not found:
        return ""
    from knowledge_base import format_block
    return format_block(found, detailed=detailed)


def _user_message(article: RawArticle) -> str:
    return f"Başlık: {article.title}\n\nKaynak Özeti: {article.summary or '(özet yok)'}"


def _scores(data: dict) -> tuple[int, int, int, int, int]:
    """Pull the five scores out, clamped to 0-10.

    The schema can't express minimum/maximum (the API rejects those keys), so
    the range lives only in the prompt — and models do sometimes overshoot.
    An unclamped 50 would be worth 500 points and wreck the ranking.
    """
    return tuple(
        max(0, min(10, int(data[f"score_{k}"]))) for k in ("a", "b", "c", "d", "e")
    )


def _quick_score(client, article: RawArticle, examples_block: str = "") -> dict:
    """Stage 1: cheap pre-filter — scores only, no Turkish analysis."""
    return llm.call_structured(
        client,
        model=llm.model(STAGE1_TIER),
        max_tokens=1500,
        system=[
            {
                "type": "text",
                "text": STAGE1_SCORING_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        user_content=examples_block + _user_message(article),
        schema=STAGE1_SCHEMA,
        tool_name="haber_puanla",
        tool_description="Haberi 5 kategoride 0-10 arası puanla.",
    )


def _deep_analyze(
    client, article: RawArticle, system_prompt: str, examples_block: str = ""
) -> dict:
    """Stage 2: deep call — full Turkish analysis + final scoring.

    Stage 2 re-scores too: stage-1 scores were only a pre-filter, this is the
    authoritative scoring that drives the final output. The same system_prompt.txt
    that's auto-tuned by the feedback loop is used here. Retrieved examples go in
    the user message so the cached system prompt stays byte-identical.
    """
    return llm.call_structured(
        client,
        model=llm.model(STAGE2_TIER),
        max_tokens=8000,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        user_content=examples_block + _user_message(article),
        schema=STAGE2_SCHEMA,
        tool_name="haber_analiz",
        tool_description="Haberi puanla ve Türkçe stratejik analiz üret.",
        effort="high",
        thinking=True,   # Claude decides when/how much to think
    )


# ── Review sample ─────────────────────────────────────────────────────────────
# Feedback only covers what the model selected, so it can never learn what it
# wrongly rejected. A few rejected articles go to Notion (not the digest) for
# Kübra to label. Mostly near-misses, plus one random Stage-1 reject: uncertainty
# sampling alone doesn't reliably beat random selection, so both are covered.
BORDERLINE_COUNT = 2
BORDERLINE_FLOOR = 30      # Stage-2 totals in [floor, min_total) count as near-misses
RANDOM_COUNT = 1


# Articles reaching deep analysis are fetched in full: from an RSS excerpt the
# model fills gaps with background knowledge, which the grounding check flags.
FULL_TEXT_CHARS = 8000
FETCH_WORKERS = 6


def _with_full_text(articles: list[RawArticle]) -> dict[int, tuple[RawArticle, Optional[str]]]:
    """id(original) -> (article to analyse, lead image). Falls back to the excerpt."""
    import dataclasses
    from concurrent.futures import ThreadPoolExecutor

    import article_text

    def one(art: RawArticle):
        try:
            fetched = article_text.fetch(art.url, timeout=15)
            return id(art), (dataclasses.replace(art, summary=fetched.text[:FULL_TEXT_CHARS]),
                             fetched.image_url)
        except Exception:
            return id(art), (art, None)

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        result = dict(pool.map(one, articles))
    fetched_count = sum(1 for a in articles if result[id(a)][0] is not a)
    print(f"   📄 Tam metin: {fetched_count}/{len(articles)} makale çekildi "
          f"(kalanı RSS özetiyle analiz edilecek)")
    return result


def _to_scored(art: RawArticle, data: dict, secim_tipi: str = "Seçildi") -> ScoredArticle:
    a, b, c, d, e = _scores(data)
    total, has_bonus = _compute_total(a, b, c, d, e)
    return ScoredArticle(
        title=art.title,
        url=art.url,
        raw_summary=art.summary,
        published=art.published,
        source=art.source,
        score_a=a, score_b=b, score_c=c, score_d=d, score_e=e,
        total_score=total,
        has_bonus=has_bonus,
        signal_level=_signal_level(total),
        ozet=data.get("ozet", ""),
        neden_onemli_sektorel=data.get("neden_onemli_sektorel", ""),
        stratejik_cikarim=data.get("stratejik_cikarim", ""),
        hedef_kitle_tahmini=_audience(data),
        secim_tipi=secim_tipi,
    )


def analyze_articles(
    raw_articles: list[RawArticle],
    min_total: float = THRESHOLD_ORTA,
    max_results: int = 10,
    knowledge_base=None,
    review_sample: Optional[list] = None,
) -> list[ScoredArticle]:
    """Two-stage scoring pipeline.

    Stage 1: cheap model scores every candidate
    Stage 2: deep model analyzes only articles passing the safety-margin gate

    With a knowledge_base, both stages see the most similar articles Kübra has
    already labelled — which is also how Stage 1, whose prompt is static, learns.

    If review_sample is a list, rejected articles chosen for Kübra's review are
    appended to it (see BORDERLINE_COUNT / RANDOM_COUNT). Returns the selection.
    """
    client = llm.get_client()
    system_prompt = _load_system_prompt()
    print(f"   Backend: {llm.backend_name()}")
    print(f"   Modeller: {llm.model(STAGE1_TIER)} → {llm.model(STAGE2_TIER)}")
    print(f"   Prompt yüklendi: {_PROMPT_FILE.name} ({len(system_prompt)} karakter)")

    neighbours: dict[int, list] = {}
    if knowledge_base is not None:
        try:
            for art, found in zip(raw_articles, knowledge_base.neighbours(raw_articles)):
                neighbours[id(art)] = found
            with_examples = sum(1 for v in neighbours.values() if v)
            print(f"   📚 {with_examples}/{len(raw_articles)} aday için benzer etiketli örnek bulundu")
        except Exception as exc:
            print(f"   [WARN] Örnek getirilemedi ({exc}) — örneksiz devam ediliyor.")
            neighbours = {}

    # ── Stage 1: pre-filter ──────────────────────────────────────────────────
    stage1_pass: list[tuple[RawArticle, float]] = []
    stage1_rejects: list[RawArticle] = []
    stage1_threshold = min_total * STAGE1_SAFETY_MARGIN
    print(f"\n   🚀 Stage 1 — {len(raw_articles)} aday hızlı puanlanıyor"
          f" (eşik ≥ {stage1_threshold:.0f})...")

    for i, art in enumerate(raw_articles, 1):
        try:
            data = _quick_score(
                client, art, knowledge_base_block(neighbours.get(id(art)), detailed=False)
            )
            a, b, c, d, e = _scores(data)
            total, _ = _compute_total(a, b, c, d, e)
            verdict = "✓ geçti" if total >= stage1_threshold else "✗ elendi"
            print(f"   [{i:2}/{len(raw_articles)}] {total:5.0f}pt {verdict} | {art.title[:60]}")
            if total >= stage1_threshold:
                stage1_pass.append((art, total))
            else:
                stage1_rejects.append(art)
        except Exception as exc:
            print(f"   [{i:2}/{len(raw_articles)}] [HATA] {exc} — {art.title[:60]}")

    # Sort by stage-1 total desc; cap deep analyses to a generous multiple of max_results
    # to avoid wasting deep calls if stage 1 is too lenient
    stage1_pass.sort(key=lambda t: t[1], reverse=True)
    deep_budget = max_results * 2
    candidates = [art for art, _ in stage1_pass[:deep_budget]]

    print(f"\n   🧠 Stage 2 — {len(candidates)} makale derin analiz...")

    # ── Stage 2: deep analysis ───────────────────────────────────────────────
    scored: list[ScoredArticle] = []
    near_misses: list[ScoredArticle] = []
    enriched = _with_full_text(candidates)
    for i, original in enumerate(candidates, 1):
        art, image_url = enriched[id(original)]
        try:
            data = _deep_analyze(
                client, art, system_prompt,
                knowledge_base_block(neighbours.get(id(original)), detailed=True),
            )
            result = _to_scored(art, data)
            result.image_url = image_url or original.image_url   # page image, else the feed's
            result.full_text = art is not original

            if result.total_score < min_total:
                print(f"   [{i:2}/{len(candidates)}] Eşik altı ({result.total_score:.0f}), elendi.")
                if result.total_score >= BORDERLINE_FLOOR:
                    near_misses.append(result)
                continue

            print(f"   [{i:2}/{len(candidates)}] {result.signal_level} | "
                  f"{result.total_score:.0f}pt | {art.title[:60]}")
            scored.append(result)
        except Exception as exc:
            print(f"   [{i:2}/{len(candidates)}] [HATA] {exc}")
            continue

    scored.sort(key=lambda a: a.total_score, reverse=True)
    selected = scored[:max_results]
    _check_grounding(client, selected)

    if review_sample is not None:
        review_sample.extend(_review_sample(
            client, system_prompt, near_misses, stage1_rejects, neighbours
        ))
    return selected


def _review_sample(client, system_prompt, near_misses, stage1_rejects, neighbours) -> list[ScoredArticle]:
    """Pick rejected articles for Kübra to label; never raises (fail open)."""
    import random

    sample: list[ScoredArticle] = []
    for art in sorted(near_misses, key=lambda a: a.total_score, reverse=True)[:BORDERLINE_COUNT]:
        art.secim_tipi = "Sınırda"
        sample.append(art)

    # Stage-1 rejects have no Turkish analysis yet, so each costs one deep call.
    for art in random.sample(stage1_rejects, min(RANDOM_COUNT, len(stage1_rejects))):
        try:
            data = _deep_analyze(
                client, art, system_prompt,
                knowledge_base_block(neighbours.get(id(art)), detailed=True),
            )
            scored_art = _to_scored(art, data, secim_tipi="Rastgele")
            scored_art.image_url = art.image_url
            sample.append(scored_art)
        except Exception as exc:
            print(f"   [WARN] Rastgele örnek analiz edilemedi ({exc})")

    if sample:
        print(f"\n   🎲 İnceleme örneklemi: " + ", ".join(
            f"{s.secim_tipi} {s.total_score:.0f}pt" for s in sample))
    return sample


def _check_grounding(client, articles: list["ScoredArticle"]) -> None:
    """Flag (never drop) selected articles whose analysis asserts unsourced facts."""
    import grounding

    print(f"\n   🔍 Uydurma kontrolü — {len(articles)} seçilen haber...")
    for art in articles:
        try:
            claims = grounding.check(client, art.title, art.raw_summary, {
                "ozet": art.ozet,
                "neden_onemli_sektorel": art.neden_onemli_sektorel,
                "stratejik_cikarim": art.stratejik_cikarim,
            })
            art.supheli_iddialar = grounding.format_claims(claims)
            if claims:
                print(f"   ⚠️  {len(claims)} şüpheli iddia | {art.title[:55]}")
                for line in art.supheli_iddialar.splitlines():
                    print(f"        {line[:160]}")
        except Exception as exc:
            print(f"   [WARN] Uydurma kontrolü yapılamadı ({exc}) | {art.title[:50]}")
