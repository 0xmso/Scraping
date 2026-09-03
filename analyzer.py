"""Claude-powered multi-dimensional article scorer and Turkish analyst.

Two-stage architecture for token efficiency:
  Stage 1 — Haiku 4.5 (cheap): score every candidate in 5 categories (JSON-only)
  Stage 2 — Opus 4.7 (deep):  full Turkish analysis for articles passing threshold

Scoring system:
  A — Yapay Zeka & LLM        ×10
  B — Fintech & Ödeme         ×10
  C — Startup Funding/M&A/IPO ×8
  D — Kripto & Web3            ×6
  E — Genel Tech & Big Tech    ×5

Thresholds:
  200+  → 🔴 KRİTİK
  120-199 → 🟠 YÜKSEK
  70-119  → 🟡 ORTA
  <70   → elenir

Cross-category bonus: 2+ categories with score ≥ 6 → ×1.25
"""

import os
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import llm
from fetcher import RawArticle

# ── Models ────────────────────────────────────────────────────────────────────
# Resolved per backend (Bedrock vs first-party) — see llm.py.
STAGE1_TIER = "fast"   # cheap pre-filter (Haiku)
STAGE2_TIER = "deep"   # deep analysis only on passing articles (Opus)

# ── Scoring constants ─────────────────────────────────────────────────────────
WEIGHTS = {"A": 10, "B": 10, "C": 8, "D": 6, "E": 5}
BONUS_MULTIPLIER = 1.25
BONUS_MIN_SCORE = 6
BONUS_MIN_CATEGORIES = 2
THRESHOLD_KRITIK = 200
THRESHOLD_YUKSEK = 120
THRESHOLD_ORTA = 70

# Stage-1 safety margin: pass to stage 2 if Haiku-scored total >= min_total * 0.7.
# Keeps borderline cases alive so Opus can re-score with full analysis.
STAGE1_SAFETY_MARGIN = 0.7

_PROMPT_FILE = Path(__file__).parent / "system_prompt.txt"


# ── Stage 1 prompt (compact, scoring only) ────────────────────────────────────
# Inlined and stable so Haiku call is fast and predictable. The auto-tuned
# system_prompt.txt only applies to the deeper Stage 2 analysis.
HAIKU_SCORING_PROMPT = """Sen bir teknoloji haber puanlayıcısısın. Verilen haber başlığı ve özetini okuyup 5 kategoride 0-10 arası puanla.

KATEGORİLER:
A = Yapay Zeka & LLM (modeller, agentic AI, AI altyapısı)
B = Fintech & Ödeme Sistemleri (embedded finance, anlık ödeme, neobank, açık bankacılık)
C = Startup Funding / M&A / IPO ($10M+ yatırım, satın alma, halka arz, unicorn)
D = Kripto & Web3 (stablecoin, CBDC, tokenizasyon, kurumsal blockchain)
E = Genel Tech & Big Tech (Apple/Google/Meta/Amazon/Microsoft ürün lansmanları)

KURALLAR:
- 0 = hiç alakası yok, 10 = o kategorinin tam kalbinde
- Spekülatif/söylenti haberler için tüm puanları 3'ün altında tut
- Sayılarla (miktar, kullanıcı sayısı, metrik) desteklenen haberlere daha yüksek puan ver

Sadece JSON döndür, başka metin ekleme."""


# ── JSON schemas for guaranteed structured output ─────────────────────────────
# Note: Numerical constraints (minimum/maximum) are NOT supported by structured
# outputs — the API rejects them. The 0-10 range is enforced via the system prompt.
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
        "neden_onemli_bankacilik": {"type": "string"},
        "stratejik_cikarim": {"type": "string"},
    },
    "required": [
        "score_a", "score_b", "score_c", "score_d", "score_e",
        "ozet", "neden_onemli_sektorel", "neden_onemli_bankacilik", "stratejik_cikarim",
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
    raw_summary: str
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
    neden_onemli_bankacilik: str = ""
    stratejik_cikarim: str = ""


def _compute_total(a: int, b: int, c: int, d: int, e: int) -> tuple[float, bool]:
    raw = a * 10 + b * 10 + c * 8 + d * 6 + e * 5
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


def _user_message(article: RawArticle) -> str:
    return f"Başlık: {article.title}\n\nKaynak Özeti: {article.summary or '(özet yok)'}"


def _parse_json_response(response) -> dict:
    """Extract JSON from the response's first *text* block.

    Must scan rather than index content[0]: with adaptive thinking the response
    can lead with a thinking block (Opus 4.6 defaults to display="summarized"),
    so content[0] is not necessarily the answer. Tolerates markdown fences.
    """
    text = next(
        (b.text for b in response.content if getattr(b, "type", None) == "text"), None
    )
    if text is None:
        raise ValueError("Yanıtta metin bloğu yok")
    raw = text.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def _quick_score(client, article: RawArticle) -> dict:
    """Stage 1: Haiku call — scores only, no Turkish analysis.

    Note: cache_control is set on the system prompt but only activates once the
    prompt exceeds 4096 tokens (current is ~400). It's a no-op for now, but
    means we get caching for free if the prompt grows via few-shot examples.
    """
    response = client.messages.create(
        model=llm.model(STAGE1_TIER),
        max_tokens=200,
        system=[
            {
                "type": "text",
                "text": HAIKU_SCORING_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        output_config={
            "format": {"type": "json_schema", "schema": STAGE1_SCHEMA}
        },
        messages=[{"role": "user", "content": _user_message(article)}],
    )
    return _parse_json_response(response)


def _deep_analyze(client, article: RawArticle, system_prompt: str) -> dict:
    """Stage 2: Opus call — full Turkish analysis + final scoring.

    Opus re-scores too: stage-1 Haiku scores were a pre-filter, this is the
    authoritative scoring that drives the final output. The same system_prompt.txt
    that's auto-tuned by the feedback loop is used here.
    """
    response = client.messages.create(
        model=llm.model(STAGE2_TIER),
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        output_config={
            "effort": "medium",   # balance cost and quality
            "format": {"type": "json_schema", "schema": STAGE2_SCHEMA},
        },
        thinking={"type": "adaptive"},   # Claude decides when/how much to think
        messages=[{"role": "user", "content": _user_message(article)}],
    )
    return _parse_json_response(response)


def analyze_articles(
    raw_articles: list[RawArticle],
    min_total: float = THRESHOLD_ORTA,
    max_results: int = 10,
) -> list[ScoredArticle]:
    """Two-stage scoring pipeline.

    Stage 1: Haiku scores every candidate (cheap)
    Stage 2: Opus deep-analyzes only articles passing the safety-margin gate
    """
    client = llm.get_client()
    system_prompt = _load_system_prompt()
    print(f"   Backend: {llm.backend_name()}")
    print(f"   Modeller: {llm.model(STAGE1_TIER)} → {llm.model(STAGE2_TIER)}")
    print(f"   Prompt yüklendi: {_PROMPT_FILE.name} ({len(system_prompt)} karakter)")

    # ── Stage 1: Haiku pre-filter ────────────────────────────────────────────
    stage1_pass: list[tuple[RawArticle, float]] = []
    stage1_threshold = min_total * STAGE1_SAFETY_MARGIN
    print(f"\n   🚀 Stage 1 (Haiku) — {len(raw_articles)} aday hızlı puanlanıyor"
          f" (eşik ≥ {stage1_threshold:.0f})...")

    for i, art in enumerate(raw_articles, 1):
        try:
            data = _quick_score(client, art)
            a, b, c, d, e = (
                int(data["score_a"]), int(data["score_b"]), int(data["score_c"]),
                int(data["score_d"]), int(data["score_e"]),
            )
            total, _ = _compute_total(a, b, c, d, e)
            verdict = "✓ geçti" if total >= stage1_threshold else "✗ elendi"
            print(f"   [{i:2}/{len(raw_articles)}] {total:5.0f}pt {verdict} | {art.title[:60]}")
            if total >= stage1_threshold:
                stage1_pass.append((art, total))
        except Exception as exc:
            print(f"   [{i:2}/{len(raw_articles)}] [HATA] {exc} — {art.title[:60]}")

    # Sort by stage-1 total desc; cap deep analyses to a generous multiple of max_results
    # to avoid wasting Opus calls if stage 1 is too lenient
    stage1_pass.sort(key=lambda t: t[1], reverse=True)
    deep_budget = max_results * 2
    candidates = [art for art, _ in stage1_pass[:deep_budget]]

    print(f"\n   🧠 Stage 2 (Opus) — {len(candidates)} makale derin analiz...")

    # ── Stage 2: Opus deep analysis ──────────────────────────────────────────
    scored: list[ScoredArticle] = []
    for i, art in enumerate(candidates, 1):
        try:
            data = _deep_analyze(client, art, system_prompt)
            a, b, c, d, e = (
                int(data["score_a"]), int(data["score_b"]), int(data["score_c"]),
                int(data["score_d"]), int(data["score_e"]),
            )
            total, has_bonus = _compute_total(a, b, c, d, e)

            if total < min_total:
                print(f"   [{i:2}/{len(candidates)}] Eşik altı ({total:.0f}), elendi.")
                continue

            level = _signal_level(total)
            print(f"   [{i:2}/{len(candidates)}] {level} | {total:.0f}pt | {art.title[:60]}")

            scored.append(
                ScoredArticle(
                    title=art.title,
                    url=art.url,
                    raw_summary=art.summary,
                    published=art.published,
                    source=art.source,
                    score_a=a, score_b=b, score_c=c, score_d=d, score_e=e,
                    total_score=total,
                    has_bonus=has_bonus,
                    signal_level=level,
                    ozet=data.get("ozet", ""),
                    neden_onemli_sektorel=data.get("neden_onemli_sektorel", ""),
                    neden_onemli_bankacilik=data.get("neden_onemli_bankacilik", ""),
                    stratejik_cikarim=data.get("stratejik_cikarim", ""),
                )
            )
        except Exception as exc:
            print(f"   [{i:2}/{len(candidates)}] [HATA] {exc}")
            continue

    scored.sort(key=lambda a: a.total_score, reverse=True)
    return scored[:max_results]
