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


def _quick_score(client, article: RawArticle) -> dict:
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
        user_content=_user_message(article),
        schema=STAGE1_SCHEMA,
        tool_name="haber_puanla",
        tool_description="Haberi 5 kategoride 0-10 arası puanla.",
    )


def _deep_analyze(client, article: RawArticle, system_prompt: str) -> dict:
    """Stage 2: deep call — full Turkish analysis + final scoring.

    Stage 2 re-scores too: stage-1 scores were only a pre-filter, this is the
    authoritative scoring that drives the final output. The same system_prompt.txt
    that's auto-tuned by the feedback loop is used here.
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
        user_content=_user_message(article),
        schema=STAGE2_SCHEMA,
        tool_name="haber_analiz",
        tool_description="Haberi puanla ve Türkçe stratejik analiz üret.",
        effort="high",
        thinking=True,   # Claude decides when/how much to think
    )


def analyze_articles(
    raw_articles: list[RawArticle],
    min_total: float = THRESHOLD_ORTA,
    max_results: int = 10,
) -> list[ScoredArticle]:
    """Two-stage scoring pipeline.

    Stage 1: cheap model scores every candidate
    Stage 2: deep model analyzes only articles passing the safety-margin gate
    """
    client = llm.get_client()
    system_prompt = _load_system_prompt()
    print(f"   Backend: {llm.backend_name()}")
    print(f"   Modeller: {llm.model(STAGE1_TIER)} → {llm.model(STAGE2_TIER)}")
    print(f"   Prompt yüklendi: {_PROMPT_FILE.name} ({len(system_prompt)} karakter)")

    # ── Stage 1: pre-filter ──────────────────────────────────────────────────
    stage1_pass: list[tuple[RawArticle, float]] = []
    stage1_threshold = min_total * STAGE1_SAFETY_MARGIN
    print(f"\n   🚀 Stage 1 — {len(raw_articles)} aday hızlı puanlanıyor"
          f" (eşik ≥ {stage1_threshold:.0f})...")

    for i, art in enumerate(raw_articles, 1):
        try:
            data = _quick_score(client, art)
            a, b, c, d, e = _scores(data)
            total, _ = _compute_total(a, b, c, d, e)
            verdict = "✓ geçti" if total >= stage1_threshold else "✗ elendi"
            print(f"   [{i:2}/{len(raw_articles)}] {total:5.0f}pt {verdict} | {art.title[:60]}")
            if total >= stage1_threshold:
                stage1_pass.append((art, total))
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
    for i, art in enumerate(candidates, 1):
        try:
            data = _deep_analyze(client, art, system_prompt)
            a, b, c, d, e = _scores(data)
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
