"""Claude-powered multi-dimensional article scorer and Turkish analyst.

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
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import anthropic

from fetcher import RawArticle

WEIGHTS = {"A": 10, "B": 10, "C": 8, "D": 6, "E": 5}
BONUS_MULTIPLIER = 1.25
BONUS_MIN_SCORE = 6
BONUS_MIN_CATEGORIES = 2
THRESHOLD_KRITIK = 200
THRESHOLD_YUKSEK = 120
THRESHOLD_ORTA = 70

SYSTEM_PROMPT = """Sen bir bankanın inovasyon ekibi için çalışan stratejik teknoloji analistisin.
Görevin: verilen haber başlığı ve özetini okuyup 5 kategoride puanlayarak Türkçe analiz üretmek.

PUANLAMA KATEGORİLERİ (her biri 0-10):
A = Yapay Zeka & LLM (yeni modeller, agentic AI, AI araçları, AI altyapısı)
B = Fintech & Ödeme Sistemleri (embedded finance, anlık ödeme, BNPL, neobank, RegTech, açık bankacılık)
C = Startup Funding / M&A / IPO ($10M+ yatırım, satın alma, halka arz, unicorn)
D = Kripto & Web3 (stablecoin, CBDC, tokenizasyon, kurumsal blockchain)
E = Genel Tech & Big Tech (Apple/Google/Meta/Amazon/Microsoft ürün lansmanları)

KURALLAR:
- 0 = hiç alakası yok, 10 = o kategorinin tam kalbinde
- Spekülatif/söylenti haberler için tüm puanları 3'ün altında tut
- Somut veri (miktar, kullanıcı sayısı, metrik) içeren haberlere daha yüksek puan ver

ÇIKTI: Aşağıdaki JSON formatında yanıt ver, başka metin ekleme:
{
  "score_a": <int 0-10>,
  "score_b": <int 0-10>,
  "score_c": <int 0-10>,
  "score_d": <int 0-10>,
  "score_e": <int 0-10>,
  "ozet": "<3-4 cümle Türkçe özet: kim, ne yaptı, hangi ölçekte, hangi bağlamda — sayılarla destekle>",
  "neden_onemli_sektorel": "<1-2 cümle: teknoloji/finans/inovasyon ekosistemi için ne anlama geliyor>",
  "neden_onemli_bankacilik": "<1-2 cümle: banka kurumu için fırsat, tehdit veya öğrenme noktası>",
  "stratejik_cikarim": "<TEK cümle: 'Bu yüzden X yapmalıyız' veya 'Bu nedenle Y izlenmeli' formatında>"
}"""


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


def _analyze_single(client: anthropic.Anthropic, article: RawArticle) -> dict:
    """Call Claude for one article. Returns parsed JSON dict."""
    user_msg = f"Başlık: {article.title}\n\nKaynak Özeti: {article.summary or '(özet yok)'}"
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw_text = response.content[0].text.strip()
    # Strip markdown code fences if present
    raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw_text)


def analyze_articles(
    raw_articles: list[RawArticle],
    min_total: float = THRESHOLD_ORTA,
    max_results: int = 10,
) -> list[ScoredArticle]:
    """Score and analyze articles via Claude API.

    Returns up to max_results articles with total_score >= min_total,
    sorted by total_score descending.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")

    client = anthropic.Anthropic(api_key=api_key)
    scored: list[ScoredArticle] = []

    for i, art in enumerate(raw_articles, 1):
        print(f"  [{i}/{len(raw_articles)}] Analiz ediliyor: {art.title[:70]}...")
        try:
            data = _analyze_single(client, art)
            a, b, c, d, e = (
                int(data.get("score_a", 0)),
                int(data.get("score_b", 0)),
                int(data.get("score_c", 0)),
                int(data.get("score_d", 0)),
                int(data.get("score_e", 0)),
            )
            total, has_bonus = _compute_total(a, b, c, d, e)

            if total < min_total:
                print(f"       → Eşik altı ({total:.0f}), elendi.")
                continue

            level = _signal_level(total)
            print(f"       → {level} | Skor: {total:.0f}")

            scored.append(
                ScoredArticle(
                    title=art.title,
                    url=art.url,
                    raw_summary=art.summary,
                    published=art.published,
                    source=art.source,
                    score_a=a,
                    score_b=b,
                    score_c=c,
                    score_d=d,
                    score_e=e,
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
            print(f"       → [HATA] {exc}")
            continue

    scored.sort(key=lambda a: a.total_score, reverse=True)
    return scored[:max_results]
