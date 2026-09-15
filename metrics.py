"""Learning metrics: is the model actually getting better at Kübra's judgement?

Written once per digest run to the Öğrenme Metrikleri database, one row per day
(re-runs on the same day update it). Every figure is over the trailing 30 days,
because Kübra labels after the run — yesterday's rate keeps moving as labels
arrive, so a snapshot is only meaningful as part of the trend.

  Kitle Uyumu         model's hidden audience guess == Kübra's Etiket
  Alakasız Oranı      selected articles she labelled Alakasız (wrong picks)
  Kaçırılan Haber     review-sample articles (rejected) she labelled relevant
  Uydurma İşaret      selected articles whose analysis was flagged
  Tam Metin           selected articles analysed from the full article
  Etiketleme Oranı    selected articles that have any label yet
  Bilgi Tabanı        all-time labelled examples available for retrieval
"""

import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import httpx

NOTION_VERSION = "2022-06-28"
WINDOW_DAYS = 30
RELEVANT = {"Dijital Ekipler", "Üst Yönetim", "İkisi De"}


@dataclass
class Metrics:
    kitle_uyumu: Optional[float]
    kitle_n: int
    alakasiz_orani: Optional[float]
    kacirilan_orani: Optional[float]
    kacirilan_n: int
    uydurma_orani: Optional[float]
    tam_metin_orani: Optional[float]
    etiketleme_orani: Optional[float]
    secilen: int
    bilgi_tabani: int


def _headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['NOTION_TOKEN']}",
            "Notion-Version": NOTION_VERSION, "Content-Type": "application/json"}


def _query(db_id: str, body: dict) -> list[dict]:
    url = f"https://api.notion.com/v1/databases/{db_id}/query"
    pages, cursor = [], None
    with httpx.Client(timeout=30) as client:
        while True:
            if cursor:
                body["start_cursor"] = cursor
            resp = client.post(url, headers=_headers(), json={**body, "page_size": 100})
            resp.raise_for_status()
            data = resp.json()
            pages += data["results"]
            if not data.get("has_more"):
                return pages
            cursor = data["next_cursor"]


def _sel(props: dict, name: str) -> str:
    v = props.get(name, {}).get("select")
    return v["name"] if v else ""


def _ratio(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def compute(rows: list[dict], knowledge_base_size: int) -> Metrics:
    """Pure computation over Articles rows (Notion page dicts) — testable offline."""
    selected, samples = [], []
    for page in rows:
        p = page["properties"]
        (samples if _sel(p, "Seçim Tipi") in ("Sınırda", "Rastgele") else selected).append(p)

    labelled_sel = [p for p in selected if _sel(p, "Etiket")]
    both = [p for p in selected + samples if _sel(p, "Etiket") and _sel(p, "Model Tahmini")]
    labelled_samples = [p for p in samples if _sel(p, "Etiket")]
    checked = [p for p in selected if _sel(p, "Uydurma Kontrolü")]

    return Metrics(
        kitle_uyumu=_ratio(sum(_sel(p, "Etiket") == _sel(p, "Model Tahmini") for p in both), len(both)),
        kitle_n=len(both),
        alakasiz_orani=_ratio(sum(_sel(p, "Etiket") == "Alakasız" for p in labelled_sel), len(labelled_sel)),
        kacirilan_orani=_ratio(sum(_sel(p, "Etiket") in RELEVANT for p in labelled_samples), len(labelled_samples)),
        kacirilan_n=len(labelled_samples),
        uydurma_orani=_ratio(sum(_sel(p, "Uydurma Kontrolü") == "Şüpheli" for p in checked), len(checked)),
        tam_metin_orani=_ratio(sum(bool(p.get("Tam Metin", {}).get("checkbox")) for p in selected), len(selected)),
        etiketleme_orani=_ratio(len(labelled_sel), len(selected)),
        secilen=len(selected),
        bilgi_tabani=knowledge_base_size,
    )


def collect() -> Metrics:
    since = (datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)).isoformat()
    articles = os.environ["NOTION_ARTICLES_DB_ID"]
    rows = _query(articles, {"filter": {"timestamp": "created_time", "created_time": {"after": since}}})
    kb = len(_query(articles, {"filter": {"or": [
        {"property": "Etiket", "select": {"is_not_empty": True}},
        {"property": "Feedback", "select": {"is_not_empty": True}},
    ]}}))
    if os.environ.get("NOTION_EXAMPLES_DB_ID"):
        kb += len(_query(os.environ["NOTION_EXAMPLES_DB_ID"],
                         {"filter": {"property": "Durum", "select": {"equals": "İşlendi"}}}))
    return compute(rows, kb)


def _pct(v: Optional[float]) -> dict:
    return {"number": v}


def write(m: Metrics, day: Optional[date] = None) -> None:
    """Upsert today's row, so a re-run replaces rather than duplicates."""
    db = os.environ["NOTION_METRICS_DB_ID"]
    day = day or datetime.now(timezone.utc).date()
    props = {
        "Tarih": {"title": [{"type": "text", "text": {"content": day.isoformat()}}]},
        "Gün": {"date": {"start": day.isoformat()}},
        "Kitle Uyumu": _pct(m.kitle_uyumu), "Kitle Uyumu (n)": {"number": m.kitle_n},
        "Alakasız Oranı": _pct(m.alakasiz_orani),
        "Kaçırılan Haber Oranı": _pct(m.kacirilan_orani), "Kaçırılan (n)": {"number": m.kacirilan_n},
        "Uydurma İşaret Oranı": _pct(m.uydurma_orani),
        "Tam Metin Oranı": _pct(m.tam_metin_orani),
        "Etiketleme Oranı": _pct(m.etiketleme_orani),
        "Seçilen Haber": {"number": m.secilen},
        "Bilgi Tabanı": {"number": m.bilgi_tabani},
    }
    existing = _query(db, {"filter": {"property": "Gün", "date": {"equals": day.isoformat()}}})
    with httpx.Client(timeout=30) as client:
        if existing:
            client.patch(f"https://api.notion.com/v1/pages/{existing[0]['id']}",
                         headers=_headers(), json={"properties": props}).raise_for_status()
        else:
            client.post("https://api.notion.com/v1/pages", headers=_headers(),
                        json={"parent": {"database_id": db}, "properties": props}).raise_for_status()


def summary(m: Metrics) -> str:
    f = lambda v: "—" if v is None else f"%{v * 100:.0f}"
    return (f"Kitle uyumu {f(m.kitle_uyumu)} (n={m.kitle_n}) · Alakasız {f(m.alakasiz_orani)} · "
            f"Kaçırılan {f(m.kacirilan_orani)} (n={m.kacirilan_n}) · Uydurma {f(m.uydurma_orani)} · "
            f"Tam metin {f(m.tam_metin_orani)} · Etiketlenen {f(m.etiketleme_orani)} · "
            f"Bilgi tabanı {m.bilgi_tabani}")


def record_safely() -> None:
    """Collect and write; never let metrics break the digest (fail open)."""
    if not os.environ.get("NOTION_METRICS_DB_ID"):
        print("   [WARN] NOTION_METRICS_DB_ID yok — metrikler yazılmadı.")
        return
    try:
        m = collect()
        write(m)
        print(f"📈 {summary(m)}")
    except Exception as exc:
        print(f"   [WARN] Metrikler yazılamadı ({exc})")
