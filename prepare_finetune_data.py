"""Prepares Bedrock fine-tuning data from Kübra's accumulated judgments.

Not distillation — Bedrock distillation only needs prompts (a teacher model
generates the answers), which would just compress Stage 1's own behaviour,
including whatever it already gets wrong. This is supervised fine-tuning:
Claude 3 Haiku (the only Anthropic model Bedrock currently fine-tunes, and
only in us-west-2 — Sonnet 5 / Haiku 4.5 / Opus 5 aren't fine-tunable there)
trained on Kübra's actual corrections, so it encodes the cases where she
disagreed with the model rather than just imitating it faster.

Scope is deliberately narrow: a binary alakalı/alakasız classifier, not a
replacement for Stage 1's 5-category scoring. The only ground truth we hold
is a correct/incorrect verdict (Feedback) or an audience label (Etiket) —
neither tells us what Kübra's A/B/C/D/E breakdown would have been, so
reconstructing one would be invented, not learned. A well-scoped binary task
is also a better fit for a small fine-tuned model than a 5-way weighted score.

Two label sources, both real ground truth:
  - Feedback (✅/❌) on articles that were selected and shown to her.
  - Etiket on review-sample rows (Seçim Tipi = Sınırda/Rastgele) — these are
    exactly the cases Stage 1 rejected. An audience label there is a "kaçırılan
    haber" Stage 1 got wrong; Alakasız confirms the rejection was right. This
    is the highest-value signal, since it's the failure mode Stage 1 can't see
    on its own.

This only writes local JSONL files — no AWS resources are created and no
training job is started. Launching one needs an S3 bucket, an IAM role Bedrock
can assume, and the us-west-2 region; that's an infrastructure/cost decision
for the user, not something this script does on its own.

Usage:
    python3 prepare_finetune_data.py [--out-dir out/finetune]
"""

import json
import os
import random
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import httpx

NOTION_VERSION = "2022-06-28"
LOOKBACK_DAYS = 90   # wider than the optimizer's 30 — fine-tuning wants volume, not recency
VAL_FRACTION = 0.15
MIN_RECOMMENDED = 32   # Bedrock's stated minimum for Claude 3 Haiku fine-tuning

SYSTEM_PROMPT = """Sen bir bankanın dijital inovasyon ekibi için haber alaka düzeyini belirleyen bir
sınıflandırıcısın. Sana bir haberin başlığı ve özeti verilecek. Görevin: bu haberin bankacılık
veya dijital ekiplerin gündemiyle alakalı olup olmadığına karar vermek.

Sadece JSON döndür: {"alakali": true veya false}"""


def _headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['NOTION_TOKEN']}",
            "Notion-Version": NOTION_VERSION, "Content-Type": "application/json"}


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
            pages += data.get("results", [])
            if not data.get("has_more"):
                return pages
            cursor = data["next_cursor"]


def _text(props: dict, name: str) -> str:
    prop = props.get(name, {})
    return "".join(b.get("plain_text", "") for b in (prop.get("title") or prop.get("rich_text") or []))


def _select(props: dict, name: str) -> str:
    v = props.get(name, {}).get("select")
    return v["name"] if v else ""


_RELEVANT_LABELS = {"Dijital Ekipler", "Üst Yönetim", "İkisi De"}


def _example(props: dict, source: str) -> Optional[dict]:
    """One (title, summary) -> alakali verdict, or None if this row has no usable label."""
    feedback = _select(props, "Feedback")
    etiket = _select(props, "Etiket")

    if feedback == "✅ Doğru seçim":
        alakali = True
    elif feedback == "❌ Yanlış seçim":
        alakali = False
    elif etiket in _RELEVANT_LABELS:
        alakali = True
    elif etiket == "Alakasız":
        alakali = False
    else:
        return None  # "⚠️ Skor yanlış" or no label — not a relevance verdict either way

    title = _text(props, "Name")
    summary = _text(props, "Özet")
    if not title:
        return None

    note = _text(props, "Feedback Notu")
    target = {"alakali": alakali}
    if note.strip():
        target["not"] = note.strip()   # only Kübra's own words — never invented

    return {
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": f"Başlık: {title}\n\nÖzet: {summary or '(özet yok)'}"},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
        "_meta": {"source": source, "alakali": alakali, "title": title},
    }


def collect_examples() -> list[dict]:
    since = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).isoformat()
    db_id = os.environ["NOTION_ARTICLES_DB_ID"]

    feedback_rows = _query_all(db_id, {"and": [
        {"property": "Feedback", "select": {"is_not_empty": True}},
        {"timestamp": "created_time", "created_time": {"after": since}},
    ]})
    sample_rows = _query_all(db_id, {"and": [
        {"or": [{"property": "Seçim Tipi", "select": {"equals": t}} for t in ("Sınırda", "Rastgele")]},
        {"property": "Etiket", "select": {"is_not_empty": True}},
        {"timestamp": "created_time", "created_time": {"after": since}},
    ]})

    examples, seen_titles = [], set()
    for rows, label in ((feedback_rows, "feedback"), (sample_rows, "review_sample")):
        for page in rows:
            ex = _example(page["properties"], label)
            if ex and ex["_meta"]["title"] not in seen_titles:
                seen_titles.add(ex["_meta"]["title"])
                examples.append(ex)
    return examples


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if k != "_meta"}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")

    out_dir = Path(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out-dir=")), "out/finetune"))

    print("📥 Notion'dan etiketli örnekler toplanıyor...")
    examples = collect_examples()

    by_source = {"feedback": 0, "review_sample": 0}
    by_label = {True: 0, False: 0}
    for ex in examples:
        by_source[ex["_meta"]["source"]] += 1
        by_label[ex["_meta"]["alakali"]] += 1

    print(f"   Toplam: {len(examples)} örnek")
    print(f"   Kaynak: Feedback {by_source['feedback']} · İnceleme örneği (kaçırılan/doğru eleme) {by_source['review_sample']}")
    print(f"   Dağılım: alakalı {by_label[True]} · alakasız {by_label[False]}")

    if len(examples) < MIN_RECOMMENDED:
        print(f"\n⚠️  Bedrock'un belirttiği minimum {MIN_RECOMMENDED} örneğin altındasınız "
              f"({len(examples)}). Dosyalar yine de yazılacak ama bir eğitim işi şu an başarısız olur.")

    random.Random(42).shuffle(examples)   # fixed seed: re-running reproduces the same split
    split = max(1, round(len(examples) * VAL_FRACTION)) if examples else 0
    val, train = examples[:split], examples[split:]

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "validation.jsonl", val)
    print(f"\n✅ {out_dir}/train.jsonl — {len(train)} örnek")
    print(f"✅ {out_dir}/validation.jsonl — {len(val)} örnek")

    print(
        "\n📋 Bir eğitim işi başlatmak için (henüz yapılmadı, AWS kaynağı ve ücret gerektirir):\n"
        "   1. us-west-2 bölgesinde bir S3 bucket (bu iki dosyayı yükleyip)\n"
        "   2. Bedrock'un bucket'a erişebileceği bir IAM rolü\n"
        "   3. Bedrock konsolu → Custom models → Fine-tune → Claude 3 Haiku, us-west-2\n"
        "   Bu üç adım altyapı/izin değişikliği ve gerçek ücret içerdiği için onay istiyorum."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
