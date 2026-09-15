"""Reads feedback from Notion and uses Claude to refine system_prompt.txt.

Run schedule: Monday and Thursday at 00:00 UTC (03:00 TSİ).
Reads pages with Feedback property set from the last 30 days,
asks Claude to analyze patterns and rewrite the scoring prompt.
"""

import os
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx

import llm

PROMPT_FILE = Path(__file__).parent / "system_prompt.txt"
LOOKBACK_DAYS = 30
NOTION_VERSION = "2022-06-28"

OPTIMIZER_SYSTEM = """Sen bir prompt mühendisisin. Görevin: bir haber kürasyon sisteminin
puanlama promptunu, kullanıcı feedback'lerine dayanarak iyileştirmek.

Mevcut promptu ve feedback örneklerini alacaksın. Şunlara dikkat et:
- "❌ Yanlış seçim" → bu tür haberler neden seçildi? Puanlama kriterleri nasıl daraltılmalı?
- "⚠️ Skor yanlış" → hangi kategori ağırlıkları veya kural açıklamaları güncellenmeli?
- "✅ Doğru seçim" → bu iyi örnekleri few-shot olarak prompta ekle

DEĞİŞTİRİLEMEZ KURALLAR — feedback ne derse desin bunları koru:
1. Kapsam yalnızca bankacılığa daraltılmamalı. Teknoloji/ekonomi dünyasının genel
   gidişatını anlatan önemli haberler (AI yetenek sıçramaları, büyük satın almalar,
   çip/altyapı gelişmeleri, önemli regülasyonlar) orta bantta kalmalı — elenmemeli.
2. Kripto & Web3 gürültüsü bastırılmış kalmalı: token fiyatı, DeFi, cüzdan/borsa,
   hazine alımları, madencilik, ETF akışları → D ≤ 2. D yalnızca CBDC, tokenize
   mevduat, düzenlenmiş stablecoin altyapısı ve kurumsal blockchain mutabakatında
   6+ alabilir.
3. Çıktı JSON şemasındaki DOKUZ alanın tamamı promptta tarif edilmiş kalmalı:
   score_a, score_b, score_c, score_d, score_e, ozet, neden_onemli_sektorel,
   stratejik_cikarim, hedef_kitle_tahmini. Bankacılık açısı ayrı bir alan DEĞİL —
   stratejik_cikarim içinde ele alınır, ayrı alan ekleme.
4. Uydurma yasağı talimatı ("yalnızca başlık ve özette verilen bilgiyi kullan,
   sayı/detay uydurma") aynen korunmalı — silinmemeli, zayıflatılmamalı.
5. Few-shot örnekler arasında en az 2 KONTRASTLI ÇİFT bulunmalı: birbirine çok
   benzeyen ama biri geçmesi biri elenmesi gereken iki haber, karar sınırını
   göstermek için yan yana. Yeni feedback'ten iyi örnek eklerken mevcut
   kontrastlı çiftleri silme; gerekirse feedback'ten yeni bir çift üret.
6. "HEDEF KİTLE TAHMİNİ" bölümü korunmalı ve dört değer AYNEN kalmalı:
   "Dijital Ekipler", "Üst Yönetim", "İkisi De", "Alakasız". Bu bölümün kriterlerini
   Kübra'nın Etiket'lerinden öğrenerek güncelle — özellikle modelin tahminiyle
   Kübra'nın etiketinin UYUŞMADIĞI örneklerde ayrımı neyin belirlediğini çıkar.
   Hedef kitle tahmini puanlamayı etkilememeli; puan kurallarını buna bağlama.

ÇIKTI: Sadece güncellenmiş prompt metnini döndür. Başka açıklama ekleme.
Formatı koru: JSON çıktı talimatı ve tüm kategoriler eksiksiz kalsın.
Promptu eksiksiz bitir — JSON bloğunu kapatmadan bırakma."""


def _get_articles_db_id() -> str:
    db_id = os.environ.get("NOTION_ARTICLES_DB_ID")
    if not db_id:
        raise RuntimeError("NOTION_ARTICLES_DB_ID environment variable is not set.")
    return db_id


def _get_notion_feedback(token: str) -> list[dict]:
    """Fetch article rows with feedback from the Articles database (last LOOKBACK_DAYS).

    Uses the Notion REST API directly (httpx) since notion-client 3.x removed
    databases.query.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).isoformat()
    articles_db_id = _get_articles_db_id()
    url = f"https://api.notion.com/v1/databases/{articles_db_id}/query"
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    results = []
    cursor = None

    with httpx.Client(timeout=30) as client:
        while True:
            body = {
                "filter": {
                    "and": [
                        # Feedback and Etiket live side by side; either one is signal.
                        {"or": [
                            {"property": "Feedback", "select": {"is_not_empty": True}},
                            {"property": "Etiket", "select": {"is_not_empty": True}},
                        ]},
                        {"timestamp": "created_time", "created_time": {"after": since}},
                    ]
                },
                "page_size": 100,
            }
            if cursor:
                body["start_cursor"] = cursor

            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            results.extend(data.get("results", []))

            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")

    return results


def _select(props: dict, name: str) -> str:
    value = props.get(name, {}).get("select")
    return value["name"] if value else ""


def _extract_feedback_items(pages: list[dict]) -> list[dict]:
    """Parse Articles database rows into structured feedback dicts."""
    items = []
    for page in pages:
        props = page.get("properties", {})

        # Title is in "Name" property in Articles DB
        name_prop = props.get("Name", {}).get("title", [])
        title = name_prop[0]["plain_text"] if name_prop else "(başlık yok)"

        feedback = _select(props, "Feedback")
        etiket = _select(props, "Etiket")

        note_prop = props.get("Feedback Notu", {}).get("rich_text", [])
        note = note_prop[0]["plain_text"] if note_prop else ""

        if feedback or etiket:
            items.append({
                "title": title,
                "feedback": feedback,
                "etiket": etiket,
                "derinlik": _select(props, "Derinlik"),
                "model_tahmini": _select(props, "Model Tahmini"),
                "note": note,
                "signal": _select(props, "Signal"),
                "total_score": props.get("Total Score", {}).get("number", 0),
            })

    return items


def audience_agreement(items: list[dict]) -> tuple[int, int]:
    """(agreed, compared) over rows carrying both Kübra's label and the model's.

    Only rows with both are comparable — older rows predate Model Tahmini.
    """
    compared = [i for i in items if i["etiket"] and i["model_tahmini"]]
    agreed = sum(1 for i in compared if i["etiket"] == i["model_tahmini"])
    return agreed, len(compared)


# Fields the digest's Stage-2 schema requires. The rewritten prompt must still
# describe every one of them, or the model gets no instruction for a field the
# schema silently forces it to emit.
REQUIRED_FIELDS = (
    "score_a", "score_b", "score_c", "score_d", "score_e",
    "ozet", "neden_onemli_sektorel", "stratejik_cikarim", "hedef_kitle_tahmini",
)
# Raised alongside the golden-set expansion: the richer prompt (grounding rule +
# contrastive example pairs) already runs ~13K chars. Token cost isn't a
# constraint here, so this ceiling exists only to catch a runaway/broken
# generation, not to cap deliberate growth.
MAX_PROMPT_CHARS = 25000


GOLDEN_FILE = Path(__file__).parent / "golden_articles.json"
# Tolerance scales with set size rather than a fixed count: models aren't
# deterministic, so a large golden set (50+) will trip a fixed low threshold
# on ordinary per-call variance, not genuine regression. ~10% headroom, floor
# of 1 so a tiny set still catches a real miss.
GOLDEN_FAILURE_RATE = 0.10


class PromptRejected(Exception):
    """Raised when a generated prompt fails validation and must not be written."""


def behavioural_check(new_prompt: str) -> None:
    """Score a fixed set of known articles with the new prompt.

    validate_prompt() only proves a rewrite is well-formed. This proves it still
    judges correctly: a prompt that is perfectly structured but, say, narrows the
    scope until nothing qualifies will pass structural checks and quietly gut the
    digest — that happened once and was only caught by hand.

    Raises PromptRejected if the prompt misjudges more than MAX_GOLDEN_FAILURES.
    """
    if not GOLDEN_FILE.exists():
        print("   [WARN] golden_articles.json yok — davranış testi atlandı.")
        return

    # Imported here so the module still loads without the digest's dependencies.
    import llm
    from analyzer import THRESHOLD_ORTA as MIN_TOTAL
    from analyzer import _compute_total, _deep_analyze, _scores
    from fetcher import RawArticle

    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    total_items = len(golden.get("must_pass", [])) + len(golden.get("must_fail", []))
    max_failures = max(1, round(total_items * GOLDEN_FAILURE_RATE))

    client = llm.get_client()
    failures: list[str] = []

    # Score the way production does — with retrieved labelled examples — so the
    # gate tests the prompt as it will actually run.
    import knowledge_base
    from analyzer import knowledge_base_block

    cases = [
        (RawArticle(title=item["title"], url="", summary=item["summary"],
                    published=None, source="golden"), should_pass)
        for bucket, should_pass in (("must_pass", True), ("must_fail", False))
        for item in golden.get(bucket, [])
    ]
    kb = knowledge_base.load_safely()
    found = kb.neighbours([a for a, _ in cases]) if kb else [[] for _ in cases]

    for (art, should_pass), neighbours in zip(cases, found):
        try:
            data = _deep_analyze(
                client, art, new_prompt, knowledge_base_block(neighbours, detailed=True)
            )
            total, _ = _compute_total(*_scores(data))
        except Exception as exc:
            failures.append(f"{art.title[:45]} — puanlanamadı ({exc})")
            continue

        passed = total >= MIN_TOTAL
        mark = "✓" if passed == should_pass else "✗"
        print(f"   {mark} {total:5.0f}pt  {art.title[:52]}")
        if passed != should_pass:
            want = f"≥{MIN_TOTAL}" if should_pass else f"<{MIN_TOTAL}"
            failures.append(f"{art.title[:45]} → {total:.0f} (beklenen {want})")

    if len(failures) > max_failures:
        raise PromptRejected(
            f"davranış testinde {len(failures)}/{total_items} hata "
            f"(tolerans: {max_failures}): " + " | ".join(failures)
        )
    if failures:
        print(f"   [WARN] {len(failures)}/{total_items} tolere edilen sapma: {failures[0]}")


# Audience gate. Below this many audience-labelled rows the measured agreement is
# noise, so the check stands down rather than rejecting on a handful of calls.
AUDIENCE_MIN_SAMPLE = 12
# How far agreement may fall versus the live prompt before a rewrite is rejected.
AUDIENCE_MAX_DROP = 0.10
AUDIENCE_SAMPLE_CAP = 40
_AUDIENCE_TARGETS = {"Dijital Ekipler", "Üst Yönetim", "İkisi De", "Alakasız"}


def audience_check(new_prompt: str, current_prompt: str) -> None:
    """Reject a rewrite that makes the model worse at Kübra's audience split.

    The golden set only checks scores, so a rewrite could quietly break the
    Dijital Ekipler / Üst Yönetim distinction and still pass. This re-guesses the
    audience for Kübra's labelled rows under both prompts and compares agreement
    with her labels. Comparing against the live prompt (not a fixed bar) keeps the
    gate meaningful while the criteria are still being learned.
    """
    import knowledge_base
    from analyzer import _audience, _deep_analyze, knowledge_base_block
    from fetcher import RawArticle

    kb = knowledge_base.load_safely()
    if kb is None:
        return
    labelled = [e for e in kb.examples
                if e.origin == "Articles" and e.label in _AUDIENCE_TARGETS]
    real_audience = [e for e in labelled if e.label != "Alakasız"]
    if len(real_audience) < AUDIENCE_MIN_SAMPLE:
        print(f"   [INFO] Kitle testi atlandı: {len(real_audience)} kitle etiketi var, "
              f"en az {AUDIENCE_MIN_SAMPLE} gerekli.")
        return

    sample = labelled[-AUDIENCE_SAMPLE_CAP:]
    articles = [RawArticle(title=e.title, url=e.url, summary=e.text, published=None,
                           source="audience-check") for e in sample]
    found = kb.neighbours(articles)
    client = llm.get_client()

    def agreement(prompt: str) -> float:
        hits = 0
        for ex, art, neighbours in zip(sample, articles, found):
            try:
                data = _deep_analyze(client, art, prompt, knowledge_base_block(neighbours, detailed=True))
                hits += _audience(data) == ex.label
            except Exception:
                pass
        return hits / len(sample)

    current = agreement(current_prompt)
    new = agreement(new_prompt)
    print(f"   Kitle uyumu: mevcut prompt %{current*100:.0f} → yeni prompt %{new*100:.0f} "
          f"({len(sample)} etiketli haber)")
    if new < current - AUDIENCE_MAX_DROP:
        raise PromptRejected(
            f"kitle uyumu düştü: %{current*100:.0f} → %{new*100:.0f} "
            f"(izin verilen düşüş %{AUDIENCE_MAX_DROP*100:.0f})"
        )


def validate_prompt(new_prompt: str, current_prompt: str) -> None:
    """Reject a rewritten prompt that would degrade the digest.

    Raises PromptRejected with the reason. The caller keeps the existing prompt.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in new_prompt]
    if missing:
        raise PromptRejected(f"eksik alan(lar): {', '.join(missing)}")

    # An unbalanced JSON block means the response was cut off mid-structure.
    if new_prompt.count("{") != new_prompt.count("}"):
        raise PromptRejected(
            f"JSON blogu kapanmamis ({new_prompt.count('{')} '{{' vs "
            f"{new_prompt.count('}')} '}}')"
        )

    # Truncation usually severs the final line mid-sentence.
    if not new_prompt.rstrip().endswith(("}", '"', ">", ".", "]")):
        tail = new_prompt.rstrip()[-40:]
        raise PromptRejected(f"cumle ortasinda bitiyor: ...{tail!r}")

    if len(new_prompt) > MAX_PROMPT_CHARS:
        raise PromptRejected(
            f"cok uzun: {len(new_prompt)} > {MAX_PROMPT_CHARS} karakter"
        )

    # A drastic shrink means the model summarized instead of rewriting.
    if len(new_prompt) < len(current_prompt) * 0.5:
        raise PromptRejected(
            f"asiri kisalmis: {len(new_prompt)} < mevcut {len(current_prompt)} / 2"
        )


def _format_item(item: dict) -> str:
    tags = [t for t in (item["feedback"], f"Etiket: {item['etiket']}" if item["etiket"] else "",
                        f"Derinlik: {item['derinlik']}" if item["derinlik"] else "") if t]
    line = f"- [{' · '.join(tags)}] \"{item['title']}\" (Skor: {item['total_score']}, {item['signal']})"
    if item["etiket"] and item["model_tahmini"] and item["etiket"] != item["model_tahmini"]:
        line += f"\n  ⚡ UYUŞMAZLIK: model \"{item['model_tahmini']}\" tahmin etti, Kübra \"{item['etiket']}\" dedi"
    if item["note"]:
        line += f"\n  Not: {item['note']}"
    return line


def _build_optimizer_prompt(current_prompt: str, feedback_items: list[dict]) -> str:
    feedback_block = "\n".join(_format_item(i) for i in feedback_items)
    agreed, compared = audience_agreement(feedback_items)
    agreement_line = (
        f"Hedef kitle tahmin uyumu: {agreed}/{compared} (%{100 * agreed // compared})"
        if compared else "Hedef kitle tahmin uyumu: henüz karşılaştırılabilir kayıt yok"
    )

    return f"""MEVCUT PROMPT:
---
{current_prompt}
---

SON {LOOKBACK_DAYS} GÜNDEKİ FEEDBACK ({len(feedback_items)} kayıt):
{agreement_line}

Etiket anlamları: "Alakasız" = yanlış seçim. "Dijital Ekipler" / "Üst Yönetim" /
"İkisi De" = doğru seçim + haberin kime yönelik olduğu. Derinlik: "Detaylı" = sunumda
1 sayfa ayrılacak kadar önemli, "Kısa" = başlık düzeyinde yeterli.

{feedback_block}

Bu feedback'lere dayanarak promptu güncelle. Özellikle:
- Yanlış seçilen (❌ veya Alakasız) haberlerin ortak özelliklerini analiz et
- Puanlama kurallarını daha isabetli hale getir
- Başarılı seçimleri few-shot örnek olarak ekle (maksimum 3 örnek)
- Kübra'nın Dijital Ekipler / Üst Yönetim ayrımından kriter çıkar ve HEDEF KİTLE
  TAHMİNİ bölümünü güncelle; ⚡ UYUŞMAZLIK işaretli örnekler en değerli sinyaldir
- Sektörel/bankacılık bakış açısı talimatlarını güçlendir"""


def run_optimization() -> str:
    notion_token = os.environ.get("NOTION_TOKEN")
    db_id = os.environ.get("NOTION_DATABASE_ID")

    if not llm.has_credentials():
        raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK veya ANTHROPIC_API_KEY eksik.")
    if not all([notion_token, db_id]):
        raise RuntimeError("NOTION_TOKEN veya NOTION_DATABASE_ID eksik.")

    client = llm.get_client()
    print(f"🔌 Backend: {llm.backend_name()}")

    # 1. Read current prompt
    current_prompt = PROMPT_FILE.read_text(encoding="utf-8")
    print(f"📄 Mevcut prompt yüklendi ({len(current_prompt)} karakter)")

    # 2. Fetch feedback from Articles database
    print("📥 Articles tablosundan feedback'ler okunuyor...")
    pages = _get_notion_feedback(notion_token)
    items = _extract_feedback_items(pages)
    print(f"   {len(items)} feedback kaydı bulundu.")

    if not items:
        print("⚠️  Hiç feedback yok — prompt güncellenmedi.")
        return current_prompt

    # Break down by type
    for label in ["✅ Doğru seçim", "❌ Yanlış seçim", "⚠️ Skor yanlış"]:
        count = sum(1 for i in items if i["feedback"] == label)
        print(f"   {label}: {count}")
    for label in ["Dijital Ekipler", "Üst Yönetim", "İkisi De", "Alakasız"]:
        count = sum(1 for i in items if i["etiket"] == label)
        print(f"   Etiket {label}: {count}")
    agreed, compared = audience_agreement(items)
    if compared:
        print(f"   🎯 Model–Kübra kitle uyumu: {agreed}/{compared} (%{100 * agreed // compared})")

    # 3. Ask Claude to optimize
    print("\n🧠 Claude ile prompt optimize ediliyor...")
    user_msg = _build_optimizer_prompt(current_prompt, items)

    response = client.messages.create(
        model=llm.model("deep"),
        max_tokens=16000,
        system=OPTIMIZER_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    # Scan for the text block — a thinking block can come first.
    new_prompt = next(
        (b.text for b in response.content if getattr(b, "type", None) == "text"), ""
    ).strip()
    if not new_prompt:
        raise RuntimeError("Model boş prompt döndürdü — güncelleme yapılmadı.")

    # 4. Guardrails — never overwrite a working prompt with a broken one.
    if response.stop_reason == "max_tokens":
        raise PromptRejected("yanıt max_tokens sınırına çarptı (kesik)")
    validate_prompt(new_prompt, current_prompt)

    print("\n🧪 Davranış testi (golden set)...")
    behavioural_check(new_prompt)
    print("\n🎯 Kitle testi...")
    audience_check(new_prompt, current_prompt)

    # 5. Save updated prompt
    PROMPT_FILE.write_text(new_prompt, encoding="utf-8")
    print(f"\n✅ system_prompt.txt güncellendi ({len(new_prompt)} karakter)")
    print(f"   Değişiklik: {len(new_prompt) - len(current_prompt):+d} karakter")
    print(f"💰 {llm.usage_summary()}")

    return new_prompt


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
    try:
        run_optimization()
    except PromptRejected as exc:
        # The existing prompt is untouched, so the digest keeps working — but
        # exit non-zero so the run is flagged rather than failing silently.
        print(f"\n❌ Üretilen prompt reddedildi: {exc}")
        print("   Mevcut prompt korundu, dijest etkilenmedi.")
        sys.exit(1)
