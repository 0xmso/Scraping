"""Monthly "Now & Next" deck — the NEXT section, generated from Kübra's labels.

Scope (agreed with the team): only NEXT. NOW — Turkish banks' product launches —
comes from a separate competitor-monitoring system and is not touched here.

Articles Kübra labelled Dijital Ekipler or İkisi De in the coverage window go in:
  Derinlik = Detaylı → one slide each (Özet bullets, two "Önemli" points, image)
  Derinlik = Kısa    → headline lines on "Öne çıkanlar" slides
Nothing is dropped to fit: every labelled article is included, only reported.

The template is the team's own deck, passed via DECK_TEMPLATE_PATH. It is never
committed: it carries internal branding and the previous month's content.

Usage:
    python3 monthly_deck.py                  # current coverage window
    python3 monthly_deck.py --sample         # layout test from recent selected articles
    python3 monthly_deck.py --out path.pptx
    python3 monthly_deck.py --ci             # no-op unless today is this month's send_date()
"""

import copy
import io
import os
import re
import sys
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches

import llm

NOTION_VERSION = "2022-06-28"
AUDIENCE = ("Dijital Ekipler", "İkisi De")

# Prototype slides in the team template (1-based), and shape positions (inches).
PROTO_COVER, PROTO_DIVIDER, PROTO_DETAIL, PROTO_LIST = 1, 11, 15, 16
MONTHS_TR = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz",
             "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]

# Text budgets derived from the template's box sizes and fonts, so copy fits.
TITLE_MAX = 60
BULLETS = 3
BULLET_MAX = 130
# Hard caps are the box limits; the model is asked for less (the *_ASK values) so
# the cap, which truncates with an ellipsis, almost never has to act. Point titles
# are set in a wide ultra-bold face: past ~20 chars they wrap onto the description.
POINT_TITLE_MAX = 20
POINT_TITLE_ASK = 18
POINT_DESC_MAX = 115
POINT_DESC_ASK = 90
LIST_LINE_MAX = 95
LIST_LINES_PER_SLIDE = 14


# ── Schedule ──────────────────────────────────────────────────────────────────

def send_date(year: int, month: int) -> date:
    """Wednesday of the week containing the month's last Friday."""
    last = date(year, month, monthrange(year, month)[1])
    last_friday = last - timedelta(days=(last.weekday() - 4) % 7)
    return last_friday - timedelta(days=2)


def coverage_window(today: date) -> tuple[date, date]:
    """(after, through): from the previous send (exclusive) to the next one (inclusive)."""
    this = send_date(today.year, today.month)
    if today > this:
        y, m = (today.year + (today.month == 12), today.month % 12 + 1)
        this = send_date(y, m)
    py, pm = (this.year - (this.month == 1), (this.month - 2) % 12 + 1)
    return send_date(py, pm), this


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class DeckArticle:
    title: str
    url: str
    ozet: str
    sektorel: str
    stratejik: str
    depth: str
    score: float
    image_url: Optional[str] = None
    source_text: str = ""


def _text(props: dict, name: str) -> str:
    prop = props.get(name, {})
    return "".join(b.get("plain_text", "") for b in (prop.get("title") or prop.get("rich_text") or []))


def _row(page: dict, depth: str) -> DeckArticle:
    p = page["properties"]
    return DeckArticle(
        title=_text(p, "Name"), url=p.get("URL", {}).get("url") or "",
        ozet=_text(p, "Özet"), sektorel=_text(p, "Sektörel"), stratejik=_text(p, "Stratejik Çıkarım"),
        depth=depth, score=p.get("Total Score", {}).get("number") or 0,
        image_url=p.get("Görsel", {}).get("url"),
    )


def _query(body: dict) -> list[dict]:
    url = f"https://api.notion.com/v1/databases/{os.environ['NOTION_ARTICLES_DB_ID']}/query"
    headers = {"Authorization": f"Bearer {os.environ['NOTION_TOKEN']}",
               "Notion-Version": NOTION_VERSION, "Content-Type": "application/json"}
    pages, cursor = [], None
    with httpx.Client(timeout=30) as client:
        while True:
            if cursor:
                body["start_cursor"] = cursor
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            pages += data["results"]
            if not data.get("has_more"):
                return pages
            cursor = data["next_cursor"]


DEPTH_EXCLUDE = "Hariç Tut"  # Etiket qualifies the article, but Kübra opted it out of this deck


def labelled_articles(after: date, through: date) -> tuple[list[DeckArticle], int]:
    """Returns (included articles, count excluded via Derinlik=Hariç Tut).

    Etiket alone decides training/audience-gate eligibility; Derinlik=Hariç Tut
    is a separate, deck-only opt-out so Kübra can keep labelling accurately
    without every Dijital Ekipler/İkisi De article bloating the presentation.
    """
    pages = _query({"page_size": 100, "filter": {"and": [
        {"or": [{"property": "Etiket", "select": {"equals": a}} for a in AUDIENCE]},
        {"property": "Date", "date": {"after": after.isoformat()}},
        {"property": "Date", "date": {"on_or_before": through.isoformat()}},
    ]}})
    out, excluded = [], 0
    for page in pages:
        depth = (page["properties"].get("Derinlik", {}).get("select") or {}).get("name")
        if depth == DEPTH_EXCLUDE:
            excluded += 1
            continue
        out.append(_row(page, depth or "Kısa"))  # unset depth → headline, never dropped
    return out, excluded


def sample_articles(detailed: int = 3, short: int = 9) -> list[DeckArticle]:
    """Layout test data: recent selected articles, top scores as Detaylı."""
    since = (datetime.now() - timedelta(days=21)).date().isoformat()
    pages = _query({"page_size": 100, "filter": {"and": [
        {"property": "Date", "date": {"on_or_after": since}},
        {"or": [{"property": "Seçim Tipi", "select": {"equals": "Seçildi"}},
                {"property": "Seçim Tipi", "select": {"is_empty": True}}]},
    ]}, "sorts": [{"property": "Total Score", "direction": "descending"}]})
    rows = [_row(p, "") for p in pages if _text(p["properties"], "Özet")]
    for i, r in enumerate(rows[:detailed + short]):
        r.depth = "Detaylı" if i < detailed else "Kısa"
    return rows[:detailed + short]


# ── Copy ──────────────────────────────────────────────────────────────────────

_RULES = f"""Sen bir banka dijital ekibinin aylık "Now & Next" sunumu için NEXT (İnovasyon Radarı)
slaytı yazan editörsün. Dil: Türkçe, kurumsal ama canlı; kısa ve somut.

UYDURMA YASAĞI: Yalnızca verilen KAYNAK metinde ve analizde geçen bilgiyi kullan.
Kaynakta olmayan sayı, isim, tarih, nitelik ("en büyük", "ilk") EKLEME.

Karakter sınırlarına kesinlikle uy, aşan metin slayta sığmaz."""

_DETAIL_SCHEMA = {
    "type": "object",
    "properties": {
        "baslik": {"type": "string"},
        "ozet_maddeleri": {"type": "array", "items": {"type": "string"}},
        "onemli": {"type": "array", "items": {
            "type": "object",
            "properties": {"baslik": {"type": "string"}, "aciklama": {"type": "string"}},
            "required": ["baslik", "aciklama"], "additionalProperties": False}},
        "sirket_alan_adi": {"type": "string"},
    },
    "required": ["baslik", "ozet_maddeleri", "onemli", "sirket_alan_adi"],
    "additionalProperties": False,
}

_LIST_SCHEMA = {
    "type": "object",
    "properties": {"gruplar": {"type": "array", "items": {
        "type": "object",
        "properties": {"baslik": {"type": "string"},
                       "maddeler": {"type": "array", "items": {"type": "string"}}},
        "required": ["baslik", "maddeler"], "additionalProperties": False}}},
    "required": ["gruplar"],
    "additionalProperties": False,
}


def _fit(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit - 1].rsplit(" ", 1)[0].rstrip(",;:—-")
    return cut + "…"


def detail_copy(client, art: DeckArticle) -> dict:
    source = art.source_text or art.ozet
    data = llm.call_structured(
        client, model=llm.model("deep"), max_tokens=4000,
        system=[{"type": "text", "text": _RULES, "cache_control": {"type": "ephemeral"}}],
        user_content=(
            f"Bu haber için bir NEXT detay slaytı yaz.\n\n"
            f"- baslik: haberi anlatan TAM bir cümle — özne ve çekimli fiil içermeli, nokta ile biter, "
            f"en fazla {TITLE_MAX} karakter (ör. \"BBVA, müşteri deneyimini YZ ile ölçmeye başladı.\"). "
            f"Sığmıyorsa detayı at, fiili atma.\n"
            f"- ozet_maddeleri: tam {BULLETS} madde, her biri en fazla {BULLET_MAX} karakter; "
            f"ne oldu, nasıl çalışıyor, ölçek/rakam\n"
            f"- onemli: tam 2 madde; baslik 1-3 kelime, en fazla {POINT_TITLE_ASK} karakter "
            f"(ör. \"Proaktif önlem\", \"Veri egemenliği\"); aciklama tek cümle, en fazla "
            f"{POINT_DESC_ASK} karakter; bir banka dijital ekibi için neden önemli\n"
            f"- sirket_alan_adi: haberin ana konusu olan şirketin resmi web alan adı (ör. "
            f"\"worldline.com\"); yalnızca emin olduğunda doldur, değilse boş bırak (\"\"). "
            f"Bu sadece logo bulmak için kullanılır, slayt metnine girmez.\n\n"
            f"BAŞLIK: {art.title}\n\nANALİZ ÖZETİ: {art.ozet}\nSEKTÖREL: {art.sektorel}\n"
            f"STRATEJİK ÇIKARIM: {art.stratejik}\n\nKAYNAK:\n{source[:8000]}"
        ),
        schema=_DETAIL_SCHEMA, tool_name="slayt_yaz", tool_description="NEXT detay slaytı içeriği.",
        effort="high", thinking=True,
    )
    bullets = [_fit(b, BULLET_MAX) for b in llm.coerce_list(data.get("ozet_maddeleri")) if str(b).strip()]
    points = [p for p in llm.coerce_list(data.get("onemli")) if isinstance(p, dict)][:2]
    return {
        "domain": _trusted_domain(data.get("sirket_alan_adi"), art),
        "baslik": _fit(data.get("baslik") or art.title, TITLE_MAX),
        "ozet": bullets[:BULLETS] or [_fit(art.ozet, BULLET_MAX)],
        "onemli": [{"baslik": _fit(p.get("baslik", ""), POINT_TITLE_MAX),
                    "aciklama": _fit(p.get("aciklama", ""), POINT_DESC_MAX)} for p in points],
    }


def list_copy(client, arts: list[DeckArticle]) -> list[dict]:
    items = "\n".join(f"{i}. {a.title} — {_fit(a.ozet, 300)}" for i, a in enumerate(arts, 1))
    data = llm.call_structured(
        client, model=llm.model("deep"), max_tokens=4000,
        system=[{"type": "text", "text": _RULES, "cache_control": {"type": "ephemeral"}}],
        user_content=(
            "Aşağıdaki haberlerin HER BİRİ için \"Öne çıkanlar\" listesine tek bir Türkçe madde yaz "
            f"(en fazla {LIST_LINE_MAX} karakter, ör. \"Kraken, kripto debit kart çıkardı.\").\n"
            "Aynı konuya ait 2+ haber varsa bir grup başlığı altında topla (ör. \"Visa ve "
            "Mastercard'ın hamleleri\"); tekil haberler başlığı boş (\"\") bir grupta kalsın. "
            "Hiçbir haberi atlama.\n\n" + items
        ),
        schema=_LIST_SCHEMA, tool_name="liste_yaz", tool_description="Öne çıkanlar maddeleri.",
    )
    groups = []
    for g in llm.coerce_list(data.get("gruplar")):
        if not isinstance(g, dict):
            continue
        lines = [_fit(x, LIST_LINE_MAX) for x in llm.coerce_list(g.get("maddeler")) if str(x).strip()]
        if lines:
            groups.append({"baslik": _fit(g.get("baslik") or "", LIST_LINE_MAX), "maddeler": lines})
    written = sum(len(g["maddeler"]) for g in groups)
    if written < len(arts):
        print(f"   [WARN] Liste {len(arts)} haberden {written} madde döndürdü — eksikler başlıkla eklendi.")
        groups.append({"baslik": "", "maddeler": [_fit(a.title, LIST_LINE_MAX) for a in arts[written:]]})
    return groups


# ── PowerPoint ────────────────────────────────────────────────────────────────

def _at(slide, x: float, y: float, tol: float = 0.15, kind: str = "text"):
    """The shape whose top-left is at (x, y) inches."""
    for sh in slide.shapes:
        if abs(Emu(sh.left or 0).inches - x) <= tol and abs(Emu(sh.top or 0).inches - y) <= tol:
            if kind == "text" and sh.has_text_frame:
                return sh
            if kind == "picture" and sh.shape_type == 13:
                return sh
    raise LookupError(f"şablonda {kind} bulunamadı: ({x}, {y})")


_R_ATTR = re.compile(r'(r:(?:embed|link|id|pict)=")(rId\d+)(")')


def duplicate_slide(prs, source):
    """Copy a slide, remapping relationship ids so images and SVGs survive."""
    new = prs.slides.add_slide(source.slide_layout)
    for shape in list(new.shapes):
        shape._element.getparent().remove(shape._element)
    mapping = {}
    for rid, rel in source.part.rels.items():
        if rel.is_external or "notesSlide" in rel.reltype or "slideLayout" in rel.reltype:
            continue
        mapping[rid] = new.part.relate_to(rel._target, rel.reltype)
    from lxml import etree
    for child in source.shapes._spTree.iterchildren():
        if child.tag in (qn("p:nvGrpSpPr"), qn("p:grpSpPr")):
            continue
        xml = etree.tostring(child).decode()
        xml = _R_ATTR.sub(lambda m: m.group(1) + mapping.get(m.group(2), m.group(2)) + m.group(3), xml)
        new.shapes._spTree.append(etree.fromstring(xml))
    bg = source._element.cSld.find(qn("p:bg"))
    if bg is not None:
        new._element.cSld.insert(0, copy.deepcopy(bg))
    return new


def _set_paragraphs(shape, lines: list[tuple[str, str]]):
    """Replace a text box's paragraphs; each line is (style, text) where style names
    a template paragraph to clone ("bullet", "header" or "plain"), keeping its look."""
    body = shape.text_frame._txBody
    paras = body.findall(qn("a:p"))
    protos = {}
    for p in paras:
        has_bullet = p.find(f"{qn('a:pPr')}/{qn('a:buChar')}") is not None
        bold = any(r.get("b") == "1" for r in p.iter(qn("a:rPr")))
        key = "bullet" if has_bullet else ("header" if bold else "plain")
        protos.setdefault(key, p)
    fallback = paras[0]
    for p in paras:
        body.remove(p)
    for style, text in lines:
        proto = protos.get(style) or protos.get("plain") or fallback
        para = copy.deepcopy(proto)
        runs = para.findall(qn("a:r"))
        for r in runs[1:]:
            para.remove(r)
        if runs:
            rpr = runs[0].find(qn("a:rPr"))
            if rpr is not None and "err" in rpr.attrib:
                del rpr.attrib["err"]
            runs[0].find(qn("a:t")).text = text
        body.append(para)


def _set_text(shape, text: str):
    """Single-line box: keep the first paragraph and run's formatting, replace the text."""
    body = shape.text_frame._txBody
    paras = body.findall(qn("a:p"))
    for p in paras[1:]:
        body.remove(p)
    runs = paras[0].findall(qn("a:r"))
    for r in runs[1:]:
        paras[0].remove(r)
    rpr = runs[0].find(qn("a:rPr"))
    if rpr is not None and "err" in rpr.attrib:
        del rpr.attrib["err"]
    runs[0].find(qn("a:t")).text = text


def _replace_picture(slide, placeholder, image_bytes: Optional[bytes], max_in: Optional[float] = None):
    """Fit an image inside the placeholder's box, centred; max_in caps its longer side."""
    left, top, box_w, box_h = placeholder.left, placeholder.top, placeholder.width, placeholder.height
    placeholder._element.getparent().remove(placeholder._element)
    if not image_bytes:
        return
    from PIL import Image
    try:
        w, h = Image.open(io.BytesIO(image_bytes)).size
    except Exception:
        return
    fit_w, fit_h = box_w, box_h
    if max_in:
        fit_w = fit_h = min(box_w, box_h, Inches(max_in))
    scale = min(fit_w / w, fit_h / h)
    pw, ph = int(w * scale), int(h * scale)
    slide.shapes.add_picture(io.BytesIO(image_bytes), left + (box_w - pw) // 2,
                             top + (box_h - ph) // 2, pw, ph)


_DOMAIN = re.compile(r"^(?:[a-z0-9-]+\.)+[a-z]{2,}$")
LOGO_MIN_PX = 128
LOGO_MAX_IN = 2.2   # logos are small square marks; stretching one to the image box blurs it


def _trusted_domain(domain, art: DeckArticle) -> Optional[str]:
    """Accept the model's company domain only if its name appears in the article.

    The domain is model knowledge, not source text. Requiring its first label
    ("worldline" for worldline.com) to occur in the title or summary keeps a
    wrong guess from putting another company's logo on the slide.
    """
    domain = str(domain or "").strip().lower().removeprefix("https://").removeprefix("http://")
    domain = domain.removeprefix("www.").split("/", 1)[0]
    if not _DOMAIN.match(domain):
        return None
    name = domain.split(".")[0].replace("-", "")
    haystack = re.sub(r"[^a-z0-9]", "", f"{art.title} {art.ozet}".lower())
    return domain if len(name) >= 3 and name in haystack else None


def _logo(domain: Optional[str]) -> Optional[bytes]:
    if not domain:
        return None
    data = _download(f"https://www.google.com/s2/favicons?domain={domain}&sz=256")
    if not data:
        return None
    from PIL import Image
    try:
        return data if min(Image.open(io.BytesIO(data)).size) >= LOGO_MIN_PX else None
    except Exception:
        return None


def _download(url: Optional[str]) -> Optional[bytes]:
    if not url:
        return None
    try:
        import article_text
        r = httpx.get(url, headers={"User-Agent": article_text.BROWSER_UA}, timeout=20, follow_redirects=True)
        r.raise_for_status()
        return r.content if r.headers.get("content-type", "").startswith("image/") else None
    except Exception:
        return None


def build_deck(template: Path, month_label: str, details: list[tuple[DeckArticle, dict]],
               list_groups: list[dict], out: Path) -> dict:
    prs = Presentation(str(template))
    originals = list(prs.slides)
    proto_detail, proto_list = originals[PROTO_DETAIL - 1], originals[PROTO_LIST - 1]
    keep = [originals[PROTO_COVER - 1], originals[PROTO_DIVIDER - 1]]
    added, missing_images, logo_count = [], 0, 0

    for art, copy_ in details:
        s = duplicate_slide(prs, proto_detail)
        _set_text(_at(s, 1.7, -0.3), copy_["baslik"])
        _set_text(_at(s, 1.7, 0.4), month_label)
        _set_paragraphs(_at(s, 2.4, 3.3), [("bullet", b) for b in copy_["ozet"]])
        # (number badge, point title, point description) per "Önemli" slot
        slots = [((11.6, 3.3), (12.7, 3.4), (12.7, 3.9)), ((11.6, 5.6), (12.7, 5.7), (12.7, 6.2))]
        for (badge_xy, title_xy, desc_xy), point in zip(slots, copy_["onemli"] + [None] * 2):
            badge, title_box, desc_box = _at(s, *badge_xy), _at(s, *title_xy), _at(s, *desc_xy)
            if point:
                _set_text(title_box, point["baslik"])
                _set_text(desc_box, point["aciklama"])
            else:
                for box in (badge, title_box, desc_box):
                    box._element.getparent().remove(box._element)
        # Article image → company logo (small, not stretched) → leave the area empty.
        image, logo = _download(art.image_url), None
        if image is None:
            logo = _logo(copy_.get("domain"))
            logo_count += logo is not None
        missing_images += image is None and logo is None
        _replace_picture(s, _at(s, 3.6, 7.1, kind="picture"), image or logo,
                         max_in=LOGO_MAX_IN if logo else None)
        added.append(s)

    # Ungrouped lines first, directly under "Öne çıkanlar": placed after a group
    # they would read as belonging to that group's header.
    list_groups = sorted(list_groups, key=lambda g: bool(g["baslik"]))
    lines = []
    for g in list_groups:
        if g["baslik"]:
            lines.append(("header", g["baslik"]))
        lines += [("bullet", m) for m in g["maddeler"]]
    pages = [lines[i:i + LIST_LINES_PER_SLIDE] for i in range(0, len(lines), LIST_LINES_PER_SLIDE)]
    for i, page in enumerate(pages):
        s = duplicate_slide(prs, proto_list)
        _set_text(_at(s, 1.7, 0.4), month_label)
        header = [("header", "Öne çıkanlar" if i == 0 else "Öne çıkanlar (devam)")]
        _set_paragraphs(_at(s, 1.7, 2.3), header + page)
        added.append(s)

    # Final order: cover, NEXT divider, detail slides, list slides; drop everything else.
    sld_list = prs.slides._sldIdLst
    by_part = {s.part: s for s in prs.slides}
    ids = {}
    for sld_id in list(sld_list):
        slide = by_part[prs.part.related_part(sld_id.rId)]
        ids[id(slide)] = sld_id
        sld_list.remove(sld_id)
    for slide in keep + added:
        sld_list.append(ids[id(slide)])
    for slide in originals:
        if slide not in keep:
            prs.part.drop_rel(ids[id(slide)].rId)

    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    return {"slides": len(keep) + len(added), "detail": len(details),
            "list_pages": len(pages), "missing_images": missing_images, "logos": logo_count}


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")

    today = date.today()

    # The workflow runs every Wednesday (cron can't express "the Wednesday of the
    # week containing the month's last Friday" directly) and relies on this to
    # no-op on the other three — send_date() is the one place that rule lives.
    if "--ci" in sys.argv and today != send_date(today.year, today.month):
        print(f"⏭️  Bugün ({today}) bu ayın gönderim günü değil "
              f"({send_date(today.year, today.month)}) — atlanıyor.")
        return 0

    template = Path(os.environ.get("DECK_TEMPLATE_PATH", ""))
    if not template.is_file():
        print("❌ DECK_TEMPLATE_PATH ayarlı değil veya dosya yok.")
        return 1

    sample = "--sample" in sys.argv
    after, through = coverage_window(today)
    month_label = f"{MONTHS_TR[through.month - 1]} {through.year}"

    if sample:
        arts = sample_articles()
        print(f"🧪 Örnek mod: son 3 haftanın en yüksek skorlu {len(arts)} haberi")
    else:
        arts, excluded = labelled_articles(after, through)
        print(f"📅 Kapsam: {after:%d.%m.%Y} sonrası → {through:%d.%m.%Y} · {len(arts)} etiketli haber"
              + (f" · {excluded} 'Hariç Tut' ile çıkarıldı" if excluded else ""))
    if not arts:
        print("Kapsamda Dijital Ekipler / İkisi De etiketli haber yok — sunum üretilmedi.")
        return 1

    detailed = sorted([a for a in arts if a.depth == "Detaylı"], key=lambda a: a.score, reverse=True)
    short = [a for a in arts if a.depth != "Detaylı"]
    print(f"   {len(detailed)} detaylı slayt · {len(short)} kısa haber")

    import article_text
    client = llm.get_client()
    details = []
    for art in detailed:
        try:
            fetched = article_text.fetch(art.url, timeout=15)
            art.source_text, art.image_url = fetched.text, art.image_url or fetched.image_url
        except Exception:
            pass
        details.append((art, detail_copy(client, art)))
        print(f"   ✍️  {details[-1][1]['baslik']}")
    groups = list_copy(client, short) if short else []

    out = Path(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")),
                    f"out/Now & Next - {month_label} - NEXT taslak.pptx"))
    stats = build_deck(template, month_label, details, groups, out)
    print(f"\n✅ {out} · {stats['slides']} slayt ({stats['detail']} detay, "
          f"{stats['list_pages']} öne çıkanlar sayfası)")
    if stats["logos"]:
        print(f"   🏷️  {stats['logos']} detay slaytında görsel yerine şirket logosu kullanıldı")
    if stats["missing_images"]:
        print(f"   ⚠️  {stats['missing_images']} detay slaytında görsel bulunamadı, alan boş bırakıldı")
    print(f"💰 {llm.usage_summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
