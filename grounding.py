"""Hallucination check for the Turkish analysis text.

Bedrock Guardrails' contextual grounding check would be the managed option, but
it only supports English, French and Spanish, and our summaries are Turkish. So
this does the same job in the style of claim-level faithfulness evaluation: a
second model extracts each factual claim from the generated text and checks it
against the source the analysis was allowed to use (title + RSS excerpt).

It flags, never filters: an article with an unsupported claim is still published,
with the claim listed in Notion for Kübra to see.
"""

import re

import llm

# Opus, not Sonnet: in side-by-side runs Sonnet 5 treated a background-knowledge
# name completion ("Khan" → "Lina Khan") as supported, Opus 5 flagged it reliably,
# and both stayed at zero on clean text. Only the ≤10 selected articles are checked.
GROUNDING_TIER = "deep"

_SYSTEM = """Sen bir doğruluk denetçisisin. Sana bir haberin KAYNAK metni ve bu kaynaktan
üretilmiş Türkçe ANALİZ metni verilecek.

Görevin: ANALİZ içindeki olgusal iddiaları (kim, ne yaptı, sayı, tutar, tarih, oran,
ölçek, ilk/en büyük gibi nitelemeler, isimler) tek tek çıkar ve her birinin KAYNAK
metinde desteklenip desteklenmediğine karar ver.

DESTEKLENİR sayılır:
- Kaynaktaki bilginin çevirisi veya eşdeğer ifadesi ("$78,000" = "78 bin dolar")
- Birim/format dönüşümü, yuvarlama ("3.2 milyar" ≈ "3 milyar doların üzerinde")
- Kaynaktan doğrudan çıkan mantıksal sonuç
- Yaygın kısaltma açılımları ve unvan çevirileri ("FTC" → "ABD Federal Ticaret
  Komisyonu", "boss" → "başkan", "CEO" → "üst yönetici")

DESTEKLENMEZ sayılır:
- Kaynakta hiç geçmeyen sayı, tarih, isim, ülke, şirket, ürün adı
- Kaynaktaki bir ismin arka plan bilgisiyle tamamlanması (kaynak "Khan" diyorsa
  "Lina Khan", kaynak "the CEO" diyorsa CEO'nun adı)
- Kaynakta olmayan nitelemeler ("ABD'nin en büyük", "ilk kez", "pazar lideri")
- Kaynakta olmayan geçmiş bağlam veya arka plan bilgisi

Şunlar olgusal iddia DEĞİLDİR, değerlendirme dışı bırak:
- Kaynağın kendisi hakkındaki ifadeler: "kaynakta detay verilmemiş", "benchmark sonucu
  belirtilmemiş" — analiz bunları bilerek, bilgi eksikliğini dürüstçe belirtmek için yazar
- Tavsiye: "bankalar izlemeli", "bu yüzden X yapmalıyız"
- Yorum ve değerlendirme: "fırsat yaratıyor", "yapısal bir dönüşüm sinyali", "rekabeti
  artırabilir", "trendin güçlendiğini gösteriyor", "önemli bir adım" — analiz alanları
  zaten yorum üretmek için var; bir yorumun kaynakta yazmaması onu uydurma yapmaz.
AMA yorum cümlesinin İÇİNDEKİ sayı, tutar, tarih, özel isim ve "en büyük / ilk / lider"
gibi nitelemeleri AYRI birer iddia olarak MUTLAKA kontrol et. Örnek: "22 milyon
kullanıcısıyla rekabeti artırabilir" → "rekabeti artırabilir" yorumdur ama "22 milyon
kullanıcı" kaynakta yoksa işaretlenir.

Şüpheli iddia yoksa listeyi boş döndür. Emin olmadığın durumda iddiayı işaretleme."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "desteklenmeyen_iddialar": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "iddia": {"type": "string"},
                    "neden": {"type": "string"},
                },
                "required": ["iddia", "neden"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["desteklenmeyen_iddialar"],
    "additionalProperties": False,
}


_NUMBER = re.compile(r"\d[\d.,]*")
_WORD_NUMBERS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "bir": "1", "iki": "2", "üç": "3", "dört": "4", "beş": "5", "altı": "6",
    "yedi": "7", "sekiz": "8", "dokuz": "9",   # no "on": collides with English "on"
}


def _significant(token: str) -> str:
    """'1,200,000' / '1.2' / '78.000' → '12' / '12' / '78': digits minus separators and zeros."""
    return re.sub(r"[.,]", "", token).lstrip("0").rstrip("0")


def unsupported_numbers(source: str, generated: str) -> list[str]:
    """Numbers in the analysis with no counterpart in the source.

    Deterministic backstop for the model check, which can talk itself out of a
    number buried in an interpretive sentence. Matching is on significant digits
    in either direction, so "$78,000" supports "78 bin" and "3.2 billion"
    supports "3,2 milyar". Scale words aren't compared, so this can miss a
    mislabelled magnitude — it catches invented figures, not unit errors.
    """
    def raw(token: str) -> str:
        return re.sub(r"[.,]", "", token).lstrip("0")

    src_words = re.findall(r"\w+", source.lower())
    src_tokens = _NUMBER.findall(source) + [_WORD_NUMBERS[w] for w in src_words if w in _WORD_NUMBERS]
    src = {(raw(t), _significant(t)) for t in src_tokens if _significant(t)}

    def supported(token: str) -> bool:
        r, sig = raw(token), _significant(token)
        for s_raw, s_sig in src:
            if r == s_raw:
                return True
            short, long_ = sorted((s_sig, sig), key=len)
            # Significant-digit prefix lets "78" match "78000" and "3,2" match
            # "3.2". It needs 2+ digits on the short side: dropping zeros turns
            # "300" into "3", which a stray "Q3" would otherwise vouch for.
            if len(short) >= 2 and long_.startswith(short):
                return True
        return False

    missing = [t.rstrip(".,") for t in _NUMBER.findall(generated)
               if _significant(t) and not supported(t)]
    return list(dict.fromkeys(missing))


def check(client, title: str, source: str, analysis: dict) -> list[dict]:
    """Return unsupported claims as [{"iddia": ..., "neden": ...}]; [] when clean."""
    generated = "\n\n".join(
        f"{label}:\n{analysis.get(key, '')}"
        for label, key in (
            ("Özet", "ozet"),
            ("Sektörel önem", "neden_onemli_sektorel"),
            ("Stratejik çıkarım", "stratejik_cikarim"),
        )
        if analysis.get(key)
    )
    data = llm.call_structured(
        client,
        model=llm.model(GROUNDING_TIER),
        max_tokens=2000,
        system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
        user_content=f"KAYNAK:\nBaşlık: {title}\n\n{source or '(özet yok)'}\n\nANALİZ:\n{generated}",
        schema=_SCHEMA,
        tool_name="iddia_denetimi",
        tool_description="Analizdeki kaynakta desteklenmeyen olgusal iddiaları listele.",
    )
    claims = _normalise(data.get("desteklenmeyen_iddialar", []))

    already = " ".join(c["iddia"] for c in claims)
    for number in unsupported_numbers(f"{title}\n{source}", generated):
        if number not in already:
            claims.append({"iddia": number, "neden": "Bu sayı kaynak metinde geçmiyor (otomatik sayı kontrolü)"})
    return claims


def _normalise(raw) -> list[dict]:
    """Coerce the claim list into dicts.

    Bedrock doesn't support strict tool use, so the schema isn't enforced: the
    model sometimes returns the array JSON-encoded as a string, items as bare
    strings, or a string with tool-call markup leaked in around the JSON
    ('<parameter name="desteklenmeyen_iddialar">[]'). Unparseable text is
    treated as no claims — turning garbage into a "claim" is itself a false flag.
    """
    claims = []
    for item in llm.coerce_list(raw):
        if isinstance(item, dict) and item.get("iddia"):
            claims.append({"iddia": str(item["iddia"]), "neden": str(item.get("neden", ""))})
        elif isinstance(item, str) and item.strip():
            claims.append({"iddia": item.strip(), "neden": ""})
    return claims


def format_claims(claims: list[dict]) -> str:
    return "\n".join(f"• {c['iddia']}" + (f" — {c['neden']}" if c["neden"] else "") for c in claims)
