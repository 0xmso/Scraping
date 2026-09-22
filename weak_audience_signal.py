"""Experiment: is Model Tahmini trustworthy enough to use as a weak audience label?

The audience gate (see prompt_optimizer.audience_check) needs 12 real Etiket
labels to even run; only 1 exists. Model Tahmini (the model's own hedef_kitle_
tahmini guess, hidden from Kübra) is set on far more rows, but using a model's
own prediction as ground truth for evaluating that same model is circular —
it can only ever confirm what the model already believes.

This script does NOT wire anything into the live optimizer. It only measures
whether Model Tahmini is even accurate enough to be worth using as a weak
signal in the first place, on the one thing we can check it against: rows
that happen to have BOTH a real Etiket and a Model Tahmini. If that agreement
rate is low, weak supervision from Model Tahmini would inject more noise than
signal, and the honest conclusion is "not yet" — not "build it anyway".

Usage:
    python3 weak_audience_signal.py
"""

import sys
from pathlib import Path

from prompt_optimizer import (
    _get_notion_feedback,
    _extract_feedback_items,
    audience_agreement,
)

# Below this, "the model agrees with itself" isn't a finding — recommend
# waiting for more overlap data rather than trusting a small-n rate either way.
MIN_OVERLAP_FOR_VERDICT = 5
# Weak supervision literature treats labeling functions as usable once they
# beat a clear majority baseline; for a 4-way categorical field a rate this
# far above chance is a reasonable, conservative bar — not a formal guarantee.
TRUST_THRESHOLD = 0.70


def main() -> int:
    import os
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")

    token = os.environ.get("NOTION_TOKEN")
    if not token:
        print("NOTION_TOKEN eksik.")
        return 1

    print("📥 Notion'dan feedback/etiket/model-tahmini okunuyor...")
    items = _extract_feedback_items(_get_notion_feedback(token))

    agreed, compared = audience_agreement(items)
    print(f"\n🎯 Gerçek uyum ölçümü (Etiket VE Model Tahmini ikisi de olan satırlar): "
          f"{agreed}/{compared}" + (f" (%{100*agreed//compared})" if compared else ""))

    weak_pool = [
        i for i in items
        if not i["etiket"] and i["model_tahmini"] and i["feedback"] == "✅ Doğru seçim"
    ]
    dist = {}
    for i in weak_pool:
        dist[i["model_tahmini"]] = dist.get(i["model_tahmini"], 0) + 1
    print(f"\n🏊 Zayıf-etiket havuzu (Etiket YOK, ✅ Doğru seçim VAR, Model Tahmini VAR): "
          f"{len(weak_pool)} satır")
    for label, n in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f"   {label}: {n}")

    print("\n📋 Karar:")
    if compared < MIN_OVERLAP_FOR_VERDICT:
        print(f"   ⏳ Örtüşen satır sayısı ({compared}) çok az — henüz bir yargıya varılamaz. "
              f"En az {MIN_OVERLAP_FOR_VERDICT} örtüşme birikince tekrar çalıştır.")
    else:
        rate = agreed / compared
        if rate >= TRUST_THRESHOLD:
            print(f"   ✅ Uyum oranı %{rate*100:.0f} ≥ %{TRUST_THRESHOLD*100:.0f} eşiği — Model Tahmini "
                  f"optimizer'a YÖNLENDİRİCİ bağlam olarak eklenebilir (asla tek başına kural "
                  f"gerekçesi olmamalı, asla audience_check gate'ini beslememeli).")
        else:
            print(f"   🛑 Uyum oranı %{rate*100:.0f} < %{TRUST_THRESHOLD*100:.0f} eşiği — Model Tahmini şu an "
                  f"kendi hatalarını doğrulamaktan öteye geçmiyor. Zayıf sinyal olarak KULLANMA; "
                  f"gerçek Etiket sayısı artana kadar bekle.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
