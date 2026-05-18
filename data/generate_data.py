# generate_data.py
# Generates training data for the router model.
# Supports manual templates and auto-generation via API.

import json
import random
import os
from pathlib import Path

OUTPUT_DIR   = Path(__file__).parent
TRAIN_FILE   = OUTPUT_DIR / "train.jsonl"
VALID_FILE   = OUTPUT_DIR / "valid.jsonl"
VALID_SPLIT  = 0.15   # 15% of data goes to validation

SYSTEM_PROMPT = (
    "You are a request classifier. Reply ONLY with JSON.\n"
    "Format: {\"domain\": \"...\", \"intent\": \"...\"}\n"
    "Domains: medizin, technik, ernaehrung, allgemein\n"
    "Intents: quick_info, research, personal_advice, how_to"
)


# Raw examples: (user_input, domain, intent)
# Add your own examples here — the more the better.
RAW_EXAMPLES = [

    # ── medizin / quick_info ──────────────────────────────────
    ("Wie viel Vitamin C brauche ich täglich?",             "medizin", "quick_info"),
    ("Was ist der normale Blutdruck?",                      "medizin", "quick_info"),
    ("Wie viel Schlaf braucht ein Erwachsener?",            "medizin", "quick_info"),
    ("Was hilft gegen Kopfschmerzen?",                      "medizin", "quick_info"),
    ("Wie hoch ist der normale Blutzucker?",                "medizin", "quick_info"),
    ("Was ist Vitamin D?",                                  "medizin", "quick_info"),
    ("Wie lange dauert eine Erkältung?",                    "medizin", "quick_info"),
    ("Was ist ein normaler Puls?",                          "medizin", "quick_info"),

    # ── medizin / research ───────────────────────────────────
    ("Ich forsche über Mangelkrankheiten, erkläre Skorbut.", "medizin", "research"),
    ("Was ist der wissenschaftliche Hintergrund von Vitamin C?", "medizin", "research"),
    ("Erkläre den Zusammenhang zwischen Schlaf und Immunsystem.", "medizin", "research"),
    ("Welche Studien gibt es zu Omega-3 Fettsäuren?",       "medizin", "research"),
    ("Wie funktioniert das Immunsystem auf zellulärer Ebene?", "medizin", "research"),
    ("Was sind die biochemischen Ursachen von Typ-2 Diabetes?", "medizin", "research"),

    # ── medizin / personal_advice ────────────────────────────
    ("Ich bin oft müde, könnte das Vitaminmangel sein?",    "medizin", "personal_advice"),
    ("Ich habe seit einer Woche Kopfschmerzen, was tun?",   "medizin", "personal_advice"),
    ("Mein Kind schläft schlecht, was kann ich tun?",       "medizin", "personal_advice"),
    ("Ich fühle mich ständig erschöpft, was könnte das sein?", "medizin", "personal_advice"),
    ("Ich habe Rückenschmerzen beim Sitzen, was hilft?",    "medizin", "personal_advice"),

    # ── technik / how_to ─────────────────────────────────────
    ("Mein Python Script gibt einen TypeError aus.",        "technik", "how_to"),
    ("Wie installiere ich Python auf dem Mac?",             "technik", "how_to"),
    ("Wie erstelle ich eine virtuelle Umgebung in Python?", "technik", "how_to"),
    ("Wie verbinde ich ein GitHub Repository lokal?",       "technik", "how_to"),
    ("Mein Git Push wird abgelehnt, was tun?",              "technik", "how_to"),
    ("Wie lese ich eine JSON Datei in Python?",             "technik", "how_to"),
    ("Wie installiere ich ein Python Paket mit pip?",       "technik", "how_to"),
    ("Wie starte ich einen lokalen Webserver?",             "technik", "how_to"),

    # ── technik / research ───────────────────────────────────
    ("Erkläre mir wie ein Transformer funktioniert.",       "technik", "research"),
    ("Was ist der Unterschied zwischen RAM und SSD?",       "technik", "research"),
    ("Wie funktioniert HTTPS Verschlüsselung?",             "technik", "research"),
    ("Erkläre den Unterschied zwischen TCP und UDP.",       "technik", "research"),
    ("Was ist Backpropagation in neuronalen Netzen?",       "technik", "research"),
    ("Wie funktioniert ein Betriebssystem intern?",         "technik", "research"),

    # ── technik / quick_info ─────────────────────────────────
    ("Was bedeutet RAM?",                                   "technik", "quick_info"),
    ("Was ist Python?",                                     "technik", "quick_info"),
    ("Was ist ein API?",                                    "technik", "quick_info"),
    ("Was ist der Unterschied zwischen CPU und GPU?",       "technik", "quick_info"),

    # ── ernaehrung / quick_info ──────────────────────────────
    ("Was kann ich mit Hähnchenbrust kochen?",              "ernaehrung", "quick_info"),
    ("Wie viele Kalorien hat ein Apfel?",                   "ernaehrung", "quick_info"),
    ("Was ist ein gesundes Frühstück?",                     "ernaehrung", "quick_info"),
    ("Wie viel Protein brauche ich täglich?",               "ernaehrung", "quick_info"),
    ("Was sind gute Quellen für Eisen?",                    "ernaehrung", "quick_info"),
    ("Wie viel Wasser sollte ich täglich trinken?",         "ernaehrung", "quick_info"),
    ("Was sind gesunde Snacks?",                            "ernaehrung", "quick_info"),

    # ── ernaehrung / how_to ──────────────────────────────────
    ("Wie koche ich Reis richtig?",                         "ernaehrung", "how_to"),
    ("Wie bereite ich Hähnchen sicher zu?",                 "ernaehrung", "how_to"),
    ("Wie mache ich einen gesunden Smoothie?",              "ernaehrung", "how_to"),
    ("Wie meal prepe ich für die ganze Woche?",             "ernaehrung", "how_to"),

    # ── ernaehrung / research ────────────────────────────────
    ("Was sagt die Forschung über Intervallfasten?",        "ernaehrung", "research"),
    ("Welche Studien gibt es zur Mittelmeer-Diät?",         "ernaehrung", "research"),
    ("Was ist der wissenschaftliche Stand zu Zucker?",      "ernaehrung", "research"),

    # ── allgemein / quick_info ───────────────────────────────
    ("Hey wie geht's?",                                     "allgemein", "quick_info"),
    ("Was kannst du für mich tun?",                         "allgemein", "quick_info"),
    ("Wer hat Deutschland gegründet?",                      "allgemein", "quick_info"),
    ("Was ist die Hauptstadt von Frankreich?",              "allgemein", "quick_info"),
    ("Wie spät ist es?",                                    "allgemein", "quick_info"),
    ("Was ist heute für ein Tag?",                          "allgemein", "quick_info"),
    ("Erzähl mir einen Witz.",                              "allgemein", "quick_info"),
    ("Was ist das Wetter heute?",                           "allgemein", "quick_info"),
]


def to_mlx_format(user_input: str, domain: str, intent: str) -> dict:
    """Convert a raw example to MLX chat format."""
    label = json.dumps({"domain": domain, "intent": intent},
                       ensure_ascii=False)
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_input},
            {"role": "assistant", "content": label},
        ]
    }


def save_jsonl(examples: list, path: Path):
    """Write examples to a .jsonl file (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(examples)} examples → {path}")


def generate():
    """Convert RAW_EXAMPLES, shuffle, split into train/valid and save."""
    converted = [to_mlx_format(inp, dom, intent)
                 for inp, dom, intent in RAW_EXAMPLES]

    random.seed(42)
    random.shuffle(converted)

    split_idx   = int(len(converted) * (1 - VALID_SPLIT))
    train_data  = converted[:split_idx]
    valid_data  = converted[split_idx:]

    save_jsonl(train_data, TRAIN_FILE)
    save_jsonl(valid_data, VALID_FILE)

    print(f"\nTotal:      {len(converted)}")
    print(f"Train:      {len(train_data)}")
    print(f"Validation: {len(valid_data)}")

    # Show label distribution
    from collections import Counter
    labels = [(ex["messages"][2]["content"]) for ex in converted]
    parsed = [json.loads(l) for l in labels]
    domains = Counter(p["domain"] for p in parsed)
    intents = Counter(p["intent"] for p in parsed)
    print(f"\nDomains: {dict(domains)}")
    print(f"Intents: {dict(intents)}")


if __name__ == "__main__":
    generate()
