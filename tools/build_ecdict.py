#!/usr/bin/env python3
"""Build a trimmed ECDICT asset (ecdict.json) for the English dictionary mode.

Source: skywind3000/ECDICT (ecdict.csv, ~770k entries, CC-BY/MIT).
We keep only common words (by BNC / COCA frequency rank) that have a Chinese
translation, and emit a compact JSON array of:

    [word(lowercase), phonetic(KK), translation(zh), definition(en), pos, tag]

Newlines inside fields are flattened to "; " so the on-device Gson parse stays
simple (same shape as the existing dictionary.json: List<List<String>>).

Usage:
    python tools/build_ecdict.py [INPUT_CSV] [OUTPUT_JSON] [--limit N]

Defaults:
    INPUT_CSV   = /tmp/ecdict_full.csv
    OUTPUT_JSON = app/src/main/assets/ecdict.json
    --limit     = 40000  (top-N words by frequency rank)
"""
import csv
import json
import sys

DEFAULT_INPUT = "/tmp/ecdict_full.csv"
DEFAULT_OUTPUT = "app/src/main/assets/ecdict.json"
DEFAULT_LIMIT = 40000


def flatten(s: str) -> str:
    # ECDICT encodes line breaks as the literal two-char sequence "\n" as well
    # as real newlines; normalise both to "; ".
    s = s.replace("\\n", "\n").replace("\r", "\n")
    return "; ".join(part.strip() for part in s.split("\n") if part.strip()).strip()


def freq_rank(row: dict) -> int:
    """Lower is more common. Combine BNC + COCA(frq); 0 means 'unknown' -> push to back."""
    ranks = []
    for key in ("bnc", "frq"):
        try:
            v = int(row.get(key) or 0)
        except ValueError:
            v = 0
        if v > 0:
            ranks.append(v)
    return min(ranks) if ranks else 10 ** 9


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = DEFAULT_LIMIT
    for a in sys.argv[1:]:
        if a.startswith("--limit"):
            limit = int(a.split("=", 1)[1]) if "=" in a else int(sys.argv[sys.argv.index(a) + 1])
    src = args[0] if len(args) > 0 else DEFAULT_INPUT
    out = args[1] if len(args) > 1 else DEFAULT_OUTPUT

    csv.field_size_limit(10 ** 7)
    rows = []
    with open(src, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = (row.get("word") or "").strip()
            translation = flatten(row.get("translation") or "")
            if not word or not translation:
                continue
            # Skip multi-word phrases and anything non-ascii in the headword.
            if " " in word or not word.isascii():
                continue
            rows.append((freq_rank(row), word, row))

    rows.sort(key=lambda t: t[0])
    seen = set()
    result = []
    for _, word, row in rows:
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append([
            key,
            (row.get("phonetic") or "").strip(),
            flatten(row.get("translation") or ""),
            flatten(row.get("definition") or ""),
            (row.get("pos") or "").strip(),
            (row.get("tag") or "").strip(),
        ])
        if len(result) >= limit:
            break

    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {len(result)} entries to {out}")


if __name__ == "__main__":
    main()
