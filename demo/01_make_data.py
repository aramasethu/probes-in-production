"""Generate a small, balanced, cleanly-labeled true/false statement dataset.

The goal is volume + clean labels, not coverage. We mechanically build
statements from facts we control, so the label is never ambiguous.

The set is deliberately FACTUAL-dominant: "truth is linearly encoded" is
strongest for facts the model reliably knows (capitals, continents, element
symbols). Arithmetic / comparison are included as a smaller, harder slice --
their truth is encoded less cleanly, which 03_sweep.py surfaces in the
per-category breakdown.

For each true statement we mint a matched false one by swapping in a wrong
(but plausible) value, keeping surface form balanced across labels.
"""
from __future__ import annotations

import json
import random

from common import DATA_PATH, SEED

# --- raw facts: country -> (capital, continent) ----------------------------

COUNTRIES = {
    "France": ("Paris", "Europe"), "Japan": ("Tokyo", "Asia"),
    "Italy": ("Rome", "Europe"), "Spain": ("Madrid", "Europe"),
    "Germany": ("Berlin", "Europe"), "Canada": ("Ottawa", "North America"),
    "Russia": ("Moscow", "Europe"), "China": ("Beijing", "Asia"),
    "Egypt": ("Cairo", "Africa"), "Brazil": ("Brasilia", "South America"),
    "Australia": ("Canberra", "Oceania"), "India": ("New Delhi", "Asia"),
    "Greece": ("Athens", "Europe"), "Portugal": ("Lisbon", "Europe"),
    "Norway": ("Oslo", "Europe"), "Sweden": ("Stockholm", "Europe"),
    "Poland": ("Warsaw", "Europe"), "Austria": ("Vienna", "Europe"),
    "Turkey": ("Ankara", "Asia"), "Thailand": ("Bangkok", "Asia"),
    "Kenya": ("Nairobi", "Africa"), "Peru": ("Lima", "South America"),
    "Cuba": ("Havana", "North America"), "Ireland": ("Dublin", "Europe"),
    "Finland": ("Helsinki", "Europe"), "Hungary": ("Budapest", "Europe"),
    "Chile": ("Santiago", "South America"), "Iran": ("Tehran", "Asia"),
    "Iraq": ("Baghdad", "Asia"), "Argentina": ("Buenos Aires", "South America"),
    "Mexico": ("Mexico City", "North America"), "Nigeria": ("Abuja", "Africa"),
    "Vietnam": ("Hanoi", "Asia"), "Indonesia": ("Jakarta", "Asia"),
    "Colombia": ("Bogota", "South America"), "Morocco": ("Rabat", "Africa"),
    "Ukraine": ("Kyiv", "Europe"), "Netherlands": ("Amsterdam", "Europe"),
    "Belgium": ("Brussels", "Europe"), "Switzerland": ("Bern", "Europe"),
    "Denmark": ("Copenhagen", "Europe"), "Ethiopia": ("Addis Ababa", "Africa"),
    "Pakistan": ("Islamabad", "Asia"), "Ghana": ("Accra", "Africa"),
    "Cambodia": ("Phnom Penh", "Asia"), "Bolivia": ("La Paz", "South America"),
    "Romania": ("Bucharest", "Europe"), "Algeria": ("Algiers", "Africa"),
    "Venezuela": ("Caracas", "South America"), "Iceland": ("Reykjavik", "Europe"),
}

ELEMENTS = {
    "gold": "Au", "silver": "Ag", "iron": "Fe", "oxygen": "O",
    "hydrogen": "H", "carbon": "C", "sodium": "Na", "potassium": "K",
    "helium": "He", "nitrogen": "N", "calcium": "Ca", "copper": "Cu",
    "zinc": "Zn", "lead": "Pb", "tin": "Sn", "mercury": "Hg",
}

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
TENS = {30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy"}


def num_to_words(n: int) -> str:
    if n <= 20:
        return ONES[n]
    if n < 30:
        return "twenty-" + ONES[n - 20]
    for base, word in TENS.items():
        if base <= n < base + 10:
            return word if n == base else f"{word}-{ONES[n - base]}"
    return str(n)


def cap(s: str) -> str:
    return s[0].upper() + s[1:]


def main() -> None:
    rng = random.Random(SEED)
    rows: list[dict] = []

    cities = [v[0] for v in COUNTRIES.values()]
    continents = sorted({v[1] for v in COUNTRIES.values()})

    for country, (city, continent) in COUNTRIES.items():
        # capital: true + corrupted (wrong capital from another country)
        rows.append({"statement": f"The capital of {country} is {city}.",
                     "label": 1, "kind": "capital"})
        wrong_city = rng.choice([c for c in cities if c != city])
        rows.append({"statement": f"The capital of {country} is {wrong_city}.",
                     "label": 0, "kind": "capital"})
        # continent: true + corrupted (wrong continent)
        rows.append({"statement": f"{country} is located in {continent}.",
                     "label": 1, "kind": "continent"})
        wrong_cont = rng.choice([c for c in continents if c != continent])
        rows.append({"statement": f"{country} is located in {wrong_cont}.",
                     "label": 0, "kind": "continent"})

    # element symbols: true + corrupted
    symbols = list(ELEMENTS.values())
    for name, sym in ELEMENTS.items():
        rows.append({"statement": f"The chemical symbol for {name} is {sym}.",
                     "label": 1, "kind": "element"})
        wrong = rng.choice([s for s in symbols if s != sym])
        rows.append({"statement": f"The chemical symbol for {name} is {wrong}.",
                     "label": 0, "kind": "element"})

    # arithmetic (addition): true + corrupted result  -- the harder slice
    seen = set()
    while sum(1 for r in rows if r["kind"] == "arith") < 40:
        a, b = rng.randint(2, 39), rng.randint(2, 39)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        c = a + b
        rows.append({"statement": f"{cap(num_to_words(a))} plus "
                     f"{num_to_words(b)} equals {num_to_words(c)}.",
                     "label": 1, "kind": "arith"})
        off = rng.choice([-3, -2, -1, 1, 2, 3])
        rows.append({"statement": f"{cap(num_to_words(a))} plus "
                     f"{num_to_words(b)} equals {num_to_words(c + off)}.",
                     "label": 0, "kind": "arith"})

    # comparisons: true + corrupted (flip the relation)  -- the harder slice
    seen.clear()
    while sum(1 for r in rows if r["kind"] == "compare") < 40:
        a, b = rng.randint(1, 70), rng.randint(1, 70)
        if a == b or (a, b) in seen:
            continue
        seen.add((a, b))
        hi, lo = max(a, b), min(a, b)
        rows.append({"statement": f"{cap(num_to_words(hi))} is greater "
                     f"than {num_to_words(lo)}.", "label": 1, "kind": "compare"})
        rows.append({"statement": f"{cap(num_to_words(lo))} is greater "
                     f"than {num_to_words(hi)}.", "label": 0, "kind": "compare"})

    rng.shuffle(rows)

    n_true = sum(r["label"] for r in rows)
    with open(DATA_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"[data] wrote {len(rows)} statements to {DATA_PATH}")
    print(f"[data] balance: {n_true} true / {len(rows) - n_true} false")
    by_kind: dict[str, int] = {}
    for r in rows:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
    print(f"[data] by kind: {by_kind}")
    print("[data] samples:")
    for r in rows[:6]:
        print(f"    [{r['label']}] {r['statement']}")


if __name__ == "__main__":
    main()
