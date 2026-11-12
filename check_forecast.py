#!/usr/bin/env python3
"""Local sanity check for a pumpfun-top5 forecast: parses it like the scorer does and scores the
order part against the practice answer's published last read."""

import json
import math
import sys

P_FLOOR = 1e-4

# Published in competition.yaml: the practice answer's top 5 at its last read.
TRUTH_LAST = [
    "5qaP7oKUqKffwwo5DaEqDoTtsBLMpdbQTxEkCDaTpump",  # Max
    "4Dgjitjn8br3LcdcBnkc9tYJ8d1cnCCJd2nsVgzZpump",  # MIHAO VI
    "BuaseKYxFH5Bcz6uoTU2JiX32hRGsGfGLTgamoWpump",  # Google
    "HzwVr7KjPhUriHK9MrJbFzbTDTPwUXGrL2Ah2wVhpump",  # NIHAONOON
    "FjFVmDDM3w7qVLS9uxLkXnDM4qCGsqXSow6d32phpump",  # HAMSTER
]


def parse(doc, places=5, max_rows=100):
    problems = []
    if set(doc.keys()) != {"probabilities", "top5", "sizes"}:
        problems.append(f"keys are {sorted(doc.keys())}")
    named = [dict() for _ in range(places)]
    sums = [0.0] * places
    for i, row in enumerate(doc.get("probabilities", [])):
        if set(row.keys()) != {"mint", "place", "p"}:
            problems.append(f"probabilities[{i}] keys")
            continue
        mint, place, p = row["mint"], row["place"], row["p"]
        if not isinstance(mint, str) or not (32 <= len(mint) <= 44):
            problems.append(f"probabilities[{i}] mint {mint!r}")
        if not isinstance(place, int) or not 1 <= place <= places:
            problems.append(f"probabilities[{i}] place {place!r}")
            continue
        if not (isinstance(p, (int, float)) and 0 <= p <= 1):
            problems.append(f"probabilities[{i}] p {p!r}")
            continue
        if mint in named[place - 1]:
            problems.append(f"{mint} twice at place {place}")
        named[place - 1][mint] = p
        sums[place - 1] += p
        if len(named[place - 1]) > max_rows:
            problems.append(f"place {place} over {max_rows} rows")
    for k, total in enumerate(sums, start=1):
        if total > 1 + 1e-9:
            problems.append(f"place {k} sums to {total}")
    top = doc.get("top5")
    if not isinstance(top, list) or len(top) != places or len(set(top)) != places:
        problems.append("top5 is not 5 distinct mints")
    return problems, named, sums


def order_score(named, sums, chart, places=5):
    total = 0.0
    detail = []
    for k in range(1, min(places, len(chart)) + 1):
        mint = chart[k - 1]
        if mint in named[k - 1]:
            p = named[k - 1][mint]
        else:
            unnamed = [c for c in chart if c not in named[k - 1]]
            left = max(0.0, 1.0 - sums[k - 1])
            p = left / len(unnamed) if unnamed else 0.0
        total += math.log(max(p, P_FLOOR))
        detail.append((k, mint[:8], round(p, 6), round(math.log(max(p, P_FLOOR)), 3)))
    return total, detail


def main(path):
    doc = json.load(open(path, encoding="utf-8"))
    problems, named, sums = parse(doc)
    print("schema problems:", problems if problems else "none")
    print("rows:", len(doc["probabilities"]), "| sizes:", len(doc["sizes"]))
    print("place sums:", [round(s, 4) for s in sums])
    print("top5:", [m[:8] for m in doc["top5"]])
    hits = len(set(doc["top5"]) & set(TRUTH_LAST))
    print(f"top5 hits vs practice truth: {hits}/5")
    score, detail = order_score(named, sums, TRUTH_LAST)
    print(f"order score on the practice last read: {score:.4f}")
    for row in detail:
        print("  place", row[0], row[1], "p", row[2], "ln", row[3])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out/forecast.json")
