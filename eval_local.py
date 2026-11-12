#!/usr/bin/env python3
"""Local backtest for the submission: cut the practice history at every mark, forecast from the marks
up to the cut, score against the marks after it with the contest's own rule (order + sizes), and
compare with the persistence baseline the house scores -40.408345 on the practice."""

import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(os.path.dirname(HERE), "kit", "pumpfun-top5-forecast-01-kit")
P_FLOOR = 1e-4

spec = importlib.util.spec_from_file_location("entry", os.path.join(HERE, "main.py"))
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def mark_rows(mark):
    return [r for r in (mark.get("rows") or []) if isinstance(r.get("mint"), str)]


def chart_of(mark):
    rows = sorted(mark_rows(mark), key=lambda r: r.get("rank") or 10**6)
    return [r["mint"] for r in rows]


def cap_of(mark, mint):
    for r in mark_rows(mark):
        if r["mint"] == mint:
            return math.log10(int(r["market_cap_lamports"])) - 9.0
    return None


def place_probabilities(forecast, chart, places=5):
    named = [dict() for _ in range(places)]
    sums = [0.0] * places
    for row in forecast["probabilities"]:
        k = row["place"] - 1
        named[k][row["mint"]] = row["p"]
        sums[k] += row["p"]
    out = []
    for k in range(places):
        mint = chart[k] if k < len(chart) else None
        if mint is None:
            break
        if mint in named[k]:
            out.append(named[k][mint])
        else:
            unnamed = [c for c in chart if c not in named[k]]
            left = max(0.0, 1.0 - sums[k])
            out.append(left / len(unnamed) if unnamed else 0.0)
    return out


def size_probabilities(forecast, edges):
    rows = [dict() for _ in range(5)]
    sums = [0.0] * 5
    for row in forecast["sizes"]:
        k = row["place"] - 1
        rows[k][row["bin"]] = row["p"]
        sums[k] += row["p"]
    bins = len(edges) + 1
    out = []
    for k in range(5):
        at = rows[k]
        def prob(bin_index):
            if bin_index in at:
                return at[bin_index]
            unnamed_bins = bins - len(at)
            left = max(0.0, 1.0 - sums[k])
            return left / unnamed_bins if unnamed_bins else 0.0
        out.append(prob)
    return out


def bin_of(value, edges):
    count = 0
    for edge in edges:
        if edge <= value:
            count += 1
        else:
            break
    return count


def score(forecast, mark, edges):
    chart = chart_of(mark)[:100]
    order = 0.0
    for place, p in enumerate(place_probabilities(forecast, chart), start=1):
        order += math.log(max(p, P_FLOOR))
    size = 0.0
    probs = size_probabilities(forecast, edges)
    for k in range(5):
        mint = chart[k] if k < len(chart) else None
        if mint is None:
            break
        cap = cap_of(mark, mint)
        if cap is None:
            continue
        size += math.log(max(probs[k](bin_of(cap, edges)), P_FLOOR))
    return order, size


def baseline(marks, cut_index, places=5, edges=None):
    """The house baseline, rebuilt exactly: persistence with 0.3/0.1 and sd 0.35 on the caps at the cut."""
    last = marks[cut_index]
    rows = sorted(mark_rows(last), key=lambda r: r.get("rank") or 10**6)
    probs, sizes = [], []
    for k in range(1, places + 1):
        if k > len(rows):
            break
        probs.append({"mint": rows[k - 1]["mint"], "place": k, "p": 0.30})
        if k > 1:
            probs.append({"mint": rows[k - 2]["mint"], "place": k, "p": 0.10})
        if k < len(rows):
            probs.append({"mint": rows[k]["mint"], "place": k, "p": 0.10})
        mean = math.log10(int(rows[k - 1]["market_cap_lamports"])) - 9.0
        for index in range(len(edges) + 1):
            lo = float("-inf") if index == 0 else edges[index - 1]
            hi = float("inf") if index == len(edges) else edges[index]
            p = entry.above(lo, mean, 0.35) - entry.above(hi, mean, 0.35)
            u = math.floor(p * 1e6)
            if u > 0:
                sizes.append({"place": k, "bin": index, "p": u / 1e6})
    return {"probabilities": probs, "top5": [r["mint"] for r in rows[:places]], "sizes": sizes}


def main():
    task = json.load(open(os.path.join(KIT, "input", "task.json"), encoding="utf-8"))
    edges = [float(e) for e in task["size"]["edges"]]
    marks = entry.load_marks(os.path.join(KIT, "input", "history.ndjson"))
    print(f"marks: {len(marks)}")

    totals = {"mine": [], "base": []}
    for cut in range(2, len(marks) - 1):
        history = marks[: cut + 1]
        future = marks[cut + 1 :]
        sub_task = dict(task)
        sub_task["reads"] = [m["at"] for m in future]
        mine = entry.build_forecast(history, sub_task)
        base = baseline(marks, cut, edges=edges)
        mine_o, mine_s = 0.0, 0.0
        base_o, base_s = 0.0, 0.0
        for mark in future:
            o, s = score(mine, mark, edges)
            mine_o += o
            mine_s += s
            o, s = score(base, mark, edges)
            base_o += o
            base_s += s
        n = len(future)
        mine_o, mine_s, base_o, base_s = mine_o / n, mine_s / n, base_o / n, base_s / n
        totals["mine"].append((mine_o, mine_s))
        totals["base"].append((base_o, base_s))
        print(
            f"cut {cut}: mine order {mine_o:7.3f} size {mine_s:7.3f} total {mine_o + mine_s:8.3f} | "
            f"base order {base_o:7.3f} size {base_s:7.3f} total {base_o + base_s:8.3f}"
        )

    for key in totals:
        orders = [o for o, _ in totals[key]]
        sizes = [s for _, s in totals[key]]
        print(f"{key}: mean order {sum(orders) / len(orders):.3f} mean size {sum(sizes) / len(sizes):.3f} "
              f"mean total {(sum(orders) + sum(sizes)) / len(orders):.3f}")


if __name__ == "__main__":
    main()
