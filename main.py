#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry.

Reads the house's 10-minute marks of every pump.fun coin launched at or after the opening
(/input/task.json, /input/history.ndjson) and writes one forecast of the top 5 places by market cap
at each of the contest's answer reads (/output/forecast.json).

Model, in one paragraph: each coin's log10 market cap and its recent trajectory are turned into a
"strength" that carries the coin forward to the read. Strength is the cap now, plus a damped share of
the recent slope (momentum fades with the horizon), plus a graduation bonus (a coin whose pool prices
it has real liquidity behind the cap, a curve coin does not), minus a drawdown penalty (a coin well
under its own peak is on the way down), plus a small age term. Strength is then given noise whose sd
grows with the horizon, and a seeded Monte Carlo over the noisy orderings gives each coin a
probability at each place; the seven reads' distributions are averaged, because the one forecast is
scored at all seven. Place sizes are a mixture of normals in log10(SOL) around the predicted place
caps, one component per read horizon, so the mixture widens as the horizons spread out.

Standard library only, no network, seeded, deterministic for a given input.
"""

import json
import math
import os
import random

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261112
DRAWS = 4000
MAX_CANDIDATES = 160
NAMED_PER_PLACE = 14
HEDGE = 0.34
SIZE_HEDGE = 0.20
SIZE_SD_BASE = 0.30
SIZE_SD_PER_HOUR = 0.055
SIZE_DRIFT_PER_HOUR = -0.03
NOISE_SD_BASE = 0.45
NOISE_SD_PER_HOUR = 0.040
MICRO = 1_000_000


def load_marks(path):
    marks = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                mark = json.loads(line)
            except json.JSONDecodeError:
                continue
            if mark.get("ok") is True and isinstance(mark.get("rows"), list):
                marks.append(mark)
    marks.sort(key=lambda m: m.get("at") or "")
    return marks


def log_cap(row):
    raw = row.get("market_cap_lamports")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return math.log10(value) - 9.0


def parse_instant(text):
    if not text or not isinstance(text, str):
        return None
    value = text.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        from datetime import datetime

        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def collect(marks):
    """mint -> list of (t, log cap, rank, venue) plus the row that created it."""
    series = {}
    created = {}
    for mark in marks:
        at = parse_instant(mark.get("at"))
        if at is None:
            continue
        for row in mark["rows"]:
            mint = row.get("mint")
            cap = log_cap(row)
            if not isinstance(mint, str) or not mint or cap is None:
                continue
            series.setdefault(mint, []).append((at, cap, row.get("rank"), row.get("venue")))
            if mint not in created:
                created[mint] = parse_instant(row.get("created_at")) or at
    return series, created


def slope_per_hour(points, window=4):
    tail = points[-window:]
    if len(tail) < 2:
        return 0.0
    (t0, c0, _, _), (t1, c1, _, _) = tail[0], tail[-1]
    hours = (t1 - t0) / 3600.0
    if hours <= 0.05:
        return 0.0
    return (c1 - c0) / hours


def features(series, created, cut):
    out = {}
    for mint, points in series.items():
        if not points:
            continue
        points = sorted(points, key=lambda p: p[0])
        at_last, cap_last, rank_last, venue_last = points[-1]
        caps = [p[1] for p in points]
        cap_max = max(caps)
        age_hours = max(0.0, (at_last - created.get(mint, at_last)) / 3600.0)
        out[mint] = {
            "cap": cap_last,
            "cap_max": cap_max,
            "drawdown": min(cap_max - cap_last, 1.6),
            "slope": slope_per_hour(points),
            "age": age_hours,
            "pool": 1.0 if venue_last == "pool" else 0.0,
            "rank": rank_last if isinstance(rank_last, int) else 10**6,
            "marks": len(points),
        }
    return out


def strength(feat, hours):
    taper = min(max(hours, 0.0), 3.0) / 3.0
    momentum = 0.42 * max(-1.3, min(1.3, feat["slope"])) * taper
    graduation = 0.30 * feat["pool"]
    penalty = -0.22 * feat["drawdown"]
    age_term = 0.015 * math.log1p(feat["age"])
    return feat["cap"] + momentum + graduation + penalty + age_term


def erfc(x):
    z = abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    r = t * math.exp(
        -z * z
        - 1.26551223
        + t
        * (
            1.00002368
            + t
            * (
                0.37409196
                + t
                * (
                    0.09678418
                    + t
                    * (
                        -0.18628806
                        + t
                        * (
                            0.27886807
                            + t
                            * (
                                -1.13520398
                                + t
                                * (1.48851587 + t * (-0.82215223 + t * 0.17087277))
                            )
                        )
                    )
                )
            )
        )
    )
    return r if x >= 0 else 2.0 - r


def above(x, mean, sd):
    if x == float("-inf"):
        return 1.0
    if x == float("inf"):
        return 0.0
    return 0.5 * erfc((x - mean) / (sd * math.sqrt(2.0)))


def bin_probabilities(mean, sd, edges):
    """Normal in log10 space, masses over the bins the task's edges cut."""
    probs = []
    for index in range(len(edges) + 1):
        lo = float("-inf") if index == 0 else edges[index - 1]
        hi = float("inf") if index == len(edges) else edges[index]
        mass = above(lo, mean, sd) - above(hi, mean, sd)
        if mass > 0:
            probs.append((index, mass))
    return probs


def build_forecast(marks, task):
    """The forecast document for these marks and this task (pure: no IO)."""
    places = int(task.get("places", 5))
    edges = [float(e) for e in task["size"]["edges"]]

    cut = max(parse_instant(m.get("at")) or 0.0 for m in marks)
    reads = [parse_instant(r) for r in task.get("reads", [])]
    horizons = [max(0.0, (r - cut) / 3600.0) for r in reads if r is not None]
    if not horizons:
        horizons = [0.0]

    series, created = collect(marks)
    feat = features(series, created, cut)
    if not feat:
        raise SystemExit("the history names no coin")

    by_strength = sorted(feat.items(), key=lambda kv: (-kv[1]["cap"], kv[0]))
    candidates = [mint for mint, _ in by_strength[:MAX_CANDIDATES]]
    if len(candidates) < places:
        raise SystemExit("fewer coins than places in the history")

    rng = random.Random(SEED)
    place_mass = [dict() for _ in range(places)]
    size_mass = [dict() for _ in range(places)]
    order_votes = {mint: 0.0 for mint in candidates}

    draws = DRAWS
    for hours in horizons:
        base = [strength(feat[mint], hours) for mint in candidates]
        sd = NOISE_SD_BASE + NOISE_SD_PER_HOUR * hours
        place_caps = [[] for _ in range(places)]
        noise = [[rng.gauss(0.0, 1.0) for _ in candidates] for _ in range(draws)]
        for draw in range(draws):
            sample = [(base[i] + sd * noise[draw][i], i) for i in range(len(candidates))]
            sample.sort(key=lambda pair: -pair[0])
            for place in range(places):
                value, index = sample[place]
                place_mass[place][candidates[index]] = place_mass[place].get(candidates[index], 0.0) + 1.0
                place_caps[place].append(base[index])
            for rank in range(places):
                order_votes[candidates[sample[rank][1]]] += 1.0 / (1.0 + rank)

        for place in range(places):
            caps = sorted(place_caps[place])
            if not caps:
                continue
            mean = caps[len(caps) // 2] + SIZE_DRIFT_PER_HOUR * hours
            size_sd = SIZE_SD_BASE + SIZE_SD_PER_HOUR * hours
            for index, mass in bin_probabilities(mean, size_sd, edges):
                size_mass[place][index] = size_mass[place].get(index, 0.0) + mass

    reads_count = float(len(horizons))
    probabilities = []
    sizes = []
    for place in range(places):
        total = sum(place_mass[place].values()) or 1.0
        share = {}
        for mint, count in place_mass[place].items():
            share[mint] = (count / total) * (1.0 - HEDGE)
        ranked = sorted(share.items(), key=lambda kv: (-kv[1], kv[0]))[:NAMED_PER_PLACE]
        kept = 0.0
        for mint, value in ranked:
            kept += value
            if value >= 1e-5:
                probabilities.append({"mint": mint, "place": place + 1, "p": round(value, 6)})
        if kept > 1.0 - HEDGE:
            pass
        bins = {}
        for index, mass in size_mass[place].items():
            bins[index] = bins.get(index, 0.0) + mass / reads_count
        bin_total = sum(bins.values()) or 1.0
        width = len(edges) + 1
        keep = 1.0 - SIZE_HEDGE
        for index in range(width):
            value = keep * (bins.get(index, 0.0) / bin_total) + SIZE_HEDGE / width
            if value >= 1e-5:
                sizes.append({"place": place + 1, "bin": index, "p": round(value, 6)})

    top5 = []
    for place in range(places):
        best = max(place_mass[place].items(), key=lambda kv: (kv[1], -len(kv[0])), default=None)
        if best is not None and best[0] not in top5:
            top5.append(best[0])
    if len(top5) < places:
        for mint, _ in sorted(order_votes.items(), key=lambda kv: (-kv[1], kv[0])):
            if mint not in top5:
                top5.append(mint)
            if len(top5) == places:
                break

    forecast = {"probabilities": probabilities, "top5": top5[:places], "sizes": sizes}
    return forecast


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as handle:
        task = json.load(handle)
    marks = load_marks(os.path.join(INPUT_DIR, "history.ndjson"))
    if not marks:
        raise SystemExit("the history holds no ok mark")
    output_name = task.get("output", "forecast.json")
    forecast = build_forecast(marks, task)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, output_name), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(forecast, sort_keys=True))
        handle.write("\n")


if __name__ == "__main__":
    main()
