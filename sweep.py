#!/usr/bin/env python3
"""Parameter sweep on the practice backtest: hedges, size spread, ranking noise."""

import importlib.util
import itertools
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(os.path.dirname(HERE), "kit", "pumpfun-top5-forecast-01-kit")

spec = importlib.util.spec_from_file_location("entry", os.path.join(HERE, "main.py"))
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)

ev_spec = importlib.util.spec_from_file_location("ev", os.path.join(HERE, "eval_local.py"))
ev = importlib.util.module_from_spec(ev_spec)
ev_spec.loader.exec_module(ev)

task = json.load(open(os.path.join(KIT, "input", "task.json"), encoding="utf-8"))
edges = [float(e) for e in task["size"]["edges"]]
marks = entry.load_marks(os.path.join(KIT, "input", "history.ndjson"))


def run():
    orders, sizes = [], []
    for cut in range(2, len(marks) - 1):
        history = marks[: cut + 1]
        future = marks[cut + 1 :]
        sub_task = dict(task)
        sub_task["reads"] = [m["at"] for m in future]
        mine = entry.build_forecast(history, sub_task)
        o_total = s_total = 0.0
        for mark in future:
            o, s = ev.score(mine, mark, edges)
            o_total += o
            s_total += s
        n = len(future)
        orders.append(o_total / n)
        sizes.append(s_total / n)
    return sum(orders) / len(orders), sum(sizes) / len(sizes)


results = []
for hedge, size_hedge, sd_base, noise in itertools.product(
    (0.10, 0.18, 0.26, 0.34), (0.05, 0.10, 0.20), (0.30, 0.38, 0.45), (0.45, 0.60, 0.75)
):
    entry.HEDGE = hedge
    entry.SIZE_HEDGE = size_hedge
    entry.SIZE_SD_BASE = sd_base
    entry.NOISE_SD_BASE = noise
    o, s = run()
    results.append((o + s, o, s, hedge, size_hedge, sd_base, noise))

results.sort(reverse=True)
print(f"{'total':>9} {'order':>8} {'size':>8}  hedge size_hedge sd_base noise")
for total, o, s, hedge, size_hedge, sd_base, noise in results[:12]:
    print(f"{total:9.3f} {o:8.3f} {s:8.3f}  {hedge:5.2f} {size_hedge:10.2f} {sd_base:7.2f} {noise:5.2f}")
