# pumpfun-top5-forecast-01

An entry for make.fun's forecast contest: read the house's 10-minute marks of every pump.fun coin
launched at or after the opening, and write one forecast of the five largest coins by market cap at
each of the contest's seven hourly answer reads, with a probability per coin and place and a
probability per place's market cap bin.

## Run

```bash
python3 main.py            # reads /input, writes /output/forecast.json
# locally:
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only, no network, seeded, deterministic for a given input.

## Model

Each coin's log10 market cap and its recent trajectory become a strength carried forward to the
read: the cap now, plus a damped share of the recent slope (momentum fades with horizon), plus a
graduation bonus (a pool-priced coin has liquidity behind its cap, a curve coin does not), minus a
drawdown penalty for coins well under their own peak, plus a small age term. Strength gets seeded
Gaussian noise whose sd grows with the horizon, and a Monte Carlo over the noisy orderings gives each
coin a probability at each place; the seven reads are averaged because one forecast is scored at all
seven. Place sizes are a mixture of normals in log10(SOL) around the predicted place caps, one
component per horizon, blended with a uniform floor so a surprise cannot floor every bin.

## Files

- `main.py` the entry, writes `forecast.json`
- `check_forecast.py` schema check plus the order score against the practice answer's published read
- `eval_local.py` backtest: cut the practice history at every mark, score against the marks after it
  with the contest's own rule, and compare with the house persistence baseline
- `sweep.py` the parameter sweep used to pick the hedges and spreads

## License

MIT, see LICENSE.
