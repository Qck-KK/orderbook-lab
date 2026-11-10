# orderbook-lab

A small, honest study of Binance USD-M perpetual microstructure (default: BTCUSDT).
It records the public order book and trades, then measures what a market maker cares about:
how wide the spread is, how much size sits near the mid, and whether trades are followed by
adverse price moves (markouts).

## Questions
1. How do spread and near-mid depth vary through the day?
2. Are larger aggressive trades more "informed", i.e. do they have larger positive markouts at 1s / 5s / 30s?
3. Does top-of-book imbalance or microprice say anything about the next few seconds of mid moves?

## Setup
```bash
pip install -e .[dev]
pytest                                   # offline tests on synthetic data
python -m orderbook_lab.recorder --symbol btcusdt --minutes 60 --out data/
python -m orderbook_lab.report --data data/ --out report.md
```

## What is measured
| Metric | Definition |
|---|---|
| Spread (bps) | (ask₀ − bid₀) / mid × 10⁴ |
| Microprice | (bid₀·askQty₀ + ask₀·bidQty₀) / (bidQty₀ + askQty₀) |
| Depth within X bps | Sum of quantity on levels within X bps of mid (top 20 levels only) |
| Imbalance | (bidQty − askQty) / (bidQty + askQty) over the first k levels |
| Markout at h | side × (mid(t+h) − mid(t−)) / mid(t−) × 10⁴, using only data before the trade for mid(t−) |
| Realised vol | Std of 1s log mid returns, annualised over 365×24×3600 s |

## Limitations (read before quoting any number)
- `depth20@100ms` is a **sampled partial book**: it can miss changes between snapshots and only sees 20 levels.
  A full diff-depth book (REST snapshot + `@depth` updates with sequence checks) is the next step.
- Exchange event time is used throughout; network latency to the recorder is stored (`recv_ns`) but not corrected.
- Markouts use overlapping windows, so per-trade values are correlated; treat means as descriptive, not significant.
- No fees, queue position or latency are modelled, so nothing here is a trading strategy.

## Layout
```
src/orderbook_lab/recorder.py   websocket -> chunked parquet (reconnects, flushes every 60s)
src/orderbook_lab/metrics.py    spread, microprice, depth, imbalance, markouts, realised vol
src/orderbook_lab/report.py     markdown summary of a recording
tests/test_metrics.py           unit tests incl. a look-ahead check for markouts
```
