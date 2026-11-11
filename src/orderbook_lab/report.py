"""Summarise a recording: spread, depth, imbalance, markouts, realised vol.

  python -m orderbook_lab.report --data data/ [--out report.md]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M


def build(data: Path) -> str:
    book = M.load(data, "book")
    trades = M.load(data, "trades")
    tob = M.top_of_book(book)
    span = M.book_span_bps(book)
    top5 = sum(book[f"bid_qty_{i}"] + book[f"ask_qty_{i}"] for i in range(5))
    top20_bid = sum(book[f"bid_qty_{i}"] for i in range(20))
    top20_ask = sum(book[f"ask_qty_{i}"] for i in range(20))
    imb = M.imbalance(book, 5)
    try:
        ref, ref_name = M.tob_from_bbo(M.load(data, "bbo")), "bookTicker mid"
    except FileNotFoundError:
        ref, ref_name = tob[["E", "mid"]], "100 ms snapshot mid (stale; markouts biased up)"
    ev = M.taker_events(trades)
    mo = M.markouts(ev, ref)
    # placebo: same times, random signs -> should be ~0; anything else is a bug or drift
    rng = np.random.default_rng(0)
    plc = ev.assign(side=rng.choice([-1, 1], len(ev)))
    mo_plc = M.markouts(plc, ref).dropna()
    mo_ok = mo.dropna()
    w = mo_ok["price"] * mo_ok["qty"]

    span_min = (tob["E"].iloc[-1] - tob["E"].iloc[0]) / 60000
    lines = [
        "# Order-book report",
        f"- snapshots: {len(book):,}, trades: {len(trades):,}, span: {span_min:.1f} min",
        f"- spread (bps): median {tob['spread_bps'].median():.3f}, p95 {tob['spread_bps'].quantile(.95):.3f}",
        f"- top 20 levels reach only {span.median():.2f} bps from mid (median), so depth is 'top-of-book depth', not a wide band",
        f"- depth, median qty: top 5 levels both sides {top5.median():.2f}; top 20 bid {top20_bid.median():.2f}, ask {top20_ask.median():.2f}",
        f"- 5-level imbalance: mean {imb.mean():+.3f}, sd {imb.std():.3f}",
        f"- realised vol (1s bars, annualised): {M.realized_vol(tob):.1%}",
        "",
        f"## Markouts by trade-size quartile (bps, signed by aggressor; reference: {ref_name})",
        f"- {len(trades):,} aggTrades collapsed into {len(ev):,} taker events (same T and side); "
        f"{(ev['n_fills'] > 1).mean():.1%} of events swept more than one level",
        M.summarize_markouts(mo_ok).round(3).to_markdown(),
        "",
        "| check | 1s | 5s | 30s |",
        "|---|---|---|---|",
        "| mean, per event | " + " | ".join(f"{mo_ok[c].mean():.3f}" for c in ("mo_1s_bps", "mo_5s_bps", "mo_30s_bps")) + " |",
        "| mean, notional-weighted | " + " | ".join(f"{np.average(mo_ok[c], weights=w):.3f}" for c in ("mo_1s_bps", "mo_5s_bps", "mo_30s_bps")) + " |",
        "| placebo (random signs) | " + " | ".join(f"{mo_plc[c].mean():.3f}" for c in ("mo_1s_bps", "mo_5s_bps", "mo_30s_bps")) + " |",
        "",
        "## Does imbalance predict the next 5s mid move?",
    ]
    # simple, honest check: correlation of imbalance with forward 5s mid return
    fwd = M.mid_at(tob, tob["E"].to_numpy() + 5000)
    fwd_ret = pd.Series((fwd - tob["mid"].to_numpy()) / tob["mid"].to_numpy() * 1e4)
    fwd_ret[tob["E"].to_numpy() + 5000 > tob["E"].iloc[-1]] = float("nan")
    lines.append(f"- corr(imb_5, fwd 5s return) = {imb.corr(fwd_ret):+.3f} (in-sample, overlapping windows; not a trading signal)")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    text = build(a.data)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
