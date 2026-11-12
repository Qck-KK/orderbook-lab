"""Summarise a recording: spread, depth, imbalance, markouts, realised vol.

  python -m orderbook_lab.report --data data_us_open [--start 15:30 --end 16:05] [--out report.md]

--start/--end are Berlin local times (HH:MM) on the recording's date.
"""
from __future__ import annotations

import argparse
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import metrics as M

TZ = ZoneInfo("Europe/Berlin")


def _ms(first_ms: int, hhmm: str) -> int:
    day = datetime.fromtimestamp(first_ms / 1000, TZ).date()
    h, m = map(int, hhmm.split(":"))
    return int(datetime.combine(day, dtime(h, m), TZ).timestamp() * 1000)


def load_window(data: Path, start: str | None = None, end: str | None = None):
    book = M.load(data, "book")
    trades = M.load(data, "trades")
    try:
        bbo = M.load(data, "bbo")
    except FileNotFoundError:
        bbo = None
    first = int(book["E"].iloc[0])
    lo = _ms(first, start) if start else -np.inf
    hi = _ms(first, end) if end else np.inf
    cut = lambda df: None if df is None else df[(df["E"] >= lo) & (df["E"] < hi)].reset_index(drop=True)
    book, trades, bbo = cut(book), cut(trades), cut(bbo)
    if len(book) == 0 or len(trades) == 0:
        raise ValueError(f"no data in window {start}-{end}")
    return book, trades, bbo


def stats(book: pd.DataFrame, trades: pd.DataFrame, bbo: pd.DataFrame | None) -> dict:
    tob = M.top_of_book(book)
    ref = M.tob_from_bbo(bbo) if bbo is not None and len(bbo) else tob[["E", "mid"]]
    ev = M.taker_events(trades)
    mo = M.markouts(ev, ref).dropna()
    rng = np.random.default_rng(0)
    plc = M.markouts(ev.assign(side=rng.choice([-1, 1], len(ev))), ref).dropna()
    w = mo["price"] * mo["qty"]
    imb = M.imbalance(book, 5)
    fwd = M.mid_at(tob, tob["E"].to_numpy() + 5000)
    fwd_ret = pd.Series((fwd - tob["mid"].to_numpy()) / tob["mid"].to_numpy() * 1e4)
    fwd_ret[tob["E"].to_numpy() + 5000 > tob["E"].iloc[-1]] = np.nan
    minutes = (tob["E"].iloc[-1] - tob["E"].iloc[0]) / 60000
    q = M.summarize_markouts(mo)
    t0 = datetime.fromtimestamp(tob["E"].iloc[0] / 1000, TZ)
    t1 = datetime.fromtimestamp(tob["E"].iloc[-1] / 1000, TZ)
    return {
        "window (Berlin)": f"{t0:%m-%d %H:%M}-{t1:%H:%M}",
        "minutes": minutes,
        "taker events/min": len(ev) / minutes,
        "% sweeps": 100 * (ev["n_fills"] > 1).mean(),
        "median event $": float((ev["price"] * ev["qty"]).median()),
        "spread bps (median)": tob["spread_bps"].median(),
        "% time spread > 1 tick": 100 * (tob["spread"] > 0.15).mean(),
        "top-5 depth BTC (median)": float(sum(book[f"bid_qty_{i}"] + book[f"ask_qty_{i}"] for i in range(5)).median()),
        "top-20 reach bps": M.book_span_bps(book).median(),
        "realised vol (ann.)": M.realized_vol(tob),
        "corr(imb5, fwd 5s)": imb.corr(fwd_ret),
        "markout 1s, per event": mo["mo_1s_bps"].mean(),
        "markout 5s, per event": mo["mo_5s_bps"].mean(),
        "markout 1s, $-weighted": float(np.average(mo["mo_1s_bps"], weights=w)),
        "markout 1s, smallest quartile": q["mo_1s_bps"].iloc[0],
        "markout 1s, largest quartile": q["mo_1s_bps"].iloc[-1],
        "placebo 1s (≈0)": plc["mo_1s_bps"].mean(),
        "_quartiles": q,
        "_n_trades": len(trades), "_n_events": len(ev), "_n_book": len(book),
    }


def build(data: Path, start: str | None = None, end: str | None = None) -> str:
    s = stats(*load_window(data, start, end))
    q = s["_quartiles"]
    lines = [
        f"# Order-book report: {data} {s['window (Berlin)']}",
        f"- snapshots: {s['_n_book']:,}, aggTrades: {s['_n_trades']:,} -> {s['_n_events']:,} taker events "
        f"({s['% sweeps']:.1f}% swept >1 level), span: {s['minutes']:.1f} min",
        f"- spread: median {s['spread bps (median)']:.3f} bps; wider than 1 tick {s['% time spread > 1 tick']:.1f}% of the time",
        f"- top 20 levels reach {s['top-20 reach bps']:.2f} bps from mid; top-5 depth (both sides) median {s['top-5 depth BTC (median)']:.2f} BTC",
        f"- realised vol (1s bars, annualised): {s['realised vol (ann.)']:.1%}",
        f"- corr(5-level imbalance, fwd 5s return) = {s['corr(imb5, fwd 5s)']:+.3f} (in-sample, overlapping windows)",
        "",
        "## Markouts per taker event by size quartile (bps, signed by aggressor; ref: bookTicker mid)",
        q.round(3).to_markdown(),
        "",
        f"- mean per event 1s/5s: {s['markout 1s, per event']:.3f} / {s['markout 5s, per event']:.3f}; "
        f"$-weighted 1s: {s['markout 1s, $-weighted']:.3f}; placebo 1s: {s['placebo 1s (≈0)']:.3f}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--start", help="Berlin time HH:MM")
    p.add_argument("--end", help="Berlin time HH:MM")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    text = build(a.data, a.start, a.end)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
