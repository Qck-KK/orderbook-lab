"""Microstructure metrics on recorded book snapshots and trades.

Conventions
  * time index: exchange event time `E` in ms
  * bps = 1e4 * x / mid
  * trade side: +1 buyer-initiated, -1 seller-initiated
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load(path, kind: str) -> pd.DataFrame:
    """Read all parquet chunks for one stream kind ('book', 'trades', 'bbo')."""
    from pathlib import Path

    files = sorted(Path(path, kind).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {kind} files under {path}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("E").drop_duplicates().reset_index(drop=True)


def top_of_book(book: pd.DataFrame) -> pd.DataFrame:
    """Mid, spread (abs and bps) and microprice from level-0 columns."""
    bp, ap = book["bid_px_0"], book["ask_px_0"]
    bq, aq = book["bid_qty_0"], book["ask_qty_0"]
    mid = (bp + ap) / 2
    out = pd.DataFrame({"E": book["E"], "mid": mid, "spread": ap - bp})
    out["spread_bps"] = 1e4 * out["spread"] / mid
    # microprice: weights the side with LESS size more (price leans toward thin side)
    out["microprice"] = (bp * aq + ap * bq) / (bq + aq)
    return out


def depth_within(book: pd.DataFrame, bps: float, n_levels: int = 20) -> pd.DataFrame:
    """Total bid and ask quantity within `bps` of mid (per snapshot)."""
    mid = (book["bid_px_0"] + book["ask_px_0"]) / 2
    lo, hi = mid * (1 - bps / 1e4), mid * (1 + bps / 1e4)
    bid = sum(book[f"bid_qty_{i}"].where(book[f"bid_px_{i}"] >= lo, 0.0) for i in range(n_levels))
    ask = sum(book[f"ask_qty_{i}"].where(book[f"ask_px_{i}"] <= hi, 0.0) for i in range(n_levels))
    return pd.DataFrame({"E": book["E"], f"bid_depth_{bps:g}bps": bid, f"ask_depth_{bps:g}bps": ask})


def imbalance(book: pd.DataFrame, levels: int = 5) -> pd.Series:
    """(bid - ask) / (bid + ask) over the first `levels` levels, in [-1, 1]."""
    b = sum(book[f"bid_qty_{i}"] for i in range(levels))
    a = sum(book[f"ask_qty_{i}"] for i in range(levels))
    return ((b - a) / (b + a)).rename(f"imb_{levels}")


def mid_at(tob: pd.DataFrame, times_ms: np.ndarray) -> np.ndarray:
    """Last observed mid at or before each time (as-of join, no look-ahead)."""
    idx = np.searchsorted(tob["E"].to_numpy(), times_ms, side="right") - 1
    mids = tob["mid"].to_numpy()
    res = np.full(len(times_ms), np.nan)
    ok = idx >= 0
    res[ok] = mids[idx[ok]]
    return res


def markouts(trades: pd.DataFrame, tob: pd.DataFrame, horizons_s=(1, 5, 30)) -> pd.DataFrame:
    """Signed mid move after each trade, in bps of the pre-trade mid.

    markout_h = side * (mid(t+h) - mid(t-)) / mid(t-) * 1e4
    Positive = price kept moving in the aggressor's direction (informed / toxic flow
    from a market maker's point of view). Trades whose horizon runs past the end of
    the book data get NaN rather than a stale value.
    """
    t = trades["E"].to_numpy()
    m0 = mid_at(tob, t - 1)  # strictly before the trade
    end = tob["E"].iloc[-1]
    out = trades[["E", "price", "qty", "side"]].copy()
    out["mid0"] = m0
    for h in horizons_s:
        th = t + int(h * 1000)
        mh = mid_at(tob, th)
        mh[th > end] = np.nan
        out[f"mo_{h}s_bps"] = trades["side"].to_numpy() * (mh - m0) / m0 * 1e4
    return out


def summarize_markouts(mo: pd.DataFrame, by_size_quantiles: int = 4) -> pd.DataFrame:
    """Mean markout by trade-size bucket (notional). Bigger trades are usually more informed."""
    mo = mo.copy()
    mo["notional"] = mo["price"] * mo["qty"]
    mo["size_bucket"] = pd.qcut(mo["notional"], by_size_quantiles, labels=False, duplicates="drop")
    cols = [c for c in mo.columns if c.startswith("mo_")]
    g = mo.groupby("size_bucket")
    res = g[cols].mean()
    res["n"] = g.size()
    res["median_notional"] = g["notional"].median()
    return res


def realized_vol(tob: pd.DataFrame, bar: str = "1s") -> float:
    """Annualised realised vol of mid log-returns sampled on `bar` bars (crypto: 365*24*3600 s/yr)."""
    s = tob.set_index(pd.to_datetime(tob["E"], unit="ms"))["mid"].resample(bar).last().ffill()
    r = np.log(s).diff().dropna()
    bars_per_year = 365 * 24 * 3600 / pd.Timedelta(bar).total_seconds()
    return float(r.std() * np.sqrt(bars_per_year))
