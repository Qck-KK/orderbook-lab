import numpy as np
import pandas as pd
import pytest

from orderbook_lab import metrics as M
from orderbook_lab.recorder import parse_message, ChunkWriter
from orderbook_lab.report import build


def make_book(times, mids, spread=1.0, qty=1.0, n=20, tick=0.5):
    rows = []
    for t, m in zip(times, mids):
        r = {"E": t, "u": t}
        for i in range(n):
            r[f"bid_px_{i}"] = m - spread / 2 - i * tick
            r[f"ask_px_{i}"] = m + spread / 2 + i * tick
            r[f"bid_qty_{i}"] = qty
            r[f"ask_qty_{i}"] = qty
        rows.append(r)
    return pd.DataFrame(rows)


def test_top_of_book_basic():
    b = make_book([0], [100.0], spread=1.0)
    tob = M.top_of_book(b)
    assert tob["mid"].iloc[0] == 100.0
    assert tob["spread_bps"].iloc[0] == pytest.approx(100.0)  # 1/100 * 1e4


def test_microprice_leans_to_thin_side():
    b = make_book([0], [100.0], spread=1.0)
    b["bid_qty_0"], b["ask_qty_0"] = 9.0, 1.0  # heavy bid, thin ask -> price likely up
    mp = M.top_of_book(b)["microprice"].iloc[0]
    assert 100.0 < mp < 100.5


def test_depth_within_counts_only_inside_band():
    b = make_book([0], [100.0], spread=0.2, tick=0.1)  # levels at 99.9, 99.8, ...
    d = M.depth_within(b, bps=25)  # band 99.75 .. 100.25 -> levels 0,1 on each side
    assert d.iloc[0, 1] == 2 and d.iloc[0, 2] == 2


def test_imbalance_range():
    b = make_book([0], [100.0])
    b["bid_qty_0"] = 4.0
    v = M.imbalance(b, 1).iloc[0]
    assert v == pytest.approx(0.6)


def test_markout_sign_and_no_lookahead():
    # mid steps up from 100 to 101 at t=1000ms
    times = np.arange(0, 40000, 100)
    mids = np.where(times < 1000, 100.0, 101.0)
    tob = M.top_of_book(make_book(times, mids))
    trades = pd.DataFrame({"E": [1000, 1000, 39500], "price": [100.5] * 3, "qty": [1.0] * 3, "side": [1, -1, 1]})
    mo = M.markouts(trades, tob, horizons_s=(1,))
    # pre-trade mid must be 100 (strictly before t=1000), not the new 101
    assert mo["mid0"].iloc[0] == 100.0
    assert mo["mo_1s_bps"].iloc[0] == pytest.approx(100.0)
    assert mo["mo_1s_bps"].iloc[1] == pytest.approx(-100.0)
    # horizon past end of data -> NaN, not stale
    assert np.isnan(mo["mo_1s_bps"].iloc[2])


def test_realized_vol_zero_for_flat_mid():
    times = np.arange(0, 10000, 100)
    tob = M.top_of_book(make_book(times, np.full(len(times), 100.0)))
    assert M.realized_vol(tob) == 0.0


def test_parse_messages():
    t = parse_message({"stream": "btcusdt@aggTrade", "data": {"E": 1, "T": 1, "a": 7, "p": "100.5", "q": "0.2", "m": True}}, 5)
    assert t[0] == "trades" and t[1]["side"] == -1 and t[1]["price"] == 100.5
    d = parse_message({"stream": "btcusdt@depth20@100ms", "data": {"E": 2, "T": 2, "u": 9,
                       "b": [["100", "1"]], "a": [["101", "2"]]}}, 5)
    assert d[0] == "book" and d[1]["ask_qty_0"] == 2.0 and np.isnan(d[1]["bid_px_1"])
    assert parse_message({"stream": "x@unknown", "data": {}}, 0) is None


def test_end_to_end_report(tmp_path):
    rng = np.random.default_rng(0)
    times = np.arange(0, 120_000, 100)
    mids = 100 + np.cumsum(rng.normal(0, 0.01, len(times)))
    w = ChunkWriter(tmp_path, "btcusdt")
    for _, r in make_book(times, mids).iterrows():
        w.add("book", r.to_dict())
    for t in range(500, 110_000, 250):
        w.add("trades", {"E": t, "price": 100.0, "qty": float(rng.uniform(0.01, 2)), "side": int(rng.choice([-1, 1]))})
    w.flush()
    text = build(tmp_path)
    assert "Markouts" in text and "spread:" in text


def test_stream_urls_are_routed():
    from orderbook_lab.recorder import stream_urls
    pub, mkt = stream_urls("BTCUSDT")
    assert "/public/stream" in pub and "depth20@100ms" in pub and "bookTicker" in pub
    assert "/market/stream" in mkt and "btcusdt@aggTrade" in mkt


def test_record_with_fake_websockets(tmp_path, monkeypatch):
    import asyncio, json, sys, types
    from orderbook_lab import recorder

    class FakeWS:
        def __init__(self, url): self.url, self.n = url, 0
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def recv(self):
            await asyncio.sleep(0.01); self.n += 1
            if "market" in self.url:
                return json.dumps({"stream": "btcusdt@aggTrade", "data": {"E": self.n, "T": self.n, "a": self.n, "p": "1", "q": "1", "m": False}})
            return json.dumps({"stream": "btcusdt@depth20@100ms", "data": {"E": self.n, "u": self.n, "b": [["1", "1"]], "a": [["2", "1"]]}})

    monkeypatch.setitem(sys.modules, "websockets", types.SimpleNamespace(connect=lambda url, **k: FakeWS(url)))
    asyncio.run(recorder.record("btcusdt", tmp_path, minutes=0.01, flush_seconds=0.2))
    assert list((tmp_path / "trades").glob("*.parquet")) and list((tmp_path / "book").glob("*.parquet"))


def test_taker_events_collapse_sweeps():
    tr = pd.DataFrame({"E": [5, 6, 6, 9], "T": [5, 5, 5, 8], "price": [100.0, 100.1, 100.2, 99.0],
                       "qty": [1.0, 1.0, 2.0, 1.0], "side": [1, 1, 1, -1]})
    ev = M.taker_events(tr)
    assert len(ev) == 2
    sweep = ev.iloc[0]
    assert sweep["E"] == 5 and sweep["qty"] == 4.0 and sweep["n_fills"] == 3
    assert sweep["price"] == pytest.approx((100 + 100.1 + 200.4) / 4)


def test_window_and_compare(tmp_path, monkeypatch):
    import sys
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from orderbook_lab import compare
    base = int(datetime(2026, 11, 12, 15, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp() * 1000)
    rng = np.random.default_rng(0)
    times = base + np.arange(0, 3_600_000, 1000)
    mids = 77000 + np.cumsum(rng.normal(0, 1, len(times)))
    w = ChunkWriter(tmp_path, "btcusdt")
    for r in make_book(times, mids, spread=0.1, tick=0.1).to_dict("records"):
        w.add("book", r)
    for t, m in zip(times, mids):
        w.add("bbo", {"E": int(t), "bid_px": m - .05, "ask_px": m + .05, "bid_qty": 1., "ask_qty": 1.})
        w.add("trades", {"E": int(t) + 500, "T": int(t) + 500, "price": m, "qty": float(rng.uniform(.01, 1)), "side": int(rng.choice([-1, 1]))})
    w.flush()
    text = build(tmp_path, "15:30", "15:45")
    assert "15:30-15:44" in text
    out = tmp_path / "cmp.md"
    monkeypatch.setattr(sys, "argv", ["x", str(tmp_path), f"{tmp_path}@15:00-15:30", "--out", str(out)])
    compare.main()
    assert out.exists() and out.with_suffix(".csv").exists()
