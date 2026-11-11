"""Record Binance USD-M futures market data to chunked parquet files.

Streams (combined endpoint):
  <sym>@depth20@100ms  partial book, top 20 levels, every 100 ms
  <sym>@aggTrade       aggregated trades (m = buyer is maker -> seller aggressor)
  <sym>@bookTicker     best bid/ask updates

Usage:
  python -m orderbook_lab.recorder --symbol btcusdt --out data/ --minutes 60

Notes:
  * Binance closes a connection after 24 h; we reconnect with backoff on any drop.
  * Rows are buffered and flushed every `--flush-seconds` to
    data/<stream>/<symbol>_<UTC timestamp>.parquet, so a crash loses at most one chunk.
  * Timestamps: `E` (event time, exchange ms) and `recv_ns` (local receive time)
    are both stored, so latency can be checked later.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger("recorder")

PUBLIC_URL = "wss://fstream.binance.com/public/stream?streams="
MARKET_URL = "wss://fstream.binance.com/market/stream?streams="
N_LEVELS = 20


def stream_urls(symbol: str) -> list[str]:
    """Since the 2026-03 routing upgrade, USD-M streams live on separate paths:
    /public for book data (depth, bookTicker), /market for trades (aggTrade).
    An unrouted connection silently gets no aggTrade data."""
    s = symbol.lower()
    return [
        PUBLIC_URL + f"{s}@depth20@100ms/{s}@bookTicker",
        MARKET_URL + f"{s}@aggTrade",
    ]


def parse_message(msg: dict, recv_ns: int) -> tuple[str, dict] | None:
    """Turn one combined-stream message into (kind, flat row). Pure function, unit-tested."""
    stream = msg.get("stream", "")
    d = msg.get("data", {})
    if stream.endswith("@aggTrade"):
        return "trades", {
            "E": d["E"], "T": d["T"], "recv_ns": recv_ns, "agg_id": d["a"],
            "price": float(d["p"]), "qty": float(d["q"]),
            # m=True: buyer is maker -> aggressor sold
            "side": -1 if d["m"] else 1,
        }
    if stream.endswith("@bookTicker"):
        return "bbo", {
            "E": d.get("E"), "T": d.get("T"), "recv_ns": recv_ns, "u": d["u"],
            "bid_px": float(d["b"]), "bid_qty": float(d["B"]),
            "ask_px": float(d["a"]), "ask_qty": float(d["A"]),
        }
    if "@depth" in stream:
        row = {"E": d["E"], "T": d.get("T"), "recv_ns": recv_ns, "u": d["u"]}
        for side, key in (("bid", "b"), ("ask", "a")):
            levels = d.get(key, [])[:N_LEVELS]
            for i in range(N_LEVELS):
                px, qty = (float(levels[i][0]), float(levels[i][1])) if i < len(levels) else (float("nan"), 0.0)
                row[f"{side}_px_{i}"] = px
                row[f"{side}_qty_{i}"] = qty
        return "book", row
    return None


class ChunkWriter:
    def __init__(self, out: Path, symbol: str):
        self.out, self.symbol = out, symbol.lower()
        self.buf: dict[str, list[dict]] = defaultdict(list)

    def add(self, kind: str, row: dict) -> None:
        self.buf[kind].append(row)

    def flush(self) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        buf, self.buf = self.buf, defaultdict(list)
        counts = []
        for kind, rows in buf.items():
            if not rows:
                continue
            d = self.out / kind
            d.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(d / f"{self.symbol}_{stamp}.parquet", index=False)
            counts.append(f"{len(rows)} {kind}")
        log.info("flushed %s", ", ".join(counts) if counts else "nothing (no data received!)")
        if "trades" not in buf:
            log.warning("no trades in this chunk - check the /market connection")


async def _consume(url: str, writer: "ChunkWriter", deadline: float) -> None:
    """Read one websocket until the deadline, reconnecting with backoff on any drop."""
    import websockets  # imported here so metrics/tests don't need it

    name = url.split("/")[3]  # 'public' or 'market'
    backoff = 1.0
    while time.monotonic() < deadline:
        try:
            async with websockets.connect(url, ping_interval=20, max_size=2**22) as ws:
                log.info("connected: %s", name)
                backoff = 1.0
                while time.monotonic() < deadline:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    parsed = parse_message(json.loads(raw), time.time_ns())
                    if parsed:
                        writer.add(*parsed)
        except Exception as e:  # noqa: BLE001 - keep recording
            log.warning("%s dropped (%s); reconnecting in %.0fs", name, e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


async def _flusher(writer: "ChunkWriter", deadline: float, flush_seconds: float) -> None:
    while time.monotonic() < deadline:
        await asyncio.sleep(min(flush_seconds, max(0.0, deadline - time.monotonic())))
        writer.flush()


async def record(symbol: str, out: Path, minutes: float, flush_seconds: float) -> None:
    writer = ChunkWriter(out, symbol)
    deadline = time.monotonic() + minutes * 60
    try:
        await asyncio.gather(
            *(_consume(u, writer, deadline) for u in stream_urls(symbol)),
            _flusher(writer, deadline, flush_seconds),
        )
    finally:
        writer.flush()  # also runs on Ctrl+C


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="btcusdt")
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--minutes", type=float, default=60)
    p.add_argument("--flush-seconds", type=float, default=60)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(record(a.symbol, a.out, a.minutes, a.flush_seconds))


if __name__ == "__main__":
    main()
