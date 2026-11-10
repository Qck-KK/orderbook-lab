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

BASE_URL = "wss://fstream.binance.com/stream?streams="
N_LEVELS = 20


def stream_url(symbol: str) -> str:
    s = symbol.lower()
    return BASE_URL + "/".join([f"{s}@depth20@100ms", f"{s}@aggTrade", f"{s}@bookTicker"])


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
        for kind, rows in self.buf.items():
            if not rows:
                continue
            d = self.out / kind
            d.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(d / f"{self.symbol}_{stamp}.parquet", index=False)
            log.info("flushed %d %s rows", len(rows), kind)
        self.buf.clear()


async def record(symbol: str, out: Path, minutes: float, flush_seconds: float) -> None:
    import websockets  # imported here so metrics/tests don't need it

    writer = ChunkWriter(out, symbol)
    deadline = time.monotonic() + minutes * 60
    last_flush = time.monotonic()
    backoff = 1.0
    while time.monotonic() < deadline:
        try:
            async with websockets.connect(stream_url(symbol), ping_interval=20, max_size=2**22) as ws:
                log.info("connected")
                backoff = 1.0
                while time.monotonic() < deadline:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    parsed = parse_message(json.loads(raw), time.time_ns())
                    if parsed:
                        writer.add(*parsed)
                    if time.monotonic() - last_flush >= flush_seconds:
                        writer.flush()
                        last_flush = time.monotonic()
        except (asyncio.TimeoutError, OSError, Exception) as e:  # noqa: BLE001 - keep recording
            log.warning("connection dropped (%s); reconnecting in %.0fs", e, backoff)
            writer.flush()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
    writer.flush()


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
