"""Figures for the README.

  python -m orderbook_lab.plots data data_us_open@14:00-15:30 data_us_open@15:30-16:05 \
      --labels "US close" "EU afternoon" "US open" --open-window data_us_open@15:00-16:05

Writes figures/markouts_by_size.png and figures/around_us_open.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import metrics as M
from .compare import parse
from .report import TZ, load_window, stats

# reference palette, light mode; first three categorical slots validate all-pairs
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "D"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def markouts_by_size(sessions, labels, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    _style(ax)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ends = []
    for i, ((_, folder, start, end), lab) in enumerate(zip(sessions, labels)):
        q = stats(*load_window(folder, start, end))["_quartiles"]
        x = np.arange(1, len(q) + 1)
        y = q["mo_1s_bps"].to_numpy()
        ax.plot(x, y, color=SERIES[i], linewidth=2, marker=MARKERS[i], markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=lab)
        ends.append((y[-1], lab, x[-1]))
    # direct labels at the line ends, nudged apart so they never overlap
    lo, hi = ax.get_ylim()
    gap = 0.07 * (hi - lo)
    ends.sort()
    placed = []
    for y, lab, x in ends:
        yy = max(y, placed[-1] + gap) if placed else y
        placed.append(yy)
        ax.annotate(lab, (x, y), xytext=(x + 0.12, yy), textcoords="data",
                    va="center", fontsize=9, color=INK)
    ax.set_ylim(lo, max(hi, placed[-1] + gap / 2))
    ax.set_xticks([1, 2, 3, 4], ["Q1\nsmallest", "Q2", "Q3", "Q4\nlargest"])
    ax.set_xlim(0.7, 5.0)
    ax.set_xlabel("Taker order size quartile (notional, within session)", color=INK2, fontsize=9)
    ax.set_ylabel("1s markout, bps (signed by aggressor)", color=INK2, fontsize=9)
    ax.set_title("Bigger taker orders move price more; in busy sessions small ones do too",
                 loc="left", color=INK, fontsize=11, pad=26)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=len(labels), borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def around_open(window, out: Path):
    _, folder, start, end = window
    book, trades, _ = load_window(folder, start, end)
    tob = M.top_of_book(book)
    ev = M.taker_events(trades)
    to_t = lambda e: pd.DatetimeIndex(pd.to_datetime(e, unit="ms", utc=True)).tz_convert(TZ)
    events = ev.set_index(to_t(ev["E"])).resample("1min").size()
    s = tob.set_index(to_t(tob["E"]))
    vol = np.log(s["mid"].resample("1s").last().ffill()).diff()
    vol_min = vol.resample("1min").std() * np.sqrt(365 * 24 * 3600) * 100
    wide = (s["spread"] > 0.15).resample("1min").mean() * 100
    # drop partial minutes at either edge (they look like activity collapses)
    t_first, t_last = s.index[0], s.index[-1]
    full = lambda ser: ser[(ser.index >= t_first.ceil("1min")) & (ser.index + pd.Timedelta("1min") <= t_last)]
    events, vol_min, wide = full(events), full(vol_min), full(wide)

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.6), sharex=True, facecolor=SURFACE)
    panels = [(events, "Taker events per minute"),
              (vol_min, "Realised vol, % annualised (1s returns)"),
              (wide, "% of time spread wider than 1 tick")]
    open_t = events.index[0].normalize() + pd.Timedelta(hours=15, minutes=30)
    for ax, (ser, title) in zip(axes, panels):
        _style(ax)
        ax.plot(ser.index, ser.values, color=SERIES[0], linewidth=2)
        ax.axvline(open_t, color=INK2, linewidth=1, linestyle="--")
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.set_ylim(bottom=0)
    axes[0].annotate("US equity open 15:30", (open_t, 0), xytext=(6, 6), textcoords="offset points",
                     fontsize=9, color=INK2, va="bottom",
                     bbox=dict(boxstyle="square,pad=0.2", fc=SURFACE, ec="none"))
    import matplotlib.dates as mdates
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ))
    axes[-1].set_xlabel("Berlin time", color=INK2, fontsize=9)
    fig.suptitle("BTCUSDT perpetual around the US equity open", x=0.02, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sessions", nargs="+")
    p.add_argument("--labels", nargs="+")
    p.add_argument("--open-window", help="FOLDER@HH:MM-HH:MM covering the US open")
    p.add_argument("--outdir", type=Path, default=Path("figures"))
    a = p.parse_args()
    if len(a.sessions) > 3:
        p.error("at most 3 sessions per chart (keeps colours distinguishable)")
    sessions = [parse(s) for s in a.sessions]
    labels = a.labels or [s[0] for s in sessions]
    a.outdir.mkdir(exist_ok=True)
    markouts_by_size(sessions, labels, a.outdir / "markouts_by_size.png")
    print("wrote", a.outdir / "markouts_by_size.png")
    if a.open_window:
        around_open(parse(a.open_window), a.outdir / "around_us_open.png")
        print("wrote", a.outdir / "around_us_open.png")


if __name__ == "__main__":
    main()
