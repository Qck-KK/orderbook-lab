"""Compare sessions side by side.

  python -m orderbook_lab.compare data_us_close data_us_open@14:00-15:30 data_us_open@15:30-16:05 --out compare.md

Each argument is FOLDER or FOLDER@HH:MM-HH:MM (Berlin time). Results are also saved as compare.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .report import load_window, stats


def parse(arg: str):
    if "@" in arg:
        folder, win = arg.split("@", 1)
        start, end = win.split("-")
        return arg, Path(folder), start, end
    return arg, Path(arg), None, None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sessions", nargs="+")
    p.add_argument("--out", type=Path, default=Path("compare.md"))
    a = p.parse_args()
    cols = {}
    for arg in a.sessions:
        name, folder, start, end = parse(arg)
        s = stats(*load_window(folder, start, end))
        cols[name] = {k: v for k, v in s.items() if not k.startswith("_")}
    df = pd.DataFrame(cols)
    df.to_csv(a.out.with_suffix(".csv"), encoding="utf-8")
    fmt = df.map(lambda v: f"{v:.3f}" if isinstance(v, float) else str(v))
    text = "# Session comparison\n\n" + fmt.to_markdown() + "\n"
    a.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
