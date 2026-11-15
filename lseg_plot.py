#!/usr/bin/env python3
"""
Query LSEG and save a Plotly chart for the course site.

The cycle, run by .github/workflows/lseg-plot.yml when "Run workflow" is clicked:
  1. log in to LSEG with a platform session (app key + username + password),
     so LSEG Workspace does not have to be open anywhere;
  2. ask for about six months of daily bars for one instrument (a RIC);
  3. save the chart as docs/lseg-plot.html, a self-contained page element;
  4. the workflow commits that file and GitHub Pages republishes the site,
     where docs/lseg.html shows it.

No credentials are written in this file. lseg_login.py reads them from the
environment variables LSEG_APP_KEY, LSEG_USERNAME and LSEG_PASSWORD (GitHub
Actions secrets) or, on a Mac, from the macOS Keychain.

Run it locally:
    python3 lseg_plot.py                        # charts NVDA.O
    LSEG_PLOT_RIC=AMD.O python3 lseg_plot.py    # any other RIC
"""

import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import lseg.data as ld
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lseg_login import open_session  # noqa: E402

OUT = "docs/lseg-plot.html"
DEFAULT_RIC = "NVDA.O"
DAYS_BACK = 183   # calendar days requested, about six months of sessions
MA_WINDOW = 20    # sessions in the moving average
FIELDS = ["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1", "ACVOL_UNS"]  # open, high, low, close, volume

# Colors from the site's "Modern serif" palette (trading_app/trading_app/theme.py),
# so the chart sits in the page.
BG, GRID, AXIS, TEXT, HEADING, MUTED = "#FFFFFF", "#EBEEF2", "#E1E5EB", "#1C2430", "#16263F", "#4E5868"
FONT = "Inter, ui-sans-serif, system-ui, -apple-system, sans-serif"
# Candles are context in ink: an up session is hollow and a down session filled,
# so the fill, not a red/green pair, carries the direction.
CANDLE_LINE, CANDLE_UP_FILL, CANDLE_DOWN_FILL = MUTED, BG, MUTED
AVERAGE, VOLUME = "#2A5FA5", "#7F8898"    # the main series, and the gray context


def fetch(ric: str) -> pd.DataFrame:
    """About six months of daily bars, ending yesterday (New York time)."""
    # LSEG's bar for the current day is unreliable (a repeated date with volume 0),
    # so the request ends at yesterday.
    end = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)
    start = end - timedelta(days=DAYS_BACK)
    print(f"Requesting {ric} daily bars, {start} to {end}")

    try:
        df = ld.get_history(universe=ric, fields=FIELDS, interval="1D",
                            start=start.isoformat(), end=end.isoformat())
    except Exception as exc:   # e.g. an unknown RIC: the library raises instead of returning nothing
        sys.exit(f"LSEG rejected the request for {ric}: {exc}")
    if df is None or df.empty:
        sys.exit(f"LSEG returned no rows for {ric}. Is the RIC right, and is it licensed?")

    # Only the fields LSEG has for this instrument come back as columns.
    missing = [f for f in FIELDS[:4] if f not in df.columns]   # open, high, low, close are required
    if missing:
        sys.exit(f"{ric} has no {', '.join(missing)} in LSEG's daily history "
                 f"(it returned {', '.join(map(str, df.columns))}); try a stock RIC such as NVDA.O.")
    if "ACVOL_UNS" not in df.columns:   # volume is optional (FX and some indices have none)
        df["ACVOL_UNS"] = float("nan")

    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="first")].dropna(subset=["TRDPRC_1"]).sort_index()
    if df.empty:
        sys.exit(f"LSEG returned rows for {ric} but no closing prices.")

    first, last = df.index[0].date(), df.index[-1].date()
    print(f"Received {len(df)} sessions, {first} to {last}")
    # An oversized request is cut short without an error, so check the span.
    if first > start + timedelta(days=10) or last < end - timedelta(days=7):
        print(f"WARNING: asked for {start} to {end} but got {first} to {last}")
    return df


def chart(df: pd.DataFrame, ric: str) -> go.Figure:
    close = df["TRDPRC_1"]
    pulled = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    span = f"{df.index[0]:%b %d, %Y} to {df.index[-1]:%b %d, %Y}"

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.76, 0.24], vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["OPEN_PRC"], high=df["HIGH_1"], low=df["LOW_1"], close=close,
        name=ric, increasing_line_color=CANDLE_LINE, increasing_fillcolor=CANDLE_UP_FILL,
        decreasing_line_color=CANDLE_LINE, decreasing_fillcolor=CANDLE_DOWN_FILL,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=close.rolling(MA_WINDOW).mean(), name=f"{MA_WINDOW}-day average",
        line=dict(color=AVERAGE, width=2), hovertemplate="%{y:.2f}<extra>20-day avg</extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=df.index, y=df["ACVOL_UNS"], name="Volume", marker_color=VOLUME, marker_opacity=0.55,
        hovertemplate="%{y:,.0f}<extra>volume</extra>",
    ), row=2, col=1)

    fig.update_layout(
        title=dict(
            text=(f"{ric} · daily bars from LSEG"
                  f"<br><span style='font-size:12px;color:{MUTED}'>"
                  f"{span} · {len(df)} sessions<br>pulled {pulled}</span>"),
            x=0.01, xanchor="left", y=0.97, yanchor="top",
            font=dict(color=HEADING, size=16),
        ),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=FONT, color=MUTED, size=13),
        hoverlabel=dict(bgcolor=BG, bordercolor=AXIS, font=dict(family=FONT, color=TEXT, size=12)),
        dragmode=False,   # on a phone the page scrolls instead of zooming
        # Legend under the chart so it never covers the title on a narrow screen.
        legend=dict(orientation="h", x=0, xanchor="left", y=-0.08, yanchor="top",
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified", showlegend=True,
        margin=dict(l=56, r=20, t=92, b=64), height=580,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, fixedrange=True, rangeslider_visible=False,
                     rangebreaks=[dict(bounds=["sat", "mon"])])   # hide weekends
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, fixedrange=True, zeroline=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def save(fig: go.Figure) -> None:
    html = fig.to_html(include_plotlyjs="cdn", full_html=True,
                       config={"displaylogo": False, "responsive": True})
    # Match the page background inside the frame, load the site's chart font and keep
    # the element out of search results.
    html = html.replace(
        "<head>",
        f'<head><meta name="robots" content="noindex">'
        f'<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">'
        f"<style>html,body{{margin:0;background:{BG}}}</style>",
        1,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {OUT}")


def main() -> None:
    ric = (os.environ.get("LSEG_PLOT_RIC") or DEFAULT_RIC).strip()
    if not re.fullmatch(r"[A-Za-z0-9._=^#/-]{1,32}", ric):
        sys.exit(f"Not a valid RIC: {ric!r}")

    session = open_session()
    try:
        df = fetch(ric)
    finally:
        session.close()
        print("LSEG session closed")
    save(chart(df, ric))


if __name__ == "__main__":
    main()
