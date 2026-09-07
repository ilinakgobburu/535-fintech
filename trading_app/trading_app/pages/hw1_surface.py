"""
Assignment 1.1 — Option Surface Lab, as a Reflex page.

This is the interactive twin of the published static page. Both are driven by
the same `lib` modules, so a number shown here and a number shown on GitHub
Pages come from one implementation and cannot drift apart.

Later homeworks add sibling modules in this package.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import reflex as rx

from .. import theme as T
from ..lib.loaders import MARK_FIELD, PRINT_FIELD, build_frames, load_payload
from ..lib.metrics import (
    fmt_money, fmt_pct, interpolation_holdout, slice_asof, sparsity_stats,
    spread_stats,
)
from ..lib import plots

CACHE = Path(__file__).resolve().parents[1] / "data" / "option_pipeline_data.pkl"


def _bootstrap() -> dict:
    """
    Load the panel once, at import, and precompute the opening view.

    The cache is static and read-only, so there is no reason to make the first
    render wait on an event round-trip. Baking the opening slice into the
    initial state means the page is correct the moment it paints, and the
    websocket is only needed for interaction.
    """
    frames = build_frames(load_payload(CACHE))
    wide = frames["wide"]

    counts = wide.groupby("date")["ric"].nunique()
    dates = [d for d, n in counts.items() if n >= 12] or list(counts.index)
    rights = sorted(wide["cp"].unique().tolist())
    cp = rights[0]

    best, chosen = -1, str(dates[-1].date())
    for d in dates:
        sl = slice_asof(wide, d, cp)
        score = len(sl) * sl["expiry"].nunique()
        if score > best:
            best, chosen = score, str(d.date())

    return {
        "frames": frames,
        "wide": wide,
        "dates": [str(d.date()) for d in dates],
        "rights": rights,
        "cp": cp,
        "asof": chosen,
        "stock_fig": plots.underlying_figure(frames["stock"], frames["underlying"]),
        "provenance": (
            f"{'SYNTHETIC' if frames['synthetic'] else 'LSEG cache'} · "
            f"pulled {frames['fetched_at']} · "
            f"fields {' + '.join(frames['fields_present'])}"
        ),
    }


_BOOT = _bootstrap()


def _slice_views(asof: str, cp: str, show_mark: bool, show_print: bool,
                 show_sheet: bool) -> dict:
    """Everything one (date, right) slice puts on screen."""
    wide = _BOOT["wide"]
    ticker = _BOOT["frames"]["underlying"]
    sl = slice_asof(wide, pd.Timestamp(asof), cp)

    stats = sparsity_stats(sl)
    sp = spread_stats(sl)
    h = interpolation_holdout(sl, MARK_FIELD)

    return {
        "n_series": stats["n_series"],
        "n_both": stats["n_both"],
        "pct": fmt_pct(stats["pct_mark_no_trade"]),
        "gap": fmt_money(stats["median_abs_diff"]),
        "spread": ("n/a" if sp["median_spread_pct"] is None
                   else f"{sp['median_spread_pct']:.1f}%"),
        "holdout": "n/a" if h is None else fmt_money(h["median_abs_err"]),
        "surface": plots.surface_figure(
            sl, asof, cp=cp, show_mark=show_mark, show_print=show_print,
            show_sheet=show_sheet, ticker=ticker),
        "compare": plots.mark_vs_print_figure(sl, ticker),
        "occ_mark": plots.occupancy_figure(sl, MARK_FIELD, cp),
        "occ_print": plots.occupancy_figure(sl, PRINT_FIELD, cp),
    }


_OPEN = _slice_views(_BOOT["asof"], _BOOT["cp"], True, True, True)


class SurfaceState(rx.State):
    """One as-of slice of the panel, plus the figures drawn from it."""

    status: str = f"{_OPEN['n_series']} series on {_BOOT['asof']}"
    provenance: str = _BOOT["provenance"]
    underlying: str = _BOOT["frames"]["underlying"]

    # controls
    asof: str = _BOOT["asof"]
    asof_options: list[str] = _BOOT["dates"]
    cp: str = _BOOT["cp"]
    cp_options: list[str] = _BOOT["rights"]
    show_mark: bool = True
    show_print: bool = True
    show_sheet: bool = True

    # the two required numbers, plus context
    pct_mark_no_trade: str = _OPEN["pct"]
    median_gap: str = _OPEN["gap"]
    n_series: int = _OPEN["n_series"]
    n_both: int = _OPEN["n_both"]
    median_spread_pct: str = _OPEN["spread"]
    holdout_err: str = _OPEN["holdout"]

    fig_surface: go.Figure = _OPEN["surface"]
    fig_compare: go.Figure = _OPEN["compare"]
    fig_occ_mark: go.Figure = _OPEN["occ_mark"]
    fig_occ_print: go.Figure = _OPEN["occ_print"]
    fig_stock: go.Figure = _BOOT["stock_fig"]

    def set_asof(self, value: str):
        self.asof = value
        self._rebuild()

    def set_cp(self, value: str):
        self.cp = value
        self._rebuild()

    def toggle_mark(self, value: bool):
        self.show_mark = value
        self._rebuild()

    def toggle_print(self, value: bool):
        self.show_print = value
        self._rebuild()

    def toggle_sheet(self, value: bool):
        self.show_sheet = value
        self._rebuild()

    def _rebuild(self):
        if not self.asof:
            return
        v = _slice_views(self.asof, self.cp, self.show_mark,
                         self.show_print, self.show_sheet)
        self.n_series = v["n_series"]
        self.n_both = v["n_both"]
        self.pct_mark_no_trade = v["pct"]
        self.median_gap = v["gap"]
        self.median_spread_pct = v["spread"]
        self.holdout_err = v["holdout"]
        self.fig_surface = v["surface"]
        self.fig_compare = v["compare"]
        self.fig_occ_mark = v["occ_mark"]
        self.fig_occ_print = v["occ_print"]
        self.status = f"{self.n_series} series on {self.asof}"


def _tile(label: str, value, accent: str = T.MARK, sub: str = "") -> rx.Component:
    body = [
        rx.text(label, size="1", color=T.TEXT_MUTED),
        rx.text(value, size="7", color=accent, weight="bold",
                font_family=T.FONT_MONO),
    ]
    if sub:
        body.append(rx.text(sub, size="1", color=T.TEXT_FAINT))
    return rx.box(
        *body,
        bg=T.PANEL, border=f"1px solid {T.LINE}", border_radius="10px",
        padding="16px 18px", flex="1", min_width="190px",
    )


def _plot(fig, height: str) -> rx.Component:
    return rx.box(
        rx.plotly(data=fig, style={"width": "100%", "height": height}),
        width="100%", bg=T.PANEL, border=f"1px solid {T.LINE}",
        border_radius="10px", padding="8px",
    )


def _toggle(label: str, checked, handler, color: str) -> rx.Component:
    return rx.hstack(
        rx.switch(checked=checked, on_change=handler),
        rx.text(label, size="2", color=color),
        spacing="2", align="center",
    )


def surface_page() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.vstack(
                rx.text("MEng FinTech · Algorithmic Trading II · Assignment 1.1",
                        size="1", color=T.TEXT_FAINT, letter_spacing="0.16em"),
                rx.heading(
                    f"Option Surface Lab", size="8", color=T.TEXT,
                    letter_spacing="-0.02em",
                ),
                rx.text(
                    "Listed options are a sparse cloud, not a filled sheet. "
                    "MID_PRICE is the mark; TRDPRC_1 is evidence somebody traded.",
                    size="3", color=T.TEXT_MUTED,
                ),
                rx.text(SurfaceState.provenance, size="1", color=T.TEXT_FAINT,
                        font_family=T.FONT_MONO),
                align="start", spacing="1", width="100%",
            ),
            rx.divider(border_color=T.LINE),

            rx.hstack(
                _tile("Mark, no trade", SurfaceState.pct_mark_no_trade, T.MARK),
                _tile(f"Median |{MARK_FIELD} − {PRINT_FIELD}|",
                      SurfaceState.median_gap, T.BOTH),
                _tile("Median spread", SurfaceState.median_spread_pct, T.PRINT,
                      "of the mark"),
                _tile("Interpolation miss", SurfaceState.holdout_err, T.BOTH,
                      "on cells we already know"),
                spacing="3", width="100%", wrap="wrap",
            ),

            rx.hstack(
                rx.text("As-of", size="2", color=T.TEXT_MUTED),
                rx.select(SurfaceState.asof_options, value=SurfaceState.asof,
                          on_change=SurfaceState.set_asof, size="2"),
                rx.text("Right", size="2", color=T.TEXT_MUTED),
                rx.select(SurfaceState.cp_options, value=SurfaceState.cp,
                          on_change=SurfaceState.set_cp, size="2"),
                rx.spacer(),
                _toggle(MARK_FIELD, SurfaceState.show_mark,
                        SurfaceState.toggle_mark, T.MARK),
                _toggle(PRINT_FIELD, SurfaceState.show_print,
                        SurfaceState.toggle_print, T.PRINT),
                _toggle("Interpolated sheet", SurfaceState.show_sheet,
                        SurfaceState.toggle_sheet, T.BOTH),
                width="100%", align="center", wrap="wrap", spacing="3",
                bg=T.PANEL, border=f"1px solid {T.LINE}", border_radius="10px",
                padding="14px 18px",
            ),

            _plot(SurfaceState.fig_surface, "640px"),
            _plot(SurfaceState.fig_compare, "450px"),
            rx.hstack(
                _plot(SurfaceState.fig_occ_mark, "360px"),
                _plot(SurfaceState.fig_occ_print, "360px"),
                width="100%", spacing="3",
            ),
            _plot(SurfaceState.fig_stock, "340px"),

            rx.text(
                "The published static build at docs/index.html carries the full "
                "written analysis and the bid-ask panels.",
                size="1", color=T.TEXT_FAINT,
            ),
            spacing="4", width="100%",
        ),
        bg=T.BASE, min_height="100vh", padding="32px 24px 64px",
    )
