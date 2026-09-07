"""
Graphical identity for the semester app.

The design has one idea behind it: this dataset is mostly *absence*, so the
palette has to make absence readable rather than treating it as background.

  - The ground is deep and desaturated. An empty cell is the default state of
    the page, not an error state.
  - Two semantic colors, reserved and never reused for anything else:

        MARK  (MID_PRICE)  cool aqua   -- ubiquitous, calm, model-ish
        PRINT (TRDPRC_1)   warm amber  -- rare, hot, someone actually traded

    Temperature carries the meaning. A warm mark means a human crossed the
    spread. A cool mark means a quote midpoint was recorded and nothing more.
  - BOTH (a series carrying a mid *and* a print) is violet -- visibly a blend
    of the two parents, which is exactly what it is.
  - Interpolation reuses the violet at low opacity, because a sheet drawn over
    holes is an assumption we imposed, not an observation we made.
  - The underlying OHLC chart uses a blue/red pair that appears nowhere else,
    so it reads as context and never competes with the option fields.

Every color here was checked with the dataviz validator against the #141A2E
plot surface rather than chosen by eye. The MARK/PRINT/BOTH trio passes the
lightness band, chroma floor, CVD separation, normal-vision floor and 3:1
contrast gates under `--pairs all` (worst pair dE 8.4 CVD / 19.8 normal); the
underlying pair passes the same gates independently (19.2 / 29.0).

Color is doubled by shape everywhere it carries meaning -- MARK draws as a
circle, PRINT as a diamond -- so the encoding survives color-vision deficiency,
grayscale printing and forced-colors mode.
"""

from __future__ import annotations

# --- ground ---------------------------------------------------------------
BASE = "#0B1020"       # page background
PANEL = "#141A2E"      # card / plot surface (all contrast checks ran against this)
PANEL_HI = "#1B2340"   # hover, raised surfaces
LINE = "#26304A"       # borders, axis lines
LINE_SOFT = "#1D2438"  # gridlines

# --- type -----------------------------------------------------------------
TEXT = "#E8ECF5"
TEXT_MUTED = "#8792AB"
TEXT_FAINT = "#5A6580"

# --- semantic trio (reserved, validated as a set) -------------------------
MARK = "#199e70"        # MID_PRICE  -- the mark
PRINT = "#c98500"       # TRDPRC_1   -- evidence of a trade
BOTH = "#9085e9"        # carries both

# Single-hue ramps for occupancy heatmaps: surface -> the field's own color.
MARK_SCALE = [[0.0, PANEL], [1.0, MARK]]
PRINT_SCALE = [[0.0, PANEL], [1.0, PRINT]]

# --- the assumption -------------------------------------------------------
INTERP = BOTH
INTERP_SCALE = [[0.0, "#1A1B3A"], [0.5, "#4A45A0"], [1.0, BOTH]]
INTERP_OPACITY = 0.22

# --- context (underlying only, never beside the trio) ---------------------
UP = "#3987e5"
DOWN = "#e66767"

WARN = "#e66767"

FONT = "Inter, ui-sans-serif, system-ui, -apple-system, sans-serif"
FONT_MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# Marker shapes double the color encoding.
MARK_SYMBOL_3D = "circle"
PRINT_SYMBOL_3D = "diamond"


def layout(**overrides) -> dict:
    """Base Plotly layout. Every figure in the app starts here."""
    base = dict(
        paper_bgcolor=BASE,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family=FONT, size=12.5),
        title=dict(
            font=dict(size=15, color=TEXT, family=FONT),
            x=0.0,
            xanchor="left",
            y=0.97,
            yanchor="top",
        ),
        margin=dict(l=56, r=24, t=64, b=48),
        hoverlabel=dict(
            bgcolor=PANEL_HI,
            bordercolor=LINE,
            font=dict(family=FONT_MONO, size=11, color=TEXT),
        ),
        legend=dict(
            bgcolor="rgba(11,16,32,0.78)",
            bordercolor=LINE,
            borderwidth=1,
            font=dict(size=11, color=TEXT),
            itemsizing="constant",
        ),
    )
    base.update(overrides)
    return base


def axis(title: str | None = None, **overrides) -> dict:
    a = dict(
        gridcolor=LINE_SOFT,
        zeroline=False,
        linecolor=LINE,
        tickfont=dict(size=11, color=TEXT_MUTED, family=FONT),
    )
    if title:
        a["title"] = dict(text=title, font=dict(size=11, color=TEXT_MUTED, family=FONT))
    a.update(overrides)
    return a


def scene_axis(title: str, **overrides) -> dict:
    a = dict(
        title=dict(text=title, font=dict(size=11, color=TEXT_MUTED, family=FONT)),
        backgroundcolor=PANEL,
        gridcolor=LINE,
        showbackground=True,
        zeroline=False,
        tickfont=dict(size=11, color=TEXT_MUTED, family=FONT),
    )
    a.update(overrides)
    return a


def caption(text: str) -> dict:
    """Small explanatory line pinned just above a plot."""
    return dict(
        text=text,
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.0,
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        font=dict(size=11, color=TEXT_FAINT, family=FONT),
    )
