"""
Graphical identity for the semester app: the light "Modern serif" theme.

The tokens are the ones in the student's style guide
(GitSite/style-guide-modern-serif.html, section 4). The CSS of every page and
these Python constants must agree, so that a chart sits in its card.

The design keeps one idea from the earlier dark theme: this dataset is mostly
*absence*, so the palette has to make absence readable rather than treating
it as background.

  - The ground is a cool off-white and every chart sits on a white card. An
    empty cell takes the card's color: it is the default state of the page,
    not an error state.
  - Two semantic chart colors, reserved for the option fields:

        MARK  (MID_PRICE)  slate blue  -- the main series, ubiquitous, model-ish
        PRINT (TRDPRC_1)   burnt orange -- the comparison, rare: someone traded

  - BOTH (a series carrying a mid *and* a print) and the interpolated sheet use
    the gray context color, because a sheet drawn over holes is an assumption
    we imposed, not an observation we made.
  - Series colors mark data only. Text, labels and legends wear the ink
    tokens; the UI (links, pressed buttons, highlights) uses ACCENT.
  - The underlying's candlesticks stay out of the trio altogether: hollow ink
    candles for an up session and filled ones for a down session, so they read
    as context and the fill, not a hue, carries the direction.

The MARK/PRINT/BOTH trio was checked with the dataviz validator on the white
plot surface: it passes the lightness band, CVD separation, normal-vision
floor and 3:1 contrast gates under `--pairs all` (worst pair dE 14.8 CVD /
16.3 normal). BOTH fails only the chroma floor, as intended: it is the gray
context color, not a third identity hue. Green and red fail the CVD gate
(dE 5.8), which is why UP and DOWN color text that carries its own +/- sign
and never a chart mark.

Color is doubled by shape everywhere it carries meaning -- MARK draws as a
circle, PRINT as a diamond -- so the encoding survives color-vision deficiency,
grayscale printing and forced-colors mode.
"""

from __future__ import annotations

# --- ground ---------------------------------------------------------------
BASE = "#F5F6F8"       # page background
PANEL = "#FFFFFF"      # card / plot surface (all contrast checks ran against this)
PANEL_HI = "#F0F2F5"   # table headers, code chips, pressed states
LINE = "#E1E5EB"       # borders, axis lines
LINE_SOFT = "#EBEEF2"  # gridlines

# --- type -----------------------------------------------------------------
TEXT = "#1C2430"       # body ink
HEADING = "#16263F"    # headings and big numbers
TEXT_MUTED = "#4E5868"
TEXT_FAINT = "#5F6878"

# --- UI accent (links, focus rings, pressed buttons; never a chart series) --
ACCENT = "#2F5D8A"
ACCENT_HOVER = "#244B70"
TINT = "#E8EEF5"
TINT_LINE = "#B9CBDD"

# --- chart series (validated as a set on PANEL) ----------------------------
SERIES_1 = "#2A5FA5"   # main
SERIES_2 = "#C26B1D"   # comparison (marks only; too light for text)
SERIES_3 = "#7F8898"   # context

# --- semantic trio (reserved) ---------------------------------------------
MARK = SERIES_1         # MID_PRICE  -- the mark
PRINT = SERIES_2        # TRDPRC_1   -- evidence of a trade
BOTH = SERIES_3         # carries both

# Single-hue ramps for occupancy heatmaps: light -> the field's own color.
# The ramps start one step off white so an empty cell still shows as a tile.
MARK_SCALE = [[0.0, PANEL_HI], [1.0, MARK]]
PRINT_SCALE = [[0.0, PANEL_HI], [1.0, PRINT]]

# --- the assumption -------------------------------------------------------
INTERP = BOTH
INTERP_SCALE = [[0.0, "#C9CED7"], [0.5, "#A4ABB7"], [1.0, BOTH]]
INTERP_OPACITY = 0.30

# --- gain / loss TEXT, always beside its +/- sign -------------------------
UP = "#1E7A46"
DOWN = "#B42318"

WARN = DOWN

# --- the underlying's candlesticks (context, never beside the trio) -------
CANDLE_LINE = TEXT_MUTED   # both directions are outlined in ink
CANDLE_UP_FILL = PANEL     # an up session is hollow
CANDLE_DOWN_FILL = TEXT_MUTED  # a down session is filled

SERIF = "'Source Serif 4', Georgia, serif"
SHADOW = "0 1px 2px rgba(22,38,63,.06), 0 4px 14px rgba(22,38,63,.05)"   # cards float, no borders
FONT = "Inter, ui-sans-serif, system-ui, -apple-system, sans-serif"
FONT_MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# Marker shapes double the color encoding.
MARK_SYMBOL_3D = "circle"
PRINT_SYMBOL_3D = "diamond"


def layout(**overrides) -> dict:
    """Base Plotly layout. Every figure in the app starts here."""
    base = dict(
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT_MUTED, family=FONT, size=12.5),
        title=dict(
            font=dict(size=15, color=HEADING, family=FONT),
            x=0.0,
            xanchor="left",
            y=0.97,
            yanchor="top",
        ),
        margin=dict(l=56, r=24, t=64, b=48),
        hoverlabel=dict(
            bgcolor=PANEL,
            bordercolor=LINE,
            font=dict(family=FONT, size=12, color=TEXT),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
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
