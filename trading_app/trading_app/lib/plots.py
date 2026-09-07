"""
Plotly figures for the surface lab.

Two rules hold across every figure here:

  1. MARK and PRINT keep their colors AND their shapes everywhere. Identity is
     never carried by color alone.
  2. Nothing invents a number. Where the data stops, the drawing stops -- the
     interpolated sheet is the one exception, and it is drawn as a translucent
     assumption with the honest cloud sitting on top of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .. import theme as T
from .loaders import MARK_FIELD, PRINT_FIELD
from .metrics import interpolate_grid, occupancy_matrix


def _empty(message: str, height: int = 420) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **T.layout(
            height=height,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[
                dict(
                    text=message,
                    xref="paper", yref="paper", x=0.5, y=0.5,
                    xanchor="center", yanchor="middle", showarrow=False,
                    font=dict(size=13, color=T.TEXT_MUTED, family=T.FONT),
                )
            ],
        )
    )
    return fig


def underlying_figure(df_stock: pd.DataFrame, ticker: str) -> go.Figure:
    """Context: the underlying. Deliberately outside the semantic palette."""
    if df_stock is None or df_stock.empty:
        return _empty("No underlying history in the cache.")

    fig = go.Figure(
        go.Candlestick(
            x=df_stock.index,
            open=df_stock["OPEN_PRC"],
            high=df_stock["HIGH_1"],
            low=df_stock["LOW_1"],
            close=df_stock["TRDPRC_1"],
            increasing=dict(line=dict(color=T.UP, width=1), fillcolor=T.UP),
            decreasing=dict(line=dict(color=T.DOWN, width=1), fillcolor=T.DOWN),
            name=ticker,
            showlegend=False,
        )
    )
    fig.update_layout(
        **T.layout(
            height=340,
            title=dict(text=f"{ticker} · underlying", x=0.0, xanchor="left"),
            xaxis={**T.axis(), "rangeslider": dict(visible=False)},
            yaxis=T.axis("Price ($)"),
            annotations=[
                T.caption(
                    "Context only. The close here is TRDPRC_1 on the stock — "
                    "a liquid print, unlike the options below."
                )
            ],
        )
    )
    return fig


def surface_figure(
    sl: pd.DataFrame,
    asof,
    cp: str = "C",
    show_mark: bool = True,
    show_print: bool = True,
    show_sheet: bool = True,
    ticker: str = "UUUU",
    max_fill_gap: float | None = 1.25,
) -> go.Figure:
    """
    Requirement 3 + 4: the 3D cloud for one as-of date, both fields together.

    X strike · Y days to expiry · Z option price.
    """
    if sl is None or sl.empty:
        return _empty("No series on this date for this right.", height=620)

    asof_txt = str(pd.Timestamp(asof).date()) if asof is not None else "all dates"
    cp_label = {"C": "Calls", "P": "Puts"}.get(cp, "Puts + calls")
    spot = sl["spot"].dropna()
    spot_val = float(spot.median()) if len(spot) else None

    fig = go.Figure()

    # The assumption goes down first so the observations sit on top of it.
    if show_sheet and show_mark:
        grid = interpolate_grid(sl, MARK_FIELD, max_fill_gap=max_fill_gap)
        if grid is not None:
            fig.add_trace(
                go.Surface(
                    x=grid["x"], y=grid["y"], z=grid["z"],
                    name="Interpolated sheet",
                    colorscale=T.INTERP_SCALE,
                    opacity=T.INTERP_OPACITY,
                    showscale=False,
                    hoverinfo="skip",
                    showlegend=True,
                    contours=dict(
                        x=dict(show=False), y=dict(show=False), z=dict(show=False)
                    ),
                )
            )

    if show_mark and sl[MARK_FIELD].notna().any():
        m = sl.dropna(subset=[MARK_FIELD])
        fig.add_trace(
            go.Scatter3d(
                x=m["strike"], y=m["dte"], z=m[MARK_FIELD],
                mode="markers",
                name=f"{MARK_FIELD} (mark)",
                marker=dict(
                    size=4, color=T.MARK, opacity=0.9,
                    symbol=T.MARK_SYMBOL_3D, line=dict(width=0),
                ),
                customdata=m[["ric", "cp"]].to_numpy(),
                hovertemplate=(
                    f"<b>{MARK_FIELD}</b> $%{{z:.3f}}<br>"
                    "K %{x:.2f} · DTE %{y}<br>%{customdata[0]}<extra></extra>"
                ),
            )
        )

    if show_print and sl[PRINT_FIELD].notna().any():
        p = sl.dropna(subset=[PRINT_FIELD])
        fig.add_trace(
            go.Scatter3d(
                x=p["strike"], y=p["dte"], z=p[PRINT_FIELD],
                mode="markers",
                name=f"{PRINT_FIELD} (traded)",
                marker=dict(
                    size=6, color=T.PRINT, opacity=1.0,
                    symbol=T.PRINT_SYMBOL_3D,
                    line=dict(width=0.5, color=T.BASE),
                ),
                customdata=p[["ric", "cp"]].to_numpy(),
                hovertemplate=(
                    f"<b>{PRINT_FIELD}</b> $%{{z:.3f}}<br>"
                    "K %{x:.2f} · DTE %{y}<br>%{customdata[0]}<extra></extra>"
                ),
            )
        )

    # Vertical plane at K = S, so "where is the money" is visible in 3D.
    if spot_val is not None and len(sl):
        z_hi = float(np.nanmax([sl[MARK_FIELD].max(), sl[PRINT_FIELD].max(), 0.01]))
        y_lo, y_hi = float(sl["dte"].min()), float(sl["dte"].max())
        if y_hi > y_lo:
            fig.add_trace(
                go.Surface(
                    x=[[spot_val, spot_val], [spot_val, spot_val]],
                    y=[[y_lo, y_hi], [y_lo, y_hi]],
                    z=[[0, 0], [z_hi, z_hi]],
                    name=f"K = S (${spot_val:.2f})",
                    colorscale=[[0, T.TEXT_FAINT], [1, T.TEXT_FAINT]],
                    opacity=0.13, showscale=False, hoverinfo="skip",
                    showlegend=True,
                )
            )

    fig.update_layout(
        **T.layout(
            height=640,
            margin=dict(l=8, r=8, t=64, b=8),
            title=dict(
                text=f"{ticker} {cp_label} · {asof_txt}"
                + (f"  ·  spot ${spot_val:.2f}" if spot_val else ""),
                x=0.0, xanchor="left",
            ),
            scene=dict(
                xaxis=T.scene_axis("Strike ($)"),
                # near-dated toward the viewer
                yaxis=T.scene_axis("Days to expiry", autorange="reversed"),
                zaxis=T.scene_axis("Option price ($)"),
                bgcolor=T.BASE,
                aspectmode="manual",
                aspectratio=dict(x=1.2, y=1.0, z=0.68),
                camera=dict(
                    eye=dict(x=1.6, y=-1.5, z=0.82),
                    center=dict(x=0, y=0, z=-0.06),
                ),
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.0,
                x=1.0, xanchor="right",
                bgcolor="rgba(11,16,32,0.78)", bordercolor=T.LINE, borderwidth=1,
                font=dict(size=11, color=T.TEXT), itemsizing="constant",
            ),
            annotations=[
                T.caption(
                    "Circles are the mark. Diamonds are actual trades. "
                    "The violet sheet is interpolated — it is an assumption, not a market."
                )
            ],
        )
    )
    return fig


def mark_vs_print_figure(sl: pd.DataFrame, ticker: str = "UUUU") -> go.Figure:
    """
    Requirement 4, in 2D: the two series side by side so a stranger can see
    they are not the same thing.

    Left  -- paired quotes against y = x. Distance off the diagonal is the gap.
    Right -- how many series carry which field at all.
    """
    if sl is None or sl.empty:
        return _empty("No series on this date.", height=430)

    both = sl.dropna(subset=[MARK_FIELD, PRINT_FIELD])
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Where both exist, they disagree",
            "Which series carry which field",
        ),
        horizontal_spacing=0.13,
        column_widths=[0.58, 0.42],
    )

    if len(both):
        lo = float(min(both[PRINT_FIELD].min(), both[MARK_FIELD].min()))
        hi = float(max(both[PRINT_FIELD].max(), both[MARK_FIELD].max()))
        pad = (hi - lo) * 0.08 if hi > lo else 0.05
        fig.add_trace(
            go.Scatter(
                x=[lo - pad, hi + pad], y=[lo - pad, hi + pad],
                mode="lines",
                line=dict(color=T.TEXT_FAINT, dash="dash", width=1),
                name="mark = print", hoverinfo="skip",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=both[PRINT_FIELD], y=both[MARK_FIELD],
                mode="markers",
                name="carries both",
                marker=dict(
                    size=9, color=T.BOTH, opacity=0.85,
                    line=dict(width=2, color=T.PANEL),  # 2px surface ring
                ),
                customdata=both[["ric", "strike", "dte", "abs_diff"]].to_numpy(),
                hovertemplate=(
                    f"{PRINT_FIELD} $%{{x:.3f}}<br>{MARK_FIELD} $%{{y:.3f}}<br>"
                    "gap $%{customdata[3]:.3f}<br>"
                    "K %{customdata[1]:.2f} · DTE %{customdata[2]}<br>"
                    "%{customdata[0]}<extra></extra>"
                ),
            ),
            row=1, col=1,
        )
        fig.update_xaxes(range=[lo - pad, hi + pad], row=1, col=1)
        fig.update_yaxes(range=[lo - pad, hi + pad], row=1, col=1)
    else:
        fig.add_annotation(
            text="No series carries both fields on this date.",
            xref="x domain", yref="y domain", x=0.5, y=0.5,
            showarrow=False, font=dict(size=12, color=T.TEXT_MUTED),
            row=1, col=1,
        )

    n_mark_only = int((sl["has_mark"] & ~sl["has_print"]).sum())
    n_both = int((sl["has_mark"] & sl["has_print"]).sum())
    n_print_only = int((sl["has_print"] & ~sl["has_mark"]).sum())
    counts = [n_mark_only, n_both, n_print_only]

    fig.add_trace(
        go.Bar(
            x=["Mark only<br>(no trade)", "Both", "Trade only<br>(no mark)"],
            y=counts,
            marker=dict(
                color=[T.MARK, T.BOTH, T.PRINT],
                line=dict(width=2, color=T.PANEL),  # 2px surface gap
            ),
            text=counts,
            textposition="outside",
            textfont=dict(color=T.TEXT, size=12, family=T.FONT),
            cliponaxis=False,
            showlegend=False,
            hovertemplate="%{x}<br><b>%{y}</b> series<extra></extra>",
        ),
        row=1, col=2,
    )
    fig.update_yaxes(range=[0, max(counts + [1]) * 1.22], row=1, col=2)

    fig.update_xaxes(**T.axis(f"{PRINT_FIELD} — last trade ($)"), row=1, col=1)
    fig.update_yaxes(**T.axis(f"{MARK_FIELD} — the mark ($)"), row=1, col=1)
    fig.update_xaxes(**T.axis(), row=1, col=2)
    fig.update_yaxes(**T.axis("Series"), row=1, col=2)

    fig.update_layout(
        **T.layout(
            height=460,
            title=dict(text=f"{ticker} · the mark is not the trade", x=0.0, xanchor="left"),
            margin=dict(l=64, r=24, t=96, b=92),
            bargap=0.45,
            # legend right, caption below: at y=1.0/x=0.0 they landed on top
            # of each other.
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, x=1.0, xanchor="right",
                bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=T.TEXT),
            ),
            annotations=[
                dict(
                    text="Every point off the dashed line is a series where the "
                         "mark and the only real trade of the day disagree.",
                    xref="paper", yref="paper", x=0.0, y=-0.24,
                    xanchor="left", yanchor="top", showarrow=False,
                    font=dict(size=10.5, color=T.TEXT_FAINT, family=T.FONT),
                )
            ],
        )
    )
    # subplot titles come through as annotations; restyle them
    for ann in fig.layout.annotations[:2]:
        ann.font = dict(size=12, color=T.TEXT, family=T.FONT)
    return fig


def occupancy_figure(sl: pd.DataFrame, field: str, cp: str = "C") -> go.Figure:
    """
    The honest picture: which (expiry, strike) cells carry a number at all.
    A dark cell never had one.
    """
    grid = occupancy_matrix(sl, field)
    if grid is None or grid.empty:
        return _empty(f"No {field} coverage on this date.", height=360)

    scale = T.MARK_SCALE if field == MARK_FIELD else T.PRINT_SCALE
    filled = float(np.nansum(grid.values)) / max(grid.size, 1)

    fig = go.Figure(
        go.Heatmap(
            z=grid.values,
            x=[f"{c:.2f}" for c in grid.columns],
            y=list(grid.index),
            colorscale=scale,
            zmin=0, zmax=1,
            showscale=False,
            xgap=2, ygap=2,  # 2px surface gap between cells
            hovertemplate=(
                "K %{x} · expiry %{y}<br>"
                "%{customdata}<extra></extra>"
            ),
            customdata=np.where(
                np.nan_to_num(grid.values) > 0, "has a number", "never had one"
            ),
        )
    )
    fig.update_layout(
        **T.layout(
            height=360,
            title=dict(
                text=f"{field} coverage · {filled:.0%} of the listed grid",
                x=0.0, xanchor="left",
            ),
            xaxis={**T.axis("Strike ($)"), "tickangle": -45, "showgrid": False},
            yaxis={**T.axis("Expiry"), "showgrid": False},
            margin=dict(l=76, r=20, t=64, b=64),
            annotations=[T.caption("Lit = a number exists. Dark = no quote that day.")],
        )
    )
    return fig
