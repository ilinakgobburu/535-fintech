"""
Reflex entry point for the semester app.

Each homework registers its own page module under `pages/`; this file only
wires them up. Adding HW1.2 means adding one import and one `add_page`.
"""

from __future__ import annotations

import reflex as rx

from . import theme as T
from .pages.hw1_surface import surface_page

app = rx.App(
    style={
        "background_color": T.BASE,
        "color": T.TEXT,
        "font_family": T.FONT,
    },
)
app.add_page(surface_page, route="/", title="Option Surface Lab")
