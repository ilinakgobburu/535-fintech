"""
LSEG login shared by the repo's scripts (lseg_plot.py, the order scripts, nvda_marks.py).

It opens a platform session (app key + username + password), so LSEG Workspace
does not need to be open anywhere. No credentials are written in this file.
They come from the environment variables LSEG_APP_KEY, LSEG_USERNAME and
LSEG_PASSWORD (GitHub Actions secrets) or, when those are not set and the
script runs on a Mac, from the macOS Keychain items lseg-app-key,
lseg-username and lseg-password.

    from lseg_login import open_session
    session = open_session()
    try:
        ...  # ld.get_history(...)
    finally:
        session.close()

One login holds one session at a time. By default a new session takes over (signs out) any other on the same
login, LSEG Workspace included. open_session(take_over=False) never does: its login fails instead while another
session holds the login (nvda_marks.py, so that the nightly mark never signs out an orders run).
"""

import os
import shutil
import subprocess
import sys

import lseg.data as ld

KEYCHAIN = {
    "LSEG_APP_KEY": "lseg-app-key",
    "LSEG_USERNAME": "lseg-username",
    "LSEG_PASSWORD": "lseg-password",
}


def credential(name: str) -> str:
    """An environment variable first (GitHub), then the macOS Keychain (laptop)."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if shutil.which("security"):
        out = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN[name], "-w"],
            capture_output=True, text=True,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    return ""


def open_session(take_over: bool = True):
    """Open an LSEG platform session and make it the default. Exits if login fails. take_over=False: fail rather
    than sign out another session on this login."""
    creds = {name: credential(name) for name in KEYCHAIN}
    missing = [name for name, value in creds.items() if not value]
    if missing:
        sys.exit("Missing " + ", ".join(missing)
                 + ": set them as GitHub Actions secrets, or save them in the macOS Keychain.")

    session = ld.session.platform.Definition(
        app_key=creds["LSEG_APP_KEY"],
        grant=ld.session.platform.GrantPassword(
            username=creds["LSEG_USERNAME"],
            password=creds["LSEG_PASSWORD"],
        ),
        signon_control=take_over,   # True: takes over any other session on this login; False: leaves it alone
    ).get_session()
    session.open()
    # The library prints login errors instead of raising them, so check the state.
    if session.open_state != ld.OpenState.Opened:
        sys.exit("LSEG login failed: check the app key, username and password."
                 + ("" if take_over else " Or another session holds this login (an orders run, LSEG Workspace): "
                                         "this run does not sign it out."))
    ld.session.set_default(session)
    print("LSEG platform session opened")
    return session
