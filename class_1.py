import os
import pickle
import datetime
import warnings
import pandas as pd
import numpy as np
import lseg.data as ld
import plotly.graph_objects as go
import reflex as rx

# Suppress internal Pandas downcasting FutureWarnings from lseg.data
warnings.filterwarnings("ignore", category=FutureWarning, module="lseg.data")

# Local cache path
CACHE_FILE = "option_pipeline_data.pkl"


# -----------------------------------------------------------------------------
# Core Data Pipeline Logic
# -----------------------------------------------------------------------------
def load_or_fetch_pipeline_data(
        ticker_stock: str = "UUUU.K",
        ticker_root: str = "UUUU",
        weeks_back: int = 12,
        strike_step: float = 0.50,
        batch_size: int = 25
) -> dict:
    """
    Checks for a local pickle cache. If not found, opens an LSEG session, queries
    stock OHLC history + expired options daily prices, and saves to disk.
    """
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached dataset from {CACHE_FILE}...")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print("Cache not found. Initializing LSEG data pull...")
    ld.open_session()

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(weeks=weeks_back)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # 1. Fetch Underlying Stock OHLC
    df_stock = ld.get_history(
        universe=[ticker_stock],
        fields=["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1"],
        start=start_str,
        end=end_str,
        interval="daily"
    )

    low_price = float(df_stock["LOW_1"].min())
    high_price = float(df_stock["HIGH_1"].max())

    # 2. Generate Candidate Expired OPRA RICs
    min_strike = np.floor(low_price / strike_step) * strike_step
    max_strike = np.ceil(high_price / strike_step) * strike_step
    strikes = np.arange(min_strike, max_strike + strike_step, strike_step)
    friday_dates = pd.date_range(start=start_str, end=end_str, freq="W-FRI")

    candidate_rics = []
    for dt in friday_dates:
        year_str = dt.strftime("%y")
        day_str = dt.strftime("%d")
        month_num = dt.month

        call_code = chr(ord('A') + month_num - 1)
        put_code = chr(ord('M') + month_num - 1)

        for strike in strikes:
            strike_str = f"{int(round(strike * 100)):05d}"
            call_base = f"{ticker_root.upper()}{call_code}{day_str}{year_str}{strike_str}.U"
            candidate_rics.append(f"{call_base}^{call_code}{year_str}")

            put_base = f"{ticker_root.upper()}{put_code}{day_str}{year_str}{strike_str}.U"
            candidate_rics.append(f"{put_base}^{put_code}{year_str}")

    # 3. Batch Query Options History
    batches = [candidate_rics[i: i + batch_size] for i in
               range(0, len(candidate_rics), batch_size)]
    history_frames = []
    fields = ["TRDPRC_1", "SETTLE"]

    for batch in batches:
        try:
            df_batch = ld.get_history(
                universe=batch,
                fields=fields,
                start=start_str,
                end=end_str,
                interval="daily"
            )
            if df_batch is not None and not df_batch.empty:
                df_clean = df_batch.dropna(how="all", axis=1)
                if not df_clean.empty:
                    history_frames.append(df_clean)
        except Exception:
            for single_ric in batch:
                try:
                    df_single = ld.get_history(
                        universe=[single_ric],
                        fields=fields,
                        start=start_str,
                        end=end_str,
                        interval="daily"
                    )
                    if df_single is not None and not df_single.empty and not df_single.dropna(
                            how="all").empty:
                        history_frames.append(df_single)
                except Exception:
                    continue

    ld.close_session()

    df_options = pd.DataFrame()
    if history_frames:
        df_options = pd.concat(history_frames, axis=1)
        df_options = df_options.loc[:, ~df_options.columns.duplicated()]

    data_payload = {
        "stock": df_stock,
        "options": df_options,
        "ticker": ticker_root,
        "fetched_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save cache locally
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data_payload, f)
    print(f"Data pipeline complete. Results cached to {CACHE_FILE}.")

    return data_payload


# -----------------------------------------------------------------------------
# Reflex App State & UI Logic
# -----------------------------------------------------------------------------
class State(rx.State):
    """Reflex application state."""
    ticker: str = "UUUU"
    status_msg: str = "Ready"
    option_count: int = 0
    fig: go.Figure = go.Figure()

    def load_data(self):
        """Loads data from cache or pipeline and renders the candlestick chart."""
        self.status_msg = "Checking local cache / fetching data..."
        payload = load_or_fetch_pipeline_data()

        df_stock = payload["stock"]
        df_options = payload["options"]
        self.option_count = len(df_options.columns)
        self.ticker = payload["ticker"]

        # Build Dark/Neon Candlestick Plotly Figure
        fig = go.Figure(data=[
            go.Candlestick(
                x=df_stock.index,
                open=df_stock["OPEN_PRC"],
                high=df_stock["HIGH_1"],
                low=df_stock["LOW_1"],
                close=df_stock["TRDPRC_1"],
                increasing_line_color="#00ffcc",  # Neon Cyan Bullish
                increasing_fillcolor="#00ffcc",
                decreasing_line_color="#ff0055",  # Neon Pink Bearish
                decreasing_fillcolor="#ff0055",
                name=self.ticker
            )
        ])

        fig.update_layout(
            template="plotly_dark",
            title=f"{self.ticker} Stock Price (12-Week OHLC History)",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font=dict(color="#e6edf3", family="monospace"),
            xaxis=dict(gridcolor="#30363d", rangeslider=dict(visible=False)),
            yaxis=dict(gridcolor="#30363d", title="Price ($)"),
            margin=dict(l=40, r=40, t=60, b=40)
        )

        self.fig = fig
        self.status_msg = f"Data Loaded ({payload['fetched_at']})"


def index() -> rx.Component:
    return rx.container(
        rx.vstack(
            # Header Block
            rx.hstack(
                rx.heading(
                    "OPTIONS STRATEGY BACKTESTER",
                    size="8",
                    color="#00ffcc",
                    style={"letter_spacing": "2px"}
                ),
                rx.spacer(),
                rx.badge(
                    State.status_msg,
                    color_scheme="cyan",
                    variant="solid"
                ),
                width="100%",
                align="center",
                padding_y="1rem"
            ),

            # Metrics Row
            rx.hstack(
                rx.card(
                    rx.text("Underlying Ticker", size="2", color="#8b949e"),
                    rx.text(State.ticker, size="6", color="#39d353",
                            weight="bold"),
                    bg="#161b22", border="1px solid #30363d", padding="1rem"
                ),
                rx.card(
                    rx.text("Traded Option Series Discovered", size="2",
                            color="#8b949e"),
                    rx.text(State.option_count, size="6", color="#00ffcc",
                            weight="bold"),
                    bg="#161b22", border="1px solid #30363d", padding="1rem"
                ),
                rx.button(
                    "Reload / Refresh Data",
                    on_click=State.load_data,
                    bg="#238636",
                    color="#ffffff",
                    _hover={"bg": "#2ea043"},
                    size="3"
                ),
                spacing="4",
                width="100%",
                align="center"
            ),

            # Candlestick Plot Card
            rx.box(
                rx.plotly(data=State.fig,
                          style={"width": "100%", "height": "500px"}),
                width="100%",
                bg="#161b22",
                border="1px solid #30363d",
                border_radius="8px",
                padding="1rem"
            ),
            spacing="5",
            width="100%"
        ),
        on_mount=State.load_data,
        background_color="#0d1117",
        min_height="100vh",
        max_width="100%",
        padding="2rem"
    )


app = rx.App()
app.add_page(index)