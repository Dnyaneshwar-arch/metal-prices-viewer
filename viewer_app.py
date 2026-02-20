from __future__ import annotations

from html import escape
from typing import List, Dict
from pathlib import Path
import re

import altair as alt
import pandas as pd
import streamlit as st

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False

from data_utils import load_config, load_sheet


# ---------- Page & CSS ----------
st.set_page_config(
    page_title="Metal Prices Dashboards",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .title { font-size: 34px; font-weight: 800; }
      .stButton>button { padding: .45rem .9rem; }
      .sheet-headline{
        margin: 8px 0 12px 0;
        font-weight: 800;
        font-size: 18px;
        color:#111827;
      }
      .summary-box{
        margin-top: 8px;
        padding: 10px 12px;
        background:#FAFCFF;
        border:1px solid #E5EDF7;
        border-radius: 10px;
        font-size: 14px;
        line-height: 1.5;
        color:#111827;
      }
      .summary-box b{ color:#0B5CAB; }
      .menu-title{ font-weight:800; font-size:16px; margin: 6px 0 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- Helpers ----------------
def _tidy(chart: alt.Chart) -> alt.Chart:
    return (
        chart
        .properties(padding={"left": 48, "right": 48, "top": 12, "bottom": 36})
        .configure_view(strokeWidth=0)
        .configure_axis(labelPadding=8, titlePadding=10, labelFontSize=12, titleFontSize=12)
        .configure_axisX(labelAngle=0)
    )


def _forecast_series(y: pd.Series, periods: int, seasonal_periods: int) -> pd.Series:
    if len(y) < 3:
        return pd.Series([float(y.iloc[-1])] * periods, index=range(periods))
    if _HAS_STATSMODELS:
        try:
            use_season = seasonal_periods > 0 and len(y) >= (seasonal_periods * 2)
            model = ExponentialSmoothing(
                y.astype(float),
                trend="add",
                seasonal="add" if use_season else None,
                seasonal_periods=seasonal_periods if use_season else None,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True)
            return pd.Series(fit.forecast(periods))
        except Exception:
            pass
    return pd.Series([float(y.iloc[-1])] * periods)


def _bar_size(n: int) -> int:
    approx_width = 1100
    size = int(approx_width / max(1, n) * 0.65)
    return max(8, min(42, size))


# ---------------- Page 1 ----------------
def render_metal_prices_page():
    st.markdown('<div class="title">Metal Prices Dashboards</div>', unsafe_allow_html=True)

    DEFAULT_START = pd.to_datetime("2025-04-01").date()
    DEFAULT_END = pd.Timestamp.today().date()

    config = load_config()
    sheets: List[Dict] = config.get("sheets", [])
    if not sheets:
        st.info("No published data yet.")
        st.stop()

    labels = [s.get("label") or s.get("sheet") or s.get("slug") for s in sheets]
    slugs = [s.get("slug") for s in sheets]

    sel_label = st.selectbox("Commodity", labels, index=0, key="commodity-select")
    slug = slugs[labels.index(sel_label)]

    df = load_sheet(slug)
    if df.empty:
        st.warning("No data for this commodity.")
        st.stop()

    min_d = pd.to_datetime(df["Month"].min()).date()
    max_d = pd.to_datetime(df["Month"].max()).date()
    def_start = max(min_d, DEFAULT_START)
    def_end = min(max_d, DEFAULT_END)

    k_from = f"applied-from-{slug}"
    k_to = f"applied-to-{slug}"
    st.session_state.setdefault(k_from, def_start)
    st.session_state.setdefault(k_to, def_end)

    ver_key = f"ver-{slug}"
    st.session_state.setdefault(ver_key, 0)
    ver = st.session_state[ver_key]

    w_from = f"from-{slug}-{ver}"
    w_to = f"to-{slug}-{ver}"

    # -------- FILTER FORM --------
    with st.form(key=f"filters-{slug}-{ver}", border=False):
        c1, c2, c3, c4, c5 = st.columns([2.6, 2.6, 0.8, 0.8, 1.4], gap="small")

        c1.date_input("From", def_start, key=w_from, format="DD/MM/YYYY")
        c2.date_input("To", def_end, key=w_to, format="DD/MM/YYYY")

        with c3:
            st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True)
            search = st.form_submit_button("Search")

        with c4:
            st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True)
            clear = st.form_submit_button("Clear")

        with c5:
            chart_type = st.selectbox(
                "Chart Type",
                ["Bar Chart", "Line Chart", "Pie Chart", "Area Chart"],
                key=f"chart-type-{slug}"
            )

    if clear:
        st.session_state[k_from] = def_start
        st.session_state[k_to] = def_end
        st.session_state[ver_key] = ver + 1
        st.rerun()

    if search:
        start_val = st.session_state[w_from]
        end_val = st.session_state[w_to]
        if start_val > end_val:
            st.error("From date must be ≤ To date.")
            st.stop()
        st.session_state[k_from] = start_val
        st.session_state[k_to] = end_val

    start = st.session_state[k_from]
    end = st.session_state[k_to]

    mask = (df["Month"].dt.date >= start) & (df["Month"].dt.date <= end)
    f = df.loc[mask].copy()
    if f.empty:
        st.info("No rows in this range.")
        st.stop()

    f = f.sort_values("Month")
    f["MonthLabel"] = pd.to_datetime(f["Month"]).dt.strftime("%b %y").str.upper()
    f["PriceTT"] = f["Price"].apply(lambda x: f"${x:,.0f}")

    actual_only = f.copy()
    domain_order = actual_only["MonthLabel"].tolist()

    placeholder = st.empty()

    if chart_type == "Bar Chart":
        chart = alt.Chart(actual_only).mark_bar().encode(
            x=alt.X("MonthLabel:N", sort=domain_order),
            y=alt.Y("Price:Q", scale=alt.Scale(zero=False)),
            tooltip=["MonthLabel", "PriceTT"]
        )

    elif chart_type == "Line Chart":
        chart = alt.Chart(actual_only).mark_line(point=True).encode(
            x=alt.X("MonthLabel:N", sort=domain_order),
            y=alt.Y("Price:Q", scale=alt.Scale(zero=False)),
            tooltip=["MonthLabel", "PriceTT"]
        )

    elif chart_type == "Area Chart":
        chart = alt.Chart(actual_only).mark_area(opacity=0.5).encode(
            x=alt.X("MonthLabel:N", sort=domain_order),
            y=alt.Y("Price:Q", scale=alt.Scale(zero=False)),
            tooltip=["MonthLabel", "PriceTT"]
        )

    elif chart_type == "Pie Chart":
        chart = alt.Chart(actual_only).mark_arc().encode(
            theta="Price:Q",
            color="MonthLabel:N",
            tooltip=["MonthLabel", "PriceTT"]
        )

    placeholder.altair_chart(_tidy(chart.properties(height=430)), use_container_width=True)


# ---------------- Sidebar ----------------
st.sidebar.markdown('<div class="menu-title">☰ Menu</div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Go to",
    ["Metal Prices", "Billet Prices"],
    index=0,
    key="nav-radio",
    label_visibility="collapsed",
)

if page == "Metal Prices":
    render_metal_prices_page()
