from __future__ import annotations

from html import escape
from typing import List, Dict

import altair as alt
import pandas as pd
import streamlit as st

from data_utils import load_config, load_sheet

# ---------- Page & CSS ----------
st.set_page_config(page_title="Metal Prices Dashboards", layout="wide")
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

      /* summary text under the chart */
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">Metal Prices Dashboards</div>', unsafe_allow_html=True)

DEFAULT_START = pd.to_datetime("2025-04-01").date()
DEFAULT_END = pd.Timestamp.today().date()

config = load_config()
sheets: List[Dict] = config.get("sheets", []) if isinstance(config, dict) else config
if not sheets:
    st.info("No published data yet.")
    st.stop()

labels = [s.get("label") or s.get("sheet") or s.get("slug") for s in sheets]
slugs = [s.get("slug") for s in sheets]

sel_label = st.selectbox("Commodity", labels, index=0)
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

with st.form(key=f"filters-{slug}-{ver}", border=False):
    c1, c2, c3, c4 = st.columns([2.6, 2.6, 0.6, 0.6], gap="small")
    c1.date_input("From (dd-mm-yyyy)", def_start, key=w_from)
    c2.date_input("To (dd-mm-yyyy)", def_end, key=w_to)
    with c3:
        st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True)
        search = st.form_submit_button("Search")
    with c4:
        st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True)
        clear = st.form_submit_button("Clear")

# Bigger heading (from config saved at publish-time)
meta = next((m for m in sheets if m.get("slug") == slug or m.get("sheet") == slug), {})
heading_text = meta.get("heading") or meta.get("label") or meta.get("sheet") or ""
st.markdown(f'<div class="sheet-headline">{escape(str(heading_text))}</div>', unsafe_allow_html=True)

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

# ------- Plot -------
f = f.sort_values("Month")
f["MonthLabel"] = pd.to_datetime(f["Month"]).dt.strftime("%b %y").str.upper()

def _bar_size(n: int) -> int:
    approx_width = 1100
    size = int(approx_width / max(1, n) * 0.65)
    return max(8, min(42, size))

# ➕ tooltip display helper: big numbers => no decimals; small numbers => 2 decimals
def _fmt_tooltip(v):
    try:
        v = float(v)
        return f"${v:,.2f}" if abs(v) < 10 else f"${v:,.0f}"
    except Exception:
        return str(v)

f["PriceTT"] = f["Price"].apply(_fmt_tooltip)

chart = (
    alt.Chart(f)
    .mark_bar(size=_bar_size(len(f)))
    .encode(
        x=alt.X("MonthLabel:N", title="Months", sort=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Price:Q", title="Price"),
        tooltip=[
            alt.Tooltip("MonthLabel:N", title="Month"),
            alt.Tooltip("PriceTT:N", title="Price"),   # ← shows like: Price $15,325  /  Price $3.05
        ],
    )
    .properties(height=430)
)
st.altair_chart(chart, use_container_width=True)

# ------- Auto summary (2–3 lines) under the chart -------
def _fmt_money(x: float) -> str:
    try:
        return f"${x:,.2f}"
    except Exception:
        return str(x)

def _fmt_mon(d) -> str:
    try:
        return pd.to_datetime(d).strftime("%b %Y")
    except Exception:
        return str(d)

# start/end
first_row = f.iloc[0]
last_row = f.iloc[-1]
start_price = float(first_row["Price"])
end_price = float(last_row["Price"])
start_mon = _fmt_mon(first_row["Month"])
end_mon = _fmt_mon(last_row["Month"])

chg_abs = end_price - start_price
chg_pct = (chg_abs / start_price * 100.0) if start_price not in (0, None) else 0.0
arrow = "↑" if chg_abs > 0 else ("↓" if chg_abs < 0 else "→")

# highs/lows
hi_idx = f["Price"].idxmax()
lo_idx = f["Price"].idxmin()
hi_price, hi_mon = float(f.loc[hi_idx, "Price"]), _fmt_mon(f.loc[hi_idx, "Month"])
lo_price, lo_mon = float(f.loc[lo_idx, "Price"]), _fmt_mon(f.loc[lo_idx, "Month"])

avg_price = float(f["Price"].mean())
rng = hi_price - lo_price
n_months = len(f)

summary_html = f"""
<div class="summary-box">
  <div><b>{start_mon}</b> to <b>{end_mon}</b>: price moved from <b>{_fmt_money(start_price)}</b> to <b>{_fmt_money(end_price)}</b> ({arrow} {chg_abs:+.2f}, {chg_pct:+.2f}%).</div>
  <div>Period high was <b>{_fmt_money(hi_price)}</b> in <b>{hi_mon}</b>; low was <b>{_fmt_money(lo_price)}</b> in <b>{lo_mon}</b> (range {_fmt_money(rng)}).</div>
  <div>Across <b>{n_months}</b> months, the average price was <b>{_fmt_money(avg_price)}</b>. These details auto-update with your date filters.</div>
</div>
"""
st.markdown(summary_html, unsafe_allow_html=True)
