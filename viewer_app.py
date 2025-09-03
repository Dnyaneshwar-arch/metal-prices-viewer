from __future__ import annotations

from html import escape
from typing import List, Dict
from pathlib import Path
import re

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

# ================================
# TITLE ROW
# ================================
st.markdown('<div class="title">Metal Prices Dashboards</div>', unsafe_allow_html=True)

# ================================
# EXISTING: SCRAP / COMMODITY DASHBOARD (unchanged)
# ================================
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

# ➕ tooltip display helper
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
            alt.Tooltip("PriceTT:N", title="Price"),
        ],
    )
    .properties(height=430)
)
st.altair_chart(chart, use_container_width=True)

# ------- Auto summary under the chart -------
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

first_row = f.iloc[0]
last_row = f.iloc[-1]
start_price = float(first_row["Price"])
end_price = float(last_row["Price"])
start_mon = _fmt_mon(first_row["Month"])
end_mon = _fmt_mon(last_row["Month"])

chg_abs = end_price - start_price
chg_pct = (chg_abs / start_price * 100.0) if start_price not in (0, None) else 0.0
arrow = "↑" if chg_abs > 0 else ("↓" if chg_abs < 0 else "→")

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

# ================================
# NEW: BILLET PRICES DASHBOARD
# ================================
st.divider()
st.markdown('<div class="title">Billet Prices</div>', unsafe_allow_html=True)

from pathlib import Path
import re

# ---- helpers for Billet Excel ----
BASE_DIR = Path(__file__).parent.resolve()

# Search in data/, data/current/, or repo root
BILLET_FILE_CANDIDATES = [
    BASE_DIR / "data" / "Billet cost.xlsx",
    BASE_DIR / "data" / "current" / "Billet cost.xlsx",  # <- your current location
    BASE_DIR / "Billet cost.xlsx",
]

# Nice labels for the dropdown
BILLET_SERIES_LABELS = [
    "Billet Price (Blast Furnace Route)",
    "Billet Price (Electric Arc Furnace)",
]

def _find_billet_file() -> Path | None:
    for p in BILLET_FILE_CANDIDATES:
        if p.exists():
            return p
    return None

def _resolve_sheet_name(xls: pd.ExcelFile, series_label: str) -> str:
    """Pick the correct sheet even if Excel truncated to 31 chars."""
    want_blast = "blast" in series_label.lower()
    for name in xls.sheet_names:
        n = name.lower()
        if want_blast and "blast" in n:
            return name
        if not want_blast and ("electric" in n or "arc" in n):
            return name
    return xls.sheet_names[0]

def _load_billet_df(series_label: str) -> pd.DataFrame:
    """
    Returns tidy df: Quarter(str), Price(float), QuarterOrder(int), QuarterLabel(str)
    """
    src = _find_billet_file()
    if not src:
        st.error("Billet Excel not found. Put **Billet cost.xlsx** in `data/` or `data/current/` (or next to this file).")
        st.stop()

    # open with openpyxl engine
    try:
        xls = pd.ExcelFile(src)
    except Exception as e:
        st.error(f"Could not open {src.name}: {e}. Ensure `openpyxl` is in requirements.txt.")
        st.stop()

    sheet_name = _resolve_sheet_name(xls, series_label)
    raw = xls.parse(sheet_name=sheet_name)

    # Fuzzy column detection
    quarter_col = next((c for c in raw.columns if re.search(r"q(u)?arter", str(c), re.I)), None)
    if quarter_col is None:
        quarter_col = next((c for c in raw.columns if re.search(r"qurter", str(c), re.I)), None)
    price_col = next((c for c in raw.columns if re.search(r"billet.*(per)?.*mt", str(c), re.I)), None)

    if quarter_col is None or price_col is None:
        st.error("Could not detect columns. Need a Quarter column and a 'Billet cost per MT' column.")
        st.stop()

    df0 = raw[[quarter_col, price_col]].rename(columns={quarter_col: "Quarter", price_col: "Price"})
    df0["Quarter"] = df0["Quarter"].astype(str).str.strip()

    # Q1-2024 / Q1 2024 ordering
    def _q_order(qs: str) -> int:
        m = re.search(r"Q(\d)\s*[- ]\s*(\d{4})", qs, flags=re.I)
        if not m:
            return 0
        q = int(m.group(1)); y = int(m.group(2))
        return y * 10 + q

    df0["QuarterOrder"] = df0["Quarter"].apply(_q_order)
    df0 = df0.sort_values("QuarterOrder")
    df0["QuarterLabel"] = df0["Quarter"].str.replace("-", " ", regex=False)
    df0["Price"] = pd.to_numeric(df0["Price"], errors="coerce")
    df0 = df0.dropna(subset=["Price"])
    st.caption(f"Source: {src.name} • sheet: {sheet_name}")
    return df0

def _fmt_int(n: float) -> str:
    try:
        return f"{float(n):,.0f}"
    except Exception:
        return str(n)

# --- UI: series dropdown
series_label = st.selectbox("Select Billet Series", BILLET_SERIES_LABELS, index=0, key="billet-series")

billet_df_full = _load_billet_df(series_label)
if billet_df_full.empty:
    st.info("No billet rows.")
    st.stop()

# Defaults for quarter range
quarters = billet_df_full["QuarterLabel"].tolist()
q_start_def, q_end_def = quarters[0], quarters[-1]
kq_from = f"billet-from-{series_label}"
kq_to   = f"billet-to-{series_label}"
st.session_state.setdefault(kq_from, q_start_def)
st.session_state.setdefault(kq_to, q_end_def)

# --- Filter form
with st.form(key=f"billet-form-{series_label}", border=False):
    c1, c2, c3, c4 = st.columns([2.6, 2.6, 0.6, 0.6], gap="small")
    c1.selectbox("From Quarter", options=quarters, index=0, key=kq_from)
    c2.selectbox("To Quarter", options=quarters, index=len(quarters)-1, key=kq_to)
    with c3:
        st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True)
        btn_go = st.form_submit_button("Search")
    with c4:
        st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True)
        btn_clear = st.form_submit_button("Clear")

if btn_clear:
    st.session_state[kq_from] = q_start_def
    st.session_state[kq_to]   = q_end_def
    st.rerun()

q_from = st.session_state[kq_from]
q_to   = st.session_state[kq_to]
i_from = quarters.index(q_from)
i_to   = quarters.index(q_to)
if i_from > i_to:
    st.error("From Quarter must be ≤ To Quarter.")
    st.stop()

billet_df = billet_df_full.iloc[i_from:i_to+1].copy()
if billet_df.empty:
    st.info("No rows in this quarter range.")
    st.stop()

# --- Chart
chart2 = (
    alt.Chart(billet_df)
    .mark_bar(size=28)
    .encode(
        x=alt.X("QuarterLabel:N", title="Quarter", sort=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Price:Q", title="Billet cost per MT"),
        tooltip=[alt.Tooltip("QuarterLabel:N", title="Quarter"),
                 alt.Tooltip("Price:Q", title="Price", format=",.0f")],
    )
    .properties(height=430)
)
st.altair_chart(chart2, use_container_width=True)

# --- Summary
first_b = billet_df.iloc[0]; last_b = billet_df.iloc[-1]
b_start = float(first_b["Price"]); b_end = float(last_b["Price"])
b_arrow = "↑" if (b_end - b_start) > 0 else ("↓" if (b_end - b_start) < 0 else "→")
b_hi = billet_df.loc[billet_df["Price"].idxmax()]
b_lo = billet_df.loc[billet_df["Price"].idxmin()]
b_avg = float(billet_df["Price"].mean())

billet_summary = f"""
<div class="summary-box">
  <div><b>{first_b['QuarterLabel']}</b> to <b>{last_b['QuarterLabel']}</b>: price moved from <b>{_fmt_int(b_start)}</b> to <b>{_fmt_int(b_end)}</b> ({b_arrow} {b_end - b_start:+.0f}).</div>
  <div>Period high was <b>{_fmt_int(b_hi['Price'])}</b> in <b>{b_hi['QuarterLabel']}</b>; low was <b>{_fmt_int(b_lo['Price'])}</b> in <b>{b_lo['QuarterLabel']}</b> (range {_fmt_int(b_hi['Price'] - b_lo['Price'])}).</div>
  <div>Across <b>{len(billet_df)}</b> quarters, the average price was <b>{_fmt_int(b_avg)}</b>. These details auto-update with your quarter filters.</div>
</div>
"""
st.markdown(billet_summary, unsafe_allow_html=True)
