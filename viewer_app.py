from __future__ import annotations

from html import escape
from typing import List, Dict
from pathlib import Path
import re

import altair as alt
import pandas as pd
import streamlit as st

# Forecasting (safe fallback)
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False

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

# ---------- Shared helpers ----------
def _tidy(chart: alt.Chart) -> alt.Chart:
    """Consistent anti-clipping / spacing / font sizing for charts."""
    return (
        chart
        .properties(padding={"left": 56, "right": 56, "top": 12, "bottom": 40})
        .configure_view(strokeWidth=0)
        .configure_axis(labelPadding=8, titlePadding=10, labelFontSize=12, titleFontSize=12)
        .configure_axisX(labelAngle=0)
    )

def _y_scale_from(vals: pd.Series):
    """If data are flat or extremely tight, pad the y-domain so bars/lines are visible."""
    try:
        s = pd.to_numeric(vals, errors="coerce").dropna()
        if s.empty:
            return alt.Scale(zero=False, nice=True)
        lo, hi = float(s.min()), float(s.max())
        if lo == hi:
            pad = max(1.0, abs(hi) * 0.03)
            return alt.Scale(domain=[lo - pad, hi + pad], nice=False, zero=False)
        # very tight range (e.g., 3.00–3.02)
        if (hi - lo) / max(1e-9, abs(hi)) < 0.01:
            pad = max(0.5, (hi - lo) * 2.5)
            return alt.Scale(domain=[lo - pad, hi + pad], nice=False, zero=False)
        return alt.Scale(zero=False, nice=True)
    except Exception:
        return alt.Scale(zero=False, nice=True)

def _bar_size(n: int) -> int:
    approx_width = 1100
    size = int(approx_width / max(1, n) * 0.65)
    return max(8, min(42, size))

def _fmt_tooltip_usd(v):
    try:
        v = float(v)
        return f"${v:,.2f}" if abs(v) < 10 else f"${v:,.0f}"
    except Exception:
        return str(v)

def _fmt_money(x: float) -> str:
    try: return f"${x:,.2f}"
    except Exception: return str(x)

def _fmt_mon(d) -> str:
    try: return pd.to_datetime(d).strftime("%b %Y")
    except Exception: return str(d)

def _fmt_inr(n: float) -> str:
    try: return f"₹{float(n):,.0f}"
    except Exception: return f"₹{n}"

def _forecast_series(y: pd.Series, periods: int, seasonal_periods: int) -> pd.Series:
    """Holt–Winters forecast with safe fallback."""
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(y) < 3:
        last = float(y.iloc[-1]) if len(y) else 0.0
        return pd.Series([last] * periods, index=range(periods))
    if _HAS_STATSMODELS:
        try:
            use_season = seasonal_periods > 0 and len(y) >= (2 * seasonal_periods)
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

# ================================
# TITLE
# ================================
st.markdown('<div class="title">Metal Prices Dashboards</div>', unsafe_allow_html=True)

# ================================
# SCRAP / COMMODITY (with forecast)
# ================================
DEFAULT_START = pd.to_datetime("2025-04-01").date()
DEFAULT_END = pd.Timestamp.today().date()

config = load_config()
sheets: List[Dict] = config.get("sheets", []) if isinstance(config, dict) else config
(if_not := (not sheets)) and st.info("No published data yet.") or None
if if_not: st.stop()

labels = [s.get("label") or s.get("sheet") or s.get("slug") for s in sheets]
slugs = [s.get("slug") for s in sheets]

sel_label = st.selectbox("Commodity", labels, index=0)
slug = slugs[labels.index(sel_label)]

# Reset filters when commodity changes
if st.session_state.get("prev-slug") != slug:
    st.session_state["prev-slug"] = slug
    st.session_state[f"applied-from-{slug}"] = None
    st.session_state[f"applied-to-{slug}"] = None
    st.session_state[f"ver-{slug}"] = 0

df = load_sheet(slug)
if df.empty:
    st.warning("No data for this commodity."); st.stop()

# Make sure types are correct
df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df = df.dropna(subset=["Month", "Price"])
if df.empty:
    st.warning("No usable rows (date/price)."); st.stop()

min_d = pd.to_datetime(df["Month"].min()).date()
max_d = pd.to_datetime(df["Month"].max()).date()
def_start = max(min_d, DEFAULT_START)
def_end = min(max_d, DEFAULT_END)

k_from = f"applied-from-{slug}"
k_to = f"applied-to-{slug}"
if st.session_state.get(k_from) is None: st.session_state[k_from] = def_start
if st.session_state.get(k_to)   is None: st.session_state[k_to]   = def_end

ver_key = f"ver-{slug}"
st.session_state.setdefault(ver_key, 0)
ver = st.session_state[ver_key]
w_from = f"from-{slug}-{ver}"
w_to = f"to-{slug}-{ver}"

with st.form(key=f"filters-{slug}-{ver}", border=False):
    c1, c2, c3, c4 = st.columns([2.6, 2.6, 0.8, 0.8], gap="small")
    c1.date_input("From (dd-mm-yyyy)", st.session_state[k_from], key=w_from)
    c2.date_input("To (dd-mm-yyyy)",   st.session_state[k_to],   key=w_to)
    with c3:
        st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True)
        search = st.form_submit_button("Search")
    with c4:
        st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True)
        reset = st.form_submit_button("Reset")

meta = next((m for m in sheets if m.get("slug") == slug or m.get("sheet") == slug), {})
heading_text = meta.get("heading") or meta.get("label") or meta.get("sheet") or ""
st.markdown(f'<div class="sheet-headline">{escape(str(heading_text))}</div>', unsafe_allow_html=True)

if reset:
    st.session_state[k_from] = def_start
    st.session_state[k_to]   = def_end
    st.session_state[ver_key] = ver + 1
    st.rerun()

if search:
    start_val = st.session_state[w_from]
    end_val = st.session_state[w_to]
    if start_val > end_val:
        st.error("From date must be ≤ To date."); st.stop()
    st.session_state[k_from] = start_val
    st.session_state[k_to]   = end_val

start = st.session_state[k_from]
end = st.session_state[k_to]

mask = (df["Month"].dt.date >= start) & (df["Month"].dt.date <= end)
f = df.loc[mask].copy().sort_values("Month")
if f.empty:
    st.info("No rows in this range."); st.stop()

f["MonthLabel"] = f["Month"].dt.strftime("%b %y").str.upper()
f["PriceTT"] = f["Price"].apply(_fmt_tooltip_usd)

# -------- Forecast (next 3 months) --------
last_month = pd.to_datetime(f["Month"].max())
future_months = pd.date_range(last_month + pd.offsets.MonthBegin(1), periods=3, freq="MS")
scrap_fc = _forecast_series(f["Price"], periods=3, seasonal_periods=12)
scrap_fc_df = pd.DataFrame({
    "Month": future_months,
    "MonthLabel": future_months.strftime("%b %y").str.upper(),
    "Price": scrap_fc.values,
    "PriceTT": scrap_fc.apply(_fmt_tooltip_usd).values,
    "is_forecast": True,
})
f_act = f.copy(); f_act["is_forecast"] = False
plot_scrap = pd.concat([f_act, scrap_fc_df], ignore_index=True)

y_scale_scrap = _y_scale_from(plot_scrap["Price"])

bars_actual = (
    alt.Chart(plot_scrap[plot_scrap["is_forecast"] == False])
    .mark_bar(size=_bar_size(len(f)))
    .encode(
        x=alt.X("MonthLabel:N", title="Months", sort=None,
                scale=alt.Scale(paddingOuter=0.35, paddingInner=0.45)),
        y=alt.Y("Price:Q", title="Price", scale=y_scale_scrap),
        tooltip=[alt.Tooltip("MonthLabel:N", title="Month"),
                 alt.Tooltip("PriceTT:N", title="Price")],
    )
)

line_all = (
    alt.Chart(plot_scrap)
    .mark_line(point=True)
    .encode(
        x=alt.X("MonthLabel:N", sort=None,
                scale=alt.Scale(paddingOuter=0.35, paddingInner=0.45)),
        y=alt.Y("Price:Q", scale=y_scale_scrap),
        detail="is_forecast:N",
        strokeDash=alt.condition(alt.datum.is_forecast, alt.value([4,3]), alt.value([1,0])),
        tooltip=[alt.Tooltip("MonthLabel:N", title="Month"),
                 alt.Tooltip("PriceTT:N", title="Price")],
    )
)

chart = _tidy((bars_actual + line_all).properties(height=430))
st.altair_chart(chart, use_container_width=True)

# -------- Summary (actuals) --------
first_row = f.iloc[0]; last_row = f.iloc[-1]
start_price = float(first_row["Price"]); end_price = float(last_row["Price"])
start_mon = _fmt_mon(first_row["Month"]); end_mon = _fmt_mon(last_row["Month"])

chg_abs = end_price - start_price
chg_pct = (chg_abs / start_price * 100.0) if start_price not in (0, None) else 0.0
arrow = "↑" if chg_abs > 0 else ("↓" if chg_abs < 0 else "→")

hi_idx = f["Price"].idxmax(); lo_idx = f["Price"].idxmin()
hi_price, hi_mon = float(f.loc[hi_idx, "Price"]), _fmt_mon(f.loc[hi_idx, "Month"])
lo_price, lo_mon = float(f.loc[lo_idx, "Price"]), _fmt_mon(f.loc[lo_idx, "Month"])

avg_price = float(f["Price"].mean()); rng = hi_price - lo_price; n_months = len(f)

st.markdown(f"""
<div class="summary-box">
  <div><b>{start_mon}</b> to <b>{end_mon}</b>: price moved from <b>{_fmt_money(start_price)}</b> to <b>{_fmt_money(end_price)}</b> ({arrow} {chg_abs:+.2f}, {chg_pct:+.2f}%).</div>
  <div>Period high was <b>{_fmt_money(hi_price)}</b> in <b>{hi_mon}</b>; low was <b>{_fmt_money(lo_price)}</b> in <b>{lo_mon}</b> (range {_fmt_money(rng)}).</div>
  <div>Across <b>{n_months}</b> months, the average price was <b>{_fmt_money(avg_price)}</b>. These details auto-update with your date filters.</div>
</div>
""", unsafe_allow_html=True)

# -------- Forecast summary (Scrap) --------
scrap_fc_pairs = [f"{d.strftime('%b %Y')}: {_fmt_money(v)}" for d, v in zip(future_months, scrap_fc)]
st.markdown(
    f"""
<div class="summary-box">
  <div><b>Forecast (next 3 months)</b> — {', '.join(scrap_fc_pairs)}.</div>
  <div>Method: Holt–Winters (additive trend{', seasonal (12)' if _HAS_STATSMODELS and len(f)>=24 else ''}); fallback = last value.</div>
</div>
""",
    unsafe_allow_html=True,
)

# ================================
# BILLET PRICES (₹ tooltip + forecast)
# ================================
st.divider()
st.markdown('<div class="title">Billet Prices</div>', unsafe_allow_html=True)

BASE_DIR = Path(__file__).parent.resolve()
BILLET_FILE_CANDIDATES = [
    BASE_DIR / "data" / "Billet cost.xlsx",
    BASE_DIR / "data" / "current" / "Billet cost.xlsx",
    BASE_DIR / "Billet cost.xlsx",
]
BILLET_SERIES_LABELS = [
    "Billet Price (Blast Furnace Route)",
    "Billet Price (Electric Arc Furnace)",
]

def _find_billet_file() -> Path | None:
    for p in BILLET_FILE_CANDIDATES:
        if p.exists(): return p
    return None

def _resolve_sheet_name(xls: pd.ExcelFile, series_label: str) -> str:
    want_blast = "blast" in series_label.lower()
    for name in xls.sheet_names:
        n = name.lower()
        if want_blast and "blast" in n: return name
        if not want_blast and ("electric" in n or "arc" in n): return name
    return xls.sheet_names[0]

def _load_billet_df(series_label: str) -> pd.DataFrame:
    src = _find_billet_file()
    if not src:
        st.error("Billet Excel not found. Put **Billet cost.xlsx** in `data/` or `data/current/` (or next to this file)."); st.stop()
    try:
        xls = pd.ExcelFile(src)
    except Exception as e:
        st.error(f"Could not open {src.name}: {e}. Ensure `openpyxl` is in requirements.txt."); st.stop()

    sheet_name = _resolve_sheet_name(xls, series_label)
    raw = xls.parse(sheet_name=sheet_name)

    quarter_col = next((c for c in raw.columns if re.search(r"q(u)?arter|qurter", str(c), re.I)), None)
    price_col   = next((c for c in raw.columns if re.search(r"billet.*(per)?.*mt", str(c), re.I)), None)
    if quarter_col is None or price_col is None:
        st.error("Need a Quarter column and a 'Billet cost per MT' column."); st.stop()

    df0 = raw[[quarter_col, price_col]].rename(columns={quarter_col: "Quarter", price_col: "Price"})
    df0["Quarter"] = df0["Quarter"].astype(str).str.strip()
    df0["Price"] = pd.to_numeric(df0["Price"], errors="coerce")
    df0 = df0.dropna(subset=["Price"])

    def _q_order(qs: str) -> int:
        m = re.search(r"Q(\d)\s*[- ]\s*(\d{4})", qs, flags=re.I)
        if not m: return 0
        q = int(m.group(1)); y = int(m.group(2))
        return y * 10 + q

    df0["QuarterOrder"] = df0["Quarter"].apply(_q_order)
    df0 = df0.sort_values("QuarterOrder")
    df0["QuarterLabel"] = df0["Quarter"].str.replace("-", " ", regex=False)
    st.caption(f"Source: {src.name} • sheet: {sheet_name}")
    return df0

series_label = st.selectbox("Select Billet Series", BILLET_SERIES_LABELS, index=0, key="billet-series")
billet_df_full = _load_billet_df(series_label)
if billet_df_full.empty:
    st.info("No billet rows."); st.stop()

quarters = billet_df_full["QuarterLabel"].tolist()
q_start_def, q_end_def = quarters[0], quarters[-1]

kq_from = f"applied-billet-from-{series_label}"
kq_to   = f"applied-billet-to-{series_label}"
st.session_state.setdefault(kq_from, q_start_def)
st.session_state.setdefault(kq_to,   q_end_def)

kq_ver = f"billet-ver-{series_label}"
st.session_state.setdefault(kq_ver, 0)
ver2 = st.session_state[kq_ver]

wq_from = f"widget-billet-from-{series_label}-{ver2}"
wq_to   = f"widget-billet-to-{series_label}-{ver2}"

def _safe_idx(val: str, opts: list[str], fallback_idx: int) -> int:
    return opts.index(val) if val in opts else fallback_idx

idx_from = _safe_idx(st.session_state[kq_from], quarters, 0)
idx_to   = _safe_idx(st.session_state[kq_to],   quarters, len(quarters)-1)

with st.form(key=f"billet-form-{series_label}-{ver2}", border=False):
    c1, c2, c3, c4 = st.columns([2.6, 2.6, 0.8, 0.8], gap="small")
    c1.selectbox("From Quarter", options=quarters, index=idx_from, key=wq_from)
    c2.selectbox("To Quarter",   options=quarters, index=idx_to,   key=wq_to)
    with c3:
        st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True)
        btn_go = st.form_submit_button("Search")
    with c4:
        st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True)
        btn_reset = st.form_submit_button("Reset")

if btn_reset:
    st.session_state[kq_from] = q_start_def
    st.session_state[kq_to]   = q_end_def
    st.session_state[kq_ver]  = ver2 + 1
    st.rerun()

if btn_go:
    sel_from = st.session_state[wq_from]; sel_to = st.session_state[wq_to]
    if quarters.index(sel_from) > quarters.index(sel_to):
        st.error("From Quarter must be ≤ To Quarter."); st.stop()
    st.session_state[kq_from] = sel_from; st.session_state[kq_to] = sel_to

q_from = st.session_state[kq_from]; q_to = st.session_state[kq_to]
i_from = quarters.index(q_from); i_to = quarters.index(q_to)
billet_df = billet_df_full.iloc[i_from:i_to+1].copy()
if billet_df.empty:
    st.info("No rows in this quarter range."); st.stop()

billet_df["PriceTT"] = billet_df["Price"].apply(_fmt_inr)

# ---- Forecast (next 2 quarters) ----
def _next_quarter_labels(last_label: str, k: int) -> list[str]:
    m = re.search(r"Q(\d)\s+(\d{4})", last_label, flags=re.I)
    if not m: return [f"Q{i+1} +" for i in range(k)]
    q = int(m.group(1)); y = int(m.group(2)); out = []
    for _ in range(k):
        q += 1
        if q == 5: q = 1; y += 1
        out.append(f"Q{q} {y}")
    return out

billet_fc = _forecast_series(billet_df["Price"], periods=2, seasonal_periods=4)
future_quarters = _next_quarter_labels(billet_df["QuarterLabel"].iloc[-1], 2)

billet_fc_df = pd.DataFrame({
    "QuarterLabel": future_quarters,
    "Price": billet_fc.values,
    "PriceTT": billet_fc.apply(_fmt_inr).values,
    "is_forecast": True,
})
b_act = billet_df.copy(); b_act["is_forecast"] = False
billet_plot = pd.concat([b_act, billet_fc_df], ignore_index=True)

y_scale_billet = _y_scale_from(billet_plot["Price"])

bars2 = (
    alt.Chart(billet_plot[billet_plot["is_forecast"] == False])
    .mark_bar(size=28)
    .encode(
        x=alt.X("QuarterLabel:N", title="Quarter", sort=None,
                scale=alt.Scale(paddingOuter=0.35, paddingInner=0.45)),
        y=alt.Y("Price:Q", title="Billet cost per MT", scale=y_scale_billet),
        tooltip=[alt.Tooltip("QuarterLabel:N", title="Quarter"),
                 alt.Tooltip("PriceTT:N",      title="Price")],
    )
)

line2 = (
    alt.Chart(billet_plot)
    .mark_line(point=True)
    .encode(
        x=alt.X("QuarterLabel:N", sort=None,
                scale=alt.Scale(paddingOuter=0.35, paddingInner=0.45)),
        y=alt.Y("Price:Q", scale=y_scale_billet),
        detail="is_forecast:N",
        strokeDash=alt.condition(alt.datum.is_forecast, alt.value([4,3]), alt.value([1,0])),
        tooltip=[alt.Tooltip("QuarterLabel:N", title="Quarter"),
                 alt.Tooltip("PriceTT:N",      title="Price")],
    )
)

chart2 = _tidy((bars2 + line2).properties(height=430))
st.altair_chart(chart2, use_container_width=True)

# ---- Summary (actuals) ----
first_b = billet_df.iloc[0]; last_b = billet_df.iloc[-1]
b_start = float(first_b["Price"]); b_end = float(last_b["Price"])
b_arrow = "↑" if (b_end - b_start) > 0 else ("↓" if (b_end - b_start) < 0 else "→")
b_hi = billet_df.loc[billet_df["Price"].idxmax()]
b_lo = billet_df.loc[billet_df["Price"].idxmin()]
b_avg = float(billet_df["Price"].mean())

st.markdown(f"""
<div class="summary-box">
  <div><b>{first_b['QuarterLabel']}</b> to <b>{last_b['QuarterLabel']}</b>: price moved from <b>{b_start:,.0f}</b> to <b>{b_end:,.0f}</b> ({b_arrow} {b_end - b_start:+.0f}).</div>
  <div>High <b>{b_hi['QuarterLabel']}</b>: <b>{b_hi['Price']:,.0f}</b>; Low <b>{b_lo['QuarterLabel']}</b>: <b>{b_lo['Price']:,.0f}</b>.</div>
  <div>Across <b>{len(billet_df)}</b> quarters, average was <b>{b_avg:,.0f}</b>. Updates with your filters.</div>
</div>
""", unsafe_allow_html=True)

# ---- Forecast summary (billet) ----
billet_fc_pairs = [f"{lbl}: {_fmt_inr(val)}" for lbl, val in zip(future_quarters, billet_fc)]
st.markdown(
    f"""
<div class="summary-box">
  <div><b>Forecast (next 2 quarters)</b> — {', '.join(billet_fc_pairs)}.</div>
  <div>Method: Holt–Winters (additive trend{', seasonal (4)' if _HAS_STATSMODELS and len(billet_df)>=8 else ''}); fallback = last value.</div>
</div>
""",
    unsafe_allow_html=True,
)
