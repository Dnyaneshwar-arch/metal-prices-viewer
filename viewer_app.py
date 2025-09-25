from __future__ import annotations

from html import escape
from typing import List, Dict
from pathlib import Path
import re

import altair as alt
import pandas as pd
import streamlit as st

# Forecasting (safe fallback if not available)
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

      /* Sidebar look */
      .menu-title{ font-weight:800; font-size:16px; margin: 6px 0 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Helpers (shared) ----------------
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

# ---------------- Page 1: Metal Prices ----------------
def render_metal_prices_page():
    st.markdown('<div class="title">Metal Prices Dashboards</div>', unsafe_allow_html=True)

    DEFAULT_START = pd.to_datetime("2025-04-01").date()
    DEFAULT_END = pd.Timestamp.today().date()

    config = load_config()
    sheets: List[Dict] = config.get("sheets", []) if isinstance(config, dict) else config
    (if_not := (not sheets)) and st.info("No published data yet.") or None
    if if_not: st.stop()

    labels = [s.get("label") or s.get("sheet") or s.get("slug") for s in sheets]
    slugs = [s.get("slug") for s in sheets]

    sel_label = st.selectbox("Commodity", labels, index=0, key="commodity-select")

    # Force a clean redraw when commodity changes
    _last_sel = st.session_state.get("_last_sel")
    if _last_sel is None:
        st.session_state["_last_sel"] = sel_label
    elif sel_label != _last_sel:
        st.session_state["_last_sel"] = sel_label
        st.session_state["chart_ver_global"] = st.session_state.get("chart_ver_global", 0) + 1
        st.rerun()

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
        c1.date_input("From (DD/MM/YYYY)", def_start, key=w_from)
        c2.date_input("To (DD/MM/YYYY)", def_end, key=w_to)
        with c3:
            st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True)
            search = st.form_submit_button("Search")
        with c4:
            st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True)
            clear = st.form_submit_button("Clear")

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
            st.error("From date must be ≤ To date."); st.stop()
        st.session_state[k_from] = start_val
        st.session_state[k_to] = end_val

    start = st.session_state[k_from]
    end = st.session_state[k_to]

    mask = (df["Month"].dt.date >= start) & (df["Month"].dt.date <= end)
    f = df.loc[mask].copy()
    if f.empty:
        st.info("No rows in this range."); st.stop()

    # Plot data
    f = f.sort_values("Month")
    f["MonthLabel"] = pd.to_datetime(f["Month"]).dt.strftime("%b %y").str.upper()

    def _fmt_tooltip(v):
        try:
            v = float(v)
            return f"${v:,.2f}" if abs(v) < 10 else f"${v:,.0f}"
        except Exception:
            return str(v)

    f["PriceTT"] = f["Price"].apply(_fmt_tooltip)

    last_month = pd.to_datetime(f["Month"].max())
    future_months = pd.date_range(last_month + pd.offsets.MonthBegin(1), periods=3, freq="MS")
    scrap_fc = _forecast_series(f["Price"].reset_index(drop=True), periods=3, seasonal_periods=12)
    scrap_fc_df = pd.DataFrame({
        "Month": future_months,
        "MonthLabel": future_months.strftime("%b %y").str.upper(),
        "Price": scrap_fc.values,
        "PriceTT": scrap_fc.apply(_fmt_tooltip).values,
        "is_forecast": True,
    })

    domain_order = f["MonthLabel"].tolist() + [
        m for m in scrap_fc_df["MonthLabel"].tolist() if m not in set(f["MonthLabel"])
    ]

    f_act = f.copy()
    f_act["is_forecast"] = False
    plot_all = pd.concat([f_act, scrap_fc_df], ignore_index=True)

    actual_only = plot_all[plot_all["is_forecast"] == False]
    forecast_only = plot_all[plot_all["is_forecast"] == True]

    bars_actual = (
        alt.Chart(actual_only)
        .mark_bar(size=_bar_size(len(actual_only)))
        .encode(
            x=alt.X(
                "MonthLabel:N",
                title="Months",
                sort=domain_order,
                axis=alt.Axis(labelAngle=0),
                scale=alt.Scale(domain=domain_order, paddingOuter=0.35, paddingInner=0.45),
            ),
            y=alt.Y("Price:Q", title="Price", scale=alt.Scale(zero=False, nice=True)),
            tooltip=[alt.Tooltip("MonthLabel:N", title="Month"),
                     alt.Tooltip("PriceTT:N", title="Price")],
        )
    )

    line_forecast = (
        alt.Chart(forecast_only)
        .mark_line(point=True, strokeDash=[4, 3])
        .encode(
            x=alt.X("MonthLabel:N", sort=domain_order,
                    scale=alt.Scale(domain=domain_order, paddingOuter=0.35, paddingInner=0.45)),
            y=alt.Y("Price:Q", scale=alt.Scale(zero=False, nice=True)),
            tooltip=[alt.Tooltip("MonthLabel:N", title="Month"),
                     alt.Tooltip("PriceTT:N", title="Price")],
        )
    )

    line_actual = (
        alt.Chart(actual_only)
        .mark_line(point=True)
        .encode(
            x=alt.X("MonthLabel:N", sort=domain_order,
                    scale=alt.Scale(domain=domain_order, paddingOuter=0.35, paddingInner=0.45)),
            y=alt.Y("Price:Q", scale=alt.Scale(zero=False, nice=True)),
            tooltip=[alt.Tooltip("MonthLabel:N", title="Month"),
                     alt.Tooltip("PriceTT:N", title="Price")],
        )
    )

    global_ver = st.session_state.get("chart_ver_global", 0)
    scrap_chart_key = f"scrap-{slug}-{start.isoformat()}-{end.isoformat()}-{ver}-{global_ver}"

    placeholder = st.empty()
    chart = _tidy((bars_actual + line_forecast + line_actual).properties(height=430)).resolve_scale(x="shared", y="shared")
    placeholder.altair_chart(chart, use_container_width=True, key=scrap_chart_key)

    # Summary
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

# ---------------- Page 2: Billet Prices ----------------
def render_billet_prices_page():
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
            if p.exists():
                return p
        return None

    def _resolve_sheet_name(xls: pd.ExcelFile, series_label: str) -> str:
        want_blast = "blast" in series_label.lower()
        for name in xls.sheet_names:
            n = name.lower()
            if want_blast and "blast" in n:
                return name
            if not want_blast and ("electric" in n or "arc" in n):
                return name
        return xls.sheet_names[0]

    def _load_billet_df(series_label: str) -> pd.DataFrame:
        src = _find_billet_file()
        if not src:
            st.error("Billet Excel not found. Put **Billet cost.xlsx** in `data/` or `data/current/` (or next to this file).")
            st.stop()
        try:
            xls = pd.ExcelFile(src)
        except Exception as e:
            st.error(f"Could not open {src.name}: {e}. Ensure `openpyxl` is in requirements.txt.")
            st.stop()

        sheet_name = _resolve_sheet_name(xls, series_label)
        raw = xls.parse(sheet_name=sheet_name)

        quarter_col = next((c for c in raw.columns if re.search(r"q(u)?arter|qurter", str(c), re.I)), None)
        price_col   = next((c for c in raw.columns if re.search(r"billet.*(per)?.*mt", str(c), re.I)), None)
        if quarter_col is None or price_col is None:
            st.error("Could not detect columns. Need a Quarter column and a 'Billet cost per MT' column.")
            st.stop()

        df0 = raw[[quarter_col, price_col]].rename(columns={quarter_col: "Quarter", price_col: "Price"})
        df0["Quarter"] = df0["Quarter"].astype(str).str.strip()

        def _q_order(qs: str) -> int:
            m = re.search(r"Q(\d)\s*[- ]\s*(\d{4})", qs, flags=re.I)
            if not m: return 0
            q = int(m.group(1)); y = int(m.group(2))
            return y * 10 + q

        df0["QuarterOrder"] = df0["Quarter"].apply(_q_order)
        df0 = df0.sort_values("QuarterOrder")
        df0["QuarterLabel"] = df0["Quarter"].str.replace("-", " ", regex=False)
        df0["Price"] = pd.to_numeric(df0["Price"], errors="coerce")
        df0 = df0.dropna(subset=["Price"])
        st.caption(f"Source: {src.name} • sheet: {sheet_name}")
        return df0

    # Billet series select with rerender enforcement (and fresh mount)
    series_label = st.selectbox("Select Billet Series", BILLET_SERIES_LABELS, index=0, key="billet-series")
    _last_billet = st.session_state.get("_last_billet_series")
    if _last_billet is None:
        st.session_state["_last_billet_series"] = series_label
    elif series_label != _last_billet:
        st.session_state["_last_billet_series"] = series_label
        st.session_state["chart_ver_global"] = st.session_state.get("chart_ver_global", 0) + 1
        st.rerun()

    billet_df_full = _load_billet_df(series_label)
    if billet_df_full.empty:
        st.info("No billet rows."); st.stop()

    quarters     = billet_df_full["QuarterLabel"].tolist()
    q_start_def  = quarters[0]
    q_end_def    = quarters[-1]

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
        c1, c2, c3, c4 = st.columns([2.6, 2.6, 0.6, 0.6], gap="small")
        c1.selectbox("From Quarter", options=quarters, index=idx_from, key=wq_from)
        c2.selectbox("To Quarter",   options=quarters, index=idx_to,   key=wq_to)
        with c3:
            st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True)
            btn_go = st.form_submit_button("Search")
        with c4:
            st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True)
            btn_clear = st.form_submit_button("Clear")

    if btn_clear:
        st.session_state[kq_from] = q_start_def
        st.session_state[kq_to]   = q_end_def
        st.session_state[kq_ver]  = ver2 + 1
        st.rerun()

    if btn_go:
        sel_from = st.session_state[wq_from]
        sel_to   = st.session_state[wq_to]
        if quarters.index(sel_from) > quarters.index(sel_to):
            st.error("From Quarter must be ≤ To Quarter."); st.stop()
        st.session_state[kq_from] = sel_from
        st.session_state[kq_to]   = sel_to

    q_from = st.session_state[kq_from]
    q_to   = st.session_state[kq_to]
    i_from = quarters.index(q_from); i_to = quarters.index(q_to)
    billet_df = billet_df_full.iloc[i_from:i_to+1].copy()
    if billet_df.empty:
        st.info("No rows in this quarter range."); st.stop()

    def _fmt_inr(n: float) -> str:
        try:
            return f"₹{float(n):,.0f}"
        except Exception:
            return f"₹{n}"

    billet_df["PriceTT"] = billet_df["Price"].apply(_fmt_inr)

    # Forecast next 2 quarters
    billet_fc = _forecast_series(billet_df["Price"].reset_index(drop=True), periods=2, seasonal_periods=4)

    def _next_quarters(last_q_label: str, k: int = 2) -> list[str]:
        m = re.search(r"Q(\d)\s+(\d{4})", last_q_label)
        if not m: return []
        q = int(m.group(1)); y = int(m.group(2))
        out = []
        for _ in range(k):
            q = 1 if q == 4 else q + 1
            if q == 1: y += 1
            out.append(f"Q{q} {y}")
        return out

    future_quarters = _next_quarters(billet_df["QuarterLabel"].iloc[-1], 2)

    billet_fc_df = pd.DataFrame({
        "QuarterLabel": future_quarters,
        "Price": billet_fc.values,
        "PriceTT": billet_fc.apply(_fmt_inr).values,
        "is_forecast": True,
    })

    b_act = billet_df.copy()
    b_act["is_forecast"] = False
    plot_all = pd.concat([b_act, billet_fc_df], ignore_index=True)

    actual_only = plot_all[plot_all["is_forecast"] == False]
    forecast_only = plot_all[plot_all["is_forecast"] == True]   # ✅ fixed filtering

    domain_order_q = billet_df["QuarterLabel"].tolist() + [q for q in future_quarters if q not in set(billet_df["QuarterLabel"])]

    bars2 = (
        alt.Chart(actual_only)
        .mark_bar(size=_bar_size(len(actual_only)))
        .encode(
            x=alt.X(
                "QuarterLabel:N",
                title="Quarter",
                sort=domain_order_q,
                axis=alt.Axis(labelAngle=0),
                scale=alt.Scale(domain=domain_order_q, paddingOuter=0.35, paddingInner=0.45),
            ),
            y=alt.Y("Price:Q", title="Billet cost per MT", scale=alt.Scale(zero=False, nice=True)),
            tooltip=[alt.Tooltip("QuarterLabel:N", title="Quarter"),
                     alt.Tooltip("PriceTT:N",      title="Price")],
        )
    )

    line2_forecast = (
        alt.Chart(forecast_only)
        .mark_line(point=True, strokeDash=[4, 3])
        .encode(
            x=alt.X("QuarterLabel:N", sort=domain_order_q,
                    scale=alt.Scale(domain=domain_order_q, paddingOuter=0.35, paddingInner=0.45)),
            y=alt.Y("Price:Q", scale=alt.Scale(zero=False, nice=True)),
            tooltip=[alt.Tooltip("QuarterLabel:N", title="Quarter"),
                     alt.Tooltip("PriceTT:N",      title="Price")],
        )
    )

    line2_actual = (
        alt.Chart(actual_only)
        .mark_line(point=True)
        .encode(
            x=alt.X("QuarterLabel:N", sort=domain_order_q,
                    scale=alt.Scale(domain=domain_order_q, paddingOuter=0.35, paddingInner=0.45)),
            y=alt.Y("Price:Q", scale=alt.Scale(zero=False, nice=True)),
            tooltip=[alt.Tooltip("QuarterLabel:N", title="Quarter"),
                     alt.Tooltip("PriceTT:N",      title="Price")],
        )
    )

    global_ver = st.session_state.get("chart_ver_global", 0)
    billet_chart_key = f"billet-{series_label}-{q_from}-{q_to}-{ver2}-{global_ver}"

    # Fresh mount to avoid stale Vega view (fixes “updates only on hover”)
    placeholder = st.empty()
    chart2 = _tidy((bars2 + line2_forecast + line2_actual).properties(height=430)).resolve_scale(x="shared", y="shared")
    placeholder.altair_chart(chart2, use_container_width=True, key=billet_chart_key)

    # -------- Billet Summary --------
    def _fmt_inr_money(x: float) -> str:
        try:
            return f"₹{x:,.0f}"
        except Exception:
            return f"₹{x}"

    first_row = billet_df.iloc[0]
    last_row = billet_df.iloc[-1]
    start_price = float(first_row["Price"])
    end_price = float(last_row["Price"])
    start_q = first_row["QuarterLabel"]
    end_q = last_row["QuarterLabel"]

    chg_abs = end_price - start_price
    chg_pct = (chg_abs / start_price * 100.0) if start_price not in (0, None) else 0.0
    arrow = "↑" if chg_abs > 0 else ("↓" if chg_abs < 0 else "→")

    hi_idx = billet_df["Price"].idxmax()
    lo_idx = billet_df["Price"].idxmin()
    hi_price, hi_q = float(billet_df.loc[hi_idx, "Price"]), billet_df.loc[hi_idx, "QuarterLabel"]
    lo_price, lo_q = float(billet_df.loc[lo_idx, "Price"]), billet_df.loc[lo_idx, "QuarterLabel"]

    avg_price = float(billet_df["Price"].mean())
    rng = hi_price - lo_price
    n_rows = len(billet_df)

    summary_html_b = f"""
    <div class="summary-box">
      <div><b>{start_q}</b> to <b>{end_q}</b>: price moved from <b>{_fmt_inr_money(start_price)}</b> to <b>{_fmt_inr_money(end_price)}</b> ({arrow} {chg_abs:+.0f}, {chg_pct:+.2f}%).</div>
      <div>Period high was <b>{_fmt_inr_money(hi_price)}</b> in <b>{hi_q}</b>; low was <b>{_fmt_inr_money(lo_price)}</b> in <b>{lo_q}</b> (range {_fmt_inr_money(rng)}).</div>
      <div>Across <b>{n_rows}</b> quarters, the average price was <b>{_fmt_inr_money(avg_price)}</b>. These details auto-update with your quarter filters.</div>
    </div>
    """
    st.markdown(summary_html_b, unsafe_allow_html=True)

    billet_fc_pairs = [f"{q}: {_fmt_inr_money(v)}" for q, v in zip(future_quarters, billet_fc)]
    st.markdown(
        f"""
    <div class="summary-box">
      <div><b>Forecast (next 2 quarters)</b> — {', '.join(billet_fc_pairs)}.</div>
      <div>Method: Holt–Winters (additive trend{', seasonal (4)' if _HAS_STATSMODELS and len(billet_df)>=8 else ''}); fallback = last value.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ---------------- Page 3: Grade Prices (uses Month text exactly as sheet) ----------------
def render_grade_prices_page():
    st.markdown('<div class="title">Grade Prices</div>', unsafe_allow_html=True)

    BASE_DIR = Path(__file__).parent.resolve()
    GRADE_FILE_CANDIDATES = [
        BASE_DIR / "data" / "Grade prices.xlsx",
        BASE_DIR / "data" / "current" / "Grade prices.xlsx",
        BASE_DIR / "Grade prices.xlsx",
    ]

    def _find_grade_file() -> Path | None:
        for p in GRADE_FILE_CANDIDATES:
            if p.exists():
                return p
        return None

    src = _find_grade_file()
    if not src:
        st.error("Grade Excel not found. Put **Grade prices.xlsx** in `data/` or `data/current/` (or next to this file).")
        st.stop()

    try:
        xls = pd.ExcelFile(src)
    except Exception as e:
        st.error(f"Could not open {src.name}: {e}. Ensure `openpyxl` is in requirements.txt.")
        st.stop()

    # 1) Build Grade dropdown from A1 of each sheet
    sheet_labels: Dict[str, str] = {}
    for sh in xls.sheet_names:
        tmp = xls.parse(sheet_name=sh, header=None, nrows=1, usecols="A")
        val = tmp.iloc[0, 0]
        sheet_labels[sh] = str(val).strip() if pd.notna(val) else sh

    label_list = [sheet_labels[s] for s in xls.sheet_names]
    sel_label = st.selectbox("Grade", label_list, index=0, key="grade-select")

    # Force clean redraw when grade changes
    _last_grade = st.session_state.get("_last_grade_sel")
    if _last_grade is None:
        st.session_state["_last_grade_sel"] = sel_label
    elif sel_label != _last_grade:
        st.session_state["_last_grade_sel"] = sel_label
        st.session_state["chart_ver_global"] = st.session_state.get("chart_ver_global", 0) + 1
        st.rerun()

    sel_sheet = xls.sheet_names[label_list.index(sel_label)]
    st.caption(f"Source: {src.name} • sheet: {sel_sheet} • A1: {sel_label}")

    # 2) Find the table and detect Month/Price columns
    #    Month detection prefers header containing 'month' OR values matching month-name patterns.
    MONTH_PAT = re.compile(
        r"^\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[\s\-_/]*\d{2,4}\s*$",
        flags=re.I,
    )

    def _load_grade_df(sheet: str) -> pd.DataFrame:
        for hdr in (0, 1, 2):
            raw = xls.parse(sheet_name=sheet, header=hdr)
            if raw.empty:
                continue

            # drop totally empty cols/rows, normalize headers
            raw = raw.dropna(how="all").dropna(axis=1, how="all")
            raw.columns = [str(c).strip() for c in raw.columns]

            # choose Month column
            month_col = None
            # (a) By header name
            for c in raw.columns:
                if re.search(r"\bmonth\b", c, re.I):
                    month_col = c
                    break
            # (b) By value pattern
            if month_col is None:
                for c in raw.columns:
                    s = raw[c].astype(str)
                    if (s.apply(lambda x: bool(MONTH_PAT.match(x))).mean() > 0.6):
                        month_col = c
                        break
            if month_col is None:
                continue  # try next header row

            # choose Price column
            price_col = None
            for c in raw.columns:
                if c == month_col:
                    continue
                if re.search(r"(price|rate|cost)", c, re.I):
                    price_col = c
                    break
            if price_col is None:
                # pick the most numeric column (not the month col)
                numeric_counts = {}
                for c in raw.columns:
                    if c == month_col:
                        continue
                    s = pd.to_numeric(raw[c], errors="coerce")
                    numeric_counts[c] = s.notna().sum()
                if numeric_counts:
                    price_col = max(numeric_counts, key=numeric_counts.get)

            if price_col is None:
                continue  # try next header row

            df = raw[[month_col, price_col]].copy()
            df.columns = ["MonthText", "Price"]
            # keep Month text EXACTLY as in sheet
            df["MonthText"] = df["MonthText"].astype(str).str.strip()
            df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
            df = df.dropna(subset=["MonthText", "Price"])

            # sort by inferred month order if possible, else keep original order
            parsed = pd.to_datetime(df["MonthText"], errors="coerce", dayfirst=True, infer_datetime_format=True)
            if parsed.notna().any():
                df["_sort"] = parsed
            else:
                df["_sort"] = range(len(df))
            df = df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)

            return df

        raise ValueError("Could not detect a Month and Price column. Please ensure the sheet has a Month column (e.g., 'Jan 24') and a numeric Price column.")

    try:
        df_full = _load_grade_df(sel_sheet)
    except Exception as e:
        st.error(str(e)); st.stop()

    # 3) From/To dropdowns using MonthText EXACTLY as sheet
    months = df_full["MonthText"].tolist()
    m_start_def, m_end_def = months[0], months[-1]

    k_from = f"grade-from-{sel_sheet}"
    k_to   = f"grade-to-{sel_sheet}"
    st.session_state.setdefault(k_from, m_start_def)
    st.session_state.setdefault(k_to,   m_end_def)

    ver_key = f"grade-ver-{sel_sheet}"
    st.session_state.setdefault(ver_key, 0)
    ver = st.session_state[ver_key]

    w_from = f"grade-w-from-{sel_sheet}-{ver}"
    w_to   = f"grade-w-to-{sel_sheet}-{ver}"

    with st.form(key=f"grade-form-{sel_sheet}-{ver}", border=False):
        c1, c2, c3, c4 = st.columns([2.6, 2.6, 0.6, 0.6], gap="small")
        c1.selectbox("From (Month)", options=months, index=months.index(st.session_state[k_from]), key=w_from)
        c2.selectbox("To (Month)",   options=months, index=months.index(st.session_state[k_to]),   key=w_to)
        with c3:
            st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True)
            btn_go = st.form_submit_button("Search")
        with c4:
            st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True)
            btn_clear = st.form_submit_button("Clear")

    if btn_clear:
        st.session_state[k_from] = m_start_def
        st.session_state[k_to]   = m_end_def
        st.session_state[ver_key] = ver + 1
        st.rerun()

    if btn_go:
        sf = st.session_state[w_from]; stv = st.session_state[w_to]
        if months.index(sf) > months.index(stv):
            st.error("From (Month) must be ≤ To (Month)."); st.stop()
        st.session_state[k_from] = sf; st.session_state[k_to] = stv

    m_from = st.session_state[k_from]; m_to = st.session_state[k_to]
    i_from = months.index(m_from); i_to = months.index(m_to)
    df = df_full.iloc[i_from:i_to+1].copy()
    if df.empty:
        st.info("No rows in this month range."); st.stop()

    # Tooltips/labels
    def _fmt_inr(v):
        try:
            return f"₹{float(v):,.0f}"
        except Exception:
            return str(v)

    df["PriceTT"] = df["Price"].apply(_fmt_inr)

    # Forecast next 3 months (labels try to continue the text pattern, else "Next 1/2/3")
    parsed = pd.to_datetime(df["MonthText"], errors="coerce", dayfirst=True, infer_datetime_format=True)
    fc_vals = _forecast_series(df["Price"].reset_index(drop=True), periods=3, seasonal_periods=12)

    if parsed.notna().all():
        last_dt = parsed.iloc[-1]
        future_dt = pd.date_range(last_dt + pd.offsets.MonthBegin(1), periods=3, freq="MS")
        future_labels = [d.strftime("%b %y").upper() for d in future_dt]
    else:
        # fallback: keep simple increasing labels
        future_labels = [f"Next {i}" for i in (1, 2, 3)]

    fc_df = pd.DataFrame({
        "MonthText": future_labels,
        "Price": fc_vals.values,
        "PriceTT": fc_vals.apply(_fmt_inr).values,
        "is_forecast": True,
    })

    act = df.copy(); act["is_forecast"] = False
    plot_all = pd.concat([act, fc_df], ignore_index=True)

    domain = df["MonthText"].tolist() + [m for m in fc_df["MonthText"].tolist() if m not in set(df["MonthText"])]

    bars = (
        alt.Chart(plot_all[plot_all["is_forecast"] == False])
        .mark_bar(size=_bar_size(len(act)))
        .encode(
            x=alt.X("MonthText:N", title="Months", sort=domain,
                    scale=alt.Scale(domain=domain, paddingOuter=0.35, paddingInner=0.45)),
            y=alt.Y("Price:Q", title="Price", scale=alt.Scale(zero=False, nice=True)),
            tooltip=[alt.Tooltip("MonthText:N", title="Month"),
                     alt.Tooltip("PriceTT:N", title="Price")],
        )
    )
    line_fc = (
        alt.Chart(plot_all[plot_all["is_forecast"] == True])
        .mark_line(point=True, strokeDash=[4,3])
        .encode(
            x=alt.X("MonthText:N", sort=domain,
                    scale=alt.Scale(domain=domain, paddingOuter=0.35, paddingInner=0.45)),
            y=alt.Y("Price:Q", scale=alt.Scale(zero=False, nice=True)),
            tooltip=[alt.Tooltip("MonthText:N", title="Month"),
                     alt.Tooltip("PriceTT:N", title="Price")],
        )
    )
    line_actual = (
        alt.Chart(plot_all[plot_all["is_forecast"] == False])
        .mark_line(point=True)
        .encode(
            x=alt.X("MonthText:N", sort=domain,
                    scale=alt.Scale(domain=domain, paddingOuter=0.35, paddingInner=0.45)),
            y=alt.Y("Price:Q", scale=alt.Scale(zero=False, nice=True)),
            tooltip=[alt.Tooltip("MonthText:N", title="Month"),
                     alt.Tooltip("PriceTT:N", title="Price")],
        )
    )

    global_ver = st.session_state.get("chart_ver_global", 0)
    chart_key = f"grade-{sel_sheet}-{m_from}-{m_to}-{ver}-{global_ver}"
    placeholder = st.empty()
    chart = _tidy((bars + line_fc + line_actual).properties(height=430)).resolve_scale(x="shared", y="shared")
    placeholder.altair_chart(chart, use_container_width=True, key=chart_key)

    # Summary box (uses Month text exactly)
    def _fmt_inr_money(x: float) -> str:
        try:
            return f"₹{x:,.0f}"
        except Exception:
            return str(x)

    first_row = df.iloc[0]; last_row = df.iloc[-1]
    s_price = float(first_row["Price"]); e_price = float(last_row["Price"])
    s_mon = first_row["MonthText"]; e_mon = last_row["MonthText"]
    chg_abs = e_price - s_price
    chg_pct = (chg_abs / s_price * 100.0) if s_price else 0.0
    arrow = "↑" if chg_abs > 0 else ("↓" if chg_abs < 0 else "→")
    hi_idx = df["Price"].idxmax(); lo_idx = df["Price"].idxmin()
    hi_price, hi_mon = float(df.loc[hi_idx, "Price"]), df.loc[hi_idx, "MonthText"]
    lo_price, lo_mon = float(df.loc[lo_idx, "Price"]), df.loc[lo_idx, "MonthText"]
    avg_price = float(df["Price"].mean()); rng = hi_price - lo_price; n = len(df)

    st.markdown(
        f"""
        <div class="summary-box">
          <div><b>{s_mon}</b> to <b>{e_mon}</b>: price moved from <b>{_fmt_inr_money(s_price)}</b> to <b>{_fmt_inr_money(e_price)}</b> ({arrow} {chg_abs:+.0f}, {chg_pct:+.2f}%).</div>
          <div>Period high was <b>{_fmt_inr_money(hi_price)}</b> in <b>{hi_mon}</b>; low was <b>{_fmt_inr_money(lo_price)}</b> in <b>{lo_mon}</b> (range {_fmt_inr_money(rng)}).</div>
          <div>Across <b>{n}</b> months, the average price was <b>{_fmt_inr_money(avg_price)}</b>. These details auto-update with your month filters.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pairs = [f"{lbl}: {_fmt_inr_money(v)}" for lbl, v in zip(future_labels, fc_vals)]
    st.markdown(
        f"""
        <div class="summary-box">
          <div><b>Forecast (next 3 months)</b> — {', '.join(pairs)}.</div>
          <div>Method: Holt–Winters (additive trend{', seasonal (12)' if _HAS_STATSMODELS and len(df)>=24 else ''}); fallback = last value.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- Sidebar Navigation ----------------
st.sidebar.markdown('<div class="menu-title">☰ Menu</div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Go to",
    ["Metal Prices", "Billet Prices", "Grade Prices"],
    index=0,
    key="nav-radio",
    label_visibility="collapsed",
)

if page == "Metal Prices":
    render_metal_prices_page()
elif page == "Billet Prices":
    render_billet_prices_page()
else:
    render_grade_prices_page()
