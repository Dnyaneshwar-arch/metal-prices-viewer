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
        # ---------------- Page 3: Grade Prices (uses Month text exactly as sheet) ----------------
def render_grade_prices_page():
    st.markdown('<div class="title">Grade Prices</div>', unsafe_allow_html=True)
    BASE_DIR = Path(__file__).parent.resolve()
    GRADE_FILE_CANDIDATES = [
        BASE_DIR / "data" / "Grade prices.xlsx",
        BASE_DIR / "data" / "current" / "Grade prices.xlsx",
        BASE_DIR / "Grade prices.xlsx",
    ]
    ]
