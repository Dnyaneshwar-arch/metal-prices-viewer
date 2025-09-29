from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Union, IO, Optional

import pandas as pd

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
CONFIG_PATH = DATA_DIR / "config.json"

EXCEL_CANDIDATES = [
    DATA_DIR / "metal prices.xlsx",
    DATA_DIR / "metal proices.xlsx",   # common typo
    DATA_DIR / "metal prices.xls",
]

def _slugify(name: str) -> str:
    s = re.sub(r"\s+", "-", str(name).strip())
    s = re.sub(r"[^A-Za-z0-9\-]+", "", s)
    return s.lower()

def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def _first_nonempty_in_row(row: pd.Series) -> str:
    for v in row.tolist():
        if pd.notna(v) and str(v).strip():
            return str(v).strip()
    return ""

def _find_excel() -> Optional[Path]:
    for p in EXCEL_CANDIDATES:
        if p.exists():
            return p
    return None

def _detect_month_price(df: pd.DataFrame) -> tuple[str, str]:
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    month_col = next((c for c in cols if c.lower() == "month"), cols[0])
    price_col = next(
        (c for c in cols if re.search(r"(price|rate|cost)", c, re.I)),
        (cols[1] if len(cols) > 1 else cols[0]),
    )
    return month_col, price_col

def _clean_df(df: pd.DataFrame, month_col: str, price_col: str) -> pd.DataFrame:
    out = df[[month_col, price_col]].rename(columns={month_col: "Month", price_col: "Price"})
    out["Month"] = pd.to_datetime(out["Month"], errors="coerce", dayfirst=True, infer_datetime_format=True)
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce")
    out = out.dropna(subset=["Month", "Price"]).sort_values("Month").reset_index(drop=True)
    return out

def _read_sheet_flex(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    # Try headers on row 2 (header=1), then row 1, then row 3
    for hdr in (1, 0, 2):
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name, header=hdr)
            if df.empty:
                continue
            mcol, pcol = _detect_month_price(df)
            return _clean_df(df, mcol, pcol)
        except Exception:
            continue
    # Fallback: treat first row as header
    df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
    if df.empty:
        return pd.DataFrame(columns=["Month", "Price"])
    df.columns = df.iloc[0].astype(str).tolist()
    df = df.iloc[1:]
    mcol, pcol = _detect_month_price(df)
    return _clean_df(df, mcol, pcol)

def publish_workbook(file_obj: Union[str, Path, IO[bytes]]) -> Dict:
    """
    Create one CSV per sheet (data/<slug>.csv) and data/config.json.
    Safe to run repeatedly (idempotent).
    """
    _ensure_data_dir()
    xls = pd.ExcelFile(file_obj)
    config: Dict[str, List[Dict]] = {"sheets": []}
    written: List[str] = []

    for sheet_name in xls.sheet_names:
        # A1 heading (for UI)
        try:
            top = pd.read_excel(xls, sheet_name=sheet_name, header=None, nrows=1)
            sheet_heading = _first_nonempty_in_row(top.iloc[0]) if not top.empty else ""
        except Exception:
            sheet_heading = ""

        df_clean = _read_sheet_flex(xls, sheet_name)
        slug = _slugify(sheet_name)
        (DATA_DIR / f"{slug}.csv").write_text(df_clean.to_csv(index=False), encoding="utf-8")
        written.append(f"{slug}.csv")

        config["sheets"].append({
            "sheet": sheet_name,
            "label": sheet_name,
            "slug": slug,
            "heading": sheet_heading,
        })

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return {"sheets_written": written, "config": CONFIG_PATH.name}

def _excel_slug_map(src: Path) -> Dict[str, str]:
    """sheet_name -> slug"""
    xls = pd.ExcelFile(src)
    return {sh: _slugify(sh) for sh in xls.sheet_names}

def _force_publish_if_excel_present():
    """
    If an Excel is present, always (re)publish CSVs + config.json so the UI matches the workbook.
    """
    src = _find_excel()
    if src:
        publish_workbook(src)

def load_config():
    """
    Always prefer the current Excel if present; else fall back to existing config.json.
    """
    _ensure_data_dir()
    src = _find_excel()
    if src:
        _force_publish_if_excel_present()
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"sheets": []}

def load_sheet(slug: str) -> pd.DataFrame:
    """
    Load data for a given slug from its CSV. If missing/mismatched, rebuild from Excel
    and try to find the right sheet by normalized name.
    """
    _ensure_data_dir()
    csv_path = DATA_DIR / f"{slug}.csv"
    if not csv_path.exists():
        # Try to rebuild from Excel and match slug to a sheet
        src = _find_excel()
        if src:
            publish_workbook(src)
            # Try direct again
            if not csv_path.exists():
                # Map normalized sheet names to slugs and try nearest match
                slugs = set(_excel_slug_map(src).values())
                # simple aliasing: replace dashes/underscores/spaces
                alt = slug.replace("_", "-").replace(" ", "-").lower()
                if alt in slugs:
                    csv_path = DATA_DIR / f"{alt}.csv"

    if not csv_path.exists():
        return pd.DataFrame(columns=["Month", "Price"])

    df = pd.read_csv(csv_path)
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Month", "Price"]).sort_values("Month").reset_index(drop=True)
    return df
