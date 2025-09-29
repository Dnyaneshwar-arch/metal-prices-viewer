from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Union, IO, Optional

import pandas as pd

# Use a robust base so it works whether you run from repo root or not
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
CONFIG_PATH = DATA_DIR / "config.json"

# Support both correct name and common typo
EXCEL_CANDIDATES = [
    DATA_DIR / "metal prices.xlsx",
    DATA_DIR / "metal proices.xlsx",   # fallback if name is misspelled
    DATA_DIR / "metal prices.xls",
]

def _slugify(name: str) -> str:
    s = re.sub(r"\s+", "-", str(name).strip())
    s = re.sub(r"[^A-Za-z0-9\\-]+", "", s)
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

def publish_workbook(file_obj: Union[str, Path, IO[bytes]]) -> Dict:
    """
    Reads the Excel workbook and writes:
      - one CSV per sheet -> data/<slug>.csv
      - a config.json listing sheets, labels, and slugs
    Assumes row 1 is headings (A1 capture), row 2 has table headers.
    """
    _ensure_data_dir()

    xls = pd.ExcelFile(file_obj)
    config: Dict[str, List[Dict]] = {"sheets": []}
    written = []

    for sheet_name in xls.sheet_names:
        # Read top row to capture heading (A1 or first non-empty in row 1)
        try:
            top = pd.read_excel(xls, sheet_name=sheet_name, header=None, nrows=1)
            sheet_heading = _first_nonempty_in_row(top.iloc[0]) if not top.empty else ""
        except Exception:
            sheet_heading = ""

        # Data starts on row 2 (header row)
        df = pd.read_excel(xls, sheet_name=sheet_name, header=1)

        cols = [str(c).strip() for c in df.columns]
        df.columns = cols

        # Detect Month/Price columns robustly
        month_col = next((c for c in df.columns if c.lower().strip() == "month"), df.columns[0])
        price_col = next(
            (c for c in df.columns if re.search(r"(price|rate|cost)", str(c), re.I)),
            (df.columns[1] if len(df.columns) > 1 else df.columns[0]),
        )

        df = df[[month_col, price_col]].rename(columns={month_col: "Month", price_col: "Price"})
        df["Month"] = pd.to_datetime(df["Month"], errors="coerce", dayfirst=True, infer_datetime_format=True)
        df = df.dropna(subset=["Month"])
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df = df.dropna(subset=["Price"])
        df = df.sort_values("Month").reset_index(drop=True)

        slug = _slugify(sheet_name)
        (DATA_DIR / f"{slug}.csv").write_text(df.to_csv(index=False), encoding="utf-8")

        config["sheets"].append(
            {
                "sheet": sheet_name,
                "label": sheet_name,
                "slug": slug,
                "heading": sheet_heading,  # saved for UI
            }
        )
        written.append(f"{slug}.csv")

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return {"sheets_written": written, "config": CONFIG_PATH.name}

def _ensure_published():
    """
    If config.json (and CSVs) are not present, try to publish from the Excel.
    """
    if CONFIG_PATH.exists():
        return
    src = _find_excel()
    if src:
        publish_workbook(src)

def load_config():
    """
    Preferred flow: read config.json created by publish_workbook().
    If it doesn't exist yet, auto-publish from the Excel (if present) and return the config.
    """
    _ensure_data_dir()
    if not CONFIG_PATH.exists():
        _ensure_published()
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # Nothing available
    return {"sheets": []}

def load_sheet(slug: str) -> pd.DataFrame:
    """
    Load data for a given slug from its CSV.
    If the CSV is missing (e.g., new Excel uploaded), try to (re)publish from Excel.
    """
    _ensure_data_dir()
    p = DATA_DIR / f"{slug}.csv"
    if not p.exists():
        # Try re-publish from Excel (maybe added/renamed)
        _ensure_published()
    if not p.exists():
        return pd.DataFrame(columns=["Month", "Price"])

    df = pd.read_csv(p)
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Month", "Price"]).sort_values("Month").reset_index(drop=True)
    return df
