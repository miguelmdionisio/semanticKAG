"""Normalized CSV for morph-kgc: profile column names, dot decimals, ISO 8601 timestamps,
and a __row__ key column."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from .models import FileProfile
from .profiler import read_clean_df
from utils.uri_policy import ROW


def _parse_dt(series: pd.Series, fmt: str | None) -> pd.Series:
    if fmt and fmt != "AMBIGUOUS":
        return pd.to_datetime(series, format=fmt, errors="coerce")
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def normalize_csv(fp: FileProfile, out_path: Path, max_rows: int = 1000) -> bool:
    """Write the CSV; True if the timestamp column parsed as datetimes."""
    df, *_ = read_clean_df(Path(fp.path), max_rows=max_rows)

    ts_iso = False
    if fp.datetime_column and fp.datetime_column in df.columns:
        parsed = _parse_dt(df[fp.datetime_column], fp.datetime_format)
        if parsed.notna().any():
            df[fp.datetime_column] = parsed.apply(
                lambda t: t.isoformat() if pd.notna(t) else ""
            )
            ts_iso = True

    df.insert(0, ROW, range(len(df)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return ts_iso
