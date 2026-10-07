from __future__ import annotations
import re
from pathlib import Path
from typing import Optional
 
import pandas as pd
 
from .models import ColumnProfile, FileProfile, SSNRole
 
import warnings

# candidate datetime column names; format detection confirms
DATETIME_NAME_PATTERNS = [
    re.compile(r"\btime\b", re.I),
    re.compile(r"\bdatetime\b", re.I),
    re.compile(r"\btimestamp\b", re.I),
    re.compile(r"\bdate\b", re.I),
    re.compile(r"\bdato\b", re.I),       # danish
    re.compile(r"\btid\b", re.I),        # danish
]

_DATETIME_FORMATS = [
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%d.%m.%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
    "%d/%m/%Y %H:%M",
    "%Y-%m-%d",
]

def _detect_datetime_format(series: pd.Series, n_sample: int = 20) -> Optional[str]:
    sample = series.dropna().astype(str).head(n_sample)
    if sample.empty:
        return None
    for fmt in _DATETIME_FORMATS:
        try:
            pd.to_datetime(sample, format=fmt)   # raises if any value fails
            return fmt
        except (ValueError, TypeError):
            continue
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pd.to_datetime(sample, dayfirst=True)
        return "AMBIGUOUS"
    except (ValueError, TypeError):
        return None
 
_DECIMAL_COMMA_RE = re.compile(r"^-?\d+(,\d+)?$")

ISO8601_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?([Zz]|[+-]\d{2}:?\d{2})?$"
)
TZ_OFFSET_RE = re.compile(r"[+-]\d{2}:?\d{2}$")
# e.g. "0 days 12:31:00"
TIMEDELTA_RE = re.compile(r"^-?\d+\s+days?\b")


def determine_sosa_role(col: ColumnProfile, file_profile: FileProfile) -> SSNRole:
    if col.name == file_profile.datetime_column:
        return SSNRole.TIMESTAMP
    if col.name.startswith("Unnamed"):
        return SSNRole.ROW_INDEX
    # secondary timestamps and durations are not measurements
    vals = col.sample_values
    if vals and all(ISO8601_RE.match(v) for v in vals):
        return SSNRole.AUXILIARY
    if vals and all(TIMEDELTA_RE.match(v) for v in vals):
        return SSNRole.AUXILIARY
    # word boundaries, so a name like "Flagey" does not match "flag"
    quality_kw = re.compile(r"\b(flag|flags|quality|qc|valid|invalid|status|error)\b", re.I)
    if col.dtype == "bool" or quality_kw.search(col.name):
        return SSNRole.QUALITY_FLAG
    if col.dtype in ("float64", "float32", "int64", "int32"):
        return SSNRole.OBSERVABLE_PROP
    return SSNRole.UNKNOWN

def _detect_delimiter(path: Path, n_lines: int = 5) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = []
        for line in f:
            line = line.rstrip("\r\n")
            if line:
                lines.append(line)
            if len(lines) >= n_lines:
                break

    if not lines:
        return ","

    for sep in ["\t", ";", ","]:
        counts = {line.count(sep) for line in lines}
        if len(counts) == 1 and counts.pop() >= 1:
            return sep

    best = max(["\t", ";", ","], key=lambda s: min(line.count(s) for line in lines))
    return best if min(line.count(best) for line in lines) > 0 else ","

def _coerce_numeric_objects(df: pd.DataFrame, decimal: str) -> None:
    """Convert mostly numeric object columns; sentinel strings such as 'OL' become NaN."""
    num_re = (
        re.compile(r"^-?\d+(?:,\d+)?$") if decimal == ","
        else re.compile(r"^-?\d+(?:\.\d+)?([eE][+-]?\d+)?$")
    )
    for col in df.columns:
        if df[col].dtype != object:
            continue
        s = df[col].dropna().astype(str)
        if s.empty:
            continue
        if s.head(500).str.match(num_re).mean() >= 0.9:
            cleaned = df[col].astype(str)
            if decimal == ",":
                cleaned = cleaned.str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(cleaned, errors="coerce")


def _has_decimal_commas(df: pd.DataFrame, sample_rows: int = 50) -> bool:
    for col in df.columns:
        if df[col].dtype != object:
            continue
        sample = df[col].dropna().astype(str).head(sample_rows)
        if sample.empty:
            continue
        hits = sample.str.match(_DECIMAL_COMMA_RE).mean()
        if hits >= 0.9 and sample.str.contains(",", regex=False).any():
            return True
    return False

def _detect_header_rows(path: Path, delimiter: str, encoding: str) -> list[int]:
    """[0, 1] for a two-row header (empty first cell, second line text, third line data), else [0]."""
    with open(path, encoding=encoding, errors="replace") as f:
        lines = [f.readline().rstrip("\r\n") for _ in range(3)]
    if len([l for l in lines if l]) < 3:
        return [0]
    first_cells = [line.split(delimiter)[0].strip() for line in lines]
    looks_numeric_or_date = re.compile(r"^\d")
    if (
        first_cells[0] == ""
        and not looks_numeric_or_date.match(first_cells[1])
        and looks_numeric_or_date.match(first_cells[2])
    ):
        return [0, 1]
    return [0]


def _flatten_columns(df: pd.DataFrame) -> None:
    if not isinstance(df.columns, pd.MultiIndex):
        return
    df.columns = [
        " ".join(
            str(level) for level in parts
            if str(level).strip() and not str(level).startswith("Unnamed")
        )
        for parts in df.columns
    ]


def _detect_encoding(path: Path) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            f.read(4096)
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"
    
def _is_datetime_column(col_name: str, series: pd.Series) -> bool:
    name_match = any(p.search(col_name) for p in DATETIME_NAME_PATTERNS)
    sample = series.dropna().astype(str).head(5)
    if sample.empty:
        return False
    value_match = all(ISO8601_RE.match(v) for v in sample)
    if not (name_match or value_match):
        return False
    return _detect_datetime_format(series) is not None

def _detect_tz(series: pd.Series) -> Optional[str]:
    sample = series.dropna().head(20).astype(str)
    for val in sample:
        m = TZ_OFFSET_RE.search(val)
        if m:
            return m.group(0)
    return None

def _detect_frequency(dt_series: pd.Series) -> Optional[str]:
    fmt = _detect_datetime_format(dt_series)
    if fmt is None:
        return None
    kwargs = {"format": fmt} if fmt != "AMBIGUOUS" else {"dayfirst": True}
    parsed = pd.to_datetime(dt_series.dropna().astype(str).head(100), **kwargs)
    diffs = parsed.diff().dropna()
    if diffs.empty:
        return None
    return str(diffs.mode().iloc[0])

def read_clean_df(path: Path, max_rows: int = 100_000) -> tuple[pd.DataFrame, str, str, str]:
    """(df, delimiter, encoding, decimal). Shared with normalize so column names match."""
    delimiter = _detect_delimiter(path)
    encoding = _detect_encoding(path)
    header_rows = _detect_header_rows(path, delimiter, encoding)
    decimal = "."
    df = pd.read_csv(path, sep=delimiter, encoding=encoding, header=header_rows,
                     nrows=max_rows, low_memory=False)
    _flatten_columns(df)
    if _has_decimal_commas(df):
        decimal = ","
        df = pd.read_csv(path, sep=delimiter, encoding=encoding, header=header_rows,
                         decimal=",", nrows=max_rows, low_memory=False)
        _flatten_columns(df)
    _coerce_numeric_objects(df, decimal)
    return df, delimiter, encoding, decimal


def profile_csv(path: Path, sample_rows: int = 5, max_rows: int = 100_000) -> FileProfile:

    df_full, delimiter, encoding, decimal = read_clean_df(path, max_rows=max_rows)
    df_sample = df_full.head(sample_rows)

    row_count = len(df_full)
    row_count_sampled = row_count == max_rows
    datetime_col: Optional[str] = None
    datetime_fmt: Optional[str] = None
    timezone: Optional[str] = None
    frequency: Optional[str] = None

    columns: list[ColumnProfile] = []

    for col in df_full.columns:
        series = df_full[col]
        is_dt = _is_datetime_column(col, series)

        col_min: Optional[float] = None
        col_max: Optional[float] = None
        dtype_str = str(series.dtype)

        if is_dt and datetime_col is None:
            datetime_col = col
            datetime_fmt = _detect_datetime_format(series)
            timezone = _detect_tz(series)
            frequency = _detect_frequency(series)

        if pd.api.types.is_numeric_dtype(series) and not is_dt:
            col_min = float(series.min()) if not series.isna().all() else None
            col_max = float(series.max()) if not series.isna().all() else None
 
        null_pct = round(series.isna().mean(), 4)
 
        sample_vals = (
            df_sample[col]
            .dropna()
            .astype(str)
            .head(sample_rows)
            .tolist()
        )
 
        columns.append(ColumnProfile(
            name=col,
            dtype=dtype_str,
            null_pct=null_pct,
            min=col_min,
            max=col_max,
            sample_values=sample_vals,
        ))
 
    fp = FileProfile(
        path=str(path),
        row_count=row_count,
        row_count_sampled=row_count_sampled,
        delimiter=delimiter,
        decimal=decimal,
        encoding=encoding,
        datetime_column=datetime_col,
        datetime_format=datetime_fmt,
        timezone=timezone,
        frequency=frequency,
        columns=columns,
    )

    for col in fp.columns:
        col.ssn_role = determine_sosa_role(col, fp)

    return fp
 

