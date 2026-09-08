"""Discover and load PO workbooks with lineage metadata."""

from __future__ import annotations

import hashlib
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import openpyxl
import pandas as pd

from parts_inflation.config import REQUIRED_SOURCE_COLUMNS, _normalize_header

logger = logging.getLogger(__name__)


@dataclass
class SourceFileInfo:
    path: Path
    size: int
    mtime: float
    sheet: str
    fingerprint: str
    nrows: int = 0
    min_date: Optional[pd.Timestamp] = None
    max_date: Optional[pd.Timestamp] = None
    warnings: list[str] = field(default_factory=list)


def file_fingerprint(path: Path) -> str:
    st = path.stat()
    h = hashlib.sha256()
    h.update(path.name.encode("utf-8"))
    h.update(str(st.st_size).encode("utf-8"))
    h.update(str(int(st.st_mtime)).encode("utf-8"))
    # Content sample for robustness without full read cost on huge files
    with path.open("rb") as f:
        h.update(f.read(65536))
        if st.st_size > 65536:
            f.seek(max(0, st.st_size - 65536))
            h.update(f.read(65536))
    return h.hexdigest()[:24]


def discover_workbooks(input_dir: Path) -> list[Path]:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in {".xlsx", ".xlsm"} and not p.name.startswith("~$")],
        key=lambda p: p.name.lower(),
    )
    if not files:
        raise FileNotFoundError(f"No .xlsx/.xlsm workbooks found in {input_dir}")
    return files


def _score_sheet(headers: list[str]) -> int:
    normalized = {_normalize_header(h) for h in headers if h is not None}
    return sum(1 for c in REQUIRED_SOURCE_COLUMNS if c in normalized)


def select_sheet(path: Path) -> tuple[str, list[str]]:
    # data_only=True avoids macros; read_only for speed
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True, keep_vba=False)
    try:
        sheet_scores: list[tuple[int, str, list[str]]] = []
        for name in wb.sheetnames:
            ws = wb[name]
            first = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), None)
            headers = list(first) if first else []
            score = _score_sheet([str(h) if h is not None else "" for h in headers])
            sheet_scores.append((score, name, [str(h) if h is not None else "" for h in headers]))
        sheet_scores.sort(reverse=True)
        best_score, best_name, best_headers = sheet_scores[0]
        if best_score < 8:
            raise ValueError(
                f"{path.name}: no sheet contains the required PO columns "
                f"(best score {best_score} on '{best_name}')"
            )
        if "Sheet2" in wb.sheetnames and best_name != "Sheet2":
            # Prefer Sheet2 when present and adequate
            for score, name, headers in sheet_scores:
                if name == "Sheet2" and score >= 8:
                    return name, headers
        return best_name, best_headers
    finally:
        wb.close()


def _read_workbook_frame(path: Path, sheet: str) -> pd.DataFrame:
    # Use pandas + openpyxl; do not execute macros
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    df.columns = [_normalize_header(c) for c in df.columns]
    # Drop fully empty rows
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def load_all_po_lines(input_dir: Path) -> tuple[pd.DataFrame, list[SourceFileInfo], list[str]]:
    """Load all workbooks; return combined frame, per-file info, and warnings."""
    files = discover_workbooks(input_dir)
    frames: list[pd.DataFrame] = []
    infos: list[SourceFileInfo] = []
    all_warnings: list[str] = []

    for path in files:
        warn: list[str] = []
        sheet, headers = select_sheet(path)
        if sheet != "Sheet2":
            msg = f"{path.name}: using sheet '{sheet}' (Sheet2 not preferred/available)"
            warn.append(msg)
            all_warnings.append(msg)
            logger.warning(msg)

        missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in {_normalize_header(h) for h in headers}]
        # After read, columns are normalized — check again on frame
        df = _read_workbook_frame(path, sheet)
        missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name} sheet '{sheet}' missing required columns: {missing}")

        df = df.copy()
        df["source_file"] = path.name
        df["source_sheet"] = sheet
        df["source_row_number"] = df.index + 2  # Excel header is row 1
        # Preserve raw part number type before any conversion
        df["raw_part_number"] = df["Part Number"]
        df["raw_part_number_type"] = df["Part Number"].map(lambda x: type(x).__name__)

        fp = file_fingerprint(path)
        st = path.stat()
        dates = pd.to_datetime(df["P.O. Date"], errors="coerce")
        info = SourceFileInfo(
            path=path,
            size=st.st_size,
            mtime=st.st_mtime,
            sheet=sheet,
            fingerprint=fp,
            nrows=len(df),
            min_date=dates.min() if dates.notna().any() else None,
            max_date=dates.max() if dates.notna().any() else None,
            warnings=warn,
        )
        infos.append(info)
        frames.append(df)
        logger.info(
            "Loaded %s rows from %s [%s] (%s to %s)",
            len(df),
            path.name,
            sheet,
            info.min_date,
            info.max_date,
        )

    combined = pd.concat(frames, ignore_index=True)
    # Overlap / gap diagnostics across files by date range
    sorted_infos = sorted(infos, key=lambda x: (x.min_date or pd.Timestamp.max))
    for a, b in zip(sorted_infos, sorted_infos[1:]):
        if a.max_date is not None and b.min_date is not None:
            if b.min_date <= a.max_date:
                msg = (
                    f"Date overlap between {a.path.name} (max {a.max_date.date()}) "
                    f"and {b.path.name} (min {b.min_date.date()})"
                )
                all_warnings.append(msg)
                logger.warning(msg)
            else:
                gap = (b.min_date - a.max_date).days - 1
                if gap > 7:
                    msg = (
                        f"Date gap of {gap} days between {a.path.name} "
                        f"(max {a.max_date.date()}) and {b.path.name} (min {b.min_date.date()})"
                    )
                    all_warnings.append(msg)
                    logger.warning(msg)

    return combined, infos, all_warnings


def profile_sources(input_dir: Path) -> pd.DataFrame:
    _, infos, warnings_ = load_all_po_lines(input_dir)
    rows = []
    for info in infos:
        rows.append(
            {
                "source_file": info.path.name,
                "sheet": info.sheet,
                "nrows": info.nrows,
                "min_date": info.min_date,
                "max_date": info.max_date,
                "size_bytes": info.size,
                "mtime": datetime.fromtimestamp(info.mtime).isoformat(timespec="seconds"),
                "fingerprint": info.fingerprint,
                "warnings": "; ".join(info.warnings),
            }
        )
    profile = pd.DataFrame(rows)
    if warnings_:
        logger.info("Profile warnings: %s", warnings_)
    return profile


def combined_fingerprint(infos: list[SourceFileInfo], cleaning_version: str) -> str:
    h = hashlib.sha256()
    h.update(cleaning_version.encode())
    for info in sorted(infos, key=lambda x: x.path.name):
        h.update(info.fingerprint.encode())
        h.update(str(info.size).encode())
    return h.hexdigest()[:32]
