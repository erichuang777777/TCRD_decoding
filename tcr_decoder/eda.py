"""Interactive EDA report generation via dataprep.eda (optional dependency).

Usage
-----
    from tcr_decoder.eda import generate_eda_report
    generate_eda_report(decoder.clean, "output_report.html", cancer_group="breast")

Requires: pip install dataprep
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

# Columns excluded from EDA profiling
_SKIP_SUFFIXES = ('_raw',)
_SKIP_PREFIXES = ('QA_', 'SSF_Raw')
_SKIP_EXACT = {'Patient_ID', 'TCODE1', 'TCODE2'}
_MAX_CARDINALITY = 60  # skip near-free-text categoricals


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return clinically meaningful columns suitable for profiling."""
    keep = []
    for col in df.columns:
        if col in _SKIP_EXACT:
            continue
        if any(col.startswith(p) for p in _SKIP_PREFIXES):
            continue
        if any(col.endswith(s) for s in _SKIP_SUFFIXES):
            continue
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        # Drop high-cardinality string columns (free-text / identifiers)
        if s.dtype == object and s.nunique() > _MAX_CARDINALITY:
            continue
        keep.append(col)
    return df[keep]


def generate_eda_report(
    df: pd.DataFrame,
    output_path: str | Path,
    cancer_group: str = 'generic',
) -> Path:
    """Generate an interactive HTML EDA report for a decoded clinical DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Decoded clinical data (``TCRDecoder.clean``).
    output_path : str | Path
        Destination file path (``.html`` extension added if absent).
    cancer_group : str
        Used for the report title only.

    Returns
    -------
    Path
        Resolved path of the saved HTML file.

    Raises
    ------
    ImportError
        If ``dataprep`` is not installed.
    """
    try:
        from dataprep.eda import create_report  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "dataprep is required for EDA reports.\n"
            "Install with:  pip install dataprep"
        ) from exc

    out = Path(output_path).with_suffix('.html')
    eda_df = _select_columns(df)

    title = f"TCR Clinical EDA — {cancer_group.replace('_', ' ').title()} (n={len(df):,})"
    report = create_report(eda_df, title=title)

    # dataprep >=0.4: save(stem) writes stem.html
    report.save(str(out.with_suffix('')))

    # Verify and locate file (dataprep appends .html automatically)
    if not out.exists():
        candidates = list(out.parent.glob(out.stem + '*.html'))
        if candidates:
            out = candidates[0]

    return out.resolve()
