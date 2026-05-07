"""
Utility functions for text cleaning, type conversion, and data normalization.
These are pure functions with no side effects — safe to call on any data.
"""

import re
import pandas as pd
import numpy as np
from typing import Callable, Dict, Optional


def strip_float_suffix(v: str) -> str:
    """Remove '.0' suffix from float-formatted strings.

    Unlike str.rstrip('.0') which strips individual characters,
    this only removes an actual '.0' ending.

    Examples:
        '120.0' → '120'   (correct)
        '0.0'   → '0'     (correct)
        '100'   → '100'   (unchanged)
        '.0'    → '.0'    (too short, unchanged)
        'C50.0' → 'C50.0' (only strips if candidate is alnum without '.')
    """
    if v.endswith('.0') and len(v) >= 3:
        candidate = v[:-2]
        # Only strip if the candidate is a simple number or alphanumeric code
        # BUT NOT if it looks like an ICD code with a dot (e.g., C50.0 → C50 would lose meaning)
        if '.' in candidate:
            return v  # Don't strip — the '.0' is part of a decimal like '3.10'
        if candidate.lstrip('-').replace('.', '', 1).isdigit() or candidate.lstrip('-').isalnum():
            return candidate
    return v


def clean_text(s: str) -> str:
    """Strip Chinese characters, normalize whitespace, clean punctuation artifacts.

    Preserves English clinical content, removes CJK characters and
    fullwidth punctuation, unwraps orphaned parentheses.
    """
    if not isinstance(s, str):
        return str(s) if not pd.isna(s) else ''
    if s.strip().lower() == 'nan':
        return ''

    # Remove entire (...) blocks that contain Chinese characters
    s = re.sub(r'\([^)]*[\u4e00-\u9fff\u3400-\u4dbf][^)]*\)', '', s)
    # Remove remaining standalone Chinese/CJK characters
    s = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\uff00-\uffef\u3000-\u303f]+', '', s)
    # Remove empty parentheses
    s = re.sub(r'\(\s*\)', '', s)
    # Clean whitespace and strip leading/trailing punctuation artifacts
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'^[\s/,\-]+', '', s)
    # Remove leading '+' only when followed by space (artifact from Chinese removal)
    s = re.sub(r'^\+\s+', '', s)
    s = re.sub(r'[\s/,\-]+$', '', s)

    # Unwrap outer parens if they are balanced
    s = _unwrap_outer_parens(s)

    # Remove unmatched leading/trailing parens
    while s.startswith(')'):
        s = s[1:].lstrip()
    while s.endswith('('):
        s = s[:-1].rstrip()
    while s.endswith(')') and s.count(')') > s.count('('):
        s = s[:-1].rstrip(' ,')
    return s.strip()


def _unwrap_outer_parens(s: str) -> str:
    """Unwrap value if entirely wrapped in balanced outer parentheses."""
    s = s.strip()
    if s.startswith('(') and s.endswith(')'):
        depth, closed_early = 0, False
        for i, c in enumerate(s):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            if depth == 0 and i < len(s) - 1:
                closed_early = True
                break
        if not closed_early:
            return s[1:-1].strip()
    return s


def en(series: pd.Series) -> pd.Series:
    """Apply clean_text to every value in a Series (English extraction)."""
    return series.fillna('').astype(str).apply(clean_text)


def clean_date(series: pd.Series) -> pd.Series:
    """Normalize date columns: remove sentinel codes, return clean date strings."""
    def _clean(v):
        v = str(v).strip()
        if not v or v.lower() in ('nan', 'nat', '', '99999999', '88888888'):
            return ''
        # Remove .0 from float dates
        v = strip_float_suffix(v)
        # Normalize slashes
        v = v.replace('/', '-')
        # Handle sentinel day (99) or month (99) → partial date
        m_partial = re.match(r'^(\d{4})-(\d{2})-99', v)
        if m_partial:
            return f'{m_partial.group(1)}-{m_partial.group(2)}'
        m_year = re.match(r'^(\d{4})-99', v)
        if m_year:
            return m_year.group(1)
        # Try to parse as full datetime
        try:
            dt = pd.to_datetime(v)
            return dt.strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            pass
        # Already a partial date
        if re.match(r'^\d{4}-\d{2}$', v):
            return v
        if re.match(r'^\d{4}$', v):
            return v
        return ''
    return series.fillna('').astype(str).apply(_clean)


def clean_numeric(series: pd.Series,
                  unknown_vals: Optional[set] = None) -> pd.Series:
    """Keep numeric values; blank out known sentinel/unknown codes.

    Args:
        series: Raw data series
        unknown_vals: Set of string values to treat as missing.
                     Default: {'999','9999','98','99','888','8888'}
    """
    if unknown_vals is None:
        unknown_vals = {'999', '9999', '98', '99', '888', '8888'}

    def _clean(v):
        v = strip_float_suffix(str(v).strip())
        if v in unknown_vals or v.lower() in ('nan', ''):
            return ''
        try:
            n = float(v)
            return str(int(n)) if n == int(n) else str(round(n, 1))
        except (ValueError, TypeError, OverflowError):
            return v

    return series.fillna('').astype(str).apply(_clean)


def clean_tnm(series: pd.Series) -> pd.Series:
    """Remove duplicated range patterns in TNM descriptions.

    e.g., 'N1a - 1-3 1-3 axillary LN' → 'N1a - 1-3 axillary LN'
    """
    def _fix(v):
        if not v:
            return v
        v = re.sub(r'(\d+[-–]\d+|\u226510|\d+) \1', r'\1', v)
        return v
    return series.fillna('').astype(str).apply(_fix)


def shorten_tnm(series: pd.Series) -> pd.Series:
    """Shorten TNM descriptions to just the code (T2, N0, M0 etc).

    'T2 - >20mm, ≤50mm Tumor >20mm, ≤50mm' → 'T2'
    'N0 - No regional LN metastasis' → 'N0'
    'Stage IIA' stays as 'Stage IIA' (no dash separator)
    '' stays as ''
    """
    def _shorten(v):
        if not v:
            return v
        # Extract the code before ' - ' separator
        if ' - ' in v:
            return v.split(' - ')[0].strip()
        # Also handle 'Tx' patterns at start with space/description
        m = re.match(r'^([TNM]\w+(?:\([^)]+\))?)\s+', v)
        if m:
            return m.group(1)
        return v
    return series.fillna('').astype(str).apply(_shorten)


def _norm(v) -> str:
    """Normalize a raw value: strip float suffix, return '' for nan/empty."""
    v = strip_float_suffix(str(v).strip())
    return '' if (not v or v.lower() == 'nan') else v


def _map_decode(
    code_map: Dict[int, str],
    fallback: str = 'Code',
) -> Callable:
    """Factory: return a Series decoder for integer-keyed lookup maps.

    Eliminates the repeated _d() closure pattern across SSF decoder functions.
    Handles NaN, float-coded integers, and unknown codes uniformly.
    """
    def _decoder(series: pd.Series) -> pd.Series:
        def _d(val) -> str:
            if pd.isna(val) or str(val).strip() in ('', 'nan'):
                return ''
            try:
                iv = int(float(str(val).strip()))
            except (ValueError, TypeError):
                return str(val).strip()
            return code_map.get(iv, f'{fallback} {iv}')
        return series.apply(_d)
    return _decoder
