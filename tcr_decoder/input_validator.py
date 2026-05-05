"""
Pre-decode input validation — catch problems BEFORE decoding.

Run automatically when TCRDecoder.load() is called, or standalone:
    python -m tcr_decoder.input_validator decoded_cancer_registry_FINAL.xlsx
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

# All raw fields referenced by core.py decode()
REQUIRED_RAW_FIELDS = [
    'PK', 'SEX', 'AGE', 'DX_YEAR', 'DXDATE', 'VISTDATE', 'SMOKING',
    'SEQ1', 'SEQ2', 'TCODE1', 'LAT95', 'MCODE', 'MCODE5', 'MCODE6',
    'MCODE6C', 'CONFER', 'CSIZE95', 'PNI', 'LVI', 'LNEXAM', 'LN_POSITI',
    'AJCC', 'CT', 'CN', 'CM', 'CSTG', 'PT', 'PN', 'PM', 'PSTG',
    'SUMSTG', 'OSTG', 'OCSTG', 'OPSTG', 'META1', 'META2', 'META3',
    'S', 'FSDATE', 'PRESTYPE', 'STYPE95', 'MINS', 'MARG95', 'MARGDIS',
    'PRESLNSCO', 'SLNSCO95', 'SSF4', 'SSF5',
    'R', 'RTAR', 'RMOD', 'EBRT', 'HTAR', 'HDOSE', 'HNO',
    'LTAR', 'LDOSE', 'LNO', 'SEQRS', 'SEQLS',
    'PREC', 'C', 'PREH', 'H', 'PREI', 'I', 'PREB', 'B',
    'PRETAR', 'TAR', 'OTH', 'PREP', 'WATCHWAITING',
    'SSF1', 'SSF2', 'SSF3', 'SSF6', 'SSF7', 'SSF8', 'SSF9', 'SSF10',
    'VSTA', 'CSTA', 'LCD', 'REDATE', 'RETYPE95', 'DIECAUSE',
    'VSTA6', 'LCD6', 'SURVY6', 'REDATE6', 'RETYPE6', 'DIECAUSE6',
    'HEIGHT', 'WEIGHT', 'KPSECOG', 'CLASS95', 'CLASSOFDIAG', 'CLASSOFTREAT',
]


class InputValidationResult:
    """Container for input validation results."""

    def __init__(self):
        self.errors: List[Dict] = []    # Blocking issues
        self.warnings: List[Dict] = []  # Non-blocking concerns
        self.info: List[Dict] = []      # Informational

    def add(self, level: str, check: str, detail: str):
        entry = {'Check': check, 'Detail': detail}
        if level == 'ERROR':
            self.errors.append(entry)
        elif level == 'WARNING':
            self.warnings.append(entry)
        else:
            self.info.append(entry)

    @property
    def is_ok(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [f'Input Validation: {len(self.errors)} errors, '
                 f'{len(self.warnings)} warnings, {len(self.info)} info']
        for e in self.errors:
            lines.append(f'  [ERROR] {e["Check"]}: {e["Detail"]}')
        for w in self.warnings:
            lines.append(f'  [WARN]  {w["Check"]}: {w["Detail"]}')
        for i in self.info:
            lines.append(f'  [INFO]  {i["Check"]}: {i["Detail"]}')
        return '\n'.join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for e in self.errors:
            rows.append({**e, 'Level': 'ERROR'})
        for w in self.warnings:
            rows.append({**w, 'Level': 'WARNING'})
        for i in self.info:
            rows.append({**i, 'Level': 'INFO'})
        return pd.DataFrame(rows, columns=['Level', 'Check', 'Detail'])


def validate_input(df: pd.DataFrame,
                   sheet_name: str = 'All_Fields_Decoded',
                   expected_patients: Optional[int] = None) -> InputValidationResult:
    """Run all pre-decode validation checks on the raw DataFrame.

    Args:
        df: The loaded DataFrame from the input Excel file
        sheet_name: Name of the sheet that was loaded
        expected_patients: If provided, warn if patient count differs
    """
    result = InputValidationResult()

    # ── 0. Shape sanity ──────────────────────────────────────
    if df is None or df.shape[0] == 0:
        result.add('ERROR', 'Empty input',
                   'Input DataFrame has 0 rows — nothing to decode.')
        return result
    if df.shape[1] == 0:
        result.add('ERROR', 'No columns',
                   'Input DataFrame has 0 columns — schema unreadable.')
        return result

    # Warn on column headers with leading / trailing whitespace (they cause
    # silent fallback-to-empty-series lookups in the decoder).
    stripped_mismatch = [
        c for c in df.columns
        if isinstance(c, str) and c != c.strip()
    ]
    if stripped_mismatch:
        result.add('WARNING', 'Whitespace in column headers',
                   f'{len(stripped_mismatch)} column(s) have leading/trailing '
                   f'whitespace: {stripped_mismatch[:5]}. They have been '
                   f'stripped on load but consider cleaning the source file.')

    # ── 1. Column existence ──────────────────────────────────
    all_cols = set(df.columns)
    missing_raw = []
    missing_dec = []
    for field in REQUIRED_RAW_FIELDS:
        raw_col = f'{field}_raw'
        dec_col = f'{field}_decoded'
        if raw_col not in all_cols:
            missing_raw.append(field)
        if dec_col not in all_cols and raw_col not in all_cols:
            missing_dec.append(field)

    if missing_raw:
        shown = missing_raw[:10]
        suffix = '...' if len(missing_raw) > len(shown) else ''
        result.add(
            'ERROR',
            'Missing columns',
            f'{len(missing_raw)} required raw columns missing: {shown}{suffix}',
        )
    else:
        result.add('INFO', 'Column schema', f'All {len(REQUIRED_RAW_FIELDS)} required fields present')

    # ── 2. Patient count ─────────────────────────────────────
    n_patients = len(df)
    result.add('INFO', 'Patient count', f'{n_patients} patients')
    if expected_patients and n_patients != expected_patients:
        result.add('WARNING', 'Patient count change',
                   f'Expected {expected_patients}, got {n_patients}')

    # ── 3. Duplicate Patient IDs ─────────────────────────────
    pk_col = 'PK_raw' if 'PK_raw' in df.columns else None
    if pk_col:
        dups = df[pk_col].dropna().duplicated()
        if dups.any():
            n_dup = dups.sum()
            result.add('ERROR', 'Duplicate Patient_ID',
                       f'{n_dup} duplicate PK values found')
        else:
            result.add('INFO', 'Patient_ID uniqueness', 'All unique')

    # ── 4. Completely empty columns ──────────────────────────
    empty_cols = []
    for col in df.columns:
        if df[col].isna().all() or (df[col].astype(str).str.strip() == '').all():
            empty_cols.append(col)
    if empty_cols:
        result.add('WARNING', 'Empty columns',
                   f'{len(empty_cols)} columns are entirely empty: {empty_cols[:5]}')

    # ── 5. Numeric field type checks ─────────────────────────
    numeric_fields = ['AGE', 'CSIZE95', 'LNEXAM', 'HDOSE', 'HNO', 'LDOSE', 'LNO',
                      'HEIGHT', 'WEIGHT', 'SURVY6', 'MARGDIS']
    for field in numeric_fields:
        raw_col = f'{field}_raw'
        if raw_col in df.columns:
            non_null = df[raw_col].dropna()
            if len(non_null) > 0:
                numeric = pd.to_numeric(non_null, errors='coerce')
                pct_numeric = numeric.notna().sum() / len(non_null) * 100
                if pct_numeric < 80:
                    result.add('WARNING', f'{field} non-numeric',
                               f'Only {pct_numeric:.0f}% of {field} values are numeric')

    # ── 6. Date field format checks ──────────────────────────
    date_fields = ['DXDATE', 'VISTDATE', 'FSDATE', 'LCD', 'LCD6', 'REDATE', 'REDATE6']
    for field in date_fields:
        raw_col = f'{field}_raw'
        if raw_col in df.columns:
            non_null = df[raw_col].dropna().astype(str)
            if len(non_null) > 0:
                parsed = pd.to_datetime(non_null, errors='coerce', format='mixed')
                n_fail = parsed.isna().sum()
                # Don't count sentinels (99, 9999) as failures
                if n_fail > len(non_null) * 0.3:
                    result.add('WARNING', f'{field} date parse',
                               f'{n_fail}/{len(non_null)} values failed date parsing')

    # ── 7. SSF code range checks ─────────────────────────────
    ssf_ranges = {
        'SSF1': (0, 999), 'SSF2': (0, 999), 'SSF3': (0, 999),
        'SSF6': (0, 999), 'SSF7': (0, 999), 'SSF10': (0, 999),
    }
    for field, (lo, hi) in ssf_ranges.items():
        raw_col = f'{field}_raw'
        if raw_col in df.columns:
            vals = pd.to_numeric(df[raw_col], errors='coerce').dropna()
            if len(vals) > 0:
                out_of_range = ((vals < lo) | (vals > hi)).sum()
                if out_of_range > 0:
                    result.add('WARNING', f'{field} out of range',
                               f'{out_of_range} values outside [{lo},{hi}]')

    # ── 8. Histology check (breast cancer = 8xxx) ────────────
    mcode_col = 'MCODE_raw'
    if mcode_col in df.columns:
        mcodes = pd.to_numeric(df[mcode_col], errors='coerce').dropna()
        non_breast = ((mcodes < 8000) | (mcodes >= 10000)).sum()
        if non_breast > 0:
            result.add('WARNING', 'Non-breast histology',
                       f'{non_breast} patients with histology code outside 8000-9999')
        n_unique = mcodes.nunique()
        result.add('INFO', 'Histology codes', f'{n_unique} unique morphology codes')

    return result


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python -m tcr_decoder.input_validator <input.xlsx>')
        sys.exit(1)
    sys.stdout.reconfigure(encoding='utf-8')
    path = sys.argv[1]
    print(f'Validating {path}...')
    df = pd.read_excel(path, sheet_name='All_Fields_Decoded')
    result = validate_input(df)
    print(result.summary())
