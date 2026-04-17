"""
Derived clinical variables — computed from decoded data for analysis.

All functions take a DataFrame and return a new DataFrame with added columns.
No columns are modified; only new columns are appended.
"""

import re
import pandas as pd
import numpy as np
from typing import Optional


def _nan_series(df: pd.DataFrame) -> pd.Series:
    """Return a NaN-filled float Series aligned to df's index."""
    return pd.Series(np.nan, index=df.index, dtype=float)


def _str_series(df: pd.DataFrame) -> pd.Series:
    """Return an empty-string Series aligned to df's index."""
    return pd.Series('', index=df.index, dtype=str)


def add_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate BMI and category from Height_cm and Weight_kg.

    Categories use WHO standard cutoffs:
        Underweight: <18.5, Normal: 18.5-24.9, Overweight: 25-29.9, Obese: >=30

    Implausible heights (<50 cm or >250 cm) and weights (<20 kg or >400 kg)
    are treated as missing — BMI = NaN for those rows rather than `inf`.
    """
    ht = pd.to_numeric(df.get('Height_cm', _nan_series(df)), errors='coerce')
    wt = pd.to_numeric(df.get('Weight_kg', _nan_series(df)), errors='coerce')
    # Guard against implausible and zero values (physically impossible ranges)
    ht = ht.where((ht >= 50) & (ht <= 250))
    wt = wt.where((wt >= 20) & (wt <= 400))
    bmi = (wt / (ht / 100) ** 2).round(1)
    df = df.copy()
    df['BMI'] = bmi
    df['BMI_Category'] = pd.cut(
        bmi,
        bins=[0, 18.5, 25, 30, 100],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese'],
        right=False,
    ).astype(str).replace('nan', '')
    return df


def add_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add standard oncology age groupings."""
    age = pd.to_numeric(df.get('Age_at_Diagnosis', _nan_series(df)), errors='coerce')
    df = df.copy()
    df['Age_Group'] = pd.cut(
        age,
        bins=[0, 40, 50, 60, 70, 80, 120],
        labels=['<40', '40-49', '50-59', '60-69', '70-79', '≥80'],
        right=False,
    ).astype(str).replace('nan', '')
    df['Age_Group_Binary'] = np.where(age <= 50, '≤50', '>50')
    df.loc[age.isna(), 'Age_Group_Binary'] = ''
    return df


def add_survival_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate survival endpoints for Kaplan-Meier analysis.

    Adds:
        OS_Months: Overall survival in months
        OS_Event: 1=dead (any cause), 0=alive/censored
        CSS_Event: 1=cancer death, 0=non-cancer/alive
        RFS_Months: Recurrence-free survival (to recurrence or death)
        RFS_Event: 1=recurrence or death, 0=censored
    """
    df = df.copy()

    # OS
    surv_yr = pd.to_numeric(df.get('Survival_Years', _nan_series(df)), errors='coerce')
    df['OS_Months'] = (surv_yr * 12).round(1)

    vital_ext = df.get('Vital_Status_Extended', _str_series(df)).fillna('').astype(str)
    df['OS_Event'] = np.where(vital_ext.str.contains('Dead', case=False, na=False), 1, 0)
    df.loc[surv_yr.isna(), 'OS_Event'] = np.nan

    # CSS — cancer-specific death
    cod = df.get('Cause_of_Death_Extended', _str_series(df)).fillna('').astype(str)
    cancer_death = (
        cod.str.contains('cancer|carcinoma|neoplasm|tumor', case=False, na=False) &
        ~cod.str.contains('Non-cancer|Not applicable', case=False, na=False)
    )
    df['CSS_Event'] = np.where(cancer_death, 1, 0)
    df.loc[surv_yr.isna(), 'CSS_Event'] = np.nan

    # RFS — recurrence-free survival
    dx = pd.to_datetime(df.get('Date_of_Diagnosis', _str_series(df))
                        .astype(str).str.replace(r'(\d{4}-\d{2})$', r'\1-15', regex=True),
                        format='mixed', errors='coerce')
    recur = pd.to_datetime(df.get('Recurrence_Date_Extended', _str_series(df)),
                           errors='coerce')
    lcd6 = pd.to_datetime(df.get('Last_Contact_Extended', _str_series(df))
                          .astype(str).str.replace(r'(\d{4}-\d{2})$', r'\1-15', regex=True),
                          format='mixed', errors='coerce')

    # RFS endpoint = min(recurrence_date, death_date/lcd6)
    recur_or_death = recur.fillna(lcd6)
    rfs_days = (recur_or_death - dx).dt.days
    df['RFS_Months'] = (rfs_days / 30.44).round(1)

    has_recurrence = df.get('Recurrence_Type_Extended', _str_series(df)).fillna('').astype(str)
    # NB: operator precedence — `&` binds tighter than `>`, so the sub-expressions
    # MUST each be parenthesised.  Without the parentheses this collapses to
    # `has_recurrence.str.len() > 0` and silently counts every "No recurrence"
    # string as a recurrence event, corrupting RFS analysis.
    recurrence_flag = (
        (has_recurrence.str.len() > 0)
        & (~has_recurrence.str.contains('No recurrence', case=False, na=True))
    )
    df['RFS_Event'] = np.where(recurrence_flag | (df['OS_Event'] == 1), 1, 0)
    df.loc[rfs_days.isna(), 'RFS_Event'] = np.nan

    return df


def add_treatment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary treatment flags and treatment modality count.

    Respects Treatment_Data_Incomplete — marks as 'Unknown (treated elsewhere)'
    for Class-3 patients.
    """
    df = df.copy()
    incomplete = df.get('Treatment_Data_Incomplete', pd.Series(False, index=df.index))

    def _any_tx(this_col: str, other_col: str, label: str) -> pd.Series:
        this = df.get(this_col, pd.Series('', index=df.index)).fillna('').astype(str)
        other = df.get(other_col, pd.Series('', index=df.index)).fillna('').astype(str)
        has = (this.str.contains('performed', case=False, na=False) |
               other.str.contains('Outside|performed', case=False, na=False))
        result = np.where(has, 'Yes', 'No')
        result = np.where(incomplete, 'Unknown (treated elsewhere)', result)
        return pd.Series(result, index=df.index)

    df['Any_Chemotherapy'] = _any_tx('Chemo_This_Hosp', 'Chemo_Other_Hosp', 'Chemo')
    df['Any_Radiation'] = np.where(
        df.get('Radiation_Performed', pd.Series('', index=df.index))
        .fillna('').str.contains('performed', case=False, na=False), 'Yes', 'No')
    df.loc[incomplete, 'Any_Radiation'] = 'Unknown (treated elsewhere)'
    df['Any_Hormone_Therapy'] = _any_tx('Hormone_This_Hosp', 'Hormone_Other_Hosp', 'Hormone')
    df['Any_Targeted_Therapy'] = _any_tx('Targeted_This_Hosp', 'Targeted_Other_Hosp', 'Targeted')
    df['Any_Immunotherapy'] = _any_tx('Immuno_This_Hosp', 'Immuno_Other_Hosp', 'Immuno')

    # Count modalities
    tx_cols = ['Any_Chemotherapy', 'Any_Radiation', 'Any_Hormone_Therapy',
               'Any_Targeted_Therapy', 'Any_Immunotherapy']
    df['Treatment_Modality_Count'] = sum(
        (df[c] == 'Yes').astype(int) for c in tx_cols)

    return df


def add_staging_simplified(df: pd.DataFrame) -> pd.DataFrame:
    """Add simplified staging columns (collapse substages)."""
    df = df.copy()

    def _simplify_stage(series: pd.Series) -> pd.Series:
        s = series.fillna('').astype(str)
        result = s.copy()
        result = result.str.replace(r'Stage I[AB]', 'Stage I', regex=True)
        result = result.str.replace(r'Stage II[AB]', 'Stage II', regex=True)
        result = result.str.replace(r'Stage III[ABC]', 'Stage III', regex=True)
        result = result.str.replace(r'Stage IV.*', 'Stage IV', regex=True)
        result = result.str.replace(r'Unknown.*|Cannot.*|Not applicable.*', '', regex=True)
        return result

    def _simplify_tnm(series: pd.Series) -> pd.Series:
        s = series.fillna('').astype(str)
        return s.str.extract(r'(T[0-4X]|Tis|N[0-3X]|M[01X])', expand=False).fillna('')

    df['Stage_Simple'] = _simplify_stage(df.get('Path_Stage', pd.Series(dtype=str)))
    df['T_Simple'] = _simplify_tnm(df.get('Path_T', pd.Series(dtype=str)))
    df['N_Simple'] = _simplify_tnm(df.get('Path_N', pd.Series(dtype=str)))
    df['M_Simple'] = _simplify_tnm(df.get('Path_M', pd.Series(dtype=str)))
    return df


def add_ln_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Removed: LN_Ratio (LN_Positive / LN_Examined) was clinically uninformative
    for most registry use cases and confusing in the output. No-op retained for
    backward compatibility with the pipeline call sequence."""
    return df


def add_er_pr_percent(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric ER/PR percentages for continuous analysis.

    Handles integer AND decimal percentages:
        'ER Positive (10%)'    → 10.0
        'ER Positive (15.3%)'  → 15.3   ← decimals preserved
        'ER Positive (90.5%)'  → 90.5
        'ER Negative (<1%)'    → 0.5    (convention for rare positives)
        'ER Negative (0%)'     → 0.0

    The earlier regex `(\\d+)%` matched only the trailing digit group of a
    decimal (e.g. '15.3%' → '3'), silently producing ER_Percent values that
    differed from the true percentage by up to 10x and corrupting every
    downstream calculator (IHC4, CTS5 eligibility, PEPI ER-points).
    """
    df = df.copy()
    for receptor in ['ER', 'PR']:
        col = f'{receptor}_Status'
        if col not in df.columns:
            continue

        # Coerce to string so non-string dtypes (Object with NaN) don't
        # raise on `.str.extract`.
        col_str = df[col].fillna('').astype(str)

        # Decimal-aware extraction: captures '10', '10.5', '0.5', '100'.
        pct = col_str.str.extract(r'([0-9]+(?:\.[0-9]+)?)\s*%', expand=False)
        df[f'{receptor}_Percent'] = pd.to_numeric(pct, errors='coerce')

        # Convention overrides
        lt1 = col_str.str.contains('<1%', na=False)
        df.loc[lt1, f'{receptor}_Percent'] = 0.5
        neg0 = col_str.str.contains(r'(?:^|[^.\d])0%', regex=True, na=False) & \
               col_str.str.contains('Negative', na=False)
        df.loc[neg0, f'{receptor}_Percent'] = 0.0

        # Out-of-range sanity clip (percentages live in [0, 100])
        pct_col = df[f'{receptor}_Percent']
        df[f'{receptor}_Percent'] = pct_col.where(
            pct_col.isna() | ((pct_col >= 0) & (pct_col <= 100))
        )
    return df


def add_dx_to_surgery_interval(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate diagnosis-to-surgery interval in days."""
    df = df.copy()
    dx = pd.to_datetime(
        df.get('Date_of_Diagnosis', _str_series(df))
        .astype(str).str.replace(r'(\d{4}-\d{2})$', r'\1-15', regex=True),
        format='mixed', errors='coerce')
    sx = pd.to_datetime(df.get('Surgery_Date', _str_series(df)), errors='coerce')
    df['Dx_to_Surgery_Days'] = (sx - dx).dt.days
    return df


# ─── Prognostic score calculators ────────────────────────────────────────────
# All score logic lives in tcr_decoder.scores (modular, individually testable).
# Thin wrappers below delegate to those classes for backward compatibility.

from tcr_decoder.scores import ScoreRegistry
from tcr_decoder.scores.npi import NPIScore
from tcr_decoder.scores.pepi import PEPIScore
from tcr_decoder.scores.ihc4 import IHC4Score
from tcr_decoder.scores.cts5 import CTS5Score
from tcr_decoder.scores.molecular_subtype import MolecularSubtype
from tcr_decoder.scores.predict import PREDICTScore


def add_pepi_score(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper → tcr_decoder.scores.PEPIScore."""
    return PEPIScore().apply(df)


def add_ihc4_score(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper → tcr_decoder.scores.IHC4Score."""
    return IHC4Score().apply(df)


def add_npi_score(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper → tcr_decoder.scores.NPIScore."""
    return NPIScore().apply(df)


def add_molecular_subtype(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper → tcr_decoder.scores.MolecularSubtype."""
    return MolecularSubtype().apply(df)


def add_cts5_score(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper → tcr_decoder.scores.CTS5Score."""
    return CTS5Score().apply(df)


def add_predict_score(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper → tcr_decoder.scores.PREDICTScore."""
    return PREDICTScore().apply(df)


def add_structural_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Apply structural derived variable calculations (Module 1 output).

    Computes all clinical variables that derive mechanically from decoded
    registry fields: BMI, age groups, survival endpoints, treatment summary,
    simplified staging, ER/PR numeric percentages, and Dx-to-surgery interval.

    These columns are prerequisites for the clinical score calculators in
    tcr_decoder.scores (Module 2).  Run this before ClinicalScoreEngine.compute().

    This function has NO dependency on tcr_decoder.scores and can be used
    independently of the scoring module.
    """
    df = add_bmi(df)
    df = add_age_groups(df)
    df = add_survival_endpoints(df)
    df = add_treatment_summary(df)
    df = add_staging_simplified(df)
    df = add_ln_ratio(df)
    df = add_er_pr_percent(df)
    df = add_dx_to_surgery_interval(df)
    return df


def add_all_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Alias for add_structural_derived() — backward-compatible entry point.

    Prognostic scores (NPI, PEPI, IHC4, CTS5, Molecular Subtype, PREDICT) are
    no longer computed here.  Use ClinicalScoreEngine from tcr_decoder.scores to
    add scores, or use TCRPipeline for the full decode + score pipeline.
    """
    return add_structural_derived(df)
