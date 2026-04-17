"""
Clinical validation rules for decoded cancer registry data.

Each validator function takes a DataFrame and returns a list of flag dicts:
    [{'Patient_ID': ..., 'Flag': ..., 'Severity': ..., 'Detail': ...}, ...]

Severity levels: CRITICAL, HIGH, MEDIUM, LOW, INFO

Design principle: ALL validators are cancer-agnostic at the top level.
Validators that require cancer-specific columns (e.g., ER_Status for breast)
silently skip if those columns are absent, so the same run_all_validators()
call works for any cancer type.
"""

import pandas as pd
import numpy as np
from typing import List, Dict


Flag = Dict[str, str]


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Get column if present, else return empty string series."""
    if name in df.columns:
        return df[name].fillna('').astype(str)
    return pd.Series([''] * len(df), index=df.index)


def _require_cols(df: pd.DataFrame, *cols: str) -> bool:
    """Return True if ALL required columns exist and are non-empty."""
    return all(c in df.columns for c in cols)


def validate_stage_m0i(df: pd.DataFrame) -> List[Flag]:
    """Flag Stage IV patients with M0(i+) — bone marrow ITC, not true metastasis."""
    if not _require_cols(df, 'Path_Stage', 'Path_M'):
        return []
    flags = []
    mask = (
        _col(df, 'Path_Stage').str.contains('Stage IV', na=False) &
        _col(df, 'Path_M').str.contains(r'M0\(i\+\)|ITC', na=False, regex=True)
    )
    for idx, row in df[mask].iterrows():
        flags.append({
            'Patient_ID': str(row['Patient_ID']),
            'Flag': 'Stage IV + M0(i+)',
            'Severity': 'HIGH',
            'Detail': (
                f"Path_Stage=Stage IV but Path_M=M0(i+) [bone marrow ITC]. "
                f"M0(i+) is NOT true distant metastasis. "
                f"Verify AJCC edition staging for M0(i+)."
            ),
        })
    return flags


def validate_m1_without_site(df: pd.DataFrame) -> List[Flag]:
    """Flag M1 patients without a recorded metastasis site."""
    if not _require_cols(df, 'Path_Stage', 'Path_M', 'Metastasis_Site_1'):
        return []
    flags = []
    no_site = (df['Metastasis_Site_1'].isna() |
               (df['Metastasis_Site_1'].astype(str).str.strip() == ''))
    mask = (
        _col(df, 'Path_Stage').str.contains('Stage IV', na=False) &
        _col(df, 'Path_M').str.contains('M1 -', na=False) &
        no_site
    )
    for idx, row in df[mask].iterrows():
        flags.append({
            'Patient_ID': str(row['Patient_ID']),
            'Flag': 'M1 without metastasis site',
            'Severity': 'MEDIUM',
            'Detail': 'Path_M=M1 but no Metastasis_Site recorded.',
        })
    return flags


def validate_surgery_consistency(df: pd.DataFrame) -> List[Flag]:
    """Flag surgery/LN/margin inconsistencies."""
    if not _require_cols(df, 'Surgery_Performed'):
        return []
    flags = []
    # Surgery at other hospital only
    if 'Any_Surgery' in df.columns:
        other_only = df[
            (_col(df, 'Surgery_Performed') == 'No surgery') &
            (_col(df, 'Any_Surgery') == 'Yes')
        ]
        for idx, row in other_only.iterrows():
            flags.append({
                'Patient_ID': str(row['Patient_ID']),
                'Flag': 'Surgery at other hospital only',
                'Severity': 'INFO',
                'Detail': (
                    f"Surgery_Performed=No surgery (this hosp), but "
                    f"Surgery_Type_Other_Hosp={str(row.get('Surgery_Type_Other_Hosp', ''))[:40]}. "
                    f"Margin/LN data reflect the other hospital surgery."
                ),
            })

    # Surgery but LN_Examined=0
    if 'LN_Examined' in df.columns and 'LN_Positive' in df.columns:
        ln_surg = df[
            (pd.to_numeric(df.get('LN_Examined', pd.Series(dtype=float)),
                           errors='coerce') == 0) &
            (_col(df, 'Surgery_Performed') == 'Surgery performed') &
            (~_col(df, 'LN_Positive').isin(['0', '']))
        ]
        for idx, row in ln_surg.iterrows():
            flags.append({
                'Patient_ID': str(row['Patient_ID']),
                'Flag': 'Surgery but LN_Examined=0',
                'Severity': 'LOW',
                'Detail': f"Surgery performed but LN_Examined=0. LN_Positive={row['LN_Positive']}.",
            })
    return flags


def validate_tumor_size_vs_tstage(df: pd.DataFrame) -> List[Flag]:
    """Flag mismatches between tumor size and T stage."""
    if not _require_cols(df, 'Path_T', 'Tumor_Size_mm'):
        return []
    flags = []
    ts = pd.to_numeric(df.get('Tumor_Size_mm', pd.Series(dtype=float)), errors='coerce')

    for idx, row in df.iterrows():
        pt = str(row.get('Path_T', ''))
        sz = ts.get(idx, np.nan)
        if pd.isna(sz) or not pt:
            continue
        if 'T1' in pt and sz > 25:
            flags.append({
                'Patient_ID': str(row['Patient_ID']),
                'Flag': 'T stage vs Tumor size mismatch',
                'Severity': 'HIGH',
                'Detail': f"Path_T={pt[:25]} but Tumor_Size={sz:.0f}mm. T1 should be ≤20mm.",
            })
        elif 'T3' in pt and sz < 40:
            flags.append({
                'Patient_ID': str(row['Patient_ID']),
                'Flag': 'T stage vs Tumor size mismatch',
                'Severity': 'MEDIUM',
                'Detail': f"Path_T={pt[:25]} but Tumor_Size={sz:.0f}mm. T3 should be >50mm.",
            })
    return flags


def validate_treatment_biomarker(df: pd.DataFrame) -> List[Flag]:
    """Flag treatment-biomarker inconsistencies (breast-specific; skips if columns absent)."""
    flags = []

    # Hormone therapy in TNBC — breast only
    if 'Molecular_Subtype' in df.columns and 'Hormone_This_Hosp' in df.columns:
        tnbc = df[df['Molecular_Subtype'].isin(['Triple Negative', 'Triple-Negative'])]
        for idx, row in tnbc.iterrows():
            if 'performed' in str(row.get('Hormone_This_Hosp', '')).lower():
                flags.append({
                    'Patient_ID': str(row['Patient_ID']),
                    'Flag': 'Hormone therapy in TNBC',
                    'Severity': 'MEDIUM',
                    'Detail': 'ER-/PR-/HER2- but received hormone therapy.',
                })

    # HER2+ without targeted therapy — breast/stomach/gastric
    if 'HER2_Status' in df.columns and 'Targeted_This_Hosp' in df.columns:
        for idx, row in df.iterrows():
            her2 = str(row.get('HER2_Status', ''))
            targeted = str(row.get('Targeted_This_Hosp', ''))
            if 'Positive' in her2 and 'No targeted' in targeted:
                flags.append({
                    'Patient_ID': str(row['Patient_ID']),
                    'Flag': 'HER2+ without targeted therapy',
                    'Severity': 'MEDIUM',
                    'Detail': f"HER2={her2[:30]} but no targeted therapy.",
                })
    return flags


def validate_er_low_positive(df: pd.DataFrame) -> List[Flag]:
    """Flag ER Low Positive (1-10%) per ASCO/CAP 2020 guidelines (breast only)."""
    if 'ER_Status' not in df.columns:
        return []
    flags = []
    mask = _col(df, 'ER_Status').str.match(r'ER Positive \(([1-9]|10)%\)', na=False)
    for idx, row in df[mask].iterrows():
        flags.append({
            'Patient_ID': str(row['Patient_ID']),
            'Flag': 'ER Low Positive (1-10%)',
            'Severity': 'INFO',
            'Detail': f"{row['ER_Status']} — per ASCO/CAP 2020, uncertain endocrine benefit.",
        })
    return flags


def validate_pm_b_convention(df: pd.DataFrame) -> List[Flag]:
    """Flag M0(i+) coding convention for post-2010 patients."""
    if 'Path_M' not in df.columns:
        return []
    flags = []
    pm_b = _col(df, 'Path_M').str.contains(r'M0\(i', na=False)
    dx_yr = pd.to_numeric(df.get('Diagnosis_Year', pd.Series(dtype=float)), errors='coerce')
    n_post2010 = (pm_b & (dx_yr >= 2010)).sum()
    total_post2010 = (dx_yr >= 2010).sum()

    if n_post2010 > 0 and total_post2010 > 0:
        pct = n_post2010 / total_post2010 * 100
        if pct > 50:
            flags.append({
                'Patient_ID': 'ALL',
                'Flag': 'PM=B coding convention (post-2010)',
                'Severity': 'INFO',
                'Detail': (
                    f"{n_post2010} of {total_post2010} post-2010 patients ({pct:.0f}%) have PM=B [M0(i+)]. "
                    f"This is likely a registry DEFAULT code, not actual ITC detection."
                ),
            })
    return flags


def validate_survival_dates(df: pd.DataFrame) -> List[Flag]:
    """Check survival dates for impossible sequences."""
    flags = []
    _empty = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns]')

    def _get_date(col: str) -> pd.Series:
        if col in df.columns:
            return pd.to_datetime(df[col], errors='coerce')
        return _empty.copy()

    dx  = _get_date('Date_of_Diagnosis')
    sx  = _get_date('Surgery_Date')
    re_ = _get_date('Recurrence_Date')

    # Surgery before diagnosis (>30 days)
    gap = (sx - dx).dt.days
    bad_gap = gap.notna() & (gap < -30)
    for idx in df[bad_gap].index:
        flags.append({
            'Patient_ID': str(df.loc[idx, 'Patient_ID']),
            'Flag': 'Surgery before diagnosis',
            'Severity': 'HIGH',
            'Detail': f"Surgery {abs(gap[idx]):.0f} days before diagnosis.",
        })

    # Recurrence before diagnosis
    rgap = (re_ - dx).dt.days
    bad_rgap = rgap.notna() & (rgap < 0)
    for idx in df[bad_rgap].index:
        flags.append({
            'Patient_ID': str(df.loc[idx, 'Patient_ID']),
            'Flag': 'Recurrence before diagnosis',
            'Severity': 'CRITICAL',
            'Detail': f"Recurrence date is before diagnosis date.",
        })
    return flags


def validate_missing_data_patterns(df: pd.DataFrame) -> List[Flag]:
    """Flag systematic missing data patterns that affect analysis."""
    flags = []
    # Column set is dynamic — only check columns that exist
    candidate_fields = {
        'Ki67_Index': 'Ki-67',
        'Nottingham_Grade': 'Nottingham',
        'LVI_SSF': 'LVI (SSF)',
        'Neoadjuvant_Response': 'Neoadjuvant response',
        'EGFR_Mutation': 'EGFR mutation',
        'ALK_Rearrangement': 'ALK rearrangement',
        'AFP_Level': 'AFP',
        'MSI_Status': 'MSI',
        'PSA_Preop': 'PSA',
        'Gleason_Score': 'Gleason score',
    }
    dx_yr = pd.to_numeric(df.get('Diagnosis_Year', pd.Series(dtype=float)), errors='coerce')

    for col, label in candidate_fields.items():
        if col not in df.columns:
            continue
        avail = df[
            df[col].notna() &
            (df[col].astype(str).str.strip() != '') &
            (~df[col].astype(str).str.contains('Unknown|Not applicable|not tested',
                                                 case=False, na=True))
        ]
        if len(avail) == 0:
            continue
        min_yr = dx_yr[avail.index].min()
        missing = df[~df.index.isin(avail.index)]
        if len(missing) > 10:
            flags.append({
                'Patient_ID': 'ALL',
                'Flag': f'{label} data availability',
                'Severity': 'INFO',
                'Detail': (
                    f"{label} available for {len(avail)}/{len(df)} patients "
                    f"(since {min_yr:.0f}). {len(missing)} patients missing — "
                    f"likely pre-SSF era or not applicable."
                ),
            })
    return flags


def validate_second_primary(df: pd.DataFrame) -> List[Flag]:
    """Flag patients with second primary tumors — treatment data may be mixed."""
    if 'Cancer_Sequence' not in df.columns:
        return []
    flags = []
    second = df[
        _col(df, 'Cancer_Sequence').str.contains('Second|2nd|Third|3rd', na=False, case=False)
    ]
    if len(second) > 0:
        flags.append({
            'Patient_ID': 'MULTIPLE',
            'Flag': 'Second primary tumor patients',
            'Severity': 'INFO',
            'Detail': (
                f"{len(second)} patients have this cancer as their 2nd+ primary. "
                f"Treatment/outcome data may include effects from the other primary."
            ),
        })
    return flags


def validate_class_of_case(df: pd.DataFrame) -> List[Flag]:
    """Flag patients where treatment data may be incomplete due to Class of Case."""
    flags = []
    if 'Treatment_Data_Incomplete' in df.columns:
        incomplete = df[df['Treatment_Data_Incomplete'] == True]
        if len(incomplete) > 0:
            flags.append({
                'Patient_ID': 'ALL',
                'Flag': 'Treatment data incomplete (Class 3)',
                'Severity': 'HIGH',
                'Detail': (
                    f"{len(incomplete)} patients were diagnosed and treated entirely at another "
                    f"hospital. Treatment fields (chemo, RT, surgery) reflect the REPORTING "
                    f"hospital only — actual treatment may differ. Exclude from treatment analyses."
                ),
            })
    return flags


def validate_radiation_dose(df: pd.DataFrame) -> List[Flag]:
    """Check radiation dose plausibility."""
    if not _require_cols(df, 'High_Dose_cGy', 'High_Dose_Fractions'):
        return []
    flags = []
    dose = pd.to_numeric(df.get('High_Dose_cGy', pd.Series(dtype=float)), errors='coerce')
    frac = pd.to_numeric(df.get('High_Dose_Fractions', pd.Series(dtype=float)), errors='coerce')

    for idx in df[dose.notna() & frac.notna()].index:
        d, f = dose[idx], frac[idx]
        if f > 0:
            per_frac = d / f
            if per_frac > 500 or per_frac < 100:
                flags.append({
                    'Patient_ID': str(df.loc[idx, 'Patient_ID']),
                    'Flag': 'Unusual RT dose per fraction',
                    'Severity': 'LOW',
                    'Detail': f"Dose={d:.0f}cGy/{f:.0f}fx = {per_frac:.0f}cGy/fx (expected 150-300).",
                })
    return flags


def validate_egfr_without_targeted(df: pd.DataFrame) -> List[Flag]:
    """Flag EGFR-mutated lung cancer patients without targeted therapy (lung-specific)."""
    if not _require_cols(df, 'EGFR_Mutation', 'Targeted_This_Hosp'):
        return []
    flags = []
    for idx, row in df.iterrows():
        egfr = str(row.get('EGFR_Mutation', ''))
        targeted = str(row.get('Targeted_This_Hosp', ''))
        if 'positive' in egfr.lower() and 'No targeted' in targeted:
            flags.append({
                'Patient_ID': str(row['Patient_ID']),
                'Flag': 'EGFR+ without targeted therapy',
                'Severity': 'MEDIUM',
                'Detail': f"EGFR={egfr[:40]} but no targeted therapy recorded.",
            })
    return flags


def validate_afp_hcc(df: pd.DataFrame) -> List[Flag]:
    """Flag liver cancer patients with very high AFP (>400 ng/mL) — suggests HCC (liver-specific)."""
    if 'AFP_Level' not in df.columns:
        return []
    flags = []
    for idx, row in df.iterrows():
        afp = str(row.get('AFP_Level', ''))
        # Extract numeric from strings like 'AFP 500 ng/mL (elevated)'
        import re
        m = re.search(r'AFP (\d+) ng/mL', afp)
        if m:
            val = int(m.group(1))
            if val > 400:
                flags.append({
                    'Patient_ID': str(row['Patient_ID']),
                    'Flag': 'Very high AFP (>400 ng/mL)',
                    'Severity': 'INFO',
                    'Detail': f"{afp[:50]} — AFP >400 ng/mL is highly specific for HCC.",
                })
    return flags


def validate_msi_immunotherapy(df: pd.DataFrame) -> List[Flag]:
    """Flag MSI-H patients without immunotherapy — potential missed indication."""
    if not _require_cols(df, 'MSI_Status', 'Immuno_This_Hosp'):
        return []
    flags = []
    for idx, row in df.iterrows():
        msi = str(row.get('MSI_Status', ''))
        immuno = str(row.get('Immuno_This_Hosp', ''))
        if 'MSI-H' in msi and 'No immuno' in immuno:
            flags.append({
                'Patient_ID': str(row['Patient_ID']),
                'Flag': 'MSI-H without immunotherapy',
                'Severity': 'INFO',
                'Detail': f"MSI-H tumor but no immunotherapy. Consider pembrolizumab eligibility (post-2017).",
            })
    return flags


def run_all_validators(df: pd.DataFrame) -> pd.DataFrame:
    """Run all validation rules and return a DataFrame of flags.

    Validators automatically skip if their required columns are absent,
    so this works for any cancer type without modification.
    """
    all_validators = [
        # Universal (work for any cancer)
        validate_stage_m0i,
        validate_m1_without_site,
        validate_surgery_consistency,
        validate_tumor_size_vs_tstage,
        validate_pm_b_convention,
        validate_survival_dates,
        validate_missing_data_patterns,
        validate_second_primary,
        validate_class_of_case,
        validate_radiation_dose,
        # Breast-specific (auto-skips if ER_Status/HER2_Status absent)
        validate_treatment_biomarker,
        validate_er_low_positive,
        # Lung-specific (auto-skips if EGFR_Mutation absent)
        validate_egfr_without_targeted,
        # Liver-specific (auto-skips if AFP_Level absent)
        validate_afp_hcc,
        # Colorectal/MSI (auto-skips if MSI_Status absent)
        validate_msi_immunotherapy,
    ]

    all_flags = []
    for validator in all_validators:
        try:
            flags = validator(df)
            all_flags.extend(flags)
        except Exception as e:
            all_flags.append({
                'Patient_ID': 'SYSTEM',
                'Flag': f'Validator error: {validator.__name__}',
                'Severity': 'CRITICAL',
                'Detail': str(e)[:200],
            })

    if not all_flags:
        return pd.DataFrame(columns=['Patient_ID', 'Flag', 'Severity', 'Detail'])

    return pd.DataFrame(all_flags)
