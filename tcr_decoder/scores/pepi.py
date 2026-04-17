"""PEPI (Preoperative Endocrine Prognostic Index) calculator."""

import re
import pandas as pd
import numpy as np

from tcr_decoder.scores.base import (
    BaseScore,
    ScoreRegistry,
    extract_ki67_numeric,
    evaluate_eligibility,
)


@ScoreRegistry.register
class PEPIScore(BaseScore):
    """Preoperative Endocrine Prognostic Index.

    Applicable to ER+ breast cancer after neoadjuvant endocrine therapy.
    Scoring based on the surgical specimen pathology.

    Points:
        pT stage:   T1/T2 = 0,  T3/T4 = 3
        pN stage:   N0 = 0,     N1-N3 = 3
        Ki67 (RFS): ≤2.7%=0, >2.7–7.3%=1, >7.3–19.7%=1, >19.7–53.1%=2, >53.1%=3
        Ki67 (BCSS):≤2.7%=0, >2.7–7.3%=1, >7.3–19.7%=2, >19.7–53.1%=3, >53.1%=3
        ER Allred:  ER% ≥ 1% (Allred 3–8) = 0,  ER% < 1% (Allred 0–2) = 3

    Risk groups:  0 = Best,  1–3 = Intermediate,  ≥4 = Worse

    ER Allred note: TCR records ER%, not Allred score. ER% < 1 maps to
    Allred 0–2 (negative); ER% ≥ 1 maps to Allred 3–8 (positive).
    This is documented as a methodological approximation.
    """

    NAME = 'PEPI Score'
    CITATION = 'Ellis MJ et al. J Natl Cancer Inst. 2008;100:1380-1388'
    REQUIRED_COLS = ['T_Simple', 'N_Simple', 'Ki67_Index', 'ER_Percent']
    OUTPUT_COLS = [
        'PEPI_pT_Points', 'PEPI_pN_Points',
        'PEPI_Ki67_RFS_Points', 'PEPI_Ki67_BCSS_Points',
        'PEPI_ER_Points',
        'PEPI_RFS_Score', 'PEPI_BCSS_Score',
        'PEPI_RFS_Group', 'PEPI_BCSS_Group',
        'PEPI_Eligibility',
    ]

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── Population eligibility ────────────────────────────────────────────
        # PEPI was derived on post-neoadjuvant endocrine-therapy residual
        # tumors in ER+ patients (Ellis 2008, ACOSOG Z1031).  The Ki67
        # cutpoints come from endocrine-resistant biology.  Applying PEPI to
        # ER-negative patients, to non-neoadjuvant-ET contexts, or to
        # metastatic/in-situ disease is meaningless.
        elig = evaluate_eligibility(
            df,
            require_invasive=True,
            require_non_metastatic=True,
            require_er_positive=True,
        )
        df['PEPI_Eligibility'] = elig.reason.where(~elig.eligible, '')

        # pT
        t = df.get('T_Simple', pd.Series('', index=df.index)).fillna('').astype(str)
        df['PEPI_pT_Points'] = t.apply(_t_points)

        # pN
        n = df.get('N_Simple', pd.Series('', index=df.index)).fillna('').astype(str)
        df['PEPI_pN_Points'] = n.apply(_n_points)

        # Ki67
        ki67 = extract_ki67_numeric(df.get('Ki67_Index', pd.Series('', index=df.index)))
        df['PEPI_Ki67_RFS_Points']  = ki67.apply(_ki67_rfs)
        df['PEPI_Ki67_BCSS_Points'] = ki67.apply(_ki67_bcss)

        # ER Allred proxy
        er_pct = pd.to_numeric(
            df.get('ER_Percent', pd.Series(dtype=float)), errors='coerce'
        )
        df['PEPI_ER_Points'] = er_pct.apply(_er_points)

        # Totals (NaN if ANY component missing)
        rfs_parts  = ['PEPI_pT_Points', 'PEPI_pN_Points', 'PEPI_Ki67_RFS_Points',  'PEPI_ER_Points']
        bcss_parts = ['PEPI_pT_Points', 'PEPI_pN_Points', 'PEPI_Ki67_BCSS_Points', 'PEPI_ER_Points']
        df['PEPI_RFS_Score']  = df[rfs_parts].sum(axis=1, skipna=False)
        df['PEPI_BCSS_Score'] = df[bcss_parts].sum(axis=1, skipna=False)

        # Mask ineligible rows to NaN
        df['PEPI_RFS_Score']  = df['PEPI_RFS_Score'].where(elig.eligible)
        df['PEPI_BCSS_Score'] = df['PEPI_BCSS_Score'].where(elig.eligible)

        df['PEPI_RFS_Group']  = df['PEPI_RFS_Score'].apply(_pepi_group)
        df['PEPI_BCSS_Group'] = df['PEPI_BCSS_Score'].apply(_pepi_group)
        ineligible = ~elig.eligible
        df.loc[ineligible, 'PEPI_RFS_Group']  = 'Not applicable'
        df.loc[ineligible, 'PEPI_BCSS_Group'] = 'Not applicable'
        return df


# ── Point helpers ──────────────────────────────────────────────────────────────

def _t_points(t: str) -> float:
    t = t.strip()
    if not t:
        return np.nan
    # Explicitly reject Tis (in situ) — PEPI is for invasive disease.
    if re.match(r'^Tis\b', t, re.IGNORECASE):
        return np.nan
    # Reject TX (unknown) — PEPI requires pathological T staging from the
    # surgical specimen.  Treating TX as "T1/T2" (0 points) silently bucket
    # unknown tumors into the favourable group.
    if re.match(r'^TX\b', t, re.IGNORECASE):
        return np.nan
    if re.match(r'^T[34]', t, re.IGNORECASE):
        return 3.0
    # T0 (pCR), T1, T2 → 0 points (per Ellis 2008)
    if re.match(r'^T[0-2]\b', t, re.IGNORECASE):
        return 0.0
    return np.nan


def _n_points(n: str) -> float:
    n = n.strip()
    if not n:
        return np.nan
    if re.match(r'^N0', n, re.IGNORECASE):
        return 0.0
    if re.match(r'^N[1-3]', n, re.IGNORECASE):
        return 3.0
    return np.nan


def _ki67_rfs(k: float) -> float:
    if pd.isna(k):
        return np.nan
    if k <= 2.7:   return 0.0
    if k <= 7.3:   return 1.0
    if k <= 19.7:  return 1.0
    if k <= 53.1:  return 2.0
    return 3.0


def _ki67_bcss(k: float) -> float:
    if pd.isna(k):
        return np.nan
    if k <= 2.7:   return 0.0
    if k <= 7.3:   return 1.0
    if k <= 19.7:  return 2.0
    if k <= 53.1:  return 3.0
    return 3.0


def _er_points(e: float) -> float:
    if pd.isna(e):
        return np.nan
    return 3.0 if e < 1.0 else 0.0


def _pepi_group(score: float) -> str:
    if pd.isna(score):
        return ''
    s = int(score)
    if s == 0:
        return 'Best (PEPI=0)'
    if s <= 3:
        return 'Intermediate (PEPI 1-3)'
    return 'Worse (PEPI ≥4)'
