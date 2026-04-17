"""Nottingham Prognostic Index (NPI) calculator.

References
----------
Galea MH, Blamey RW, Elston CE, Ellis IO.  The Nottingham Prognostic Index
in primary breast cancer.  Breast Cancer Res Treat. 1992;22(3):207-219.

Blamey RW, Pinder SE, Ball GR, Ellis IO, Elston CW, Mitchell MJ, Haybittle JL.
Reading the prognosis of the individual with breast cancer.
Eur J Cancer. 2007;43(10):1545-1547.

Formula
-------
    NPI = 0.2 × tumor_diameter_cm + lymph_node_stage + grade

Lymph node stage (3-point category):
    1 = 0 positive nodes (N0)
    2 = 1–3 positive nodes
    3 = ≥4 positive nodes

Risk groups (Blamey 2007 six-group refinement):
    Excellent   NPI ≤ 2.4    (EPG — ~96% 10-y BC-specific survival)
    Good        2.41 – 3.4   (GPG — ~93%)
    Moderate I  3.41 – 4.4   (MPG-1 — ~81%)
    Moderate II 4.41 – 5.4   (MPG-2 — ~74%)
    Poor        5.41 – 6.4   (PPG — ~51%)
    Very Poor   > 6.4        (VPG — ~19%)

Note
----
The simpler three-group Galea 1992 scheme (Good ≤ 3.4 / Moderate 3.41–5.4 /
Poor > 5.4) is a strict subset of the Blamey 2007 breakdown and can be
obtained by merging Excellent+Good, Moderate I+II, and Poor+Very Poor.
Earlier versions of this file used five categories with cutoffs (3.4, 4.4,
5.4, 6.4) but labelled them Excellent/Good/Moderate I/Moderate II/Poor —
these labels were shifted one step down relative to Blamey 2007 and have
been corrected.

Intended population
-------------------
NPI was derived and validated on invasive breast cancer with full
pathological staging (post-surgery).  It is NOT applicable to:
    * DCIS / in-situ disease (Tis)
    * stage IV / metastatic disease
    * pre-surgical staging (clinical-only) where pN is unknown
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tcr_decoder.scores.base import (
    BaseScore,
    ScoreRegistry,
    extract_grade_numeric,
    evaluate_eligibility,
)


@ScoreRegistry.register
class NPIScore(BaseScore):
    """Nottingham Prognostic Index for invasive breast cancer prognosis."""

    NAME = 'Nottingham Prognostic Index (NPI)'
    CITATION = 'Galea MH et al. Breast Cancer Res Treat. 1992;22:207-219'
    REQUIRED_COLS = ['Tumor_Size_mm', 'LN_Positive_Count', 'Nottingham_Grade']
    OUTPUT_COLS = ['NPI_Score', 'NPI_Group', 'NPI_Eligibility']

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── Population eligibility gate ───────────────────────────────────────
        # Reject DCIS (Tis) and stage IV (M1) — NPI is an invasive-disease,
        # loco-regional-staging tool.
        elig = evaluate_eligibility(
            df,
            require_invasive=True,
            require_non_metastatic=True,
        )
        df['NPI_Eligibility'] = elig.reason.where(~elig.eligible, '')

        # ── Inputs (use index-aware fallbacks) ────────────────────────────────
        size_mm = pd.to_numeric(
            df.get('Tumor_Size_mm', pd.Series(np.nan, index=df.index)),
            errors='coerce',
        ).where(lambda x: (x >= 0) & (x < 888))   # 888/999 and negatives → NaN
        size_cm = size_mm / 10.0

        ln_pos = pd.to_numeric(
            df.get('LN_Positive_Count', pd.Series(np.nan, index=df.index)),
            errors='coerce',
        ).where(lambda x: x >= 0)                 # negative → NaN
        ln_stage = ln_pos.apply(_ln_stage)

        grade = extract_grade_numeric(
            df.get('Nottingham_Grade', pd.Series('', index=df.index))
        )
        # Grade must be 1/2/3; anything else → NaN
        grade = grade.where(grade.isin([1.0, 2.0, 3.0]))

        npi = (0.2 * size_cm + ln_stage + grade).round(2)
        # Mask ineligible rows to NaN
        npi = npi.where(elig.eligible)

        df['NPI_Score'] = npi
        df['NPI_Group'] = npi.apply(_npi_group)
        # Override ineligible rows
        df.loc[~elig.eligible, 'NPI_Group'] = 'Not applicable'
        return df


def _ln_stage(n: float) -> float:
    if pd.isna(n):
        return np.nan
    if n == 0:
        return 1.0
    if n <= 3:
        return 2.0
    return 3.0


def _npi_group(v: float) -> str:
    """Blamey 2007 six-group risk stratification.

    Cutoffs 2.4 / 3.4 / 4.4 / 5.4 / 6.4 partition the NPI into six
    prognostic groups with monotonically decreasing 10-year BC-specific
    survival (~96 / 93 / 81 / 74 / 51 / 19 %).
    """
    if pd.isna(v):
        return ''
    if v <= 2.4:
        return 'Excellent (NPI \u22642.4)'
    if v <= 3.4:
        return 'Good (NPI 2.41\u20133.4)'
    if v <= 4.4:
        return 'Moderate I (NPI 3.41\u20134.4)'
    if v <= 5.4:
        return 'Moderate II (NPI 4.41\u20135.4)'
    if v <= 6.4:
        return 'Poor (NPI 5.41\u20136.4)'
    return 'Very Poor (NPI >6.4)'
