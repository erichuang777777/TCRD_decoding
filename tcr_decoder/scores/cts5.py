"""CTS5 (Clinical Treatment Score post-5 years) calculator.

Published by Sestak et al. (2018) for estimating the risk of late distant
recurrence (years 5–10) in oestrogen-receptor-positive breast cancer
patients who have completed 5 years of adjuvant endocrine therapy.

Reference
---------
Sestak I, Cuzick J, Dowsett M, et al.  Integration of Clinical Variables
for the Prediction of Late Distant Recurrence in Patients With Estrogen
Receptor-Positive Breast Cancer Treated With 5 Years of Endocrine Therapy:
CTS5.  J Clin Oncol. 2018;36(19):1941-1948.

Formula (Sestak 2018, final derivation from pooled ATAC + BIG 1-98):

    CTS5 = 0.438 × nodes_cat
           + 0.988 × (
                 0.093 × size_cap
               − 0.001 × size_cap²
               + 0.375 × grade
               + 0.017 × age
             )

Where:
    nodes_cat = 5-point category of positive lymph nodes (Sestak 2018):
                    0 → 0
                    1 → 1
                    2–3 → 2
                    4–9 → 3
                    ≥10 → 4
    size_cap  = min(Tumor_Size_mm, 30)  — sizes > 30 mm are truncated
                to 30 mm per the published derivation.
    grade     = Nottingham histological grade (1, 2, or 3).
    age       = age at diagnosis in years.

Risk groups (published cutoffs):
    Low          CTS5 < 3.13
    Intermediate 3.13 ≤ CTS5 ≤ 3.86
    High         CTS5 > 3.86
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tcr_decoder.scores.base import (
    BaseScore,
    ScoreRegistry,
    extract_grade_numeric,
    evaluate_eligibility,
)

logger = logging.getLogger(__name__)


# ─── Published Sestak 2018 coefficients ────────────────────────────────────────

_BETA_NODES: float = 0.438
_BETA_OUTER: float = 0.988
_BETA_SIZE: float = 0.093
_BETA_SIZE_SQ: float = 0.001
_BETA_GRADE: float = 0.375
_BETA_AGE: float = 0.017

_SIZE_CAP_MM: float = 30.0

_LOW_CUT: float = 3.13
_HIGH_CUT: float = 3.86


def _cts5_nodal_category(n: float) -> float:
    """Convert positive LN count to CTS5 5-point category (Sestak 2018).

    0   → 0
    1   → 1
    2–3 → 2
    4–9 → 3
    ≥10 → 4
    NaN → NaN
    """
    if pd.isna(n):
        return np.nan
    if n < 0:           # defensive: negative LN count is impossible
        return np.nan
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    if n <= 3:
        return 2.0
    if n <= 9:
        return 3.0
    return 4.0


def _cts5_group(v: float) -> str:
    """Return the published CTS5 risk group label."""
    if pd.isna(v):
        return ''
    if v < _LOW_CUT:
        return 'Low (CTS5 <3.13, ~<5% late recurrence)'
    if v <= _HIGH_CUT:
        return 'Intermediate (CTS5 3.13\u20133.86)'
    return 'High (CTS5 >3.86, ~>10% late recurrence)'


@ScoreRegistry.register
class CTS5Score(BaseScore):
    """Clinical Treatment Score post-5 years (CTS5).

    Cites: Sestak I et al. J Clin Oncol. 2018;36:1941-1948.

    Only clinically meaningful for ER+ patients who have completed 5 years
    of endocrine therapy.  Results are uninformative for ER− or HER2+
    disease.  Recommended filter: ER_Percent ≥ 1 % AND
    Any_Hormone_Therapy = 'Yes'.
    """

    NAME = 'CTS5 Score'
    CITATION = 'Sestak I et al. J Clin Oncol. 2018;36:1941-1948'
    REQUIRED_COLS = [
        'Age_at_Diagnosis',
        'Tumor_Size_mm',
        'LN_Positive_Count',
        'Nottingham_Grade',
    ]
    OUTPUT_COLS = ['CTS5_Score', 'CTS5_Group', 'CTS5_Eligibility']

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── Population eligibility gate ───────────────────────────────────────
        # CTS5 is ONLY meaningful for ER+ patients who have completed (or are
        # completing) 5 years of adjuvant endocrine therapy, in non-metastatic
        # invasive disease.  Applying it to ER-negative, TNBC, or non-endocrine
        # treated patients produces clinically meaningless numbers.
        elig = evaluate_eligibility(
            df,
            require_invasive=True,
            require_non_metastatic=True,
            require_er_positive=True,
            require_endocrine_therapy=True,
        )
        df['CTS5_Eligibility'] = elig.reason.where(~elig.eligible, '')

        age = pd.to_numeric(
            df.get('Age_at_Diagnosis', pd.Series(np.nan, index=df.index)),
            errors='coerce',
        )
        size_mm = pd.to_numeric(
            df.get('Tumor_Size_mm', pd.Series(np.nan, index=df.index)),
            errors='coerce',
        ).where(lambda x: (x >= 0) & (x < 888))     # 888/999 and negatives → NaN
        size_cap = size_mm.clip(upper=_SIZE_CAP_MM)

        nodes_raw = pd.to_numeric(
            df.get('LN_Positive_Count', pd.Series(np.nan, index=df.index)),
            errors='coerce',
        )
        nodes_cat = nodes_raw.apply(_cts5_nodal_category)

        grade = extract_grade_numeric(
            df.get('Nottingham_Grade', pd.Series('', index=df.index))
        )
        # Grade must be 1, 2, or 3 per the published formula; reject anything else
        grade = grade.where(grade.isin([1.0, 2.0, 3.0]))

        cts5 = (
            _BETA_NODES * nodes_cat
            + _BETA_OUTER * (
                _BETA_SIZE * size_cap
                - _BETA_SIZE_SQ * size_cap ** 2
                + _BETA_GRADE * grade
                + _BETA_AGE * age
            )
        ).round(3)

        # Mask ineligible rows to NaN — never report a CTS5 for patients
        # outside the model's training population.
        cts5 = cts5.where(elig.eligible)

        df['CTS5_Score'] = cts5
        df['CTS5_Group'] = cts5.apply(_cts5_group)
        # For ineligible rows, override the group with the reason for audit
        ineligible_mask = ~elig.eligible
        df.loc[ineligible_mask, 'CTS5_Group'] = 'Not applicable'
        return df
