"""IHC4 score calculator.

Combined ER / PgR / Ki-67 / HER2 immunohistochemical prognostic score for
early ER+ breast cancer (10-year distant recurrence risk).

Reference
---------
Cuzick J, Dowsett M, Pineda S, et al.  Prognostic Value of a Combined
Estrogen Receptor, Progesterone Receptor, Ki-67, and Human Epidermal
Growth Factor Receptor 2 Immunohistochemical Score and Comparison With
the Genomic Health Recurrence Score in Early Breast Cancer.
J Clin Oncol. 2011;29(32):4273-4278.

Formula (per the canonical genefu R package ihc4() port):

    IHC4 = 94.7 × (
              + 0.586 × HER2
              − 0.100 × ER10
              − 0.079 × PgR10
              + 0.240 × ln(1 + 10 × Ki67)
           )

Where:
    ER10  = ER Allred-style 0–10 score (we use ER_Percent / 10 as proxy)
    PgR10 = PR Allred-style 0–10 score (we use PR_Percent / 10 as proxy)
    Ki67  = Ki-67 labelling index expressed as a fraction (0–1),
            so Ki67 = Ki67_percent / 100
    HER2  = 1 if HER2 positive, 0 if negative, NaN if equivocal/unknown

Higher IHC4 → greater 10-year distant-recurrence risk.
Yields NaN when any input (ER, PR, Ki67, HER2) is missing or equivocal.

Note on the ER/PR scale
------------------------
The genefu reference implementation requires ER and PGR on a 0–10 scale.
The Cuzick 2011 paper itself uses Allred scores, which are 0–8.  A number of
published re-implementations (including ours) use the % / 10 proxy, which
yields 0–10 values that are highly correlated with Allred but not identical.
If Allred scores become available in the registry they should be used
directly.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tcr_decoder.scores.base import (
    BaseScore,
    ScoreRegistry,
    extract_ki67_numeric,
    her2_binary,
    evaluate_eligibility,
)

logger = logging.getLogger(__name__)


# ─── Published coefficients (genefu R package, ihc4()) ─────────────────────────
_MULTIPLIER: float = 94.7
_BETA_HER2: float = 0.586
_BETA_ER: float = -0.100
_BETA_PGR: float = -0.079
_BETA_KI67: float = 0.240
_KI67_LOG_MULT: float = 10.0


@ScoreRegistry.register
class IHC4Score(BaseScore):
    """IHC4 score — combined IHC prognostic index for ER+ breast cancer.

    Cites: Cuzick J et al. JCO 2011;29:4273-4278.
    Implemented to match the canonical genefu R package ihc4() function.
    """

    NAME = 'IHC4 Score'
    CITATION = 'Cuzick J et al. J Clin Oncol. 2011;29:4273-4278 (genefu port)'
    REQUIRED_COLS = ['ER_Percent', 'PR_Percent', 'Ki67_Index', 'HER2_Status']
    OUTPUT_COLS = ['IHC4_Score', 'IHC4_Eligibility']

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── Population eligibility gate ───────────────────────────────────────
        # IHC4 was derived and validated exclusively in the ATAC ER+ cohort.
        # Applying it to ER-negative disease produces a number but the
        # calibration is meaningless.  Also gate on invasive non-metastatic.
        elig = evaluate_eligibility(
            df,
            require_invasive=True,
            require_non_metastatic=True,
            require_er_positive=True,
        )
        df['IHC4_Eligibility'] = elig.reason.where(~elig.eligible, '')

        er_pct = pd.to_numeric(
            df.get('ER_Percent', pd.Series(np.nan, index=df.index)),
            errors='coerce',
        )
        pr_pct = pd.to_numeric(
            df.get('PR_Percent', pd.Series(np.nan, index=df.index)),
            errors='coerce',
        )
        ki67_pct = extract_ki67_numeric(
            df.get('Ki67_Index', pd.Series('', index=df.index))
        )
        her2 = her2_binary(
            df.get('HER2_Status', pd.Series('', index=df.index))
        )

        # Clip to plausible ranges (defensive — warn on out-of-range values)
        er_pct_clipped = er_pct.clip(lower=0, upper=100)
        pr_pct_clipped = pr_pct.clip(lower=0, upper=100)
        ki67_pct_clipped = ki67_pct.clip(lower=0, upper=100)

        n_out_of_range = (
            (er_pct != er_pct_clipped).sum()
            + (pr_pct != pr_pct_clipped).sum()
            + (ki67_pct != ki67_pct_clipped).sum()
        )
        if n_out_of_range > 0:
            logger.warning(
                'IHC4: %d input values were out of the 0-100%% range and '
                'have been clipped.  Check ER_Percent, PR_Percent, and '
                'Ki67_Index for data-quality issues.',
                int(n_out_of_range),
            )

        er10 = er_pct_clipped / 10.0
        pgr10 = pr_pct_clipped / 10.0
        ki67_frac = ki67_pct_clipped / 100.0

        ihc4 = _MULTIPLIER * (
            _BETA_HER2 * her2
            + _BETA_ER * er10
            + _BETA_PGR * pgr10
            + _BETA_KI67 * np.log(1.0 + _KI67_LOG_MULT * ki67_frac)
        )
        # Never report an IHC4 for ineligible patients
        df['IHC4_Score'] = ihc4.where(elig.eligible).round(2)
        return df
