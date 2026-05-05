"""Molecular subtype classifier (St. Gallen 2013 / 2015 surrogate IHC criteria).

References
----------
Goldhirsch A, Winer EP, Coates AS, et al.  Personalizing the treatment of
women with early breast cancer: highlights of the St Gallen International
Expert Consensus on the Primary Therapy of Early Breast Cancer 2013.
Ann Oncol. 2013;24(9):2206-2223.

Coates AS, Winer EP, Goldhirsch A, et al.  Tailoring therapies —
improving the management of early breast cancer: St Gallen International
Expert Consensus on the Primary Therapy of Early Breast Cancer 2015.
Ann Oncol. 2015;26(8):1533-1546.  (This consensus firmed the Ki-67 cutoff
at ~20% and kept the PR ≥ 20% criterion for Luminal A.)

Surrogate criteria (combined 2013/2015)
---------------------------------------
    Luminal A-like      ER+  AND  HER2−  AND  Ki67 < 20%  AND  PR ≥ 20%
    Luminal B-like/HER2−  ER+  AND  HER2−  AND  (Ki67 ≥ 20% OR PR < 20%)
    Luminal B-like/HER2+  ER+  AND  HER2+   (any Ki67 / PR)
    HER2-Enriched       ER−  AND  PR−  AND  HER2+
    Triple Negative     ER−  AND  PR−  AND  HER2−

Notes
-----
1. HR-negative classification requires BOTH ER and PR to be known and < 1%.
   Classifying a patient as TNBC when only PR (or only ER) was measured is
   a silent mis-call — the missing receptor could well be positive.
2. The PR ≥ 20% criterion for Luminal A (Prat 2013, endorsed in St. Gallen
   2013/2015) is applied when PR is known.  If PR is missing but Ki67 is
   low and HER2−, the case is reported as 'Luminal (PR unknown)' rather
   than silently defaulted to Luminal A.
"""

import pandas as pd
import numpy as np

from tcr_decoder.scores.base import (
    BaseScore,
    ScoreRegistry,
    extract_ki67_numeric,
    evaluate_eligibility,
    _HER2_POS_PAT,
    _HER2_NEG_PAT,
)


@ScoreRegistry.register
class MolecularSubtype(BaseScore):
    """Intrinsic molecular subtype by surrogate IHC markers (St. Gallen 2013/2015).

    Positive cutoffs:
        ER / PR positive:  percent ≥ 1%      (ASCO/CAP 2020 update)
        PR "high" for Lum A:  ≥ 20%           (Prat 2013, St. Gallen 2013/2015)
        Ki67 high:         ≥ 20%             (St. Gallen 2015 consensus)
        HER2 positive:     IHC 3+ or ISH amplified (decoded HER2_Status)

    Intermediate labels returned when data is incomplete:
        'Luminal (HER2 unknown)'      HR+, HER2 equivocal / not tested
        'Luminal (Ki67 unknown)'      HR+, HER2−, Ki67 missing
        'Luminal (PR unknown)'        ER+, HER2−, Ki67 low, PR missing
        'Non-Luminal (HER2 unknown)'  ER− and PR− (both known), HER2 unknown
        'HR status incomplete'        Only one of ER/PR known and it is negative
        'Not applicable'              DCIS / stage IV / non-breast
    """

    NAME = 'Molecular Subtype (St. Gallen 2013)'
    CITATION = ('Goldhirsch A et al. Ann Oncol. 2013;24:2206-2223; '
                'Coates AS et al. Ann Oncol. 2015;26:1533-1546')
    REQUIRED_COLS = ['ER_Percent', 'HER2_Status']
    OUTPUT_COLS = ['Molecular_Subtype']

    # Cutoffs
    HR_POSITIVE_PCT = 1.0     # ASCO/CAP 2020 update
    PR_HIGH_PCT     = 20.0    # Luminal A (Prat 2013 / St. Gallen 2013/2015)
    KI67_HIGH_PCT   = 20.0    # St. Gallen 2015

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── Eligibility gate — reject DCIS (Tis) / stage IV ───────────────────
        # Intrinsic subtype is defined for INVASIVE breast cancer only.
        elig = evaluate_eligibility(
            df,
            require_invasive=True,
            require_non_metastatic=True,
        )

        er_pct = pd.to_numeric(df.get('ER_Percent', pd.Series(np.nan, index=df.index)), errors='coerce')
        pr_pct = pd.to_numeric(df.get('PR_Percent', pd.Series(np.nan, index=df.index)), errors='coerce')
        ki67   = extract_ki67_numeric(df.get('Ki67_Index', pd.Series('', index=df.index)))

        her2_str   = df.get('HER2_Status', pd.Series('', index=df.index)).fillna('').astype(str)
        her2_pos   = her2_str.str.contains(_HER2_POS_PAT, case=False, na=False, regex=True)
        her2_neg   = her2_str.str.contains(_HER2_NEG_PAT, case=False, na=False, regex=True)
        her2_known = her2_pos | her2_neg

        er_known = er_pct.notna()
        pr_known = pr_pct.notna()
        er_pos   = er_known & (er_pct >= self.HR_POSITIVE_PCT)
        pr_pos   = pr_known & (pr_pct >= self.HR_POSITIVE_PCT)

        # HR-positive: either receptor positive (conservative — a single
        # positive receptor is sufficient for endocrine therapy eligibility).
        hr_pos = er_pos | pr_pos

        # HR-negative: BOTH receptors must be known and both below 1%.
        # A single known-negative receptor is NOT enough — the other one
        # could be positive.  Silent mis-call to TNBC is unacceptable.
        er_neg_known = er_known & ~er_pos
        pr_neg_known = pr_known & ~pr_pos
        hr_neg       = er_neg_known & pr_neg_known

        # "HR status incomplete" — one receptor known-negative, the other
        # missing.  Cannot commit to HR- (could be ER-/PR+ or ER+/PR-).
        hr_incomplete = (er_neg_known ^ pr_neg_known) & ~hr_pos

        ki67_hi  = ki67 >= self.KI67_HIGH_PCT          # NaN propagates
        ki67_lo  = ki67.notna() & (ki67 < self.KI67_HIGH_PCT)

        # PR criterion for Luminal A vs Luminal B/HER2−
        pr_high     = pr_known & (pr_pct >= self.PR_HIGH_PCT)
        pr_low      = pr_known & (pr_pct < self.PR_HIGH_PCT)

        # Build the classification ladder.  Order matters: earlier
        # conditions shadow later ones via np.select.
        conditions = [
            # 0. No data at all
            ~(er_known | pr_known | her2_known),

            # 1. Luminal B/HER2+ — any HR+, HER2+
            hr_pos & her2_pos,

            # 2. Luminal A — HR+, HER2−, Ki67 low, PR high (≥20%)
            hr_pos & her2_neg & ki67_lo & pr_high,

            # 3. Luminal B/HER2− — HR+, HER2−, (Ki67 high OR PR low)
            hr_pos & her2_neg & (ki67_hi | pr_low),

            # 4. Luminal, HER2−, Ki67 low, PR unknown → can't decide A vs B
            hr_pos & her2_neg & ki67_lo & ~pr_known,

            # 5. Luminal, HER2−, Ki67 unknown
            hr_pos & her2_neg & ki67.isna(),

            # 6. Luminal, HER2 unknown
            hr_pos & ~her2_known,

            # 7. HER2-Enriched — both HR-, HER2+
            hr_neg & her2_pos,

            # 8. Triple Negative — both HR-, HER2-
            hr_neg & her2_neg,

            # 9. Non-luminal, HER2 unknown — both HR-, HER2 unknown
            hr_neg & ~her2_known,

            # 10. Only one receptor known-negative — cannot commit
            hr_incomplete,
        ]
        choices = [
            '',
            'Luminal B / HER2+',
            'Luminal A',
            'Luminal B / HER2\u2212',
            'Luminal (PR unknown)',
            'Luminal (Ki67 unknown)',
            'Luminal (HER2 unknown)',
            'HER2-Enriched',
            'Triple Negative',
            'Non-Luminal (HER2 unknown)',
            'HR status incomplete',
        ]
        subtype = pd.Series(np.select(conditions, choices, default=''), index=df.index)
        # Override ineligible rows (DCIS / stage IV) with 'Not applicable'
        subtype[~elig.eligible] = 'Not applicable'
        df['Molecular_Subtype'] = subtype
        return df
