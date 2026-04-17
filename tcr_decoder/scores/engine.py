"""
Clinical score engine — standalone Module 2 public API.

ClinicalScoreEngine can operate on ANY DataFrame that contains the
standard decoded columns (ER_Percent, HER2_Status, Tumor_Size_mm, etc.).
It does NOT require that the DataFrame was produced by TCRDecoder.

Two usage modes
---------------
compute(df)
    Expects prerequisite columns already present (ER_Percent, T_Simple,
    Any_Hormone_Therapy, etc.).  Use this when df comes from TCRDecoder.decode()
    or any other source that already has structural derived columns.

compute_standalone(df)
    Fully self-contained.  Runs add_structural_derived() internally first,
    then applies all registered score calculators.  Use this when df comes from
    an external system (e.g. another hospital, a research database) that has
    the raw decoded field columns but not the structural derived ones.

Selective scoring
-----------------
Both methods accept an optional scores parameter: a list of score names
(matching BaseScore.NAME) to restrict computation to specific calculators.
Pass scores=None (default) to apply all registered calculators.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from tcr_decoder.scores.base import ScoreRegistry

logger = logging.getLogger(__name__)


class ClinicalScoreEngine:
    """Standalone clinical risk calculator engine (Module 2).

    Computes all registered prognostic scores for breast cancer patients.
    Safe for non-breast data — each calculator returns 'Not applicable'
    when its required columns are absent.

    Examples
    --------
    # With a DataFrame from TCRDecoder (has structural derived cols):
    from tcr_decoder.scores import ClinicalScoreEngine
    engine = ClinicalScoreEngine()
    scored_df = engine.compute(decoded_df)

    # With an external DataFrame (raw decoded cols, no structural derived):
    scored_df = engine.compute_standalone(external_df)

    # Selective — only NPI and PREDICT:
    scored_df = engine.compute(df, scores=['Nottingham Prognostic Index (NPI)',
                                            'PREDICT Breast v3.0'])

    # List available calculators:
    for info in ClinicalScoreEngine.list_scores():
        print(info['name'], '—', info['citation'])
    """

    @staticmethod
    def list_scores() -> list[dict]:
        """Return metadata for all registered calculators."""
        return ScoreRegistry.list_scores()

    def compute(
        self,
        df: pd.DataFrame,
        scores: Optional[list[str]] = None,
        *,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Add prognostic scores to a DataFrame that already has structural derived columns.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain prerequisite columns produced by add_structural_derived()
            (ER_Percent, PR_Percent, T_Simple, N_Simple, Age_Group, Any_Hormone_Therapy,
            Any_Chemotherapy, Any_Radiation, Dx_to_Surgery_Days, etc.).
            Typically the output of TCRDecoder.decode().

        scores : list[str] or None
            Names of specific calculators to apply (matching BaseScore.NAME).
            Pass None (default) to apply all registered calculators.

        verbose : bool
            If True (default), emit a per-calculator summary via the logger
            showing how many rows produced a non-null numeric output for each
            score.  Set False for silent operation (e.g. in batch pipelines).

        Returns
        -------
        pd.DataFrame
            Copy of df with score columns appended.
        """
        result = df.copy()
        n_total = len(result)

        # Resolve which calculators to run
        all_classes = {s.NAME: s for s in ScoreRegistry._scores}
        if scores is None:
            selected = list(all_classes.values())
        else:
            selected = []
            for name in scores:
                if name not in all_classes:
                    raise KeyError(
                        f'Score {name!r} not found. '
                        f'Available: {sorted(all_classes.keys())}'
                    )
                selected.append(all_classes[name])

        # Run each calculator and collect per-score stats for the summary
        stats: list[dict] = []
        for score_cls in selected:
            score = score_cls()
            before = set(result.columns)
            result = score.apply(result)
            new_cols = [c for c in score.OUTPUT_COLS if c in result.columns]

            if n_total == 0:
                n_computed = 0
            else:
                # A row counts as "computed" if AT LEAST one output column
                # is a real numeric/string result (not NaN, not 'Not applicable').
                mask = pd.Series(False, index=result.index)
                for col in new_cols:
                    s = result[col]
                    is_value = s.notna() & (s.astype(str) != 'Not applicable') & (s.astype(str) != '')
                    mask = mask | is_value
                n_computed = int(mask.sum())

            stats.append({
                'name': score.NAME,
                'computed': n_computed,
                'total': n_total,
            })

        if verbose and n_total > 0:
            self._emit_summary(stats, n_total)

        return result

    @staticmethod
    def _emit_summary(stats: list[dict], n_total: int) -> None:
        """Log a human-readable summary of each score's coverage."""
        logger.info('ClinicalScoreEngine computed %d prognostic scores on %d rows:',
                    len(stats), n_total)
        for s in stats:
            pct = 100.0 * s['computed'] / s['total'] if s['total'] else 0.0
            marker = '' if s['computed'] > 0 else '  [!] all rows missing/not-applicable'
            logger.info('  - %-40s %4d / %d rows (%.1f%%)%s',
                        s['name'], s['computed'], s['total'], pct, marker)

    def compute_standalone(
        self,
        df: pd.DataFrame,
        scores: Optional[list[str]] = None,
        *,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Full self-contained processing for DataFrames from external sources.

        Runs add_structural_derived() to create prerequisite columns, then
        applies all requested score calculators.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain core decoded columns: ER_Status, PR_Status, HER2_Status,
            Ki67_Index, Nottingham_Grade, Tumor_Size_mm, LN_Positive_Count,
            Age_at_Diagnosis, Path_T, Path_N, Path_M, Path_Stage, and
            treatment columns (Chemo_This_Hosp, etc.).

        scores : list[str] or None
            Names of specific calculators to run, or None for all.

        Returns
        -------
        pd.DataFrame
            df with both structural derived columns AND score columns appended.
        """
        from tcr_decoder.derived import add_structural_derived
        result = add_structural_derived(df)
        return self.compute(result, scores=scores, verbose=verbose)
