"""
TCRPipeline — Full decode + score pipeline.

Composes Module 1 (TCRDecoder) and Module 2 (ClinicalScoreEngine) into a
single convenient entry point.  For users who want the complete experience:
raw TCR Excel → decoded clinical DataFrame + prognostic scores → Excel report.

Usage
-----
# One-liner:
from tcr_decoder.pipeline import TCRPipeline
TCRPipeline('raw_data.xlsx').run('output_with_scores.xlsx')

# Step-by-step for inspection at each stage:
pipeline = TCRPipeline('raw_data.xlsx')
pipeline.run_decode()               # Step 1: decode (no scores yet)
decoded_df = pipeline.decoded       # inspect decoded output
pipeline.run_score()                # Step 2: add scores
scored_df  = pipeline.scored        # inspect scored output
pipeline.export('output.xlsx')      # Step 3: export

# Selective scoring:
pipeline = TCRPipeline('raw_data.xlsx')
pipeline.run('output.xlsx', scores=['NPI Score', 'PREDICT Breast v3.0'])

Module independence
-------------------
Both modules can be used independently:

  # Module 1 only:
  from tcr_decoder import TCRDecoder
  decoded_df = TCRDecoder('data.xlsx').decode().clean

  # Module 2 only (external DataFrame):
  from tcr_decoder.scores import ClinicalScoreEngine
  scored_df = ClinicalScoreEngine().compute_standalone(any_df)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from tcr_decoder.core import TCRDecoder
from tcr_decoder.scores.engine import ClinicalScoreEngine


class TCRPipeline:
    """Full TCR decode + clinical score pipeline.

    Wraps TCRDecoder (Module 1) and ClinicalScoreEngine (Module 2) with a
    clean orchestration interface.

    Parameters
    ----------
    input_path : str or Path
        Path to the TCR Excel file.
    sheet_name : str
        Sheet to read (default: 'All_Fields_Decoded').
    cancer_group : str or None
        Override cancer type auto-detection. One of: breast, lung, colorectum,
        liver, cervix, stomach, thyroid, prostate, nasopharynx, endometrium,
        generic.  None = auto-detect from TCODE1.
    """

    def __init__(
        self,
        input_path: Union[str, Path],
        sheet_name: str = 'All_Fields_Decoded',
        cancer_group: Optional[str] = None,
    ) -> None:
        self.decoder = TCRDecoder(input_path, sheet_name, cancer_group)
        self.engine  = ClinicalScoreEngine()
        self._scored_df: Optional[pd.DataFrame] = None

    # ── Stage accessors ───────────────────────────────────────────────────────

    @property
    def decoded(self) -> pd.DataFrame:
        """Decoded DataFrame after run_decode() (no scores yet)."""
        return self.decoder.clean

    @property
    def scored(self) -> pd.DataFrame:
        """Scored DataFrame after run_score()."""
        if self._scored_df is None:
            raise RuntimeError('Call run_score() or run() first.')
        return self._scored_df

    # ── Pipeline stages ───────────────────────────────────────────────────────

    def run_decode(self, skip_input_check: bool = False) -> 'TCRPipeline':
        """Stage 1 — Load and decode the TCR Excel file.

        Produces decoded fields + structural derived variables (ER_Percent,
        T_Simple, OS_Months, Any_Chemotherapy, etc.).  No scores yet.
        """
        self.decoder.load(skip_input_check).decode().validate()
        return self

    def run_score(
        self,
        scores: Optional[list[str]] = None,
    ) -> 'TCRPipeline':
        """Stage 2 — Apply clinical score calculators.

        Parameters
        ----------
        scores : list[str] or None
            Restrict to specific calculators by name, or None for all.
        """
        self._scored_df = self.engine.compute(self.decoded, scores=scores)
        self.decoder._clean_df = self._scored_df   # keeps export() in sync
        return self

    def export(self, output_path: Union[str, Path]) -> Path:
        """Stage 3 — Export to multi-sheet Excel workbook."""
        return self.decoder.export(output_path)

    # ── Convenience ──────────────────────────────────────────────────────────

    def run(
        self,
        output_path: Union[str, Path],
        scores: Optional[list[str]] = None,
        skip_input_check: bool = False,
    ) -> Path:
        """One-liner: decode → score → export.

        Parameters
        ----------
        output_path : str or Path
            Destination Excel file path.
        scores : list[str] or None
            Restrict to specific score calculators, or None for all.
        skip_input_check : bool
            Skip pre-decode input validation.
        """
        return (
            self.run_decode(skip_input_check)
                .run_score(scores)
                .export(output_path)
        )
