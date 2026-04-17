"""
Base class and registry for breast cancer prognostic score calculators.

Every calculator:
  - inherits from BaseScore
  - declares NAME, CITATION, REQUIRED_COLS, OUTPUT_COLS
  - implements calculate(df) → df  (returns copy, never mutates)
  - is registered via ScoreRegistry.register()

Usage:
    from tcr_decoder.scores import ScoreRegistry
    df = ScoreRegistry.apply_all(df)
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ─── Clinical constants (shared) ───────────────────────────────────────────────

# ER positivity threshold per ASCO/CAP 2020 guidelines (Allison K et al).
# Tumors with ER 1-10% are "ER Low Positive" and remain ER+.
ER_POSITIVE_THRESHOLD_PCT: float = 1.0

# PREDICT Breast web tool bounds
PREDICT_AGE_MIN: float = 25.0
PREDICT_AGE_MAX: float = 85.0


# ─── Eligibility gate ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EligibilityResult:
    """Per-row eligibility verdict for a clinical score.

    eligible : pd.Series of bool — True if the row should be scored
    reason   : pd.Series of str  — empty if eligible, else the first
               failing criterion (one sentence, human-readable)
    """
    eligible: pd.Series
    reason: pd.Series


def evaluate_eligibility(
    df: pd.DataFrame,
    *,
    require_invasive: bool = False,
    require_non_metastatic: bool = False,
    require_er_positive: bool = False,
    require_female: bool = False,
    require_endocrine_therapy: bool = False,
    predict_age_bounds: bool = False,
) -> EligibilityResult:
    """Evaluate clinical applicability of a score to each row of df.

    This centralises the population-eligibility logic so that every
    calculator can refuse to produce a number for a patient who is
    outside the published model's training population.  Calculators
    that run silently on ineligible patients produce numbers that
    LOOK valid but carry no clinical meaning — a worse failure mode
    than crashing.

    Parameters
    ----------
    require_invasive : reject rows where Path_T starts with 'Tis' /
        'pTis' / contains 'in situ'.  Breast NPI, PEPI, PREDICT, and
        Molecular Subtype are defined for invasive disease only.
    require_non_metastatic : reject rows where Path_M or M_Simple
        contains 'M1' or where Path_Stage is stage IV.  PREDICT, CTS5,
        IHC4, PEPI are non-metastatic tools.
    require_er_positive : reject rows where ER_Percent < 1 %.  CTS5,
        IHC4, PEPI apply only to ER+ disease.
    require_female : reject rows where Sex != 'Female'.  All six
        calculators were trained on female-only cohorts.
    require_endocrine_therapy : reject rows where Any_Hormone_Therapy
        != 'Yes'.  CTS5 is conditional on 5y of completed endocrine
        therapy.
    predict_age_bounds : reject rows where Age_at_Diagnosis is outside
        the PREDICT Breast web-tool bounds [25, 85].
    """
    n = len(df)
    eligible = pd.Series(True, index=df.index)
    reason = pd.Series('', index=df.index, dtype=object)

    def _fail(mask: pd.Series, msg: str) -> None:
        """Mark rows with mask=True as ineligible (only if not already)."""
        nonlocal eligible, reason
        fresh = mask & (reason == '')
        reason[fresh] = msg
        eligible = eligible & ~mask

    # ── Sex gate ──────────────────────────────────────────────────────────────
    if require_female and 'Sex' in df.columns:
        sex = df['Sex'].fillna('').astype(str).str.strip().str.lower()
        not_female = (sex != '') & (sex != 'female') & (sex != 'f')
        _fail(not_female,
              'Sex is not Female (score not validated in male breast cancer)')

    # ── Invasive (reject Tis / DCIS / in situ) ───────────────────────────────
    if require_invasive:
        t_col = (
            df['Path_T'] if 'Path_T' in df.columns
            else df['T_Simple'] if 'T_Simple' in df.columns
            else pd.Series('', index=df.index)
        )
        t_str = t_col.fillna('').astype(str).str.lower()
        is_situ = (
            t_str.str.contains(r'\btis\b', regex=True, na=False)
            | t_str.str.contains('in situ', na=False)
            | t_str.str.contains('dcis', na=False)
            | t_str.str.contains('ptis', na=False)
        )
        _fail(is_situ,
              'In situ disease (Tis/DCIS); score is for invasive breast cancer')

    # ── Non-metastatic gate (reject M1) ──────────────────────────────────────
    if require_non_metastatic:
        # Check Path_M, M_Simple, AND Clinical_M — any one indicating M1 rejects
        m_candidates: list[pd.Series] = []
        for col in ('Path_M', 'M_Simple', 'Clinical_M'):
            if col in df.columns:
                m_candidates.append(df[col].fillna('').astype(str).str.upper())
        # Match 'M1' optionally preceded by p/c/y (TNM prefix) and optionally
        # followed by a letter suffix (a/b/c).  Rejects 'pM1', 'cM1', 'M1a',
        # but does NOT match 'M10' (impossible in TNM anyway).  Does NOT match
        # 'M0', 'MX', 'pM0'.
        m1_pat = r'(?:^|[^0-9A-Za-z])M1(?![0-9])'
        is_m1 = pd.Series(False, index=df.index)
        for m_str in m_candidates:
            is_m1 = is_m1 | m_str.str.contains(m1_pat, regex=True, na=False)
        _fail(is_m1,
              'Distant metastasis (M1); score is not validated in stage IV disease')

        # Also reject if Path_Stage / Stage_Simple / Clinical_Stage indicates stage IV
        for col in ('Path_Stage', 'Stage_Simple', 'Clinical_Stage'):
            if col in df.columns:
                stage = df[col].fillna('').astype(str).str.upper()
                is_stage_iv = stage.str.contains(r'STAGE\s*IV', regex=True, na=False)
                _fail(is_stage_iv,
                      'Stage IV disease; score is not validated in metastatic disease')

    # ── ER-positive gate ──────────────────────────────────────────────────────
    if require_er_positive and 'ER_Percent' in df.columns:
        er = pd.to_numeric(df['ER_Percent'], errors='coerce')
        # ER_Percent < 1% → ER-negative per ASCO/CAP 2020
        is_er_neg = er.notna() & (er < ER_POSITIVE_THRESHOLD_PCT)
        _fail(is_er_neg,
              f'ER-negative (ER < {ER_POSITIVE_THRESHOLD_PCT:g}%); '
              f'score is for ER-positive disease only')

    # ── Endocrine therapy gate ───────────────────────────────────────────────
    if require_endocrine_therapy and 'Any_Hormone_Therapy' in df.columns:
        ht = df['Any_Hormone_Therapy'].fillna('').astype(str).str.strip().str.lower()
        no_ht = ht == 'no'
        unknown_ht = (ht != 'yes') & (ht != 'no')
        _fail(no_ht,
              'Did not receive endocrine therapy; CTS5 is conditional on 5y ET completion')
        _fail(unknown_ht,
              'Endocrine therapy status unknown; cannot apply CTS5 safely')

    # ── PREDICT age bounds ────────────────────────────────────────────────────
    if predict_age_bounds and 'Age_at_Diagnosis' in df.columns:
        age = pd.to_numeric(df['Age_at_Diagnosis'], errors='coerce')
        out_of_range = age.notna() & (
            (age < PREDICT_AGE_MIN) | (age > PREDICT_AGE_MAX)
        )
        _fail(out_of_range,
              f'Age outside PREDICT web-tool bounds '
              f'[{PREDICT_AGE_MIN:g}, {PREDICT_AGE_MAX:g}]')

    return EligibilityResult(eligible=eligible, reason=reason)


# ─── Shared helpers ────────────────────────────────────────────────────────────

def _normalize_fullwidth_digits(s: pd.Series) -> pd.Series:
    """Convert full-width digits/punctuation (U+FF10..FF19, FF05 %) to ASCII.

    Legacy Taiwan / Japan registry exports sometimes contain full-width
    digits such as '２０％'.  Without normalization these silently fail to
    parse and become NaN.
    """
    # Full-width digits 0-9 at U+FF10 .. U+FF19
    trans = {chr(0xFF10 + i): str(i) for i in range(10)}
    trans['\uFF05'] = '%'   # full-width percent sign
    trans['\uFF0C'] = ','   # full-width comma
    trans['\uFF0E'] = '.'   # full-width full stop
    return s.map(lambda x: ''.join(trans.get(c, c) for c in str(x)))


def extract_ki67_numeric(series: pd.Series) -> pd.Series:
    """Extract numeric Ki67 % from decoded Ki67_Index strings.

    Normalises several formats:
        '15% (Intermediate)'    → 15.0
        '0.5%'                  → 0.5
        '15'  (bare integer)    → 15.0   (CSV imports may drop the %)
        '<1%'                   → 0.5    (conventional "rare positive")
        '10-20%'                → 15.0   (midpoint of the range)
        '25,5%'                 → 25.5   (European decimal comma)
        '25.5%'                 → 25.5
        '≈20%'                  → 20.0
        '２０％' (full-width)   → 20.0
        'Unknown' / 'high'/ ''  → NaN

    Values that cannot be parsed, or that fall outside [0, 100], return NaN.
    """
    # Handle pandas nullable numeric dtypes (Int64, Float64) directly — they
    # can be passed through as numeric without going through the string path.
    if pd.api.types.is_numeric_dtype(series):
        result = pd.to_numeric(series, errors='coerce').astype(float)
        return result.where((result >= 0) & (result <= 100))

    s = series.fillna('').astype(str).str.strip()

    # Normalise full-width characters first (Taiwan / Japan legacy exports)
    s = _normalize_fullwidth_digits(s)

    # Normalise decimal commas → dots  (e.g. '25,5%' → '25.5%')
    s = s.str.replace(r'(\d),(\d)', r'\1.\2', regex=True)

    result = pd.Series(np.nan, index=series.index, dtype=float)

    # Case A: '<1%'  → encode as 0.5 (common convention)
    lt1 = s.str.match(r'^\s*<\s*1\s*%?\s*$')
    result[lt1] = 0.5

    # Case B: range 'A-B%' or 'A–B%' → midpoint
    rng = s.str.extract(
        r'^\s*([\d.]+)\s*[-\u2013\u2014]\s*([\d.]+)\s*%?\s*$',
        expand=True,
    )
    rng_mask = rng[0].notna() & rng[1].notna() & result.isna()
    if rng_mask.any():
        lo = pd.to_numeric(rng[0][rng_mask], errors='coerce')
        hi = pd.to_numeric(rng[1][rng_mask], errors='coerce')
        midpoint = (lo + hi) / 2.0
        result.loc[rng_mask] = midpoint

    # Case C: single number like '15%', '15.5%', or bare '15'
    # The trailing % is optional to accommodate CSV imports where it has
    # been stripped.  To avoid false positives on random numeric codes,
    # we still anchor to start/end of string.
    single = s.str.extract(
        r'^\s*[\u2248~]?\s*([\d.]+)\s*%?\s*(?:\(.*\))?\s*$',
        expand=False,
    )
    single_num = pd.to_numeric(single, errors='coerce')
    single_mask = single_num.notna() & result.isna()
    result[single_mask] = single_num[single_mask]

    # Also try '15% (Intermediate)' style where trailing qualifier is any text
    qualified = s.str.extract(r'^\s*([\d.]+)\s*%', expand=False)
    qual_num = pd.to_numeric(qualified, errors='coerce')
    qual_mask = qual_num.notna() & result.isna()
    result[qual_mask] = qual_num[qual_mask]

    # Loose fallback: find ANY number followed by % (e.g. 'approximately 25%').
    # Requires that the number is NOT preceded by a minus sign (so '-5%'
    # rejects) and is not part of a larger token (so '150%' rejects because
    # the leading '1' is attached to nothing that brings it into range).
    # Use a capturing group with a non-digit/non-minus preceding char.
    loose = s.str.extract(r'(?:^|[^-\d.])([\d.]+)\s*%', expand=False)
    loose_num = pd.to_numeric(loose, errors='coerce')
    loose_mask = loose_num.notna() & result.isna()
    result[loose_mask] = loose_num[loose_mask]

    # Reject out-of-range values
    result = result.where((result >= 0) & (result <= 100))
    return result


# HER2 status parsing — accepts em-dash (—, U+2014), en-dash (–, U+2013),
# and ASCII hyphen (-) as the separator between IHC result and interpretation.
_DASH = r'[\u2014\u2013\-]'
_HER2_POS_PAT = (
    rf'{_DASH}\s*Positive'
    r'|\bCISH Positive\b'
    r'|\bOther test.*Positive\b'
)
_HER2_NEG_PAT = (
    rf'{_DASH}\s*Negative'
    r'|\bCISH Negative\b'
    r'|\bOther test.*Negative\b'
)


def her2_binary(series: pd.Series) -> pd.Series:
    """Return 1.0 (positive), 0.0 (negative), NaN (equivocal/empty) from HER2_Status."""
    s = series.fillna('').astype(str)
    is_pos = s.str.contains(_HER2_POS_PAT, case=False, na=False, regex=True)
    is_neg = s.str.contains(_HER2_NEG_PAT, case=False, na=False, regex=True)
    result = pd.Series(np.nan, index=series.index)
    result[is_pos] = 1.0
    result[is_neg & ~is_pos] = 0.0
    return result


_GRADE_ROMAN_MAP = {'I': 1.0, 'II': 2.0, 'III': 3.0}
_GRADE_UNKNOWN_PAT = re.compile(
    r'^\s*(unknown|not\s*assessed|not\s*tested|未知|不明|not applicable|n/a|NA)\s*$',
    re.IGNORECASE,
)


def extract_grade_numeric(series: pd.Series) -> pd.Series:
    """Extract Nottingham grade (1/2/3) from a Nottingham_Grade text value.

    Accepts multiple common formats:
        'Grade 2'                         → 2.0
        'Grade II'                        → 2.0
        'Score 6 → Grade 2 (Moderate)'    → 2.0
        'G2'                              → 2.0
        '2'                               → 2.0
        'II'                              → 2.0
        'Unknown' / '不明' / 'N/A' / ''   → NaN
        'Grade 4' / 'Grade 0'             → NaN  (invalid, not 1-3)

    Numeric dtypes (Int64, Float64) pass through after range validation.
    """
    # Numeric dtypes: convert and validate
    if pd.api.types.is_numeric_dtype(series):
        num = pd.to_numeric(series, errors='coerce').astype(float)
        return num.where(num.isin([1.0, 2.0, 3.0]))

    s = series.fillna('').astype(str).str.strip()
    result = pd.Series(np.nan, index=series.index, dtype=float)

    # Explicit unknowns → NaN (short-circuit)
    is_unknown = s.map(lambda x: bool(_GRADE_UNKNOWN_PAT.match(x)))
    s_clean = s.where(~is_unknown, '')

    # Pattern 1: 'Grade 1/2/3' explicit digit
    m1 = s_clean.str.extract(r'Grade\s*([123])\b', expand=False, flags=re.IGNORECASE)
    num1 = pd.to_numeric(m1, errors='coerce')
    result[num1.notna()] = num1[num1.notna()]

    # Pattern 2: 'Grade I/II/III' Roman
    m2 = s_clean.str.extract(r'Grade\s*(I{1,3})\b', expand=False, flags=re.IGNORECASE)
    roman_mask = m2.notna() & result.isna()
    if roman_mask.any():
        mapped = m2[roman_mask].str.upper().map(_GRADE_ROMAN_MAP)
        result.loc[roman_mask] = mapped

    # Pattern 3: bare 'G1' / 'G2' / 'G3'
    m3 = s_clean.str.extract(r'^\s*G([123])\b', expand=False, flags=re.IGNORECASE)
    num3 = pd.to_numeric(m3, errors='coerce')
    mask3 = num3.notna() & result.isna()
    result[mask3] = num3[mask3]

    # Pattern 4: bare digit '1'/'2'/'3' only (no 'Grade' prefix)
    m4 = s_clean.str.extract(r'^\s*([123])\s*$', expand=False)
    num4 = pd.to_numeric(m4, errors='coerce')
    mask4 = num4.notna() & result.isna()
    result[mask4] = num4[mask4]

    # Pattern 5: bare Roman numeral 'II' alone
    m5 = s_clean.str.extract(r'^\s*(I{1,3})\s*$', expand=False)
    mask5 = m5.notna() & result.isna()
    if mask5.any():
        mapped = m5[mask5].str.upper().map(_GRADE_ROMAN_MAP)
        result.loc[mask5] = mapped

    return result


# ─── Base class ────────────────────────────────────────────────────────────────

class BaseScore(ABC):
    """Abstract base for a prognostic score calculator.

    Subclasses must set class attributes and implement calculate().
    """

    NAME: ClassVar[str] = ''
    CITATION: ClassVar[str] = ''
    REQUIRED_COLS: ClassVar[list[str]] = []   # at least one must be present to run
    OUTPUT_COLS: ClassVar[list[str]] = []      # columns this calculator produces

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute score and return df copy with new columns appended.

        Must:
          - call df.copy() before mutating
          - use df.get(col, fallback) for all source columns
          - return NaN / '' for rows with missing inputs
          - never raise on missing columns
        """
        ...

    def is_applicable(self, df: pd.DataFrame) -> bool:
        """True if at least one REQUIRED_COL is present in df."""
        return any(c in df.columns for c in self.REQUIRED_COLS)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Safe wrapper: run calculate() or populate OUTPUT_COLS with NaN/sentinels.

        If the calculator is not applicable (REQUIRED_COLS absent), every
        OUTPUT_COL is set to the string 'Not applicable' so downstream code
        can distinguish structural non-applicability from missing values.

        If calculate() raises, a warning is emitted via both the logger and
        `warnings.warn` (so notebook users without logging configured still
        see it), and every OUTPUT_COL is FORCEFULLY overwritten with NaN.
        This prevents stale values from a previous partial run from leaking
        through.
        """
        import warnings as _warnings

        if not self.is_applicable(df):
            df = df.copy()
            for col in self.OUTPUT_COLS:
                df[col] = 'Not applicable'
            return df
        try:
            return self.calculate(df)
        except Exception as exc:
            msg = (
                f'Clinical score {self.NAME!r} failed on this DataFrame '
                f'({type(exc).__name__}: {exc}).  All {len(self.OUTPUT_COLS)} '
                f'output columns have been set to NaN.'
            )
            logger.exception(msg)
            _warnings.warn(msg, RuntimeWarning, stacklevel=2)
            df = df.copy()
            # ALWAYS overwrite — never leave stale pre-existing values.
            for col in self.OUTPUT_COLS:
                df[col] = np.nan
            return df


# ─── Registry ──────────────────────────────────────────────────────────────────

class ScoreRegistry:
    """Central registry for all prognostic score calculators.

    Usage:
        @ScoreRegistry.register
        class MyScore(BaseScore):
            ...

        df = ScoreRegistry.apply_all(df)
        names = ScoreRegistry.list_scores()
    """

    _scores: ClassVar[list[type[BaseScore]]] = []

    @classmethod
    def register(cls, score_cls: type[BaseScore]) -> type[BaseScore]:
        """Class decorator — add score to registry.

        Idempotent: re-importing or reloading the same score module will NOT
        produce a duplicate registration.  Without this guard, a notebook
        user who calls ``importlib.reload(tcr_decoder.scores.npi)`` would
        end up with NPIScore registered twice and every NPI result would
        silently be computed (and logged) twice.

        A registration by *name* (NAME) is treated as the same score even
        if the class object is a different Python object — this keeps
        module reloading safe.
        """
        # Dedupe by class identity AND by NAME to cover both
        # "same object re-registered" and "module reloaded → new class object".
        for existing in cls._scores:
            if existing is score_cls:
                return score_cls
            if existing.NAME and existing.NAME == score_cls.NAME:
                # Replace the old class in-place with the new one so reloads
                # pick up the newer implementation.
                idx = cls._scores.index(existing)
                cls._scores[idx] = score_cls
                return score_cls
        cls._scores.append(score_cls)
        return score_cls

    @classmethod
    def list_scores(cls) -> list[dict]:
        """Return metadata for all registered calculators."""
        return [
            {
                'name': s.NAME,
                'citation': s.CITATION,
                'required_cols': s.REQUIRED_COLS,
                'output_cols': s.OUTPUT_COLS,
            }
            for s in cls._scores
        ]

    @classmethod
    def apply_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Apply every registered score in registration order."""
        for score_cls in cls._scores:
            df = score_cls().apply(df)
        return df

    @classmethod
    def apply_one(cls, name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Apply a single named score."""
        for score_cls in cls._scores:
            if score_cls.NAME == name:
                return score_cls().apply(df)
        raise KeyError(f'Score {name!r} not found. Available: {[s.NAME for s in cls._scores]}')
