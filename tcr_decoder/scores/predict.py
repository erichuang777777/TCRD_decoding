"""PREDICT Breast v3.0 prognostic score calculator.

Implements the PREDICT Breast v3.0 competing-risks Cox model as a BaseScore
subclass.  The core mathematics are ported from the official Cambridge / Winton
Centre R package (WintonCentre/predictv30r, R/benefits30.R).

References
----------
Wishart GC et al. Breast Cancer Res. 2010;12:R18  (PREDICT original)
Candido dos Reis FJ et al. Breast Cancer Res. 2017;19:86  (PREDICT v2)
Jenkins G et al. JNCO Open. 2023;6:e220898  (PREDICT v3.0)
WintonCentre/predictv30r (GitHub, benefits30.R)  — canonical R source

Model overview
--------------
Three sub-models run simultaneously for each patient:

  1. Breast cancer mortality (ER+ or ER- FP-transformed Cox model)
  2. Other-cause mortality  (Gompertz-like model)
  3. Competing-risks combination via Fine-Gray approach

Treatment effects (log-HRs) for hormone therapy, chemotherapy,
trastuzumab, radiotherapy, and bisphosphonate are applied on top of the
baseline PI to compute post-treatment survival curves.

Input mapping from TCR columns
------------------------------
  Age_at_Diagnosis   → age (years)
  Tumor_Size_mm      → size (mm)
  LN_Positive_Count  → positive lymph nodes (count)
  Nottingham_Grade   → grade 1/2/3 (extracted from decoded string)
  ER_Percent         → ER status (≥1 % = positive)
  PR_Percent         → PR status (≥1 % = positive)
  HER2_Status        → HER2 binary (decoded string)
  Ki67_Index         → Ki67 binary (≥14 % = high, per PREDICT convention)
  Any_Hormone_Therapy → hormone treatment flag
  Any_Chemotherapy   → chemotherapy flag (2nd-gen default)
  Any_Targeted_Therapy + HER2+ → trastuzumab flag
  Any_Radiation      → radiotherapy flag

Columns not in TCR data
-----------------------
  screen (clinical vs screen-detected): default 0 (clinical)
  smoker (current smoker):              default 0 (non-smoker)
  bisphosphonate:                        default 0 (not recorded)
  heart_gy (Gy to heart):               default 1.0 if radiotherapy
  chemo generation:                      default 2 (2nd-gen) if any chemo

Output columns
--------------
  PREDICT_5yr_Surv     : 5-year overall survival (%) with actual treatments
  PREDICT_10yr_Surv    : 10-year overall survival (%) with actual treatments
  PREDICT_5yr_BrMort   : 5-year breast cancer mortality (%) with actual treatments
  PREDICT_10yr_BrMort  : 10-year breast cancer mortality (%) with actual treatments
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import numpy as np
import pandas as pd

from tcr_decoder.scores.base import (
    BaseScore,
    ScoreRegistry,
    extract_ki67_numeric,
    her2_binary,
    extract_grade_numeric,
    evaluate_eligibility,
    _HER2_POS_PAT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# v3.0 MODEL CONSTANTS  (from WintonCentre/predictv30r  R/benefits30.R)
# ---------------------------------------------------------------------------

# ER+ fractional polynomial centering constants
_ER_POS = {
    'age_beta_1':        34.53642,
    'age_beta_2':       -34.20342,
    'age_mfp1_center':   0.0287449295,   # (age/10)^-2 – center
    'age_mfp2_center':   0.0510121013,   # (age/10)^-2 * log(age/10) – center
    'size_beta':         0.7530729,
    'size_mfp_center':   1.545233938,    # log(size/100) + center
    'nodes_beta':        0.7060723,
    'nodes_mfp_center':  1.387566896,    # log((nodes+1)/10) + center
    'grade_beta':        0.746655,
    'screen_beta':      -0.22763366,
    'her2_pos_beta':     0.2413,
    'her2_neg_beta':    -0.0762,
    'ki67_pos_beta':     0.14904,
    'ki67_neg_beta':    -0.11333,
    'pr_pos_er_pos_beta': -0.0619,
    'pr_neg_er_pos_beta':  0.2624,
}

# ER- fractional polynomial centering constants
_ER_NEG = {
    'age_beta_1':        0.0089827,
    'age_beta_2':        0.0,
    'age_mfp1_center':  56.3254902,      # age – center
    'size_beta':         2.093446,
    'size_mfp_center':   0.5090456276,   # (size/100)^0.5 – center
    'nodes_beta':        0.6260541,
    'nodes_mfp_center':  1.086916249,    # log((nodes+1)/10) + center
    'grade_beta':        1.129091,       # applied to binary (grade>=2)
    'screen_beta':       0.0,
    'her2_pos_beta':     0.2413,
    'her2_neg_beta':    -0.0762,
    'ki67_pos_beta':     0.0,
    'ki67_neg_beta':     0.0,
    'pr_pos_er_neg_beta': -0.2231,
    'pr_neg_er_neg_beta':  0.0296,
}

# Other (non-breast) mortality model
_OTHER = {
    'age_coef':        0.0698252,
    'age_sq_center':  34.23391957,
    'h_oth_intercept': -6.052919,
    'h_oth_log_t':      1.079863,
    'h_oth_sqrt_t':     0.3255321,
}

# Baseline breast hazard
_BASELINE_BR = {
    'er_pos_intercept':   0.7424402,
    'er_pos_t_neg05':    -7.527762,
    'er_pos_logt_neg05': -1.812513,
    'er_neg_intercept':  -1.156036,
    'er_neg_t_neg2':      0.4707332,
    'er_neg_t_neg1':     -3.51355,
}

# Radiotherapy / competing-risks constants (v3.0)
_RAD = {
    'r_prop':      0.69,
    'r_breast':    0.82,
    'r_other_v30': 1.04,
    'c_other':     1.2,
}

# Smoking model constants
_SMOKING = {
    'smoker_prop': 0.1,
    'smoker_rr':   2.0,
    'cvd_prop':    0.25,
}

# ---------------------------------------------------------------------------
# INTERNAL SCALAR CORE  (ported from benefits30.R)
# ---------------------------------------------------------------------------

def _smoker_beta(smoker: int) -> float:
    sp  = _SMOKING['smoker_prop']
    rr  = _SMOKING['smoker_rr']
    cvd = _SMOKING['cvd_prop']
    rr_acm = cvd * rr + 1 - cvd
    denom   = 1 - sp + rr_acm * sp
    return math.log(rr_acm / denom) if smoker else math.log(1.0 / denom)


def _r_base(r_other: float) -> tuple[float, float]:
    rp = _RAD['r_prop']
    rb = _RAD['r_breast']
    r_base_br  = math.log(1.0 / ((1 - rp) + rp * rb))
    r_base_oth = math.log(1.0 / ((1 - rp) + rp * (r_other ** 2)))
    return r_base_br, r_base_oth


def _predict_v30_scalar(
    age_start:  float,
    size:       float,
    nodes:      float,
    grade:      float,
    er:         int,
    her2:       int,
    ki67:       int,
    pr:         int,
    generation: int,
    horm:       int,
    traz:       int,
    bis:        int,
    radio:      int,
    screen:     int = 0,
    smoker:     int = 0,
    heart_gy:   float = 1.0,
) -> Optional[dict]:
    """Scalar v3.0 computation for a single patient.

    Parameters
    ----------
    grade : 1/2/3/9 — 9 is imputed to 2.13 (unknown).
    her2  : 1=pos, 0=neg, 9=unknown.
    ki67  : 1=high(≥14%), 0=low, 9=unknown.
    pr    : 1=pos, 0=neg, 9=unknown.
    screen: 0=clinical, 1=screen-detected, 2=unknown→0.204.

    Returns
    -------
    dict with keys  'pred_cum_all', 'pred_cum_br'  (arrays length 15)
    for the patient's actual treatment combination, or None if essential
    data (age, size, nodes, grade, er) is missing / NaN / out of range.

    Returns None for:
        - any essential input NaN / None
        - age_start <= 0 or age_start > 120 (implausible; FP transforms diverge)
        - size <= 0 (log(0) crash in ER+ model)
        - nodes < 0 (log of negative in ER+ and ER- models)
        - er not in {0, 1} (cannot select model)
    """
    # ---- 0. Input validation — refuse rather than crash on bad data ---------
    for v, name in [(age_start, 'age_start'), (size, 'size'),
                    (nodes, 'nodes'), (grade, 'grade')]:
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None

    if age_start <= 0 or age_start > 120:
        return None
    if size <= 0:          # Cuzick FP transforms use log(size/100) — size must be >0
        return None
    if nodes < 0:          # LN count cannot be negative; log((nodes+1)/10) would crash
        return None
    if er not in (0, 1):
        return None

    # ---- 1. Imputation -------------------------------------------------------
    screen_v = 0.204 if screen == 2 else float(screen)
    grade_v  = 2.13  if grade == 9  else float(grade)

    time = np.arange(1, 16, dtype=float)

    # ---- 2. Grade variable (ER- uses binary ≥2) ------------------------------
    grade_val = grade_v if er == 1 else (1.0 if grade_v >= 2 else 0.0)

    # ---- 3. FP transforms ----------------------------------------------------
    if er == 1:
        P = _ER_POS
        age_mfp_1  = (age_start / 10) ** -2 - P['age_mfp1_center']
        age_mfp_2  = (age_start / 10) ** -2 * math.log(age_start / 10) - P['age_mfp2_center']
        size_mfp   = math.log(size / 100) + P['size_mfp_center']
        nodes_mfp  = math.log((nodes + 1) / 10) + P['nodes_mfp_center']
        screen_beta = P['screen_beta']
    else:
        P = _ER_NEG
        age_mfp_1  = age_start - P['age_mfp1_center']
        age_mfp_2  = 0.0
        size_mfp   = (size / 100) ** 0.5 - P['size_mfp_center']
        nodes_mfp  = math.log((nodes + 1) / 10) + P['nodes_mfp_center']
        screen_beta = P['screen_beta']

    # ---- 4. Biomarker betas --------------------------------------------------
    her2_beta = (
        P['her2_pos_beta'] if her2 == 1 else
        P['her2_neg_beta'] if her2 == 0 else 0.0
    )

    if   ki67 == 1 and er == 1: ki67_beta = _ER_POS['ki67_pos_beta']
    elif ki67 == 0 and er == 1: ki67_beta = _ER_POS['ki67_neg_beta']
    else:                        ki67_beta = 0.0

    if   pr == 1 and er == 1: pr_beta = _ER_POS['pr_pos_er_pos_beta']
    elif pr == 0 and er == 1: pr_beta = _ER_POS['pr_neg_er_pos_beta']
    elif pr == 1 and er == 0: pr_beta = _ER_NEG['pr_pos_er_neg_beta']
    elif pr == 0 and er == 0: pr_beta = _ER_NEG['pr_neg_er_neg_beta']
    else:                      pr_beta = 0.0

    # ---- 5. Radiotherapy baseline correction ---------------------------------
    r_other = _RAD['r_other_v30']
    r_base_br, r_base_oth = _r_base(r_other)

    # ---- 6. Smoking ----------------------------------------------------------
    smoker_beta = _smoker_beta(smoker)

    # ---- 7. Other mortality PI -----------------------------------------------
    c_oth = math.log(_RAD['c_other']) if generation > 0 else 0.0
    r_oth = math.log(r_other) * heart_gy if radio else 0.0
    mi = (
        _OTHER['age_coef'] * ((age_start / 10) ** 2 - _OTHER['age_sq_center'])
        + r_base_oth + smoker_beta + c_oth + r_oth
    )

    # ---- 8. Breast cancer PI -------------------------------------------------
    pi = (
        P['age_beta_1'] * age_mfp_1
        + P['age_beta_2'] * age_mfp_2
        + P['size_beta']  * size_mfp
        + P['nodes_beta'] * nodes_mfp
        + P['grade_beta'] * grade_val
        + screen_beta * screen_v
        + her2_beta + ki67_beta + pr_beta
        + r_base_br
    )

    # ---- 9. Treatment log-HRs ------------------------------------------------
    c_rx = 0.0 if generation == 0 else (-0.248 if generation == 2 else -0.446)
    h    = -0.3857 if (horm == 1 and er == 1) else 0.0
    h10  = np.where(time <= 10, h, -0.26 + h) if h != 0.0 else np.zeros(15)
    t_rx = -0.3567 if (her2 == 1 and traz == 1) else 0.0
    b    = -0.198  if bis == 1 else 0.0
    r_br = math.log(_RAD['r_breast']) if radio else 0.0   # log(0.82)

    # Full treatment combination for this patient
    # (h10 = extended 5yr+ hormone; h = standard)
    use_h10 = horm == 1 and er == 1    # extended hormone benefit
    h_vec = h10 if use_h10 else np.full(15, h)
    rx_pi = h_vec + r_br + c_rx + t_rx + b + pi

    # ---- 10. Other mortality baseline ----------------------------------------
    base_m_cum_oth = np.exp(
        _OTHER['h_oth_intercept']
        + _OTHER['h_oth_log_t'] * np.log(time)
        + _OTHER['h_oth_sqrt_t'] * time ** 0.5
    )
    s_cum_oth = np.exp(-np.exp(mi) * base_m_cum_oth)
    m_cum_oth = 1 - s_cum_oth
    m_oth = np.diff(np.concatenate([[0.0], m_cum_oth]))

    # ---- 11. Baseline breast mortality ---------------------------------------
    if er == 1:
        base_m_cum_br = np.exp(
            _BASELINE_BR['er_pos_intercept']
            + _BASELINE_BR['er_pos_t_neg05']    / time ** 0.5
            + _BASELINE_BR['er_pos_logt_neg05'] * np.log(time) / time ** 0.5
        )
    else:
        base_m_cum_br = np.exp(
            _BASELINE_BR['er_neg_intercept']
            + _BASELINE_BR['er_neg_t_neg2'] / time ** 2
            + _BASELINE_BR['er_neg_t_neg1'] / time
        )

    base_m_br = np.diff(np.concatenate([[0.0], base_m_cum_br]))

    # ---- 12. Competing-risks survival (with actual treatment) ----------------
    m_br_annual  = base_m_br * np.exp(rx_pi)
    m_cum_br     = np.cumsum(m_br_annual)
    s_cum_br     = np.exp(-m_cum_br)
    m_cum_br_risk = 1 - s_cum_br
    m_br_risk    = np.diff(np.concatenate([[0.0], m_cum_br_risk]))

    m_cum_all = 1 - s_cum_oth * s_cum_br
    m_all     = np.diff(np.concatenate([[0.0], m_cum_all]))

    prop_br      = np.where(
        (m_br_risk + m_oth) > 0,
        m_br_risk / (m_br_risk + m_oth),
        0.0,
    )
    pred_m_br   = prop_br * m_all
    pred_cum_br  = np.cumsum(pred_m_br)
    pred_cum_all = pred_cum_br + np.cumsum(m_all - pred_m_br)

    return {
        'pred_cum_all': pred_cum_all,   # all-cause mortality (cumulative, %)
        'pred_cum_br':  pred_cum_br,    # breast mortality (cumulative, %)
    }


# ---------------------------------------------------------------------------
# BaseScore SUBCLASS
# ---------------------------------------------------------------------------

@ScoreRegistry.register
class PREDICTScore(BaseScore):
    """PREDICT Breast v3.0 — competing-risks survival model.

    Computes 5- and 10-year overall survival and breast cancer mortality
    predictions incorporating the patient's actual adjuvant treatments.

    Applicable only when both ER status and tumour size/nodes/grade are
    available.  For non-breast cancer cases (missing ER_Percent, HER2_Status,
    Tumor_Size_mm, LN_Positive_Count, and Nottingham_Grade), outputs are set
    to 'Not applicable'.

    Assumptions for data not recorded in the registry
    --------------------------------------------------
    * Detection method (screen vs clinical): clinical (screen=0)
    * Smoking status: non-smoker (smoker=0)
    * Bisphosphonate use: no (bis=0)
    * Radiotherapy heart dose: 1.0 Gy if radiotherapy received
    * Chemotherapy generation: 2nd-generation if any chemotherapy recorded
    """

    NAME = 'PREDICT Breast v3.0'
    CITATION = (
        'Jenkins G et al. JNCO Open. 2023;6:e220898. '
        'Model source: WintonCentre/predictv30r (benefits30.R)'
    )
    REQUIRED_COLS = ['ER_Percent', 'Tumor_Size_mm', 'LN_Positive_Count']
    OUTPUT_COLS = [
        'PREDICT_5yr_Surv',
        'PREDICT_10yr_Surv',
        'PREDICT_5yr_BrMort',
        'PREDICT_10yr_BrMort',
        'PREDICT_Eligibility',   # audit: why row was skipped (empty if eligible)
    ]

    # Ki67 positivity threshold used by PREDICT v3.0 web tool (≥14 % = high)
    KI67_CUTOFF = 14.0

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── Population eligibility gate ───────────────────────────────────────
        # PREDICT v3.0 is only validated for non-metastatic invasive breast
        # cancer in adult women aged 25-85.  Applying it outside this
        # population produces clinically meaningless numbers.
        elig = evaluate_eligibility(
            df,
            require_invasive=True,
            require_non_metastatic=True,
            require_female=True,
            predict_age_bounds=True,
        )

        # Record the ineligibility reason alongside the score for audit
        df['PREDICT_Eligibility'] = elig.reason.where(~elig.eligible, '')

        # ── Extract inputs ────────────────────────────────────────────────────
        age   = pd.to_numeric(df.get('Age_at_Diagnosis',
                                     pd.Series(np.nan, index=df.index)),
                              errors='coerce')
        size  = pd.to_numeric(df.get('Tumor_Size_mm',
                                     pd.Series(np.nan, index=df.index)),
                              errors='coerce')
        nodes = pd.to_numeric(df.get('LN_Positive_Count',
                                     pd.Series(np.nan, index=df.index)),
                              errors='coerce')
        grade_raw = extract_grade_numeric(
            df.get('Nottingham_Grade', pd.Series('', index=df.index))
        )
        # Unknown grade → 9 (will be imputed to 2.13 inside core)
        grade = grade_raw.fillna(9).astype(float)

        er_pct = pd.to_numeric(df.get('ER_Percent',
                                      pd.Series(np.nan, index=df.index)),
                               errors='coerce')
        pr_pct = pd.to_numeric(df.get('PR_Percent',
                                      pd.Series(np.nan, index=df.index)),
                               errors='coerce')
        ki67_num = extract_ki67_numeric(
            df.get('Ki67_Index', pd.Series('', index=df.index))
        )
        her2_num = her2_binary(
            df.get('HER2_Status', pd.Series('', index=df.index))
        )

        # Binary flags: 1 / 0 / 9(unknown)
        er_flag   = _to_binary_flag(er_pct >= 1.0, er_pct.isna())
        pr_flag   = _to_binary_flag(pr_pct >= 1.0, pr_pct.isna())
        ki67_flag = _to_binary_flag(ki67_num >= self.KI67_CUTOFF, ki67_num.isna())
        her2_flag = _to_binary_flag(her2_num == 1.0, her2_num.isna())

        # ── Treatment flags ───────────────────────────────────────────────────
        # Three states per treatment: Yes / No / Unknown.  Class-3 patients
        # (Treatment_Data_Incomplete=True) have Any_* = 'Unknown (treated elsewhere)';
        # for those rows PREDICT cannot honestly estimate a post-treatment
        # survival and we return NaN.  Silently mapping 'Unknown' to 'No' would
        # systematically inflate predicted mortality.
        def _tx_state(col: str) -> pd.Series:
            s = (
                df.get(col, pd.Series('', index=df.index))
                .fillna('').astype(str).str.strip().str.lower()
            )
            # 'yes' -> 1, 'no' -> 0, anything else (incl. 'unknown...') -> NaN
            result = pd.Series(np.nan, index=df.index, dtype=float)
            result[s == 'yes'] = 1.0
            result[s == 'no']  = 0.0
            return result

        horm_state  = _tx_state('Any_Hormone_Therapy')
        chemo_state = _tx_state('Any_Chemotherapy')
        radio_state = _tx_state('Any_Radiation')
        traz_state  = _tx_state('Any_Targeted_Therapy')

        # Row is "treatment-resolvable" iff all four flags are known.
        tx_resolvable = (
            horm_state.notna() & chemo_state.notna()
            & radio_state.notna() & traz_state.notna()
        )

        # ── Row-by-row computation ────────────────────────────────────────────
        # Use positional iloc to avoid non-unique-index ambiguity.
        n = len(df)
        surv_5    = np.full(n, np.nan, dtype=float)
        surv_10   = np.full(n, np.nan, dtype=float)
        brmort_5  = np.full(n, np.nan, dtype=float)
        brmort_10 = np.full(n, np.nan, dtype=float)

        n_skipped_er_unknown = 0
        n_skipped_tx_unknown = 0
        n_skipped_bad_input  = 0
        n_skipped_ineligible = 0
        n_computed = 0

        for i in range(n):
            # Population eligibility (M1 / Tis / male / out-of-age)
            if not elig.eligible.iloc[i]:
                n_skipped_ineligible += 1
                continue

            # ER must be known to select the breast model
            if er_flag.iloc[i] == 9:
                n_skipped_er_unknown += 1
                continue

            # Treatment must be fully resolved — Class-3 Unknown rows are skipped
            if not tx_resolvable.iloc[i]:
                n_skipped_tx_unknown += 1
                continue

            result = _predict_v30_scalar(
                age_start  = float(age.iloc[i])   if pd.notna(age.iloc[i])   else float('nan'),
                size       = float(size.iloc[i])  if pd.notna(size.iloc[i])  else float('nan'),
                nodes      = float(nodes.iloc[i]) if pd.notna(nodes.iloc[i]) else float('nan'),
                grade      = float(grade.iloc[i]),
                er         = int(er_flag.iloc[i]),
                her2       = int(her2_flag.iloc[i]),
                ki67       = int(ki67_flag.iloc[i]),
                pr         = int(pr_flag.iloc[i]),
                generation = int(chemo_state.iloc[i]) * 2,
                horm       = int(horm_state.iloc[i]),
                traz       = int(traz_state.iloc[i]) * int(her2_flag.iloc[i] == 1),
                bis        = 0,
                radio      = int(radio_state.iloc[i]),
                screen     = 0,
                smoker     = 0,
                heart_gy   = 1.0,
            )

            if result is None:
                n_skipped_bad_input += 1
                continue

            pca = result['pred_cum_all']
            pcb = result['pred_cum_br']
            surv_5[i]    = round(100.0 * (1 - pca[4]),  1)
            surv_10[i]   = round(100.0 * (1 - pca[9]),  1)
            brmort_5[i]  = round(100.0 * pcb[4], 1)
            brmort_10[i] = round(100.0 * pcb[9], 1)
            n_computed += 1

        df['PREDICT_5yr_Surv']    = surv_5
        df['PREDICT_10yr_Surv']   = surv_10
        df['PREDICT_5yr_BrMort']  = brmort_5
        df['PREDICT_10yr_BrMort'] = brmort_10

        # Warn the user about skipped rows so missing outputs aren't silent.
        n_any_skipped = (
            n_skipped_ineligible + n_skipped_er_unknown
            + n_skipped_tx_unknown + n_skipped_bad_input
        )
        if n_any_skipped:
            logger.warning(
                'PREDICT v3.0: computed %d/%d rows. '
                'Skipped: %d population-ineligible (M1/Tis/male/age bounds), '
                '%d ER-unknown, %d treatment-unknown (Class-3), '
                '%d invalid numeric inputs.',
                n_computed, n,
                n_skipped_ineligible, n_skipped_er_unknown,
                n_skipped_tx_unknown, n_skipped_bad_input,
            )

        return df


# ---------------------------------------------------------------------------
# UTILITY
# ---------------------------------------------------------------------------

def _to_binary_flag(positive_mask: pd.Series, unknown_mask: pd.Series) -> pd.Series:
    """Convert boolean masks to PREDICT int codes: 1 (pos), 0 (neg), 9 (unknown)."""
    result = pd.Series(9, index=positive_mask.index, dtype=int)
    result[~unknown_mask & positive_mask]  = 1
    result[~unknown_mask & ~positive_mask] = 0
    return result
