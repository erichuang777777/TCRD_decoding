"""
PREDICT Breast v3.0 / v3.1 / v3.2 — Python Implementation
===========================================================

Faithfully ported from the official WintonCentre/predictv30r R package
(master branch, fetched 2026-04-15) and cross-checked against
pengpclab/PREDICTv3 (main branch).

Sources
-------
Primary   : https://github.com/WintonCentre/predictv30r
            R/benefits30.R   — v3.0 original (hardcoded coefficients)
            R/benefits31.R   — v3.1 (CSV-based coefficients, new transforms)
            R/benefits32.R   — v3.2 (v3.1 logic + revised r.base constants)
            R/benefits3010.R — 10-year conditional extension for v3.0
            inst/extdata/coefficients_v3.csv — authoritative CSV values

Secondary : https://github.com/pengpclab/PREDICTv3
            R/PREDICTv3.R    — standalone implementation

IMPORTANT NOTES ON MODEL VERSIONS
----------------------------------
v3.0 (benefits30.R / Predict-v3.0-2022-08-17.R)
    - Uses HARDCODED fractional polynomial transforms (v2.2-style centering)
    - r.other = 1.04   (Taylor 2017)
    - HER2 is a FIXED scalar (no time-varying effect)
    - "year" term is absent from the breast cancer PI

v3.1 (benefits31.R / Predict_v3.1_2023_05_23.R)
    - Uses CSV coefficients with NEW transforms:
      * Age ER+: (age-24)/100)^-0.5 and ((age-24)/100)^2   [age centred at 24]
      * Age ER-: (age-24)/100 and ((age-24)/100)*log((age-24)/100)
      * Size ER+: 1 - exp(-size/20)    ER-: log(size)
      * Nodes: log(nodes+1)  [both ER+ and ER-]
    - HER2 is TIME-VARYING for ER+ (lookup table, decays to 0 by year 9)
    - Adds "year" covariate (year-2000) to both breast and other mortality PI
    - r.other = 1.02   (revised from 1.04)
    - h10 extra benefit = -0.301 (was -0.26 in v3.0)

v3.2 (benefits32.R)
    - Identical to v3.1 but uses hardcoded baseline adjustments:
      r.base.br = 0.7 (ER+) or 0.55 (ER-)   [instead of computed from r.prop]
      r.base.oth = -0.062                     [instead of computed from r.prop]

All versions use the same treatment log-HRs (c, h, t, b) and smoking model.
"""

import math
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# SECTION 1 — coefficients_v3.csv  (authoritative — from WintonCentre repo)
# ---------------------------------------------------------------------------
# Fetched from:
#   inst/extdata/coefficients_v3.csv   (current, used by v3.1 / v3.2)
#   inst/extdata/coefficients_v3_v1_old.csv  (old — used for reference only)
#
# Row order matches the CSV exactly (1-indexed in R → 0-indexed here).

COEFFS_V3 = {
    # ER-negative breast mortality model
    "ag1_er0":  1.7560679383953317,   # row 1  — age term 1 beta (ER-)
    "ag2_er0":  4.555274634880487,    # row 2  — age term 2 beta (ER-)
    "sz1_er0":  0.7443277265210039,   # row 3  — size beta (ER-)
    "nd1_er0":  0.6308590364310884,   # row 4  — nodes beta (ER-)
    "gr1_er0":  0.3462287839000938,   # row 5  — grade beta (ER-)
    "sc1_er0": -0.2110969326598672,   # row 6  — screen beta (ER-)
    "yr1_er0": -0.045504610416172736, # row 7  — year beta (ER-)

    # ER-positive breast mortality model
    "ag1_er1":  0.1958986203588129,   # row 8  — age term 1 beta (ER+)
    "ag2_er1":  2.929125051895787,    # row 9  — age term 2 beta (ER+)
    "sz1_er1":  2.27417420673498,     # row 10 — size beta (ER+)
    "nd1_er1":  0.6724828209061207,   # row 11 — nodes beta (ER+)
    "gr1_er1":  0.7050339629739382,   # row 12 — grade beta (ER+)
    "sc1_er1": -0.32039363353015476,  # row 13 — screen beta (ER+)
    "yr1_er1": -0.048251370712776756, # row 14 — year beta (ER+)

    # Other (non-breast) mortality model
    "ag_ot_ea_beta_1":  4.211443507724736,  # row 15 — other mortality age term 1
    "ag_ot_ea_beta_2": -31.41202815365277,  # row 16 — other mortality age term 2
    "yr_ot_ea_beta":   -0.021186462111818898, # row 17 — other mortality year beta

    # Baseline hazard for ER- breast mortality
    # exp(h0_br_i + h0_br_t1*(t/10)^-1 + h0_br_t2*(t/10)^-1 * log(t/10))
    "h0_br_i":  -3.0153139765165835,  # row 18
    "h0_br_t1": -0.5755380524044251,  # row 19
    "h0_br_t2": -0.1028439317178058,  # row 20

    # Baseline hazard for ER+ breast mortality
    # exp(h1_br_i + h1_br_t1*(t/10)^-0.5 + h1_br_t2*(t/10)^-0.5 * log(t/10))
    "h1_br_i":  -2.3193510039281637,  # row 21
    "h1_br_t1": -3.622538332641392,   # row 22
    "h1_br_t2": -0.542240944493945,   # row 23

    # Baseline hazard for other (non-breast) mortality
    # exp(h_ot_i + h_ot_t1*log(t/10) + h_ot_t2*(t/10))
    "h_ot_i":  -4.845654758283992,    # row 24
    "h_ot_t1":  1.341348310005262,    # row 25
    "h_ot_t2":  0.49539394353046057,  # row 26
}

# Old coefficients (coefficients_v3_v1_old.csv) — kept for reference
COEFFS_V3_OLD = {
    "ag_ot_ea_beta_1":  3.768865429383992,
    "ag_ot_ea_beta_2": -30.60582416224846,
    "yr_ot_ea_beta":   -0.02342200416022824,
    "h_ot_i":  -4.736930334618728,
    "h_ot_t1":  1.3078560827584007,
    "h_ot_t2":  0.49716395288310283,
    # breast coefficients are identical to current CSV
}

# ---------------------------------------------------------------------------
# SECTION 2 — HARDCODED v3.0 coefficients (from benefits30.R / Predict-v3.0)
# ---------------------------------------------------------------------------
# These are the v3.0 fractional polynomial terms and centering constants
# that differ from the v3.1/v3.2 parameterisation.

# ER+ breast mortality (age centred so that the mean is ~0)
#   age.mfp.1 = (age.start/10)^-2 - 0.0287449295
#   age.mfp.2 = (age.start/10)^-2 * log(age.start/10) - 0.0510121013
V30_ER_POS = {
    "age_beta_1":   34.53642,
    "age_beta_2":  -34.20342,
    "age_mfp1_center": 0.0287449295,   # subtracted from (age/10)^-2
    "age_mfp2_center": 0.0510121013,   # subtracted from (age/10)^-2 * log(age/10)
    "size_beta":    0.7530729,
    "size_mfp_center": 1.545233938,    # log(size/100) + 1.545233938
    "nodes_beta":   0.7060723,
    "nodes_mfp_center": 1.387566896,   # log((nodes+1)/10) + 1.387566896
    "grade_beta":   0.746655,
    "screen_beta": -0.22763366,
    # HER2 (ER+, fixed scalar — NOT time-varying in v3.0)
    "her2_pos_beta":  0.2413,
    "her2_neg_beta": -0.0762,
    # KI67 (ER+ only)
    "ki67_pos_beta":  0.14904,
    "ki67_neg_beta": -0.11333,
    # PR
    "pr_pos_er_pos_beta": -0.0619,
    "pr_neg_er_pos_beta":  0.2624,
}

# ER- breast mortality (v3.0)
#   age.mfp.1 = age.start - 56.3254902  (linear, no power transform)
#   age.mfp.2 = 0
V30_ER_NEG = {
    "age_beta_1":   0.0089827,
    "age_beta_2":   0.0,
    "age_mfp1_center": 56.3254902,     # age.start - 56.3254902
    "size_beta":    2.093446,
    "size_mfp_center": 0.5090456276,   # (size/100)^0.5 - 0.5090456276
    "nodes_beta":   0.6260541,
    "nodes_mfp_center": 1.086916249,   # log((nodes+1)/10) + 1.086916249
    "grade_beta":   1.129091,          # applied to binary (grade>=2 → 1, else 0)
    "screen_beta":  0.0,               # screen has no effect for ER-
    # HER2 (ER-) — not listed separately in v3.0, same scalar as ER+
    "her2_pos_beta":  0.2413,
    "her2_neg_beta": -0.0762,
    # KI67 (ER-, no effect in v3.0)
    "ki67_pos_beta":  0.0,
    "ki67_neg_beta":  0.0,
    # PR (ER-)
    "pr_pos_er_neg_beta": -0.2231,
    "pr_neg_er_neg_beta":  0.0296,
}

# v3.0 OTHER MORTALITY (non-breast)
#   mi = 0.0698252 * ((age.start/10)^2 - 34.23391957) + ...
V30_OTHER = {
    "age_coef":       0.0698252,
    "age_sq_center": 34.23391957,      # subtracted from (age/10)^2
    # Baseline hazard: exp(-6.052919 + 1.079863*log(t) + 0.3255321*t^0.5)
    "h_oth_intercept": -6.052919,
    "h_oth_log_t":      1.079863,
    "h_oth_sqrt_t":     0.3255321,
}

# v3.0 BASELINE BREAST HAZARD
#   ER+: exp(0.7424402 - 7.527762/t^0.5 - 1.812513*log(t)/t^0.5)
#   ER-: exp(-1.156036 + 0.4707332/t^2 - 3.51355/t)
V30_BASELINE_BR = {
    "er_pos_intercept": 0.7424402,
    "er_pos_t_neg05":  -7.527762,
    "er_pos_logt_neg05": -1.812513,
    "er_neg_intercept": -1.156036,
    "er_neg_t_neg2":    0.4707332,
    "er_neg_t_neg1":   -3.51355,
}

# ---------------------------------------------------------------------------
# SECTION 3 — HER2 TIME-VARYING LOOKUP TABLE (v3.1 / v3.2 ER+ only)
# ---------------------------------------------------------------------------
# From benefits31.R / benefits32.R:
#   her2==1 & er==1 ~ c(0.608, .532, .457, .382, .307, .231, .156, .081, .006, 0, 0, 0, 0, 0, 0)
#   her2==0 & er==1 ~ c(-.053, -.046, -.040, -.033, -.027, -.020, -.014, -.007, 0, 0, 0, 0, 0, 0, 0)
#   her2==1 & er==0 ~ 0.2316  (constant)
#   her2==0 & er==0 ~ -0.08589  (constant)
#   her2==9 (missing) → 0

HER2_TV_ER_POS = {
    "her2_pos": [0.608, 0.532, 0.457, 0.382, 0.307, 0.231, 0.156, 0.081, 0.006,
                 0.0,   0.0,   0.0,   0.0,   0.0,   0.0],  # years 1-15
    "her2_neg": [-0.053, -0.046, -0.040, -0.033, -0.027, -0.020, -0.014, -0.007,
                  0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0],  # years 1-15
}
HER2_SCALAR_ER_NEG = {
    "her2_pos":  0.2316,
    "her2_neg": -0.08589,
}

# ---------------------------------------------------------------------------
# SECTION 4 — TREATMENT LOG-HRS (identical across all v3.x versions)
# ---------------------------------------------------------------------------
TREATMENT = {
    # Chemotherapy generation effect on breast mortality
    "chemo_gen2":  -0.248,   # log-HR for 2nd-generation chemo
    "chemo_gen3":  -0.446,   # log-HR for 3rd-generation chemo
    # Hormone therapy (ER+ only, 5-year standard course)
    "horm_5yr":    -0.3857,  # log-HR during years 1-10
    # Extended hormone therapy extra benefit (years 11-15)
    # v3.0: h10 = rep(h, 10) then rep(-0.26 + h, 5)  → extra = -0.26
    # v3.1/v3.2: h10 = rep(h, 10) then rep(-0.301 + h, 5) → extra = -0.301
    "horm_ext_extra_v30":   -0.26,   # years 11-15 extra log-HR (v3.0)
    "horm_ext_extra_v31":   -0.301,  # years 11-15 extra log-HR (v3.1/v3.2)
    # Trastuzumab (HER2+ only)
    "trastuzumab":  -0.3567,
    # Bisphosphonate (post-menopausal only)
    "bisphosphonate": -0.198,
    # Radiotherapy effect on breast mortality
    # log(r.breast) = log(0.82) ≈ -0.19845
    "r_breast_rh":   0.82,   # relative hazard; log(0.82) applied when radio==1
}

# ---------------------------------------------------------------------------
# SECTION 5 — SMOKING / LIFESTYLE MODEL (identical across all versions)
# ---------------------------------------------------------------------------
SMOKING = {
    "smoker_prop":  0.1,   # proportion of population that are current smokers
    "smoker_rr":    2.0,   # relative risk of non-breast mortality in smokers
    "cvd_prop":     0.25,  # proportion of non-breast mortality due to smoking-related disease
}

# ---------------------------------------------------------------------------
# SECTION 6 — RADIOTHERAPY BASELINE ADJUSTMENTS
# ---------------------------------------------------------------------------
# These numbers differ between v3.0 and v3.1/v3.2.
# v3.0 and v3.1 compute r.base.br / r.base.oth analytically from r.prop:
RAD_BASELINE = {
    # v3.0 uses r.other=1.04; v3.1 uses 1.02; v3.2 uses hardcoded constants
    "r_prop":    0.69,   # proportion of population receiving radiotherapy
    "r_breast":  0.82,   # relative hazard breast mortality (Darby et al)
    "r_other_v30":  1.04,  # relative hazard other mortality per Gy (v3.0, Taylor 2017)
    "r_other_v31":  1.02,  # relative hazard other mortality per Gy (v3.1/v3.2)
    # v3.2 overrides the computed values with hardcoded constants:
    "r_base_br_er_pos_v32":  0.70,   # ER+
    "r_base_br_er_neg_v32":  0.55,   # ER-
    "r_base_oth_v32":       -0.062,
    # c.other: relative hazard non-breast mortality from chemotherapy (Kerr 2022)
    "c_other":   1.2,
}

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def _smoker_beta(smoker: int) -> float:
    """
    Compute smoking log-HR for non-breast mortality.
    smoker=0 → never/ex-smoker, smoker=1 → current smoker.

    Formula (from all v3.x R files):
        smoker.rr.acm = cvd_prop * smoker_rr + 1 - cvd_prop
        beta(non-smoker) = log(1 / (1 - smoker_prop + smoker_rr_acm * smoker_prop))
        beta(smoker)     = log(smoker_rr_acm / (1 - smoker_prop + smoker_rr_acm * smoker_prop))
    """
    sp  = SMOKING["smoker_prop"]
    rr  = SMOKING["smoker_rr"]
    cvd = SMOKING["cvd_prop"]
    rr_acm = cvd * rr + 1 - cvd           # = 0.25*2 + 0.75 = 1.25
    denom = 1 - sp + rr_acm * sp           # = 0.9 + 0.125 = 1.025
    if smoker == 0:
        return math.log(1.0 / denom)       # ≈ -0.02469
    else:
        return math.log(rr_acm / denom)    # ≈ 0.19845  (log(1.25/1.025))


def _r_base(r_other: float) -> tuple:
    """
    Compute r.base.br and r.base.oth from the population radiotherapy assumptions.
    Used by v3.0 and v3.1.

    r.base.br  = log(1 / ((1-r.prop) + r.prop * r.breast))
    r.base.oth = log(1 / ((1-r.prop) + r.prop * (r.other^2)))   # 2 Gy average heart dose
    """
    rp = RAD_BASELINE["r_prop"]
    rb = RAD_BASELINE["r_breast"]
    r_base_br  = math.log(1.0 / ((1 - rp) + rp * rb))
    r_base_oth = math.log(1.0 / ((1 - rp) + rp * (r_other ** 2)))
    return r_base_br, r_base_oth


# ---------------------------------------------------------------------------
# SECTION 7 — MISSING DATA IMPUTATION RULES
# ---------------------------------------------------------------------------
# Encoded directly in the R code (all versions):
#
# screen == 2   → replace with 0.204   (unknown screen detection)
# grade == 9    → replace with 2.13    (unknown grade; applied BEFORE grade.val)
# her2 == 9     → beta = 0             (unknown HER2 → no HER2 adjustment)
# ki67 == 9     → beta = 0             (unknown Ki67)
# pr == 9       → beta = 0  (v3.0 uses ifelse chain that defaults to 0 for missing)
#
# In v3.0: grade.val (for ER-) = ifelse(grade>=2, 1, 0)  → after imputation 2.13>=2 → 1
# In v3.1: grade enters the PI directly as a continuous value (already imputed to 2.13)

MISSING_DATA = {
    "screen_unknown_value":  0.204,   # replaces screen==2
    "grade_unknown_value":   2.13,    # replaces grade==9
    "her2_unknown_beta":     0.0,     # her2==9 → no HER2 contribution
    "ki67_unknown_beta":     0.0,
    "pr_unknown_beta":       0.0,
}


# ---------------------------------------------------------------------------
# SECTION 8 — CORE v3.0 IMPLEMENTATION
# ---------------------------------------------------------------------------

def predict_v30(
    age_start:  float = 65,
    screen:     int   = 0,      # 0=clinical, 1=screen, 2=unknown→0.204
    size:       float = 25,     # mm
    grade:      float = 2,      # 1/2/3/9(unknown)
    nodes:      float = 2,      # number positive nodes
    er:         int   = 1,      # 1=ER+, 0=ER-
    her2:       int   = 0,      # 1=HER2+, 0=HER2-, 9=unknown
    ki67:       int   = 1,      # 1=KI67+, 0=KI67-, 9=unknown
    pr:         int   = 1,      # 1=PR+, 0=PR-, 9=unknown
    generation: int   = 2,      # 0=none, 2=2nd-gen chemo, 3=3rd-gen chemo
    horm:       int   = 1,      # 1=hormone therapy, 0=no
    traz:       int   = 0,      # 1=trastuzumab, 0=no
    bis:        int   = 1,      # 1=bisphosphonate, 0=no
    radio:      int   = 1,      # 1=radiotherapy, 0=no
    heart_gy:   float = 1.0,    # Gy radiation to heart
    smoker:     int   = 1,      # 0=never/ex, 1=current
) -> dict:
    """
    PREDICT Breast v3.0 — port of WintonCentre/predictv30r R/benefits30.R

    Returns a dict matching the R function's locals30 environment with keys:
      pi, mi, base_m_cum_br (array, 15 years), base_m_cum_oth (array, 15 years),
      s_cum_br (dict of 15-year arrays keyed by treatment combo),
      s_cum_oth (15-year array),
      pred_cum_all (dict),
      benefits (dict)  ← 15x11 benefit matrix by treatment combination
    """
    # ---- 1. Missing data imputation -----------------------------------------
    screen = 0.204 if screen == 2 else screen
    grade  = 2.13  if grade == 9  else grade

    # ---- 2. Time vector (years 1..15) ---------------------------------------
    time = np.arange(1, 16, dtype=float)          # [1,2,...,15]

    # ---- 3. Grade variable (ER- uses binary >=2) ----------------------------
    if er == 1:
        grade_val = grade
    else:
        grade_val = 1.0 if grade >= 2 else 0.0

    # ---- 4. FP transforms & coefficients (v3.0 hardcoded) ------------------
    if er == 1:
        age_mfp_1  = (age_start / 10) ** -2 - V30_ER_POS["age_mfp1_center"]
        age_beta_1 = V30_ER_POS["age_beta_1"]
        age_mfp_2  = (age_start / 10) ** -2 * math.log(age_start / 10) - V30_ER_POS["age_mfp2_center"]
        age_beta_2 = V30_ER_POS["age_beta_2"]
        size_mfp   = math.log(size / 100) + V30_ER_POS["size_mfp_center"]
        size_beta  = V30_ER_POS["size_beta"]
        nodes_mfp  = math.log((nodes + 1) / 10) + V30_ER_POS["nodes_mfp_center"]
        nodes_beta = V30_ER_POS["nodes_beta"]
        grade_beta = V30_ER_POS["grade_beta"]
        screen_beta = V30_ER_POS["screen_beta"]
    else:
        age_mfp_1  = age_start - V30_ER_NEG["age_mfp1_center"]
        age_beta_1 = V30_ER_NEG["age_beta_1"]
        age_mfp_2  = 0.0
        age_beta_2 = V30_ER_NEG["age_beta_2"]
        size_mfp   = (size / 100) ** 0.5 - V30_ER_NEG["size_mfp_center"]
        size_beta  = V30_ER_NEG["size_beta"]
        nodes_mfp  = math.log((nodes + 1) / 10) + V30_ER_NEG["nodes_mfp_center"]
        nodes_beta = V30_ER_NEG["nodes_beta"]
        grade_beta = V30_ER_NEG["grade_beta"]
        screen_beta = V30_ER_NEG["screen_beta"]

    # HER2 (fixed scalar, v3.0)
    if her2 == 1:
        her2_beta = V30_ER_POS["her2_pos_beta"]    # 0.2413
    elif her2 == 0:
        her2_beta = V30_ER_POS["her2_neg_beta"]    # -0.0762
    else:  # her2 == 9 unknown
        her2_beta = 0.0

    # KI67 (ER+ only)
    if ki67 == 1 and er == 1:
        ki67_beta = V30_ER_POS["ki67_pos_beta"]    # 0.14904
    elif ki67 == 0 and er == 1:
        ki67_beta = V30_ER_POS["ki67_neg_beta"]    # -0.11333
    else:
        ki67_beta = 0.0

    # PR
    if   pr == 1 and er == 1:  pr_beta = V30_ER_POS["pr_pos_er_pos_beta"]   # -0.0619
    elif pr == 0 and er == 1:  pr_beta = V30_ER_POS["pr_neg_er_pos_beta"]   # 0.2624
    elif pr == 1 and er == 0:  pr_beta = V30_ER_NEG["pr_pos_er_neg_beta"]   # -0.2231
    elif pr == 0 and er == 0:  pr_beta = V30_ER_NEG["pr_neg_er_neg_beta"]   # 0.0296
    else:                      pr_beta = 0.0

    # ---- 5. Smoking beta ---------------------------------------------------
    smoker_beta = _smoker_beta(smoker)

    # ---- 6. Baseline radiotherapy adjustment (v3.0 uses r.other=1.04) ------
    r_other = RAD_BASELINE["r_other_v30"]   # 1.04
    r_base_br, r_base_oth = _r_base(r_other)
    c_other = RAD_BASELINE["c_other"]       # 1.2

    # ---- 7. Other mortality PI (mi) ----------------------------------------
    c_oth = 0.0 if generation == 0 else math.log(c_other)
    r_oth = 0.0 if radio == 0 else math.log(r_other) * heart_gy
    mi = (V30_OTHER["age_coef"] * ((age_start / 10) ** 2 - V30_OTHER["age_sq_center"])
          + r_base_oth + smoker_beta + c_oth + r_oth)

    # ---- 8. Breast cancer PI (pi) ------------------------------------------
    pi = (age_beta_1 * age_mfp_1 + age_beta_2 * age_mfp_2
          + size_beta * size_mfp + nodes_beta * nodes_mfp
          + grade_beta * grade_val + screen_beta * screen
          + her2_beta + ki67_beta + pr_beta + r_base_br)

    # ---- 9. Treatment log-HRs ----------------------------------------------
    c_rx = 0.0 if generation == 0 else (-0.248 if generation == 2 else -0.446)
    h    = -0.3857 if (horm == 1 and er == 1) else 0.0
    h10  = np.where(np.arange(1, 16) <= 10, h, -0.26 + h) if h != 0 else np.zeros(15)
    t_rx = -0.3567 if (her2 == 1 and traz == 1) else 0.0
    b    = -0.198  if bis == 1 else 0.0
    r_br = math.log(RAD_BASELINE["r_breast"]) if radio == 1 else 0.0  # log(0.82)

    # Build the treatment matrix (11 combinations, matching benefits30.R rxbase)
    # Column order: surg, h, hr, hrc, hrct, hrctb, h10, h10r, h10rc, h10rct, h10rctb
    rx_base = {
        "surg":    np.zeros(15),
        "h":       np.full(15, h),
        "hr":      np.full(15, h + r_br),
        "hrc":     np.full(15, h + r_br + c_rx),
        "hrct":    np.full(15, h + r_br + c_rx + t_rx),
        "hrctb":   np.full(15, h + r_br + c_rx + t_rx + b),
        "h10":     h10,
        "h10r":    h10 + r_br,
        "h10rc":   h10 + r_br + c_rx,
        "h10rct":  h10 + r_br + c_rx + t_rx,
        "h10rctb": h10 + r_br + c_rx + t_rx + b,
    }
    rx = {k: v + pi for k, v in rx_base.items()}

    # ---- 10. Baseline other mortality (Gompertz-like) ----------------------
    # exp(-6.052919 + 1.079863*log(t) + 0.3255321*t^0.5)
    h_oth_i   = V30_OTHER["h_oth_intercept"]  # -6.052919
    h_oth_log = V30_OTHER["h_oth_log_t"]      #  1.079863
    h_oth_sq  = V30_OTHER["h_oth_sqrt_t"]     #  0.3255321
    base_m_cum_oth = np.exp(h_oth_i + h_oth_log * np.log(time) + h_oth_sq * time ** 0.5)

    # Cumulative other mortality survival
    # WINTON FIX: c_oth and r_oth already included in mi, so do NOT add again
    s_cum_oth = np.exp(-np.exp(mi) * base_m_cum_oth)
    m_cum_oth = 1 - s_cum_oth
    m_oth = np.diff(np.concatenate([[0], m_cum_oth]))

    # ---- 11. Baseline breast mortality -------------------------------------
    if er == 1:
        # exp(0.7424402 - 7.527762/t^0.5 - 1.812513*log(t)/t^0.5)
        base_m_cum_br = np.exp(
            V30_BASELINE_BR["er_pos_intercept"]
            + V30_BASELINE_BR["er_pos_t_neg05"] / time ** 0.5
            + V30_BASELINE_BR["er_pos_logt_neg05"] * np.log(time) / time ** 0.5
        )
    else:
        # exp(-1.156036 + 0.4707332/t^2 - 3.51355/t)
        base_m_cum_br = np.exp(
            V30_BASELINE_BR["er_neg_intercept"]
            + V30_BASELINE_BR["er_neg_t_neg2"] / time ** 2
            + V30_BASELINE_BR["er_neg_t_neg1"] / time
        )

    base_m_br = np.diff(np.concatenate([[0], base_m_cum_br]))

    # ---- 12. Annual breast mortality by treatment --------------------------
    results = {}
    for rx_name, rx_pi in rx.items():
        m_br_annual = base_m_br * np.exp(rx_pi)          # annual breast mort rate
        m_cum_br    = np.cumsum(m_br_annual)              # cumulative
        s_cum_br    = np.exp(-m_cum_br)
        m_cum_br_risk = 1 - s_cum_br
        m_br_risk   = np.diff(np.concatenate([[0], m_cum_br_risk]))

        m_cum_all   = 1 - s_cum_oth * s_cum_br
        m_all       = np.diff(np.concatenate([[0], m_cum_all]))

        prop_br     = m_br_risk / (m_br_risk + m_oth)
        pred_m_br   = prop_br * m_all
        pred_cum_br = np.cumsum(pred_m_br)
        pred_m_oth  = m_all - pred_m_br
        pred_cum_all = pred_cum_br + np.cumsum(pred_m_oth)

        results[rx_name] = {
            "pred_cum_all":   pred_cum_all,
            "pred_cum_br":    pred_cum_br,
            "s_cum_br":       s_cum_br,
        }

    # ---- 13. Benefits (% reduction in all-cause mortality vs surgery) ------
    surg_cum = results["surg"]["pred_cum_all"]
    benefits = {}
    for rx_name in rx.keys():
        if rx_name == "surg":
            benefits["surg"] = 100 * (1 - surg_cum)  # baseline survival
        else:
            benefits[rx_name] = 100 * (surg_cum - results[rx_name]["pred_cum_all"])

    return {
        "pi": pi,
        "mi": mi,
        "r_base_br":  r_base_br,
        "r_base_oth": r_base_oth,
        "smoker_beta": smoker_beta,
        "her2_beta":  her2_beta,
        "ki67_beta":  ki67_beta,
        "pr_beta":    pr_beta,
        "c_rx": c_rx, "h": h, "t_rx": t_rx, "b": b, "r_br": r_br,
        "base_m_cum_oth": base_m_cum_oth,
        "base_m_cum_br":  base_m_cum_br,
        "s_cum_oth": s_cum_oth,
        "m_oth":     m_oth,
        "results":   results,
        "benefits":  benefits,
    }


# ---------------------------------------------------------------------------
# SECTION 9 — CORE v3.1 / v3.2 IMPLEMENTATION
# ---------------------------------------------------------------------------

def _her2_beta_v31(her2: int, er: int) -> np.ndarray:
    """Return the 15-year HER2 beta vector (v3.1/v3.2 time-varying for ER+)."""
    if her2 == 1 and er == 1:
        return np.array(HER2_TV_ER_POS["her2_pos"])
    elif her2 == 0 and er == 1:
        return np.array(HER2_TV_ER_POS["her2_neg"])
    elif her2 == 1 and er == 0:
        return np.full(15, HER2_SCALAR_ER_NEG["her2_pos"])
    elif her2 == 0 and er == 0:
        return np.full(15, HER2_SCALAR_ER_NEG["her2_neg"])
    else:  # her2 == 9 unknown
        return np.zeros(15)


def predict_v31(
    year:       int   = 2017,   # year of diagnosis (fixed at 2017 for web tool)
    age_start:  float = 50,
    screen:     int   = 0,
    size:       float = 20,
    grade:      float = 2,
    nodes:      float = 2,
    er:         int   = 1,
    her2:       int   = 0,
    ki67:       int   = 1,
    pr:         int   = 1,
    generation: int   = 3,
    horm:       int   = 1,
    traz:       int   = 1,
    bis:        int   = 1,
    radio:      int   = 1,
    heart_gy:   float = 4.0,
    smoker:     int   = 1,
    version:    str   = "v31",  # "v31" or "v32"
) -> dict:
    """
    PREDICT Breast v3.1 / v3.2 — port of WintonCentre/predictv30r R/benefits31.R
    and R/benefits32.R.

    v3.1 vs v3.2 difference:
      v3.1: r.base.br and r.base.oth computed from r.prop formula
      v3.2: r.base.br and r.base.oth hardcoded (0.7/0.55/−0.062)

    Key differences from v3.0:
      - age is centred at 24 (age_net = age_start - 24)
      - FP transforms use new parameterisation (see below)
      - HER2 is time-varying for ER+
      - 'year' covariate added
      - r.other = 1.02 (v3.1/v3.2) not 1.04
      - h10 extra = -0.301 (not -0.26)
      - Other mortality baseline uses CSV-based coefficients (new parameterisation)
    """
    C = COEFFS_V3

    # ---- 1. Missing data ---------------------------------------------------
    screen = 0.204 if screen == 2 else screen
    grade  = 2.13  if grade == 9  else grade

    # ---- 2. Time vector ----------------------------------------------------
    time = np.arange(1, 16, dtype=float)
    age_net = age_start - 24   # centred age used in v3.1 transforms

    # ---- 3. FP transforms (v3.1 parameterisation) --------------------------
    if er == 1:
        age_mfp_1   = (age_net / 100) ** (-0.5)
        age_beta_1  = C["ag1_er1"]           # 0.1958986203588129
        age_mfp_2   = (age_net / 100) ** 2
        age_beta_2  = C["ag2_er1"]           # 2.929125051895787
        size_mfp    = 1 - math.exp(-size / 20)
        size_beta   = C["sz1_er1"]           # 2.27417420673498
        nodes_beta  = C["nd1_er1"]           # 0.6724828209061207
        grade_beta  = C["gr1_er1"]           # 0.7050339629739382
        screen_beta = C["sc1_er1"]           # -0.32039363353015476
        yr_br_beta  = C["yr1_er1"]           # -0.048251370712776756
    else:
        age_mfp_1   = age_net / 100
        age_beta_1  = C["ag1_er0"]           # 1.7560679383953317
        age_mfp_2   = (age_net / 100) * math.log(age_net / 100)
        age_beta_2  = C["ag2_er0"]           # 4.555274634880487
        size_mfp    = math.log(size)
        size_beta   = C["sz1_er0"]           # 0.7443277265210039
        nodes_beta  = C["nd1_er0"]           # 0.6308590364310884
        grade_beta  = C["gr1_er0"]           # 0.3462287839000938
        screen_beta = C["sc1_er0"]           # -0.2110969326598672
        yr_br_beta  = C["yr1_er0"]           # -0.045504610416172736

    nodes_mfp = math.log(nodes + 1)   # same for both ER+ and ER-

    # HER2 (time-varying for ER+)
    her2_beta_arr = _her2_beta_v31(her2, er)   # 15-element array

    # KI67
    if ki67 == 1 and er == 1:
        ki67_beta = 0.14904
    elif ki67 == 0 and er == 1:
        ki67_beta = -0.11333
    else:
        ki67_beta = 0.0

    # PR
    if   pr == 1 and er == 1:  pr_beta = -0.0619
    elif pr == 0 and er == 1:  pr_beta =  0.2624
    elif pr == 1 and er == 0:  pr_beta = -0.2231
    elif pr == 0 and er == 0:  pr_beta =  0.0296
    else:                      pr_beta =  0.0

    # ---- 4. Smoking -----------------------------------------------------------
    smoker_beta = _smoker_beta(smoker)

    # ---- 5. Baseline adjustments --------------------------------------------
    r_other  = RAD_BASELINE["r_other_v31"]   # 1.02
    c_other  = RAD_BASELINE["c_other"]       # 1.2

    if version == "v32":
        r_base_br  = RAD_BASELINE["r_base_br_er_neg_v32"] if er == 0 else RAD_BASELINE["r_base_br_er_pos_v32"]
        r_base_oth = RAD_BASELINE["r_base_oth_v32"]        # -0.062
    else:
        r_base_br, r_base_oth = _r_base(r_other)

    # ---- 6. Other mortality PI (mi) -----------------------------------------
    c_oth = 0.0 if generation == 0 else math.log(c_other)
    r_oth = math.log(r_other) * heart_gy   # NOTE: in v3.1 r.oth is NOT conditional on radio

    mi = (C["ag_ot_ea_beta_1"] * (age_net / 100) ** 3
          + C["ag_ot_ea_beta_2"] * (age_net / 100) ** 3 * math.log(age_net / 100)
          + C["yr_ot_ea_beta"] * (year - 2000)
          + r_base_oth + smoker_beta)

    # ---- 7. Breast cancer PI (pi) — time-varying due to HER2 ---------------
    pi_scalar = (age_beta_1 * age_mfp_1 + age_beta_2 * age_mfp_2
                 + size_beta * size_mfp + nodes_beta * nodes_mfp
                 + grade_beta * grade
                 + screen_beta * screen + yr_br_beta * (year - 2000)
                 + ki67_beta + pr_beta + r_base_br)
    pi_arr = pi_scalar + her2_beta_arr   # 15-element array

    # ---- 8. Treatment log-HRs -----------------------------------------------
    c_rx = 0.0 if generation == 0 else (-0.248 if generation == 2 else -0.446)
    h    = -0.3857 if (horm == 1 and er == 1) else 0.0
    h10_vec = np.concatenate([np.full(10, h), np.full(5, -0.301 + h)]) if h != 0 else np.zeros(15)
    t_rx = -0.3567 if (her2 == 1 and traz == 1) else 0.0
    b    = -0.198  if bis == 1 else 0.0
    r_br = math.log(RAD_BASELINE["r_breast"]) if radio == 1 else 0.0

    # Build the full 48-column treatment matrix (matching benefits31.R)
    _s = np.zeros(15)
    _r = np.full(15, r_br)
    _h = np.full(15, h)
    _c = np.full(15, c_rx)
    _t = np.full(15, t_rx)
    _b = np.full(15, b)

    rx_combos = {
        "s":       _s,
        "r":       _r,
        "rh":      _r + _h,
        "rc":      _r + _c,
        "rt":      _r + _t,
        "rb":      _r + _b,
        "rhc":     _r + _h + _c,
        "rht":     _r + _h + _t,
        "rhb":     _r + _h + _b,
        "rct":     _r + _c + _t,
        "rcb":     _r + _c + _b,
        "rtb":     _r + _t + _b,
        "rhct":    _r + _h + _c + _t,
        "rhcb":    _r + _h + _c + _b,
        "rhtb":    _r + _h + _t + _b,
        "rctb":    _r + _c + _t + _b,
        "rhctb":   _r + _h + _c + _t + _b,
        "h":       _h,
        "hc":      _h + _c,
        "ht":      _h + _t,
        "hb":      _h + _b,
        "hct":     _h + _c + _t,
        "hcb":     _h + _c + _b,
        "htb":     _h + _t + _b,
        "hctb":    _h + _c + _t + _b,
        "c":       _c,
        "ct":      _c + _t,
        "cb":      _c + _b,
        "ctb":     _c + _t + _b,
        "t":       _t,
        "tb":      _t + _b,
        "rh10":    _r + h10_vec,
        "rh10c":   _r + h10_vec + _c,
        "rh10t":   _r + h10_vec + _t,
        "rh10b":   _r + h10_vec + _b,
        "rh10ct":  _r + h10_vec + _c + _t,
        "rh10cb":  _r + h10_vec + _c + _b,
        "rh10tb":  _r + h10_vec + _t + _b,
        "rh10ctb": _r + h10_vec + _c + _t + _b,
        "h10":     h10_vec,
        "h10c":    h10_vec + _c,
        "h10t":    h10_vec + _t,
        "h10b":    h10_vec + _b,
        "h10ct":   h10_vec + _c + _t,
        "h10cb":   h10_vec + _c + _b,
        "h10tb":   h10_vec + _t + _b,
        "h10ctb":  h10_vec + _c + _t + _b,
    }
    # pi.rx = rx + pi (time-varying pi due to HER2)
    pi_rx = {k: v + pi_arr for k, v in rx_combos.items()}

    # mi.rx: other mortality index includes c.oth and r.oth only when chemo/radio
    # See benefits31.R mi.rx tibble — r.oth always added when radio, c.oth when chemo
    def _mi_rx(has_chemo: bool) -> np.ndarray:
        base = mi + r_oth   # r.oth always included (not conditional on radio in v3.1)
        return np.full(15, base + c_oth if has_chemo else base)

    mi_rx = {}
    for k in rx_combos:
        has_c = "c" in k
        mi_rx[k] = _mi_rx(has_c)
    # No radio, no chemo combos
    for k in ("s", "h", "h10", "ht", "hb", "htb", "t", "tb"):
        mi_rx[k] = np.full(15, mi)    # no r.oth, no c.oth for surgery/horm/traz/bis only

    # ---- 9. Baseline other mortality (v3.1 parameterisation) ----------------
    # exp(h_ot_i + h_ot_t1*log(t/10) + h_ot_t2*(t/10))
    base_m_cum_oth = np.exp(
        C["h_ot_i"]
        + C["h_ot_t1"] * np.log(time / 10)
        + C["h_ot_t2"] * (time / 10)
    )
    base_m_oth = np.diff(np.concatenate([[0], base_m_cum_oth]))

    # ---- 10. Baseline breast mortality (v3.1 parameterisation) --------------
    if er == 1:
        # exp(h1_br_i + h1_br_t1*(t/10)^-0.5 + h1_br_t2*(t/10)^-0.5*log(t/10))
        base_m_cum_br = np.exp(
            C["h1_br_i"]
            + C["h1_br_t1"] * (time / 10) ** (-0.5)
            + C["h1_br_t2"] * (time / 10) ** (-0.5) * np.log(time / 10)
        )
    else:
        # exp(h0_br_i + h0_br_t1*(t/10)^-1 + h0_br_t2*(t/10)^-1*log(t/10))
        base_m_cum_br = np.exp(
            C["h0_br_i"]
            + C["h0_br_t1"] * (time / 10) ** (-1)
            + C["h0_br_t2"] * (time / 10) ** (-1) * np.log(time / 10)
        )
    base_m_br = np.diff(np.concatenate([[0], base_m_cum_br]))

    # ---- 11. Compute mortality by treatment combo ---------------------------
    results = {}
    for rx_name in rx_combos:
        # Other mortality
        m_oth_annual    = base_m_oth * np.exp(mi_rx[rx_name])
        m_cum_oth_rx    = np.cumsum(m_oth_annual)
        s_cum_oth_rx    = np.exp(-m_cum_oth_rx)
        m_cum_oth_risk  = 1 - s_cum_oth_rx
        m_oth_risk      = np.diff(np.concatenate([[0], m_cum_oth_risk]))

        # Breast mortality
        m_br_annual    = base_m_br * np.exp(pi_rx[rx_name])
        m_cum_br_rx    = np.cumsum(m_br_annual)
        s_cum_br_rx    = np.exp(-m_cum_br_rx)
        m_cum_br_risk  = 1 - s_cum_br_rx
        m_br_risk      = np.diff(np.concatenate([[0], m_cum_br_risk]))

        # Combined
        m_cum_all  = 1 - s_cum_oth_rx * s_cum_br_rx
        m_all      = np.diff(np.concatenate([[0], m_cum_all]))

        prop_br    = m_br_risk / (m_br_risk + m_oth_risk)
        pred_m_br  = prop_br * m_all
        pred_cum_br = np.cumsum(pred_m_br)
        pred_m_oth = m_all - pred_m_br
        pred_cum_all = pred_cum_br + np.cumsum(pred_m_oth)

        results[rx_name] = {
            "pred_cum_all": pred_cum_all,
            "pred_cum_br":  pred_cum_br,
            "s_cum_br":     s_cum_br_rx,
            "s_cum_oth":    s_cum_oth_rx,
            "surv_pct":     100 * (1 - pred_cum_all),
        }

    # ---- 12. Benefits summary (matching benefits31.R tibble) ----------------
    surg = results["s"]["pred_cum_all"]
    benefits = {
        "r_benefit":      100 * (surg - results["r"]["pred_cum_all"]),
        "h_benefit":      100 * (results["r"]["pred_cum_all"] - results["rh"]["pred_cum_all"]),
        "c_benefit":      100 * (results["rh"]["pred_cum_all"] - results["rhc"]["pred_cum_all"]),
        "t_benefit":      100 * (results["rhc"]["pred_cum_all"] - results["rhct"]["pred_cum_all"]),
        "b_benefit":      100 * (results["rhct"]["pred_cum_all"] - results["rhctb"]["pred_cum_all"]),
        "total_benefit":  100 * (surg - results["rhctb"]["pred_cum_all"]),
        "r10_benefit":    100 * (surg - results["r"]["pred_cum_all"]),
        "h10_benefit":    100 * (results["r"]["pred_cum_all"] - results["rh10"]["pred_cum_all"]),
        "c10_benefit":    100 * (results["rh10"]["pred_cum_all"] - results["rh10c"]["pred_cum_all"]),
        "t10_benefit":    100 * (results["rh10c"]["pred_cum_all"] - results["rh10ct"]["pred_cum_all"]),
        "b10_benefit":    100 * (results["rh10ct"]["pred_cum_all"] - results["rh10ctb"]["pred_cum_all"]),
        "total10_benefit": 100 * (surg - results["rh10ctb"]["pred_cum_all"]),
    }

    return {
        "pi_scalar": pi_scalar,
        "pi_arr":    pi_arr,
        "mi":        mi,
        "r_base_br":  r_base_br,
        "r_base_oth": r_base_oth,
        "c_oth": c_oth, "r_oth": r_oth,
        "smoker_beta": smoker_beta,
        "her2_beta_arr": her2_beta_arr,
        "ki67_beta": ki67_beta,
        "pr_beta":   pr_beta,
        "c_rx": c_rx, "h": h, "h10_vec": h10_vec, "t_rx": t_rx, "b": b, "r_br": r_br,
        "base_m_cum_oth": base_m_cum_oth,
        "base_m_cum_br":  base_m_cum_br,
        "results": results,
        "benefits": benefits,
    }


def predict_v32(**kwargs) -> dict:
    """PREDICT Breast v3.2 — same as v3.1 but with hardcoded baseline adjustments."""
    kwargs["version"] = "v32"
    return predict_v31(**kwargs)


# ---------------------------------------------------------------------------
# SECTION 10 — CONVENIENCE FUNCTIONS
# ---------------------------------------------------------------------------

def survival_at_year(result: dict, year: int, rx_combo: str = "rhctb") -> float:
    """
    Return % survival at a given year (1-15) for a treatment combination.
    rx_combo examples: 's' (surgery only), 'rhctb' (all treatments), 'rh10ctb'.
    """
    return float(result["results"][rx_combo]["surv_pct"][year - 1])


def benefits_at_year(result: dict, year: int) -> dict:
    """Return % benefit of each treatment at a given year."""
    return {k: float(v[year - 1]) for k, v in result["benefits"].items()}


# ---------------------------------------------------------------------------
# SECTION 11 — QUICK VALIDATION (matching R seed-100 test case in benefits30.R)
# ---------------------------------------------------------------------------
# From benefits30.R:
#   s100 <- function() { benefits30(age.start=33, bis=1, er=1, generation=0,
#     grade=1, heart.gy=7, her2=1, horm=0, ki67=1, nodes=3, pr=1,
#     radio=1, screen=1, size=21, smoker=1, traz=1) }

if __name__ == "__main__":
    print("=" * 70)
    print("PREDICT Breast v3.0 — seed-100 test case")
    print("=" * 70)
    r30 = predict_v30(
        age_start=33, screen=1, size=21, grade=1, nodes=3, er=1,
        her2=1, ki67=1, pr=1, generation=0, horm=0, traz=1, bis=1,
        radio=1, heart_gy=7, smoker=1,
    )
    print(f"pi          = {r30['pi']:.6f}")
    print(f"mi          = {r30['mi']:.6f}")
    print(f"r_base_br   = {r30['r_base_br']:.6f}")
    print(f"r_base_oth  = {r30['r_base_oth']:.6f}")
    print(f"smoker_beta = {r30['smoker_beta']:.6f}")
    print(f"\nBaseline surgery survival (year 10): "
          f"{r30['benefits']['surg'][9]:.2f}%")

    print("\n" + "=" * 70)
    print("PREDICT Breast v3.1 — default parameters (age=50, ER+, generation=3)")
    print("=" * 70)
    r31 = predict_v31(
        year=2017, age_start=50, screen=0, size=20, grade=2, nodes=2,
        er=1, her2=0, ki67=1, pr=1, generation=3, horm=1, traz=1,
        bis=1, radio=1, heart_gy=4, smoker=1,
    )
    print(f"pi_scalar   = {r31['pi_scalar']:.6f}")
    print(f"mi          = {r31['mi']:.6f}")
    print(f"r_base_br   = {r31['r_base_br']:.6f}")
    print(f"r_base_oth  = {r31['r_base_oth']:.6f}")

    b10 = benefits_at_year(r31, 10)
    print("\nBenefits at year 10:")
    for k, v in b10.items():
        print(f"  {k:20s}: {v:.2f}%")

    print("\n" + "=" * 70)
    print("Key model parameters summary")
    print("=" * 70)
    print("\ncoefficients_v3.csv (current, used by v3.1/v3.2):")
    for k, v in COEFFS_V3.items():
        print(f"  {k:25s}: {v}")
