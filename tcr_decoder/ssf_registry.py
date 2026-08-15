"""
Multi-cancer SSF (Site-Specific Factor) routing registry.

SSF1-SSF10 have completely different clinical meanings for each cancer type.
This module maps ICD-O-3 topography codes (TCODE1) to the correct SSF
column names, decoders, and clinical descriptions.

Usage:
    from tcr_decoder.ssf_registry import get_ssf_profile, detect_cancer_group

    group = detect_cancer_group('C50.1')       # → 'breast'
    profile = get_ssf_profile(group)            # → SSFProfile object

Cancer group coverage:
    breast      C50.x   — ER/PR/HER2/Ki67/Nottingham (fully custom decoders)
    lung        C34.x   — nodules/VPI/ECOG/effusion/mediastinal LN/EGFR(alpha)/ALK/adeno-component/nodule-count
    colorectum  C18-C21 — CEA value/CEA vs normal/regression/CRM/BRAF/RAS(KRAS+NRAS)/obstruction/perforation/distance-anus/MSI
    liver       C22.x   — AFP/Ishak fibrosis/Child-Pugh/creatinine/bilirubin/INR/HBsAg/Anti-HCV
    cervix      C53.x   — SCC antigen value/SCC vs normal (SSF3-10 not defined)
    stomach     C16.x   — CEA value/CEA vs normal/H.pylori/tumor depth/LVI (SSF6-10 not defined)
    thyroid     C73.x   — focality/vascular invasion/extrathyroidal extension
    prostate    C61.x   — PSA/Gleason/lymphovascular invasion/margins
    generic             — numeric passthrough for all SSF1-10
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from tcr_decoder.utils import _map_decode, strip_float_suffix
from tcr_decoder.codemap import CodeMap
from tcr_decoder import encoders as _enc


# ─────────────────────────────────────────────────────────────────────────────
# Shared sentinel codes (merged into every cancer-specific _map_decode decoder)
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SSFFieldDef:
    """Definition of one SSF field for a specific cancer type."""
    raw_field: str          # source raw field name (SSF1_raw … SSF10_raw)
    column_name: str        # output column name in the clean DataFrame
    description: str        # clinical description
    decoder: Optional[Callable] = None   # custom decoder; None → generic numeric
    unit: str = ''          # unit string for display (e.g., 'ng/mL', '%')
    encoder: Optional[Callable] = None   # inverse of decoder; None → generic numeric encode


@dataclass
class SSFProfile:
    """Complete SSF profile for one cancer group."""
    cancer_group: str
    site_label: str         # human-readable (e.g., 'Breast Cancer')
    site_codes: Tuple[str, ...]   # ICD-O-3 prefixes (e.g., ('C50',))
    fields: Dict[str, SSFFieldDef]  # 'SSF1'…'SSF10' → SSFFieldDef
    notes: str = ''


# ─────────────────────────────────────────────────────────────────────────────
# Generic SSF decoder (numeric passthrough with sentinel handling)
# ─────────────────────────────────────────────────────────────────────────────

def _generic_ssf(series: pd.Series, field_name: str = '',
                 unit: str = '') -> pd.Series:
    """Generic SSF decoder: numeric value or standard sentinel text.

    Sentinel codes (888, 900-902, 988, 998, 999) are decoded to text;
    all other values are returned as numeric strings.
    """
    SENTINELS = {
        888: 'Not applicable (conversion)',
        900: 'No laboratory test done (clinical assessment only)',
        901: 'No laboratory test done (radiographic assessment only)',
        902: 'Not documented in medical record',
        988: 'Not applicable',
        998: 'Not applicable - information not collected for this case',
        999: 'Unknown / not stated',
    }
    def _decode_one(val) -> str:
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        if iv in SENTINELS:
            return SENTINELS[iv]
        suffix = f' {unit}' if unit else ''
        return f'{iv}{suffix}'

    return series.apply(_decode_one)


# ─────────────────────────────────────────────────────────────────────────────
# Breast-cancer SSF decoders (imported from decoders.py)
# ─────────────────────────────────────────────────────────────────────────────

def _breast_ssf_decoder_factory():
    """Lazy import of breast-specific decoders to avoid circular imports."""
    from tcr_decoder.decoders import (
        decode_er_pr, decode_ki67, decode_her2, decode_nottingham,
        decode_ssf3_neoadj, decode_sentinel,
    )
    return {
        'SSF1': lambda s: decode_er_pr(s, receptor='ER'),
        'SSF2': lambda s: decode_er_pr(s, receptor='PR'),
        'SSF3': decode_ssf3_neoadj,
        'SSF4': lambda s: decode_sentinel(s, kind='examined'),
        'SSF5': lambda s: decode_sentinel(s, kind='positive'),
        'SSF6': decode_nottingham,
        'SSF7': decode_her2,
        'SSF8': _decode_paget,
        'SSF9': _decode_lvi,
        'SSF10': decode_ki67,
    }


# Official range (Cancer-SSF-Manual breast SSF8, p.142/148): 000, 010, 988, 999.
_PAGET_MAP = CodeMap({
    0:   'No Paget disease',
    10:  'Paget disease present',
    988: 'Not applicable (specimen excludes nipple/areola)',
    999: 'Unknown / not documented',
}, width=3)


def _decode_paget(series: pd.Series) -> pd.Series:
    """SSF8 for breast: Paget disease of the nipple."""
    return _PAGET_MAP.decode(series)


# Official range (Cancer-SSF-Manual breast SSF9, p.143): 000,010,988,990,999.
_LVI_BREAST_MAP = CodeMap({
    0:   'No lymphovascular invasion',
    10:  'Lymphovascular invasion present',
    988: 'Not applicable',
    990: 'No residual tumor (LVI not assessable after neoadjuvant therapy)',
    999: 'Unknown / not documented',
}, width=3)


def _decode_lvi(series: pd.Series) -> pd.Series:
    """SSF9 for breast: Lymphovascular invasion (LVI)."""
    return _LVI_BREAST_MAP.decode(series)


# ─────────────────────────────────────────────────────────────────────────────
# Lung-cancer SSF decoders
# ─────────────────────────────────────────────────────────────────────────────

_LUNG_SSF1_MAP = CodeMap({
    0:   'No separate ipsilateral tumor nodules; in situ',
    10:  'Separate nodule(s) — ipsilateral same lobe',
    20:  'Separate nodule(s) — ipsilateral different lobe',
    30:  'Separate nodule(s) — both same and different lobe (ipsilateral)',
    40:  'Separate nodule(s) — ipsilateral, lobe unknown',
    999: 'Unknown / not documented',
}, width=3)


def _decode_lung_ssf1_nodules(series: pd.Series) -> pd.Series:
    """SSF1 for lung: Separate tumor nodules / ipsilateral lung."""
    return _LUNG_SSF1_MAP.decode(series)


_LUNG_SSF2_MAP = CodeMap({
    0:   'PL0 — No visceral pleural invasion (elastic layer not reached)',
    10:  'PL1 — Invasion to elastic layer of visceral pleura',
    20:  'PL2 — Invasion to surface of visceral pleura',
    30:  'PL3 — Invasion through to parietal pleura',
    40:  'Pleural invasion present; PL level not specified',
    988: 'Not applicable (no surgery to primary site)',
    999: 'Unknown / not documented',
}, width=3)


def _decode_lung_ssf2_vpi(series: pd.Series) -> pd.Series:
    """SSF2 for lung: Visceral pleural invasion (PL0-PL3)."""
    return _LUNG_SSF2_MAP.decode(series)


_LUNG_SSF3_MAP = CodeMap({
    0:   'ECOG 0 — Fully active (KPS 100)',
    1:   'ECOG 1 — Light work only (KPS 80-90)',
    2:   'ECOG 2 — Self-care, up >50% of day (KPS 60-70)',
    3:   'ECOG 3 — Limited self-care, confined >50% of day (KPS 40-50)',
    4:   'ECOG 4 — Completely disabled (KPS 10-30)',
    5:   'ECOG 5 — Death (KPS 0)',
    988: 'Not applicable',
    998: 'Not assessed',
    999: 'Unknown / not documented',
}, width=3)


def _decode_lung_ssf3_ecog(series: pd.Series) -> pd.Series:
    """SSF3 for lung: Performance status (ECOG/KPS) before treatment."""
    return _LUNG_SSF3_MAP.decode(series)


_LUNG_SSF4_MAP = CodeMap({
    0:   'No malignant pleural effusion (imaging/cytology negative; or non-malignant cause confirmed)',
    11:  'Imaging: effusion present; no cytology; physician considers malignant',
    12:  'Imaging: effusion present; cytology negative/atypical; physician considers malignant',
    13:  'Cytology confirmed malignant pleural effusion',
    14:  'Imaging: effusion present; no cytology; physician does NOT consider malignant',
    15:  'Imaging: effusion present; cytology negative/atypical; physician does NOT consider malignant',
    988: 'Not applicable — M0 case',
    999: 'Unknown / not documented',
}, width=3)


def _decode_lung_ssf4_pleural_effusion(series: pd.Series) -> pd.Series:
    """SSF4 for lung: Malignant pleural effusion."""
    return _LUNG_SSF4_MAP.decode(series)


def _decode_lung_ssf5_mediastinal(series: pd.Series) -> pd.Series:
    """SSF5 for lung: Mediastinal LN sampling/dissection (N2 nodes, 8 stations)."""
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        if iv == 988:
            return 'Not applicable (small cell lung cancer or no surgery)'
        if iv == 999:
            return 'Unknown / not documented; stations dissected but location unclear'
        if 0 <= iv <= 8:
            return f'{iv} mediastinal LN station(s) sampled/dissected' if iv > 0 else 'No mediastinal LN sampling or dissection'
        return f'Code {iv}'
    return series.apply(_d)


_LUNG_EGFR_LETTER_MAP = {
    'A': 'Exon 19 deletion',
    'B': 'Exon 21 L858R',
    'C': 'Exon 18 E709',
    'D': 'Exon 18 G719X',
    'E': 'Exon 20 insertion',
    'F': 'Exon 20 S768I',
    'G': 'Exon 20 T790M',
    'H': 'Exon 21 L861',
    'U': 'Other mutation',
    'V': 'Mutated (type NOS)',
    'X': 'No mutation',
    'Z': 'Uninterpretable result',
}


def _decode_lung_egfr(series: pd.Series) -> pd.Series:
    """SSF6 for lung: EGFR gene mutation (3-character alphabetic code).

    The TCR encodes up to 3 concurrent EGFR mutations as a 3-letter string.
    Each letter position represents one mutation:
      A=Exon19del  B=L858R  C=E709  D=G719X  E=Exon20ins
      F=S768I  G=T790M  H=L861  U=other  V=mutated(NOS)
      X=no mutation  Z=uninterpretable
    Sentinels: 999 = unknown / not tested
    """
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        s = str(val).strip().upper()
        if s in ('999', ''):
            return 'Unknown / not tested'
        if len(s) == 3:
            mutations = [_LUNG_EGFR_LETTER_MAP.get(c, f'?({c})') for c in s if c != 'X']
            if not mutations:
                return 'EGFR — No mutation (XXX)'
            return 'EGFR — ' + ' + '.join(mutations)
        # Single letter or numeric fallback
        if s in _LUNG_EGFR_LETTER_MAP:
            return 'EGFR — ' + _LUNG_EGFR_LETTER_MAP[s]
        return f'EGFR code: {val}'
    return series.apply(_d)


_LUNG_ALK_MAP = CodeMap({
    10:  'ALK positive — rearrangement/translocation present',
    20:  'ALK negative — no rearrangement',
    30:  'ALK test performed; result uninterpretable',
    999: 'Unknown / not tested / no ALK test ordered',
}, width=3)


def _decode_lung_alk(series: pd.Series) -> pd.Series:
    """SSF7 for lung: ALK gene translocation."""
    return _LUNG_ALK_MAP.decode(series)


_LUNG_SSF8_MAP = CodeMap({
    0:   'None of: micropapillary / solid / cribriform components',
    1:   'Micropapillary only',
    2:   'Solid only',
    3:   'Micropapillary + Solid',
    4:   'Cribriform / complex gland only',
    5:   'Micropapillary + Cribriform',
    6:   'Solid + Cribriform',
    7:   'Micropapillary + Solid + Cribriform',
    988: 'Not applicable (CIS; non-NM adenocarcinoma; no curative surgery; neoadjuvant before surgery)',
    999: 'Unknown / not documented',
}, width=3)


def _decode_lung_ssf8_adeno(series: pd.Series) -> pd.Series:
    """SSF8 for lung: Specific lung adenocarcinoma component (micropapillary/solid/cribriform).

    Codes are additive bitmask: micropapillary=1, solid=2, cribriform=4.
    """
    return _LUNG_SSF8_MAP.decode(series)


def _decode_lung_ssf9_nodules(series: pd.Series) -> pd.Series:
    """SSF9 for lung: Tumor nodule count (for early-stage multi-tumor cases)."""
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        if iv == 988:
            return 'Not applicable (T0; non-stage 0-2; T3N0; no surgery; single tumor; no external data)'
        if iv == 999:
            return 'Unknown / not documented'
        if 2 <= iv <= 20:
            return f'{iv} tumor nodules'
        if iv == 21:
            return '>20 tumor nodules'
        return f'Code {iv}'
    return series.apply(_d)


# ─────────────────────────────────────────────────────────────────────────────
# Colorectal SSF decoders
# ─────────────────────────────────────────────────────────────────────────────

def _decode_msi(series: pd.Series) -> pd.Series:
    """Microsatellite instability (MSI) status."""
    return _MSI_MAP.decode(series)


def _decode_kras(series: pd.Series) -> pd.Series:
    """KRAS mutation status."""
    return _KRAS_MAP.decode(series)


# ─────────────────────────────────────────────────────────────────────────────
# Liver/HCC SSF decoders
# ─────────────────────────────────────────────────────────────────────────────

def _decode_liver_afp(series: pd.Series) -> pd.Series:
    """SSF1 for liver: AFP (alpha-fetoprotein) with TCR-specific encoding.

    For dx year >=2021: A00-A99 = 1-99 ng/mL actual value; 010-099 = 100-999 ng/mL (unit dropped);
    100-987 = 1000-9879 ng/mL / 10; 991-993 = instrument saturation codes.
    """
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        s = str(val).strip().upper()
        # Alphabetic A-codes (2021+ only): A01-A99 = actual integer 1-99 ng/mL;
        # A00 specifically means "<1 ng/mL" (e.g. an actual value of 0.91
        # ng/mL is coded A00), not literally zero/undetectable -- per
        # Cancer-SSF-Manual (liver SSF1, p.81): "AFP檢驗結果實際數值為0.91
        # ng/ml，請編碼A00".
        if len(s) == 3 and s[0] == 'A' and s[1:].isdigit():
            num = int(s[1:])
            if num == 0:
                return 'AFP <1 ng/mL (A-code, 2021+ scheme)'
            return f'AFP {num} ng/mL (A-code, 2021+ scheme)'
        try:
            iv = int(float(s))
        except (ValueError, TypeError):
            return str(val).strip()
        if iv == 988:
            return 'Not applicable (treated at another hospital, no external data)'
        if iv == 991:
            return 'AFP > instrument max (400-6000 ng/mL range)'
        if iv == 992:
            return 'AFP > instrument max (6001-9879 ng/mL range)'
        if iv == 993:
            return 'AFP >=9880 ng/mL (or instrument max >=9880)'
        if iv == 999:
            return 'Unknown / not tested before first treatment'
        if 0 <= iv <= 9:
            return f'AFP ~{iv * 10} ng/mL (pre-2021 code: value /10, rounded)'
        if 10 <= iv <= 99:
            return f'AFP {iv * 10}-{iv * 10 + 9} ng/mL (100-999 range code)'
        if 100 <= iv <= 987:
            return f'AFP ~{iv * 10} ng/mL (1000-9879 range code)'
        return f'AFP code {iv}'
    return series.apply(_d)


_LIVER_FIBROSIS_MAP = CodeMap({
    0:   'Ishak F0 — No fibrosis',
    1:   'Ishak F1 — Some portal areas expanded; short fibrous septa',
    2:   'Ishak F2 — Most portal areas expanded; short fibrous septa',
    3:   'Ishak F3 — Most portal areas expanded; occasional P-P bridging',
    4:   'Ishak F4 — Marked P-P and P-C bridging',
    5:   'Ishak F5 — Marked bridging with occasional nodules (incomplete cirrhosis)',
    6:   'Ishak F6 — Cirrhosis (probable or definite)',
    7:   'No pathology report; imaging (US/CT/MRI) shows cirrhosis',
    8:   'No pathology report; imaging (US/CT/MRI) shows no cirrhosis',
    988: 'Not applicable (no accessible pathology or imaging data)',
    999: 'Unknown / Ishak score not used',
}, width=3)


def _decode_liver_fibrosis(series: pd.Series) -> pd.Series:
    """SSF2 for liver: Liver fibrosis grade (Ishak score)."""
    return _LIVER_FIBROSIS_MAP.decode(series)


_CHILD_PUGH_MAP = CodeMap({
    105: 'Child-Pugh Class A, Score 5',
    106: 'Child-Pugh Class A, Score 6',
    199: 'Child-Pugh Class A, Score unknown',
    207: 'Child-Pugh Class B, Score 7',
    208: 'Child-Pugh Class B, Score 8',
    209: 'Child-Pugh Class B, Score 9',
    299: 'Child-Pugh Class B, Score unknown',
    310: 'Child-Pugh Class C, Score 10',
    311: 'Child-Pugh Class C, Score 11',
    312: 'Child-Pugh Class C, Score 12',
    313: 'Child-Pugh Class C, Score 13',
    314: 'Child-Pugh Class C, Score 14',
    315: 'Child-Pugh Class C, Score 15',
    399: 'Child-Pugh Class C, Score unknown',
    999: 'Class and score both unknown / not assessed',
}, fallback='Child-Pugh code', width=3)


def _decode_child_pugh(series: pd.Series) -> pd.Series:
    """SSF3 for liver: Child-Pugh class and score.

    3-char code: first digit = class (1=A, 2=B, 3=C),
    next two = score (05-15) or 99 for class only.
    """
    return _CHILD_PUGH_MAP.decode(series)


def _decode_lab_value_10x(series: pd.Series, analyte: str, unit: str) -> pd.Series:
    """Generic decoder for lab values coded as value x 10 (one decimal place).

    Used for: Creatinine (mg/dL), Total Bilirubin (mg/dL).
    Codes: 001=<=0.1, 002-986=0.2-98.6, 987=>=98.7, 988=N/A, 999=unknown.
    """
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        if iv == 988:
            return 'Not applicable (treated at another hospital, no external data)'
        if iv == 999:
            return f'{analyte} unknown / not tested'
        if iv == 1:
            return f'{analyte} <=0.1 {unit}'
        if 2 <= iv <= 986:
            return f'{analyte} {iv / 10:.1f} {unit}'
        if iv == 987:
            return f'{analyte} >=98.7 {unit}'
        return f'{analyte} code {iv}'
    return series.apply(_d)


def _decode_liver_inr(series: pd.Series) -> pd.Series:
    """SSF6 for liver: INR (prothrombin time, coded as value x 10)."""
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        if iv == 988:
            return 'Not applicable (treated at another hospital, no external data)'
        if iv == 997:
            return 'INR >6.0 (or above instrument maximum)'
        if iv == 999:
            return 'INR unknown / not tested'
        if 1 <= iv <= 60:
            return f'INR {iv / 10:.1f}'
        return f'INR code {iv}'
    return series.apply(_d)


_HBSAG_MAP = CodeMap({
    0:   'Not tested; no HBV carrier history',
    1:   'Not tested; HBV carrier history documented',
    10:  'Negative; no HBV carrier history',
    11:  'Negative; HBV carrier history documented',
    20:  'Positive',
    999: 'Unknown / not documented',
}, width=3)


def _decode_hbsag(series: pd.Series) -> pd.Series:
    """SSF7 for liver: HBsAg (hepatitis B surface antigen) with history."""
    return _HBSAG_MAP.decode(series)


_ANTI_HCV_MAP = CodeMap({
    0:   'Not tested; no HCV infection history',
    1:   'Not tested; HCV infection history documented',
    10:  'Negative; no HCV infection history',
    11:  'Negative; HCV infection history (treated / SVR)',
    20:  'Positive (Anti-HCV positive and/or HCV RNA positive)',
    999: 'Unknown / not documented',
}, width=3)


def _decode_anti_hcv(series: pd.Series) -> pd.Series:
    """SSF8 for liver: Anti-HCV (hepatitis C antibody/antigen/RNA) with history."""
    return _ANTI_HCV_MAP.decode(series)


# ─────────────────────────────────────────────────────────────────────────────
# Prostate SSF decoders
# ─────────────────────────────────────────────────────────────────────────────

_PSA_TIER_MAP = {
    981: (98.0, 199.9), 982: (200.0, 299.9), 983: (300.0, 399.9),
    984: (400.0, 499.9), 985: (500.0, 599.9), 986: (600.0, 699.9),
    987: (700.0, 799.9), 989: (800.0, 899.9), 990: (900.0, 999.9),
}
_PSA_THOUSANDS_MAP = {991: 1, 992: 2, 993: 3, 994: 4, 995: 5, 996: 6, 997: 7}


def _decode_psa(series: pd.Series) -> pd.Series:
    """PSA (prostate-specific antigen) in ng/mL.

    Codebook: Cancer-SSF-Manual (prostate), SSF1, p.183-184.
    Official range 001-999, with no 000: 001 = <=0.1 ng/mL, 002-979 = value
    x10 (0.2-97.9 ng/mL), 980 = 98.0 ng/mL (legacy, dx year 100-104 only),
    981-987 + 989-990 = 100-wide tiers from 98.0 to 999.9 ng/mL, 988 = not
    applicable (first course started at another hospital with no lab value),
    991-997 = 1000-wide tiers from 1000 to 7999 ng/mL, 998 = >=8000 ng/mL,
    999 = unknown.
    """
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        if iv == 1:
            return 'PSA <=0.1 ng/mL'
        if iv == 988:
            return 'Not applicable (first course at another hospital, no lab value)'
        if iv == 998:
            return 'PSA >=8000 ng/mL'
        if iv == 999:
            return 'Unknown; not documented'
        if iv == 980:
            return 'PSA 98.0 ng/mL (legacy code, dx year 100-104 only)'
        if iv in _PSA_TIER_MAP:
            lo, hi = _PSA_TIER_MAP[iv]
            return f'PSA {lo:.1f}-{hi:.1f} ng/mL'
        if iv in _PSA_THOUSANDS_MAP:
            k = _PSA_THOUSANDS_MAP[iv]
            return f'PSA {k * 1000}-{k * 1000 + 999} ng/mL'
        if 2 <= iv <= 979:
            return f'PSA {iv/10:.1f} ng/mL'
        return f'Code {iv}'
    return series.apply(_d)


def _decode_gleason_patterns(series: pd.Series, specimen: str) -> pd.Series:
    """Gleason primary + secondary PATTERN codes (prostate SSF2 / SSF4).

    Codebook: Cancer-SSF-Manual (prostate) p.179-180 (SSF2, needle core
    biopsy / TURP) and p.183-184 (SSF4, radical prostatectomy / autopsy).
    Range for both: 011-015,019,021-025,029,031-035,039,041-045,049,
    051-055,059,099,988,999.

    The code is NOT a sum: the tens digit is the primary pattern (1-5) and
    the units digit is the secondary pattern (1-5, or 9 = unknown); 099 means
    both unknown. '034' is therefore "primary 3, secondary 4" (Gleason 3+4),
    not "Gleason score 34".
    """
    na_reason = ('no needle biopsy/TURP performed, or PIN III'
                 if specimen == 'biopsy' else
                 'no radical prostatectomy/autopsy, PIN III, no residual '
                 'tumour, or neoadjuvant therapy given before surgery')
    procedure = ('needle biopsy/TURP' if specimen == 'biopsy'
                 else 'radical prostatectomy/autopsy')

    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        s = strip_float_suffix(str(val).strip())
        if not s.isdigit():
            return s
        iv = int(s)
        if iv == 988:
            return f'Not applicable ({na_reason})'
        if iv == 999:
            return f'Unknown ({procedure} performed but pattern not stated)'
        if iv == 99:
            return 'Primary pattern unknown, secondary pattern unknown'
        primary, secondary = iv // 10, iv % 10
        if 1 <= primary <= 5 and 1 <= secondary <= 5:
            return (f'Gleason {primary}+{secondary} '
                    f'(primary {primary}, secondary {secondary})')
        if 1 <= primary <= 5 and secondary == 9:
            return f'Primary pattern {primary}, secondary pattern unknown'
        return f'Code {s}'

    return series.apply(_d)


def _decode_gleason_score(series: pd.Series, specimen: str) -> pd.Series:
    """Gleason SCORE (prostate SSF3 / SSF5). Range 002-010, 988, 999."""
    na_reason = ('no needle biopsy/TURP performed, or PIN III'
                 if specimen == 'biopsy' else
                 'no radical prostatectomy/autopsy, PIN III, no residual '
                 'tumour, or neoadjuvant therapy given before surgery')
    procedure = ('needle biopsy/TURP' if specimen == 'biopsy'
                 else 'radical prostatectomy/autopsy')
    grade_group = {
        2: 1, 3: 1, 4: 1, 5: 1, 6: 1,   # ISUP/WHO 2016 Grade Group 1 = score <=6
        7: 2, 8: 4, 9: 5, 10: 5,
    }

    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        s = strip_float_suffix(str(val).strip())
        if not s.isdigit():
            return s
        iv = int(s)
        if iv == 988:
            return f'Not applicable ({na_reason})'
        if iv == 999:
            return f'Unknown ({procedure} performed but score not stated)'
        if 2 <= iv <= 10:
            gg = grade_group[iv]
            note = ' or 3' if iv == 7 else ''
            return f'Gleason score {iv} — Grade Group {gg}{note}'
        return f'Code {s}'

    return series.apply(_d)


def _decode_biopsy_cores(series: pd.Series, kind: str) -> pd.Series:
    """Number of biopsy cores examined (SSF6) / positive (SSF7).

    Codebook p.187 (SSF6, range 001-100,988,999) and p.188 (SSF7, range
    000-100,988,998,999). SSF6 has no 000 and no 998; SSF7 has both, and
    they mean different things, so the two share no vocabulary.
    """
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        s = strip_float_suffix(str(val).strip())
        if not s.isdigit():
            return s
        iv = int(s)
        if iv == 988:
            return ('Not applicable (no biopsy performed; TURP only)'
                    if kind == 'examined' else
                    'Not applicable (no biopsy performed; TURP only, or PIN III)')
        if iv == 998 and kind == 'positive':
            return 'Positive on biopsy, but the positive-core count may not be accurate'
        if iv == 999:
            return ('Biopsy performed but the number of cores examined is unknown'
                    if kind == 'examined' else
                    'Biopsy positive but the number of positive cores is unknown')
        if kind == 'positive' and iv == 0:
            return 'All cores negative'
        if 1 <= iv <= 100:
            return f'{iv} core(s) {kind}'
        return f'Code {s}'

    return series.apply(_d)


_PROSTATE_T_STAGING_MAP = CodeMap({
    0:   'Neither digital rectal exam nor imaging performed',
    10:  'Digital rectal exam only; no imaging',
    20:  'TRUS or erMRI imaging only; no digital rectal exam',
    30:  'Digital rectal exam plus TRUS or erMRI imaging',
    40:  'MRI only; neither digital rectal exam nor TRUS/erMRI',
    50:  'Digital rectal exam plus MRI; no TRUS/erMRI',
    988: 'Not applicable',
    999: 'Not documented in the medical record / unknown',
}, width=3)


def _decode_prostate_t_staging(series: pd.Series) -> pd.Series:
    """SSF8 for prostate: how the clinical T stage was determined (p.191)."""
    return _PROSTATE_T_STAGING_MAP.decode(series)


# ─────────────────────────────────────────────────────────────────────────────
# Colorectum / stomach / cervix fields that had no code table of their own
# (they used the generic numeric fallback until their 編碼範圍 was verified)
# ─────────────────────────────────────────────────────────────────────────────

def _decode_scc_lab_value(series: pd.Series) -> pd.Series:
    """SCC antigen lab value in ng/mL (cervix SSF1, p.149).

    Same x10 scheme as CEA: 001 = <=0.1, 002-986 = 0.2-98.6 ng/mL,
    987 = >=98.7, 988 = not applicable, 999 = unknown.
    """
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        s = strip_float_suffix(str(val).strip())
        if not s.isdigit():
            return s
        iv = int(s)
        if iv == 988:
            return 'Not applicable (treated at another hospital, no lab value)'
        if iv == 999:
            return 'SCC antigen unknown / not tested'
        if iv == 1:
            return 'SCC antigen <=0.1 ng/mL'
        if 2 <= iv <= 986:
            return f'SCC antigen {iv / 10:.1f} ng/mL'
        if iv == 987:
            return 'SCC antigen >=98.7 ng/mL'
        return f'SCC antigen code {iv}'
    return series.apply(_d)


def _decode_tumor_depth(series: pd.Series) -> pd.Series:
    """Measured tumour depth in 0.1 mm units (stomach SSF4, p.42).

    000 = <0.1 mm, 001-979 = actual depth x10, 980 = >=98 mm,
    988 = not applicable, 998 = no surgical specimen / neoadjuvant therapy,
    999 = the pathology report has no depth figure.
    """
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        s = strip_float_suffix(str(val).strip())
        if not s.isdigit():
            return s
        iv = int(s)
        if iv == 988:
            return 'Not applicable (non-pathological T1 case, GIST/NETs)'
        if iv == 998:
            return 'No surgical specimen, or neoadjuvant therapy given first'
        if iv == 999:
            return 'Depth not stated in the pathology report'
        if iv == 0:
            return 'Tumour depth <0.1 mm'
        if 1 <= iv <= 979:
            return f'Tumour depth {iv / 10:.1f} mm'
        if iv == 980:
            return 'Tumour depth >=98 mm'
        return f'Code {s}'
    return series.apply(_d)


def _decode_crm(series: pd.Series) -> pd.Series:
    """Circumferential resection margin (colorectum SSF4, p.51/67)."""
    specials = {
        980: 'CRM >=98 mm',
        988: 'Not applicable (no surgery, local excision only, GIST/NETs, or '
             'operated at another hospital with no CRM data)',
        990: 'No residual tumour in the specimen',
        991: 'CRM negative, distance not stated',
        992: 'CRM 1-2 mm',
        993: 'CRM 2-3 mm',
        994: 'CRM 3-4 mm',
        995: 'CRM 4-5 mm',
        996: 'CRM >5 mm',
        999: 'CRM unknown / not documented / only distal and proximal margins '
             'reported',
    }

    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        s = strip_float_suffix(str(val).strip())
        if not s.isdigit():
            return s
        iv = int(s)
        if iv in specials:
            return specials[iv]
        if 0 <= iv <= 979:
            return f'CRM {iv / 10:.1f} mm'
        return f'Code {s}'
    return series.apply(_d)


def _decode_distance_to_anus(series: pd.Series) -> pd.Series:
    """Distance from the lower tumour edge to the anus (rectum SSF9, p.75)."""
    specials = {
        988: 'Not applicable (rectosigmoid junction C19.9, GIST/NETs, or first '
             'course given at another hospital with no report)',
        991: 'Tumour in the upper third of the rectum',
        992: 'Tumour in the middle third of the rectum',
        993: 'Tumour in the lower third of the rectum',
        999: 'Not documented / not assessed before treatment',
    }

    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        s = strip_float_suffix(str(val).strip())
        if not s.isdigit():
            return s
        iv = int(s)
        if iv in specials:
            return specials[iv]
        if iv == 0:
            return 'Less than 1 mm from the anus'
        if 1 <= iv <= 150:
            return f'{iv} mm from the anus'
        return f'Code {s}'
    return series.apply(_d)


def _range_only_code(code: str, where: str) -> str:
    """Label for a code listed in a field's 編碼範圍 but absent from its table.

    The manual does this in a few places (colon SSF8 980, rectum SSF5
    990/991, uterine SSF3 987). Decoding it explicitly keeps the value
    reversible and visible instead of inventing a meaning for it.
    """
    return (f'Code {code} — listed in the 編碼範圍 but undefined in the code '
            f'table ({where}; confirm with the registry)')


_TUMOR_REGRESSION_MAP = CodeMap({
    0:   'TRG 0 — complete response, no residual tumour cells',
    10:  'TRG 1 — moderate response, single or rare residual tumour cells',
    20:  'TRG 2 — minimal response, residual tumour with fibrosis',
    30:  'TRG 3 — poor response, extensive residual tumour',
    988: 'Not applicable (no preoperative therapy, no surgery, GIST/NETs, or '
         'no histological confirmation)',
    990: 'Responded, but the degree of response is not further specified',
    999: 'Treatment response unknown / not documented',
}, width=3)


def _decode_tumor_regression(series: pd.Series) -> pd.Series:
    """SSF3 for colorectum: tumour regression grade (p.49/65)."""
    return _TUMOR_REGRESSION_MAP.decode(series)


_BRAF_MAP = CodeMap({
    0:   'BRAF wild type / no mutation',
    1:   'BRAF mutation including V600E (c.1799T>A)',
    2:   'BRAF mutation other than V600E',
    6:   'BRAF mutated, codon not specified',
    7:   'BRAF test performed but the report is uninterpretable',
    988: 'Not applicable (GIST/NETs/high-grade dysplasia, or tested at another '
         'hospital with no result)',
    990: _range_only_code('990', 'rectum SSF5, p.69'),
    991: _range_only_code('991', 'rectum SSF5, p.69'),
    999: 'Not documented / not tested',
}, width=3)


def _decode_braf(series: pd.Series) -> pd.Series:
    """SSF5 for colorectum: BRAF mutation (p.53/69)."""
    return _BRAF_MAP.decode(series)


_OBSTRUCTION_MAP = CodeMap({
    0:   'No bowel obstruction on imaging or at surgery',
    10:  'Bowel obstruction found on imaging or at surgery',
    988: 'Not applicable (no imaging and no surgery, or GIST/NETs/high-grade '
         'dysplasia)',
    999: 'Not documented / unknown',
}, width=3)


def _decode_obstruction(series: pd.Series) -> pd.Series:
    """SSF7 for colorectum: intestinal obstruction (p.57/73)."""
    return _OBSTRUCTION_MAP.decode(series)


_PERFORATION_MAP = CodeMap({
    0:   'No bowel perforation on imaging or at surgery',
    10:  'Bowel perforation found on imaging or at surgery',
    980: _range_only_code('980', 'colon SSF8, p.58'),
    988: 'Not applicable (no imaging and no surgery, or GIST/NETs/high-grade '
         'dysplasia)',
    999: 'Not documented / unknown',
}, width=3)


def _decode_perforation(series: pd.Series) -> pd.Series:
    """SSF8 for colorectum: intestinal perforation (p.58/74)."""
    return _PERFORATION_MAP.decode(series)


# ─────────────────────────────────────────────────────────────────────────────
# Esophagus (pp.31-36), Pancreas (pp.91-100), Ovary (pp.163-167),
# Bladder (pp.169-173)
# ─────────────────────────────────────────────────────────────────────────────

_ESO_PETCT_MAP = CodeMap({
    0:   'No PET-CT performed',
    20:  'PET-CT before the first course of treatment only',
    30:  'PET-CT after the first course of treatment only',
    40:  'PET-CT both before and after the first course of treatment',
    988: 'Not applicable',
    999: 'Not documented / unknown',
}, width=3)


_ESO_MIE_MAP = CodeMap({
    0:   'Esophagectomy was not minimally invasive',
    10:  'Minimally invasive esophagectomy (MIE) performed',
    988: 'Not applicable',
    999: 'Not documented / unknown',
}, width=3)


_ESO_PATH_RESPONSE_MAP = CodeMap({
    0:   'No viable cancer cells (complete response, score 0)',
    1:   'Single cells or rare small groups of cancer cells (near complete '
         'response, score 1)',
    2:   'Residual cancer with evident tumour regression, more than single '
         'cells or rare small groups (partial response, score 2)',
    3:   'Extensive residual cancer with no evident regression (poor or no '
         'response, score 3)',
    10:  'Pathology shows complete disappearance of the tumour',
    20:  'Pathology shows the tumour shrank by >=50%',
    30:  'Pathology shows the tumour shrank by <50%',
    40:  'Pathology shows the tumour did not shrink',
    988: 'Not applicable (no neoadjuvant therapy, no surgery, or GIST/sarcoma)',
    990: 'Tumour responded after neoadjuvant therapy, degree not specified',
    999: 'Neoadjuvant therapy given but the response is unknown / not documented',
}, width=3)


_ESO_RT_RESPONSE_MAP = CodeMap({
    10:  'Clinical complete response (CR, 100%)',
    20:  'Clinical partial response (PR, moderate response, 50%)',
    30:  'Clinical stable disease (SD, minimal response)',
    40:  'Clinical progressive disease (PD, poor response)',
    988: 'Not applicable (no radiotherapy, chemotherapy only, surgery '
         'performed, GIST/sarcoma, or distant-site radiotherapy only)',
    990: 'Tumour responded after treatment, degree not specified',
    999: 'Treatment given but the response is unknown / not documented',
}, width=3)


def _decode_ca19_9(series: pd.Series) -> pd.Series:
    """Pancreas SSF3: CA 19-9 lab value (p.95-96, 100-102)."""
    tiers = {
        980: '98.0-99.9 U/mL', 981: '100.0-199.9 U/mL', 982: '200.0-299.9 U/mL',
        983: '300.0-399.9 U/mL', 984: '400.0-499.9 U/mL', 985: '500.0-599.9 U/mL',
        986: '600.0-699.9 U/mL', 987: '700.0-799.9 U/mL', 989: '800.0-899.9 U/mL',
        990: '900.0-999.9 U/mL', 991: '1000.0-1999.9 U/mL',
        992: '2000.0-2999.9 U/mL', 993: '3000.0-3999.9 U/mL',
        994: '4000.0-4999.9 U/mL', 995: '5000.0-5999.9 U/mL',
        996: '6000.0-6999.9 U/mL', 997: '>=7000.0 U/mL',
    }

    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        s = strip_float_suffix(str(val).strip())
        if not s.isdigit():
            return s
        iv = int(s)
        if iv == 988:
            return ('Not applicable (first course started at another hospital '
                    'with no lab value)')
        if iv == 999:
            return 'CA 19-9 unknown / not tested before the first treatment'
        if iv in tiers:
            return f'CA 19-9 {tiers[iv]}'
        if iv == 0:
            return 'CA 19-9 0.0 U/mL'
        if iv == 1:
            return 'CA 19-9 <=0.1 U/mL'
        if 2 <= iv <= 979:
            return f'CA 19-9 {iv / 10:.1f} U/mL'
        return f'Code {s}'
    return series.apply(_d)


def _decode_mitotic_count(series: pd.Series) -> pd.Series:
    """Pancreas SSF5: mitotic count for neuroendocrine tumours (p.98-99)."""
    specials = {
        110: 'Mitotic grade 1 (<2 mitoses per 10 HPF / 2 mm2)',
        120: 'Mitotic grade 2 (2-20 mitoses per 10 HPF / 2 mm2)',
        130: 'Mitotic grade 3 (>20 mitoses per 10 HPF / 2 mm2)',
        988: 'Not applicable (histology is not a neuroendocrine tumour)',
        999: 'Not documented / unknown',
    }

    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        s = strip_float_suffix(str(val).strip())
        if not s.isdigit():
            return s
        iv = int(s)
        if iv in specials:
            return specials[iv]
        if 0 <= iv <= 21:
            return f'{iv} mitoses per 10 HPF (per 2 mm2)'
        return f'Code {s}'
    return series.apply(_d)


def _decode_hba1c(series: pd.Series) -> pd.Series:
    """Pancreas SSF6: HbA1c (p.100-101).

    First character = diabetes history (0 = none, 1 = documented), characters
    2-3 = the HbA1c value: 01 = 0.1%, 02-94 = 0.2-9.4%, 95 = 9.5-9.9%,
    96 = 10.0-10.9%, 97 = 11.0-11.9%, 98 = >=12.0%, 99 = unknown/not tested.
    """
    value_text = {
        95: '9.5-9.9%', 96: '10.0-10.9%', 97: '11.0-11.9%', 98: '>=12.0%',
        99: 'value unknown / not tested',
    }

    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        s = strip_float_suffix(str(val).strip())
        if not s.isdigit():
            return s
        if s == '988':
            return ('Not applicable (tested at another hospital with neither '
                    'the result nor the diabetes history available)')
        if s == '999':
            return 'Unknown / not tested'
        if len(s) == 3 and s[0] in '01':
            history = ('diabetes history documented' if s[0] == '1'
                       else 'no diabetes history')
            value = int(s[1:])
            if value in value_text:
                return f'HbA1c {value_text[value]}; {history}'
            if 1 <= value <= 94:
                return f'HbA1c {value / 10:.1f}%; {history}'
        return f'Code {s}'
    return series.apply(_d)


def _decode_ca125(series: pd.Series, timing: str) -> pd.Series:
    """Ovary SSF1/SSF2: serum CA-125 before / after treatment (p.165-166)."""
    tiers = {
        901: '901-1000 U/mL', 902: '1001-2000 U/mL', 903: '2001-3000 U/mL',
        904: '3001-4000 U/mL', 905: '4001-5000 U/mL', 906: '5001-6000 U/mL',
        907: '6001-7000 U/mL', 908: '7001-8000 U/mL', 909: '8001-9000 U/mL',
        910: '9001-10000 U/mL', 920: '10001-20000 U/mL',
        930: '20001-30000 U/mL', 931: '>=30001 U/mL',
    }
    when = 'before treatment' if timing == 'pre' else 'lowest after treatment'

    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        s = strip_float_suffix(str(val).strip())
        if not s.isdigit():
            return s
        iv = int(s)
        if iv == 988:
            return f'Not applicable (CA-125 {when} not available)'
        if iv == 999:
            return f'CA-125 {when} unknown / not documented'
        if iv in tiers:
            return f'CA-125 {when}: {tiers[iv]}'
        if 1 <= iv <= 900:
            return f'CA-125 {when}: {iv} U/mL'
        return f'Code {s}'
    return series.apply(_d)


_OVARY_RESIDUAL_MAP = CodeMap({
    0:   'No residual tumour',
    10:  'Residual tumour <=1 cm, no preoperative chemotherapy',
    20:  'Residual tumour <=1 cm, preoperative chemotherapy given',
    30:  'Residual tumour >1 cm, no preoperative chemotherapy',
    40:  'Residual tumour >1 cm, preoperative chemotherapy given',
    988: 'Not applicable (no surgery)',
    990: 'Gross residual tumour, size not recorded, no preoperative chemotherapy',
    991: 'Gross residual tumour, size not recorded, preoperative chemotherapy given',
    999: 'Not documented / unknown',
}, width=3)


_BLADDER_GRADE_MAP = CodeMap({
    10:  'WHO/ISUP low grade',
    20:  'WHO/ISUP high grade',
    988: 'Not applicable (no pathology, or histology is not urothelial)',
    999: 'Not graded by WHO/ISUP / not documented',
}, width=3)


_BLADDER_ENE_MAP = CodeMap({
    0:   'No regional lymph node involvement',
    10:  'Regional node involvement without extranodal extension',
    20:  'Regional node involvement with extranodal extension',
    30:  'Regional node involvement, extranodal extension unknown',
    988: 'Not applicable',
    999: 'Nodal involvement unknown / not assessable / not documented',
}, width=3)


_BLADDER_MUSCULARIS_MAP = CodeMap({
    0:   'Specimen does not include muscularis propria',
    10:  'Specimen includes muscularis propria',
    988: 'Not applicable (no TURBT, TURBT at another hospital with no report, '
         'or diagnostic TURBT only)',
    999: 'The specimen report does not state whether muscularis propria is '
         'included / unknown',
}, width=3)


# ─────────────────────────────────────────────────────────────────────────────
# Head & neck SSF decoders (Cancer-SSF-Manual pp.3-29)
#
# Applies to lip/oral cavity/tongue/salivary/oropharynx/nasopharynx/
# hypopharynx/larynx (C00-C14, C32), to mucosal melanoma of the sinonasal
# tract (C30.0, C31.0-C31.1) and to cervical nodes with an unknown primary
# (C76.0). Which of the ten SSFs a given sub-site actually records varies
# (manual pp.3-6); a field a sub-site does not record is coded 988.
# ─────────────────────────────────────────────────────────────────────────────

def _decode_hn_node_size(series: pd.Series) -> pd.Series:
    """SSF1: size of the involved cervical lymph node, in mm (p.7-8)."""
    specials = {
        0:   'No cervical lymph node involvement (N0)',
        987: 'Involved node >=987 mm',
        988: 'Not applicable (treated at another hospital, no data available)',
        990: 'Microscopic focus/microinvasion only; node not palpable and size '
             'not stated',
        991: 'Node smaller than 10 mm (exact size not available)',
        992: 'Node 10-20 mm (exact size not available)',
        993: 'Node 20-30 mm (exact size not available)',
        994: 'Node 30-40 mm (exact size not available)',
        995: 'Node 40-50 mm (exact size not available)',
        996: 'Node 50-60 mm (exact size not available)',
        997: 'Node larger than 60 mm (exact size not available)',
        999: 'Node status unknown / not documented / not assessable',
    }

    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        s = strip_float_suffix(str(val).strip())
        if not s.isdigit():
            return s
        iv = int(s)
        if iv in specials:
            return specials[iv]
        if 1 <= iv <= 986:
            return f'Involved node {iv} mm'
        return f'Code {s}'
    return series.apply(_d)


_HN_ECE_MAP = CodeMap({
    0:   'No extracapsular extension of cervical nodes',
    1:   'Clinically possible extracapsular extension (fixed/confluent/adherent '
         'nodes) but not stated in the pathology report',
    2:   'Clinically extracapsular extension, then neoadjuvant therapy, and the '
         'post-operative pathology reports no nodal involvement or no ECE',
    5:   'Pathology report states extracapsular extension is present',
    988: 'Not applicable (no cervical node involvement, or treated at another '
         'hospital with no data)',
    999: 'Nodal status unknown / not documented / not assessable',
}, width=3)


def _decode_hn_ece(series: pd.Series) -> pd.Series:
    """SSF2: extracapsular extension of cervical nodes (p.10-11)."""
    return _HN_ECE_MAP.decode(series)


# SSF3-SSF6 share one positional scheme: each of the three characters stands
# for one nodal region, 0 = not involved, 1 = involved, 8 = involved as part
# of cross-region disease that cannot be attributed to that region alone.
_HN_LEVEL_REGIONS = {
    'SSF3': ('Level I', 'Level II', 'Level III'),
    'SSF4': ('Level IV', 'Level V', 'Retropharyngeal'),
    'SSF5': ('Level VI', 'Level VII', 'Facial'),
    'SSF6': ('Lateral pharyngeal', 'Parotid', 'Suboccipital/retroauricular'),
}
_HN_LEVEL_STATUS = {
    '0': 'not involved',
    '1': 'involved',
    '8': 'cross-region involvement, cannot be localised',
}


def _decode_hn_levels(series: pd.Series, ssf_key: str) -> pd.Series:
    """SSF3-SSF6: extent of nodal involvement by region (p.12-19)."""
    regions = _HN_LEVEL_REGIONS[ssf_key]

    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        s = strip_float_suffix(str(val).strip())
        if s == '988':
            return ('Not applicable (examined or treated at another hospital '
                    'with no data available)')
        if s == '999':
            return ('Nodal status for these regions unknown / not documented / '
                    'not assessable')
        if len(s) == 3 and all(c in _HN_LEVEL_STATUS for c in s):
            if s == '000':
                return f'{regions[0]}/{regions[1]}/{regions[2]}: none involved (N0)'
            return '; '.join(f'{r}: {_HN_LEVEL_STATUS[c]}'
                             for r, c in zip(regions, s))
        return f'Code {s}'
    return series.apply(_d)


def _decode_hn_tumor_depth(series: pd.Series) -> pd.Series:
    """SSF7: measured tumour depth in the pathology report (p.20-21)."""
    specials = {
        980: 'Tumour depth >=98 mm',
        987: 'Carcinoma in situ',
        988: 'Not applicable (not an oral cavity case; or treated at another '
             'hospital with no data)',
        990: 'Microinvasion or focus/foci only, depth not described',
        997: 'Re-excision specimen has no residual tumour and the earlier '
             'report did not state a depth',
        998: 'No surgical specimen, or radiation/systemic therapy given first',
        999: 'The pathology report has no depth figure',
    }

    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        s = strip_float_suffix(str(val).strip())
        if not s.isdigit():
            return s
        iv = int(s)
        if iv in specials:
            return specials[iv]
        if iv == 0:
            return 'Tumour depth 0 mm'
        if 1 <= iv <= 979:
            return f'Tumour depth {iv / 10:.1f} mm'
        return f'Code {s}'
    return series.apply(_d)


def _decode_hn_margin(series: pd.Series) -> pd.Series:
    """SSF8: closest distance from tumour to the surgical margin (p.22-23)."""
    specials = {
        0:   'Margin positive, or <1 mm with the margin status not stated',
        980: 'Margin distance >=98 mm',
        987: 'Carcinoma in situ',
        988: 'Not applicable (not an oral cavity case; or treated at another '
             'hospital with no data)',
        990: 'Re-excision specimen has no residual tumour; margin distance '
             'not clear',
        998: 'No surgical specimen, or radiation/systemic therapy given first',
        999: 'The pathology report has no margin measurement',
    }

    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        s = strip_float_suffix(str(val).strip())
        if not s.isdigit():
            return s
        iv = int(s)
        if iv in specials:
            return specials[iv]
        if 1 <= iv <= 979:
            return f'Margin negative, {iv / 10:.1f} mm'
        return f'Code {s}'
    return series.apply(_d)


# SSF9 is a 3-character composite (p.24-25/30-31): first character = the
# clinician's overall ENE call, second = what imaging showed, third = what
# the physical examination showed.
_HN_ENE_FINAL = {
    '0': 'ENE(-)',
    '1': 'ENE(+)',
    '8': 'overt ENE not used (AJCC 8th ed. ch.9/10/14 staging)',
    '9': 'ENE status not described',
}
_HN_ENE_DETAIL = {
    '0': 'ENE(-)',
    '1': 'ENE(+) with the defining features described',
    '2': 'ENE(+) but without the defining features described',
    '9': 'not described / not performed',
}


def _decode_hn_ene_clinical(series: pd.Series) -> pd.Series:
    """SSF9: clinical extranodal extension (p.24-25)."""
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        s = strip_float_suffix(str(val).strip())
        if s == '988':
            return 'Not applicable (clinical N category is cN0)'
        if s == '998':
            return ('Not applicable (treated at another hospital with no data, '
                    'or diagnosed only after surgery)')
        if (len(s) == 3 and s[0] in _HN_ENE_FINAL
                and s[1] in _HN_ENE_DETAIL and s[2] in _HN_ENE_DETAIL):
            return (f'Overall: {_HN_ENE_FINAL[s[0]]}; '
                    f'imaging: {_HN_ENE_DETAIL[s[1]]}; '
                    f'physical exam: {_HN_ENE_DETAIL[s[2]]}')
        return f'Code {s}'
    return series.apply(_d)


def _decode_hn_ene_pathological(series: pd.Series) -> pd.Series:
    """SSF10: pathological extranodal extension (p.29-30).

    First character: 0 = nodal metastasis without ENE, 1 = ENE <=2 mm,
    2 = ENE >2 mm, 3 = ENE present with the distance unknown. Characters 2-3
    carry the distance (01-98 = 0.1-9.8 mm, 99 = not stated).
    """
    specials = {
        '000': 'Nodal metastasis confirmed, no pathological ENE',
        '199': 'Pathological ENE <=2 mm, exact distance not available',
        '210': 'Pathological ENE >=9.9 mm',
        '299': 'Pathological ENE >2 mm, exact distance not available (includes '
               'gross/macroscopic ENE without a measurement)',
        '399': 'Pathological ENE present, distance not available',
        '988': 'Not applicable (pN0 after biopsy/excision/dissection, no nodal '
               'surgery, or pNx with no lymphoid tissue)',
        '998': 'Nodal biopsy/surgery at another hospital, ENE status unknown',
        '999': 'Not assessable / not assessed / nodal involvement with ENE '
               'status unknown',
    }

    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        s = strip_float_suffix(str(val).strip())
        if s in specials:
            return specials[s]
        if len(s) == 3 and s.isdigit():
            head, dist = s[0], int(s[1:])
            if head == '1' and 1 <= dist <= 20:
                return f'Pathological ENE <=2 mm, measured {dist / 10:.1f} mm'
            if head == '2' and 21 <= dist <= 98:
                return f'Pathological ENE >2 mm, measured {dist / 10:.1f} mm'
        return f'Code {s}'
    return series.apply(_d)


# ─────────────────────────────────────────────────────────────────────────────
# Uterine corpus (endometrium) SSF decoders
# Cancer-SSF-Manual pp.151-161. SSF1/SSF2 reuse the breast ER/PR scheme;
# SSF7-SSF20 are not collected for this site and are coded 988.
# ─────────────────────────────────────────────────────────────────────────────

_UTERUS_FIGO_GRADE_MAP = CodeMap({
    1:   'Non-squamous/non-morular solid growth pattern <=5% (FIGO Grade 1)',
    2:   'Non-squamous/non-morular solid growth pattern 6-50% (FIGO Grade 2)',
    3:   'Non-squamous/non-morular solid growth pattern >50% (FIGO Grade 3)',
    # 987 appears in this field's 編碼範圍 line (p.157) but has no row in the
    # code table on p.163. Decoded explicitly rather than guessed, so a file
    # containing it is neither silently dropped nor given a made-up meaning.
    987: 'Code 987 — listed in the 編碼範圍 but undefined in the code table '
         '(confirm with the registry)',
    988: 'Not applicable',
    999: 'Unknown / not documented in the medical record',
}, width=3)


def _decode_uterus_figo_grade(series: pd.Series) -> pd.Series:
    """SSF3 for uterine corpus: % non-endometrioid cell type = FIGO grade."""
    return _UTERUS_FIGO_GRADE_MAP.decode(series)


_POLE_MAP = CodeMap({
    10:  'POLE mutation detected',
    20:  'No POLE mutation detected',
    30:  'POLE test performed but the result is uninterpretable',
    999: 'Not documented / unknown / not tested',
}, width=3)


def _decode_pole(series: pd.Series) -> pd.Series:
    """SSF4 for uterine corpus: POLE gene mutation."""
    return _POLE_MAP.decode(series)


_MSI_UTERUS_MAP = CodeMap({
    0:   'MSS / microsatellite stable; MMR proficient (pMMR)',
    10:  'MSI-L — low instability',
    20:  'MSI-H — high instability; or MMR deficient (dMMR)',
    988: 'Not applicable (GIST/NETs/high-grade dysplasia, or tested at another '
         'hospital with no result available)',
    999: 'Not documented / not tested / MSI indeterminate or equivocal',
}, width=3)


def _decode_msi_uterus(series: pd.Series) -> pd.Series:
    """SSF5 for uterine corpus: MSI / MMR status."""
    return _MSI_UTERUS_MAP.decode(series)


_P53_MAP = CodeMap({
    10:  'p53 protein expression abnormal',
    20:  'p53 protein expression normal',
    30:  'p53 protein test performed but the result is uninterpretable',
    999: 'TP53 gene test only with no p53 protein test / not documented / '
         'not tested',
}, width=3)


def _decode_p53(series: pd.Series) -> pd.Series:
    """SSF6 for uterine corpus: p53 tumour suppressor protein."""
    return _P53_MAP.decode(series)


_NOT_COLLECTED_MAP = CodeMap({
    988: 'Not applicable — this SSF is not collected for this cancer site',
}, width=3)


def _decode_not_collected(series: pd.Series) -> pd.Series:
    """An SSF the manual does not collect for this site: always 988 (p.1)."""
    return _NOT_COLLECTED_MAP.decode(series)


# ─────────────────────────────────────────────────────────────────────────────
# Thyroid SSF decoders
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Stomach / Colorectum / Cervix SSF decoders
# ─────────────────────────────────────────────────────────────────────────────

def _decode_cea_lab_value(series: pd.Series) -> pd.Series:
    """CEA lab value in ng/mL (SSF1 for stomach/colorectum/pancreas).

    Codes: 001=<=0.1, 002-986=0.2-98.6 ng/mL, 987=>=98.7, 988=N/A, 999=unknown.
    """
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        if iv == 988:
            return 'Not applicable (GIST/NETs or treated at external hospital)'
        if iv == 999:
            return 'CEA unknown / not tested'
        if iv == 1:
            return 'CEA <=0.1 ng/mL'
        if 2 <= iv <= 986:
            return f'CEA {iv / 10:.1f} ng/mL'
        if iv == 987:
            return 'CEA >=98.7 ng/mL'
        return f'CEA code {iv}'
    return series.apply(_d)


_CEA_NORMAL_MAP = CodeMap({
    10:  'CEA positive — above normal range',
    20:  'CEA negative — within normal range',
    30:  'CEA borderline — uncertain positive/negative',
    988: 'Not applicable',
    999: 'Unknown / CEA not tested',
}, width=3)


def _decode_cea_normal(series: pd.Series) -> pd.Series:
    """CEA vs. normal range (SSF2 for stomach/colorectum/pancreas)."""
    return _CEA_NORMAL_MAP.decode(series)


_H_PYLORI_MAP = CodeMap({
    0:   'H. pylori negative (all tests)',
    1:   'H. pylori positive — histology',
    2:   'H. pylori positive — bacterial culture',
    3:   'H. pylori positive — rapid urease test (RUT)',
    4:   'H. pylori positive — serology (antibody)',
    5:   'H. pylori positive — urea breath test (UBT)',
    6:   'H. pylori positive — stool antigen (HpSA)',
    7:   'H. pylori positive — PCR',
    8:   'H. pylori positive — method not specified',
    10:  'H. pylori positive — >=2 methods confirmed',
    988: 'Not applicable (GIST or NETs)',
    999: 'Unknown / not tested',
}, width=3)


def _decode_h_pylori(series: pd.Series) -> pd.Series:
    """SSF3 for stomach: H. pylori infection status and detection method."""
    return _H_PYLORI_MAP.decode(series)


_STOMACH_LVI_MAP = CodeMap({
    0:   'No lymphovascular invasion',
    10:  'Lymphovascular invasion present',
    988: 'Not applicable',
    990: 'No residual tumor (LVI not assessable after neoadjuvant)',
    999: 'Unknown / not documented',
}, width=3)


def _decode_stomach_lvi(series: pd.Series) -> pd.Series:
    """SSF5 for stomach: Lymphovascular invasion (LVI)."""
    return _STOMACH_LVI_MAP.decode(series)


_RAS_KRAS_NRAS_MAP = {
    '0': 'wild-type',
    '1': 'Codon 12 mutation',
    '2': 'Codon 13 mutation',
    '3': 'Codon 61 mutation',
    '4': 'Multi-codon mutation (>=2 codons, >=1 of 12/13/61)',
    '5': 'Non-12/13/61 mutation',
    '6': 'Mutated (codon NOS)',
    '7': 'Uninterpretable',
    '9': 'Not tested',
}


def _decode_ras_mutation(series: pd.Series) -> pd.Series:
    """SSF6 for colorectum: RAS (KRAS + NRAS) combined 3-character code.

    Position 1 = KRAS: 0=WT, 1=Codon12, 2=Codon13, 3=Codon61, 4=multi-codon,
                        5=non-12/13/61, 6=mutated NOS, 7=uninterpretable, 9=not tested
    Position 2 = NRAS: same scheme as KRAS
    Position 3 = always '8' (filler)
    Special: 988=N/A(GIST/NETs/no external data), 998=not documented/not tested
    """
    KRAS_NRAS = _RAS_KRAS_NRAS_MAP
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        s = str(val).strip()
        # Numeric sentinels
        try:
            iv = int(float(s))
            if iv == 988:
                return 'Not applicable (GIST/NETs/high-grade dysplasia or no external data)'
            if iv == 998:
                return 'RAS result not documented / not tested'
        except (ValueError, TypeError):
            pass
        # 3-char code e.g. "008", "128", "998"
        if len(s) == 3 and s[2] in ('8', '9'):
            k_code = s[0]
            n_code = s[1]
            k = KRAS_NRAS.get(k_code, f'?({k_code})')
            n = KRAS_NRAS.get(n_code, f'?({n_code})')
            return f'KRAS: {k} | NRAS: {n}'
        return f'RAS code: {val}'
    return series.apply(_d)


_MSI_CRC_MAP = CodeMap({
    0:   'MSS / Microsatellite stable; MMR proficient (pMMR)',
    10:  'MSI-L — Low instability',
    20:  'MSI-H — High instability; or MMR deficient (dMMR)',
    988: 'Not applicable (GIST/NETs/high-grade dysplasia or no external data)',
    999: 'Unknown / not tested; MSI indeterminate/equivocal',
}, width=3)


def _decode_msi_crc(series: pd.Series) -> pd.Series:
    """SSF10 for colorectum: MSI/MMR status."""
    return _MSI_CRC_MAP.decode(series)


_SCC_ANTIGEN_NORMAL_MAP = CodeMap({
    10:  'SCC antigen positive — above normal range',
    20:  'SCC antigen negative — within normal range',
    30:  'SCC antigen borderline',
    988: 'Not applicable',
    999: 'Unknown / not tested',
}, width=3)


def _decode_scc_antigen_normal(series: pd.Series) -> pd.Series:
    """SSF2 for cervix: SCC antigen vs. normal range."""
    return _SCC_ANTIGEN_NORMAL_MAP.decode(series)


# ─────────────────────────────────────────────────────────────────────────────
# SSF Profile registry
# ─────────────────────────────────────────────────────────────────────────────

def _build_profiles() -> Dict[str, SSFProfile]:
    """Build all cancer group SSF profiles.

    Returns dict of cancer_group → SSFProfile.
    Decoders are initialized lazily to avoid circular imports.
    """
    _bd = _breast_ssf_decoder_factory()

    profiles: Dict[str, SSFProfile] = {}

    # ── BREAST ────────────────────────────────────────────────────────────────
    profiles['breast'] = SSFProfile(
        cancer_group='breast',
        site_label='Breast Cancer',
        site_codes=('C50',),
        fields={
            'SSF1':  SSFFieldDef('SSF1', 'ER_Status', 'Estrogen receptor status',
                                 decoder=_bd['SSF1']),
            'SSF2':  SSFFieldDef('SSF2', 'PR_Status', 'Progesterone receptor status',
                                 decoder=_bd['SSF2']),
            'SSF3':  SSFFieldDef('SSF3', 'Neoadjuvant_Response', 'Neoadjuvant therapy response',
                                 decoder=_bd['SSF3']),
            'SSF4':  SSFFieldDef('SSF4', 'Sentinel_LN_Examined', 'Sentinel LN examined count',
                                 decoder=_bd['SSF4']),
            'SSF5':  SSFFieldDef('SSF5', 'Sentinel_LN_Positive', 'Sentinel LN positive count',
                                 decoder=_bd['SSF5']),
            'SSF6':  SSFFieldDef('SSF6', 'Nottingham_Grade', 'Nottingham/BR combined score and grade',
                                 decoder=_bd['SSF6']),
            'SSF7':  SSFFieldDef('SSF7', 'HER2_Status', 'HER2 IHC+ISH combined status',
                                 decoder=_bd['SSF7']),
            'SSF8':  SSFFieldDef('SSF8', 'Pagets_Disease', "Paget's disease of nipple",
                                 decoder=_bd['SSF8']),
            'SSF9':  SSFFieldDef('SSF9', 'LVI_SSF', 'Lymphovascular invasion (SSF source)',
                                 decoder=_bd['SSF9']),
            'SSF10': SSFFieldDef('SSF10', 'Ki67_Index', 'Ki-67 proliferation index',
                                 decoder=_bd['SSF10']),
        },
        notes='Full custom decoders for all breast SSF fields per Taiwan SSF Manual.',
    )

    # ── LUNG ──────────────────────────────────────────────────────────────────
    profiles['lung'] = SSFProfile(
        cancer_group='lung',
        site_label='Lung Cancer',
        site_codes=('C34',),
        fields={
            'SSF1':  SSFFieldDef('SSF1', 'Separate_Tumor_Nodules',
                                 'Separate tumor nodules / ipsilateral lung',
                                 decoder=_decode_lung_ssf1_nodules),
            'SSF2':  SSFFieldDef('SSF2', 'Visceral_Pleural_Invasion',
                                 'Visceral pleural invasion (PL0-PL3)',
                                 decoder=_decode_lung_ssf2_vpi),
            'SSF3':  SSFFieldDef('SSF3', 'Performance_Status_SSF3',
                                 'Performance status (ECOG/KPS) before treatment',
                                 decoder=_decode_lung_ssf3_ecog),
            'SSF4':  SSFFieldDef('SSF4', 'Malignant_Pleural_Effusion',
                                 'Malignant pleural effusion',
                                 decoder=_decode_lung_ssf4_pleural_effusion),
            'SSF5':  SSFFieldDef('SSF5', 'Mediastinal_LN_Sampling',
                                 'Mediastinal LN sampling/dissection (N2 stations sampled)',
                                 decoder=_decode_lung_ssf5_mediastinal),
            'SSF6':  SSFFieldDef('SSF6', 'EGFR_Mutation',
                                 'EGFR gene mutation (3-char alphabetic code)',
                                 decoder=_decode_lung_egfr),
            'SSF7':  SSFFieldDef('SSF7', 'ALK_Translocation',
                                 'ALK gene translocation',
                                 decoder=_decode_lung_alk),
            'SSF8':  SSFFieldDef('SSF8', 'Adenocarcinoma_Component',
                                 'Specific lung adenocarcinoma component (micropapillary/solid/cribriform)',
                                 decoder=_decode_lung_ssf8_adeno),
            'SSF9':  SSFFieldDef('SSF9', 'Tumor_Nodule_Count',
                                 'Tumor nodule count (for early-stage multi-tumor cases)',
                                 decoder=_decode_lung_ssf9_nodules),
            'SSF10': SSFFieldDef('SSF10', 'SSF10_Lung',
                                 'SSF10 — Not defined in TCR codebook for lung',
                                 decoder=_decode_not_collected),
        },
        notes='Per 2025 TCR codebook: SSF1=nodules, SSF2=VPI, SSF3=ECOG, SSF4=effusion, SSF5=mediastinal LN, SSF6=EGFR (alpha), SSF7=ALK, SSF8=adeno component, SSF9=nodule count.',
    )

    # ── COLORECTUM ────────────────────────────────────────────────────────────
    profiles['colorectum'] = SSFProfile(
        cancer_group='colorectum',
        site_label='Colorectal Cancer',
        site_codes=('C18', 'C19', 'C20', 'C21'),
        fields={
            'SSF1':  SSFFieldDef('SSF1', 'CEA_Lab_Value',
                                 'CEA lab value (ng/mL)',
                                 decoder=_decode_cea_lab_value, unit='ng/mL'),
            'SSF2':  SSFFieldDef('SSF2', 'CEA_vs_Normal',
                                 'CEA vs. normal range (above/below/borderline)',
                                 decoder=_decode_cea_normal),
            'SSF3':  SSFFieldDef('SSF3', 'Tumor_Regression_Grade',
                                 'Tumor regression grade',
                                 decoder=_decode_tumor_regression),
            'SSF4':  SSFFieldDef('SSF4', 'Circumferential_Resection_Margin',
                                 'Circumferential resection margin',
                                 decoder=_decode_crm),
            'SSF5':  SSFFieldDef('SSF5', 'BRAF_Mutation',
                                 'BRAF mutation status',
                                 decoder=_decode_braf),
            'SSF6':  SSFFieldDef('SSF6', 'RAS_Mutation',
                                 'RAS (KRAS+NRAS) combined mutation code (3-char)',
                                 decoder=_decode_ras_mutation),
            'SSF7':  SSFFieldDef('SSF7', 'Intestinal_Obstruction',
                                 'Intestinal obstruction',
                                 decoder=_decode_obstruction),
            'SSF8':  SSFFieldDef('SSF8', 'Intestinal_Perforation',
                                 'Intestinal perforation',
                                 decoder=_decode_perforation),
            'SSF9':  SSFFieldDef('SSF9', 'Distance_to_Anus',
                                 'Distance to anus (rectum/rectosigmoid C19/C20 only; colon C18/C21 = 988)',
                                 decoder=_decode_distance_to_anus, unit='mm'),
            'SSF10': SSFFieldDef('SSF10', 'MSI_MMR_Status',
                                 'MSI/MMR status',
                                 decoder=_decode_msi_crc),
        },
        notes='Per 2025 TCR codebook: SSF1=CEA value, SSF2=CEA vs normal, SSF6=RAS(KRAS+NRAS) 3-char, SSF10=MSI/MMR. SSF9 for rectum only.',
    )

    # ── LIVER / HCC ───────────────────────────────────────────────────────────
    profiles['liver'] = SSFProfile(
        cancer_group='liver',
        site_label='Liver Cancer (HCC/Cholangio)',
        site_codes=('C22',),
        fields={
            'SSF1':  SSFFieldDef('SSF1', 'AFP_Level',
                                 'Alpha-fetoprotein (AFP) level — TCR-specific coding',
                                 decoder=_decode_liver_afp, unit='ng/mL'),
            'SSF2':  SSFFieldDef('SSF2', 'Liver_Fibrosis_Ishak',
                                 'Liver fibrosis grade (Ishak score F0-F6)',
                                 decoder=_decode_liver_fibrosis),
            'SSF3':  SSFFieldDef('SSF3', 'Child_Pugh_Score',
                                 'Child-Pugh class and score (3-char code)',
                                 decoder=_decode_child_pugh),
            'SSF4':  SSFFieldDef('SSF4', 'Creatinine',
                                 'Creatinine (mg/dL x10)',
                                 decoder=lambda s: _decode_lab_value_10x(s, 'Creatinine', 'mg/dL'),
                                 unit='mg/dL'),
            'SSF5':  SSFFieldDef('SSF5', 'Total_Bilirubin',
                                 'Total bilirubin (mg/dL x10)',
                                 decoder=lambda s: _decode_lab_value_10x(s, 'Total bilirubin', 'mg/dL'),
                                 unit='mg/dL'),
            'SSF6':  SSFFieldDef('SSF6', 'INR',
                                 'INR (prothrombin time, coded as value x10)',
                                 decoder=_decode_liver_inr),
            'SSF7':  SSFFieldDef('SSF7', 'HBsAg',
                                 'HBsAg (hepatitis B surface antigen) with carrier history',
                                 decoder=_decode_hbsag),
            'SSF8':  SSFFieldDef('SSF8', 'Anti_HCV',
                                 'Anti-HCV (hepatitis C antibody/antigen/RNA) with history',
                                 decoder=_decode_anti_hcv),
            'SSF9':  SSFFieldDef('SSF9', 'SSF9_Liver',
                                 'SSF9 — Not defined in TCR codebook for liver (988 for all)',
                                 decoder=_decode_not_collected),
            'SSF10': SSFFieldDef('SSF10', 'SSF10_Liver',
                                 'SSF10 — Not defined in TCR codebook for liver (988 for all)',
                                 decoder=_decode_not_collected),
        },
        notes='Per 2025 TCR codebook: SSF1=AFP (complex encoding), SSF2=Ishak fibrosis, SSF3=Child-Pugh (3-char), SSF4=Creatinine, SSF5=Bilirubin, SSF6=INR, SSF7=HBsAg, SSF8=Anti-HCV.',
    )

    # ── CERVIX ────────────────────────────────────────────────────────────────
    profiles['cervix'] = SSFProfile(
        cancer_group='cervix',
        site_label='Cervical Cancer',
        site_codes=('C53',),
        fields={
            'SSF1':  SSFFieldDef('SSF1', 'SCC_Antigen_Lab_Value',
                                 'SCC antigen lab value (generic numeric)',
                                 decoder=_decode_scc_lab_value),
            'SSF2':  SSFFieldDef('SSF2', 'SCC_Antigen_vs_Normal',
                                 'SCC antigen vs. normal range',
                                 decoder=_decode_scc_antigen_normal),
            'SSF3':  SSFFieldDef('SSF3', 'SSF3_Cervix',
                                 'SSF3 — Not defined in TCR codebook for cervix (988 for all)',
                                 decoder=_decode_not_collected),
            'SSF4':  SSFFieldDef('SSF4', 'SSF4_Cervix',
                                 'SSF4 — Not defined in TCR codebook for cervix (988 for all)',
                                 decoder=_decode_not_collected),
            'SSF5':  SSFFieldDef('SSF5', 'SSF5_Cervix',
                                 'SSF5 — Not defined in TCR codebook for cervix (988 for all)',
                                 decoder=_decode_not_collected),
            'SSF6':  SSFFieldDef('SSF6', 'SSF6_Cervix',
                                 'SSF6 — Not defined in TCR codebook for cervix (988 for all)',
                                 decoder=_decode_not_collected),
            'SSF7':  SSFFieldDef('SSF7', 'SSF7_Cervix',
                                 'SSF7 — Not defined in TCR codebook for cervix (988 for all)',
                                 decoder=_decode_not_collected),
            'SSF8':  SSFFieldDef('SSF8', 'SSF8_Cervix',
                                 'SSF8 — Not defined in TCR codebook for cervix (988 for all)',
                                 decoder=_decode_not_collected),
            'SSF9':  SSFFieldDef('SSF9', 'SSF9_Cervix',
                                 'SSF9 — Not defined in TCR codebook for cervix (988 for all)',
                                 decoder=_decode_not_collected),
            'SSF10': SSFFieldDef('SSF10', 'SSF10_Cervix',
                                 'SSF10 — Not defined in TCR codebook for cervix (988 for all)',
                                 decoder=_decode_not_collected),
        },
        notes='Per 2025 TCR codebook: only SSF1 (SCC antigen value) and SSF2 (SCC vs normal) are defined. SSF3-10 all carry 988.',
    )

    # ── STOMACH ───────────────────────────────────────────────────────────────
    profiles['stomach'] = SSFProfile(
        cancer_group='stomach',
        site_label='Gastric Cancer',
        site_codes=('C16',),
        fields={
            'SSF1':  SSFFieldDef('SSF1', 'CEA_Lab_Value',
                                 'CEA lab value (ng/mL)',
                                 decoder=_decode_cea_lab_value, unit='ng/mL'),
            'SSF2':  SSFFieldDef('SSF2', 'CEA_vs_Normal',
                                 'CEA vs. normal range (above/below/borderline)',
                                 decoder=_decode_cea_normal),
            'SSF3':  SSFFieldDef('SSF3', 'H_Pylori_Status',
                                 'H. pylori infection status and detection method',
                                 decoder=_decode_h_pylori),
            'SSF4':  SSFFieldDef('SSF4', 'Tumor_Depth_Path',
                                 'Tumor depth in pathology (0.1mm units)',
                                 decoder=_decode_tumor_depth, unit='0.1mm'),
            'SSF5':  SSFFieldDef('SSF5', 'LVI_Stomach',
                                 'Lymphovascular invasion (LVI)',
                                 decoder=_decode_stomach_lvi),
            'SSF6':  SSFFieldDef('SSF6', 'SSF6_Stomach',
                                 'SSF6 — Not defined in TCR codebook for stomach (988 for all)',
                                 decoder=_decode_not_collected),
            'SSF7':  SSFFieldDef('SSF7', 'SSF7_Stomach',
                                 'SSF7 — Not defined in TCR codebook for stomach (988 for all)',
                                 decoder=_decode_not_collected),
            'SSF8':  SSFFieldDef('SSF8', 'SSF8_Stomach',
                                 'SSF8 — Not defined in TCR codebook for stomach (988 for all)',
                                 decoder=_decode_not_collected),
            'SSF9':  SSFFieldDef('SSF9', 'SSF9_Stomach',
                                 'SSF9 — Not defined in TCR codebook for stomach (988 for all)',
                                 decoder=_decode_not_collected),
            'SSF10': SSFFieldDef('SSF10', 'SSF10_Stomach',
                                 'SSF10 — Not defined in TCR codebook for stomach (988 for all)',
                                 decoder=_decode_not_collected),
        },
        notes='Per 2025 TCR codebook: SSF1=CEA value, SSF2=CEA vs normal, SSF3=H.pylori method, SSF4=tumor depth (0.1mm), SSF5=LVI. SSF6-10 not defined (all 988).',
    )

    # ── THYROID ───────────────────────────────────────────────────────────────
    # THYROID -- the Cancer-SSF-Manual (p.1) lists the sites that collect
    # SSFs: 頭頸、食道、胃、結腸直腸、肝、胰、肺、乳、子宮頸、子宮體、卵巢、
    # 膀胱、攝護腺、淋巴瘤、白血病. Thyroid (C73) is NOT among them, and the
    # manual's rule for any site it does not cover is "SSF1-SSF20 應編碼988".
    # This profile used to define ten invented fields (tumour focality,
    # extrathyroidal extension, BRAF, thyroglobulin, ...) that exist in the
    # clinical literature but not in the TCR code book, so decoding a real
    # thyroid file produced confident-looking labels for codes that are all
    # supposed to be 988.
    profiles['thyroid'] = SSFProfile(
        cancer_group='thyroid',
        site_label='Thyroid Cancer (no TCR SSFs)',
        site_codes=('C73',),
        fields={
            f'SSF{i}': SSFFieldDef(
                f'SSF{i}', f'SSF{i}_Thyroid',
                f'SSF{i} — thyroid is not an SSF-collecting site; every SSF '
                f'is coded 988 (manual p.1)',
                decoder=_decode_not_collected)
            for i in range(1, 11)
        },
        notes='Thyroid is not in the code book list of SSF-collecting sites '
              '(p.1), so SSF1-SSF20 must all be 988. Clinical factors such as '
              'focality, extrathyroidal extension or BRAF status are simply '
              'not part of the Taiwan registry SSF schema for this site.',
    )

    # ── PROSTATE ──────────────────────────────────────────────────────────────
    profiles['prostate'] = SSFProfile(
        cancer_group='prostate',
        site_label='Prostate Cancer',
        site_codes=('C61',),
        fields={
            # Field assignments follow the Cancer-SSF-Manual (prostate),
            # pp.175-191. An earlier version of this profile had SSF2-SSF8
            # shifted onto unrelated concepts (positive cores, margins,
            # seminal vesicle invasion, ...), which made every decoded
            # prostate column clinically wrong.
            'SSF1':  SSFFieldDef('SSF1', 'PSA_Lab_Value',
                                 'PSA lab value, highest within 3 months before '
                                 'pathological diagnosis (p.183)',
                                 decoder=_decode_psa, unit='ng/mL'),
            'SSF2':  SSFFieldDef('SSF2', 'Gleason_Pattern_Biopsy',
                                 "Gleason primary + secondary pattern on needle "
                                 "core biopsy/TURP (p.179)",
                                 decoder=lambda s: _decode_gleason_patterns(s, 'biopsy')),
            'SSF3':  SSFFieldDef('SSF3', 'Gleason_Score_Biopsy',
                                 "Gleason score on needle core biopsy/TURP (p.181)",
                                 decoder=lambda s: _decode_gleason_score(s, 'biopsy')),
            'SSF4':  SSFFieldDef('SSF4', 'Gleason_Pattern_Prostatectomy',
                                 'Gleason primary + secondary pattern on radical '
                                 'prostatectomy/autopsy (p.183)',
                                 decoder=lambda s: _decode_gleason_patterns(s, 'prostatectomy')),
            'SSF5':  SSFFieldDef('SSF5', 'Gleason_Score_Prostatectomy',
                                 'Gleason score on radical prostatectomy/autopsy (p.185)',
                                 decoder=lambda s: _decode_gleason_score(s, 'prostatectomy')),
            'SSF6':  SSFFieldDef('SSF6', 'Biopsy_Cores_Examined',
                                 'Number of biopsy cores examined (p.187)',
                                 decoder=lambda s: _decode_biopsy_cores(s, 'examined')),
            'SSF7':  SSFFieldDef('SSF7', 'Biopsy_Cores_Positive',
                                 'Number of positive biopsy cores (p.188)',
                                 decoder=lambda s: _decode_biopsy_cores(s, 'positive')),
            'SSF8':  SSFFieldDef('SSF8', 'Clinical_T_Staging_Method',
                                 'How the clinical T stage was determined (p.191)',
                                 decoder=_decode_prostate_t_staging),
            'SSF9':  SSFFieldDef('SSF9', 'SSF9_Prostate',
                                 'SSF9 — not defined in the TCR codebook for '
                                 'prostate (988 for all)',
                                 decoder=None),
            'SSF10': SSFFieldDef('SSF10', 'SSF10_Prostate',
                                 'SSF10 — not defined in the TCR codebook for '
                                 'prostate (988 for all)',
                                 decoder=None),
        },
        notes='SSF1-SSF8 per Cancer-SSF-Manual pp.175-191; SSF9-SSF20 are not '
              'collected for prostate and should be 988. Gleason pattern codes '
              'are primary*10+secondary, NOT the summed score (034 = 3+4).',
    )

    # ── ESOPHAGUS ─────────────────────────────────────────────────────────────
    profiles['esophagus'] = SSFProfile(
        cancer_group='esophagus',
        site_label='Esophageal Cancer',
        site_codes=('C15',),
        fields={
            'SSF1': SSFFieldDef('SSF1', 'PET_CT_Performed',
                                'PET-CT examination (p.33)',
                                decoder=_ESO_PETCT_MAP.decode),
            'SSF2': SSFFieldDef('SSF2', 'Minimally_Invasive_Esophagectomy',
                                'Minimally invasive esophagectomy (p.34)',
                                decoder=_ESO_MIE_MAP.decode),
            'SSF3': SSFFieldDef('SSF3', 'Neoadjuvant_Path_Response',
                                'Pathological response after neoadjuvant '
                                'therapy (p.35)',
                                decoder=_ESO_PATH_RESPONSE_MAP.decode),
            'SSF4': SSFFieldDef('SSF4', 'Radiotherapy_Clinical_Response',
                                'Clinical response after radiotherapy (p.36)',
                                decoder=_ESO_RT_RESPONSE_MAP.decode),
            **{f'SSF{i}': SSFFieldDef(
                f'SSF{i}', f'SSF{i}_Esophagus',
                f'SSF{i} — not collected for esophagus (988)',
                decoder=_decode_not_collected) for i in range(5, 11)},
        },
        notes='Cancer-SSF-Manual pp.31-36: esophagus collects SSF1-SSF4 only.',
    )

    # ── PANCREAS ──────────────────────────────────────────────────────────────
    profiles['pancreas'] = SSFProfile(
        cancer_group='pancreas',
        site_label='Pancreatic Cancer',
        site_codes=('C25',),
        fields={
            'SSF1': SSFFieldDef('SSF1', 'CEA_Lab_Value_Pancreas',
                                'CEA lab value (p.93)',
                                decoder=_decode_cea_lab_value, unit='ng/mL'),
            'SSF2': SSFFieldDef('SSF2', 'CEA_vs_Normal_Pancreas',
                                'CEA vs. normal range (p.94)',
                                decoder=_decode_cea_normal),
            'SSF3': SSFFieldDef('SSF3', 'CA19_9',
                                'Carbohydrate antigen 19-9 (p.95)',
                                decoder=_decode_ca19_9, unit='U/mL'),
            'SSF4': SSFFieldDef('SSF4', 'Ki67_Pancreas',
                                'Ki-67 for neuroendocrine tumours (p.97)',
                                decoder=lambda s: _breast_ssf_decoder_factory()['SSF10'](s)),
            'SSF5': SSFFieldDef('SSF5', 'Mitotic_Count',
                                'Mitotic count for neuroendocrine tumours (p.98)',
                                decoder=_decode_mitotic_count),
            'SSF6': SSFFieldDef('SSF6', 'HbA1c',
                                'Glycated haemoglobin with diabetes history (p.100)',
                                decoder=_decode_hba1c),
            **{f'SSF{i}': SSFFieldDef(
                f'SSF{i}', f'SSF{i}_Pancreas',
                f'SSF{i} — not collected for pancreas (988)',
                decoder=_decode_not_collected) for i in range(7, 11)},
        },
        notes='Cancer-SSF-Manual pp.91-101: pancreas collects SSF1-SSF6. SSF4 '
              'uses the same Ki-67 code scheme as breast SSF10.',
    )

    # ── OVARY ─────────────────────────────────────────────────────────────────
    profiles['ovary'] = SSFProfile(
        cancer_group='ovary',
        site_label='Ovarian Cancer',
        site_codes=('C56',),
        fields={
            'SSF1': SSFFieldDef('SSF1', 'CA125_Pre_Treatment',
                                'Serum CA-125 before treatment (p.165)',
                                decoder=lambda s: _decode_ca125(s, 'pre'),
                                unit='U/mL'),
            'SSF2': SSFFieldDef('SSF2', 'CA125_Post_Treatment',
                                'Lowest serum CA-125 after treatment (p.166)',
                                decoder=lambda s: _decode_ca125(s, 'post'),
                                unit='U/mL'),
            'SSF3': SSFFieldDef('SSF3', 'Residual_Tumor_After_Surgery',
                                'Residual tumour status and size after surgery (p.167)',
                                decoder=_OVARY_RESIDUAL_MAP.decode),
            **{f'SSF{i}': SSFFieldDef(
                f'SSF{i}', f'SSF{i}_Ovary',
                f'SSF{i} — not collected for ovary (988)',
                decoder=_decode_not_collected) for i in range(4, 11)},
        },
        notes='Cancer-SSF-Manual pp.163-167: ovary collects SSF1-SSF3 only.',
    )

    # ── BLADDER ───────────────────────────────────────────────────────────────
    profiles['bladder'] = SSFProfile(
        cancer_group='bladder',
        site_label='Bladder Cancer',
        site_codes=('C67',),
        fields={
            'SSF1': SSFFieldDef('SSF1', 'WHO_ISUP_Grade',
                                'WHO/ISUP grade (p.171)',
                                decoder=_BLADDER_GRADE_MAP.decode),
            'SSF2': SSFFieldDef('SSF2', 'Regional_Node_ENE_Bladder',
                                'Extranodal extension of regional nodes (p.172)',
                                decoder=_BLADDER_ENE_MAP.decode),
            'SSF3': SSFFieldDef('SSF3', 'Muscularis_Propria_Specimen',
                                'Muscularis propria in the pathology specimen (p.173)',
                                decoder=_BLADDER_MUSCULARIS_MAP.decode),
            **{f'SSF{i}': SSFFieldDef(
                f'SSF{i}', f'SSF{i}_Bladder',
                f'SSF{i} — not collected for bladder (988)',
                decoder=_decode_not_collected) for i in range(4, 11)},
        },
        notes='Cancer-SSF-Manual pp.169-173: bladder collects SSF1-SSF3 only.',
    )

    # ── HEAD & NECK ───────────────────────────────────────────────────────────
    profiles['head_neck'] = SSFProfile(
        cancer_group='head_neck',
        site_label='Head and Neck Cancer',
        site_codes=('C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07',
                    'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14',
                    'C30', 'C31', 'C32', 'C760'),
        fields={
            'SSF1':  SSFFieldDef('SSF1', 'Cervical_Node_Size',
                                 'Size of the involved cervical lymph node, mm (p.7)',
                                 decoder=_decode_hn_node_size, unit='mm'),
            'SSF2':  SSFFieldDef('SSF2', 'Cervical_Node_ECE',
                                 'Extracapsular extension of cervical nodes (p.10)',
                                 decoder=_decode_hn_ece),
            'SSF3':  SSFFieldDef('SSF3', 'Node_Levels_I_III',
                                 'Extent of involvement, levels I-III (p.12)',
                                 decoder=lambda s: _decode_hn_levels(s, 'SSF3')),
            'SSF4':  SSFFieldDef('SSF4', 'Node_Levels_IV_V_Retropharyngeal',
                                 'Extent of involvement, levels IV-V and '
                                 'retropharyngeal (p.14)',
                                 decoder=lambda s: _decode_hn_levels(s, 'SSF4')),
            'SSF5':  SSFFieldDef('SSF5', 'Node_Levels_VI_VII_Facial',
                                 'Extent of involvement, levels VI-VII and '
                                 'facial nodes (p.16)',
                                 decoder=lambda s: _decode_hn_levels(s, 'SSF5')),
            'SSF6':  SSFFieldDef('SSF6', 'Node_Lateral_Parotid_Suboccipital',
                                 'Extent of involvement, lateral pharyngeal / '
                                 'parotid / suboccipital nodes (p.18)',
                                 decoder=lambda s: _decode_hn_levels(s, 'SSF6')),
            'SSF7':  SSFFieldDef('SSF7', 'Tumor_Depth_HN',
                                 'Measured tumour depth in the pathology report (p.20)',
                                 decoder=_decode_hn_tumor_depth),
            'SSF8':  SSFFieldDef('SSF8', 'Surgical_Margin_Distance',
                                 'Closest tumour-to-margin distance (p.22)',
                                 decoder=_decode_hn_margin),
            'SSF9':  SSFFieldDef('SSF9', 'ENE_Clinical',
                                 'Clinical extranodal extension (p.24)',
                                 decoder=_decode_hn_ene_clinical),
            'SSF10': SSFFieldDef('SSF10', 'ENE_Pathological',
                                 'Pathological extranodal extension (p.29)',
                                 decoder=_decode_hn_ene_pathological),
        },
        notes='Cancer-SSF-Manual pp.3-29. Which SSFs a sub-site records varies '
              '(e.g. lip records SSF1, SSF7, SSF9-10; C76.0 records SSF1, '
              'SSF3-6, SSF9-10); a field a sub-site does not record is 988. '
              'C30.0/C31.0-C31.1 belong here only for mucosal melanoma.',
    )

    # NASOPHARYNX -- there is no separate nasopharynx schema in the TCR
    # manual: C11 is part of 頭頸部癌症 and uses the head & neck SSF fields
    # above (cervical node size / ECE / level involvement / depth / margin /
    # ENE). The profile that used to sit here defined EBV VCA IgA, plasma EBV
    # DNA, cranial nerve palsy and so on -- real clinical factors, but not
    # what the registry records in SSF1-SSF10 for this site, so decoding a
    # real nasopharyngeal file produced confident labels for the wrong data.


    # ── ENDOMETRIUM ───────────────────────────────────────────────────────────
    profiles['endometrium'] = SSFProfile(
        cancer_group='endometrium',
        site_label='Uterine Corpus (Endometrial) Cancer',
        site_codes=('C54',),
        fields={
            # Field assignments follow the Cancer-SSF-Manual (子宮體癌),
            # pp.151-161. An earlier version had SSF3-SSF10 shifted: POLE sat
            # on SSF8 instead of SSF4, MSI on SSF7 instead of SSF5 and p53 on
            # SSF9 instead of SSF6, so every decoded column was mislabelled.
            'SSF1':  SSFFieldDef('SSF1', 'ER_Endometrium',
                                 'Estrogen receptor assay (p.153) — same code '
                                 'scheme as breast SSF1',
                                 decoder=lambda s: _breast_ssf_decoder_factory()['SSF1'](s)),
            'SSF2':  SSFFieldDef('SSF2', 'PR_Endometrium',
                                 'Progesterone receptor assay (p.155) — same '
                                 'code scheme as breast SSF2',
                                 decoder=lambda s: _breast_ssf_decoder_factory()['SSF2'](s)),
            'SSF3':  SSFFieldDef('SSF3', 'FIGO_Grade_Uterus',
                                 '% non-endometrioid cell type in mixed '
                                 'histology = FIGO grade (p.157)',
                                 decoder=_decode_uterus_figo_grade),
            'SSF4':  SSFFieldDef('SSF4', 'POLE_Mutation',
                                 'POLE gene mutation (p.158)',
                                 decoder=_decode_pole),
            'SSF5':  SSFFieldDef('SSF5', 'MSI_MMR_Status_Uterus',
                                 'Microsatellite instability / MMR status (p.159)',
                                 decoder=_decode_msi_uterus),
            'SSF6':  SSFFieldDef('SSF6', 'p53_Status',
                                 'p53 tumour suppressor protein (p.161)',
                                 decoder=_decode_p53),
            'SSF7':  SSFFieldDef('SSF7', 'SSF7_Endometrium',
                                 'SSF7 — not collected for uterine corpus (988)',
                                 decoder=_decode_not_collected),
            'SSF8':  SSFFieldDef('SSF8', 'SSF8_Endometrium',
                                 'SSF8 — not collected for uterine corpus (988)',
                                 decoder=_decode_not_collected),
            'SSF9':  SSFFieldDef('SSF9', 'SSF9_Endometrium',
                                 'SSF9 — not collected for uterine corpus (988)',
                                 decoder=_decode_not_collected),
            'SSF10': SSFFieldDef('SSF10', 'SSF10_Endometrium',
                                 'SSF10 — not collected for uterine corpus (988)',
                                 decoder=_decode_not_collected),
        },
        notes='SSF1-SSF6 per Cancer-SSF-Manual pp.151-161; SSF7-SSF20 are not '
              'collected for this site and must be 988. The FIGO 2023 molecular '
              'classes are derived from POLE (SSF4), MMR (SSF5) and p53 (SSF6) '
              '-- the registry has no separate field for the class itself.',
    )

    # ── GENERIC FALLBACK ──────────────────────────────────────────────────────
    profiles['generic'] = SSFProfile(
        cancer_group='generic',
        site_label='Cancer (generic)',
        site_codes=(),
        fields={
            f'SSF{i}': SSFFieldDef(
                f'SSF{i}', f'SSF{i}',
                f'Site-specific factor {i} (generic numeric)',
                decoder=lambda s, i=i: _generic_ssf(s, f'SSF{i}'),
            )
            for i in range(1, 11)
        },
        notes=(
            'Generic fallback: SSF1-10 are decoded as numeric codes with standard sentinels. '
            'No cancer-specific clinical interpretation. '
            'Add a new SSFProfile to ssf_registry.py to enable full decoding for this cancer type.'
        ),
    )

    return profiles


# Build profiles once at module load
_PROFILES: Dict[str, SSFProfile] = _build_profiles()


# ─────────────────────────────────────────────────────────────────────────────
# Wire encoders (inverse of decoders) onto each profile field.
#
# CodeMap-backed fields get their CodeMap's .encode for free -- decode and
# encode read the exact same dict, so they can never drift apart. Bespoke
# composite/scaled/combinatorial fields (ER/PR, HER2, Ki-67, AFP, EGFR, RAS,
# ...) use the matching encode_* function from tcr_decoder.encoders, which
# mirrors the corresponding decode_*/​_decode_* function's exact output
# format. Fields with no custom decoder (decoder=None) fall back to
# encode_generic_ssf in apply_ssf_encode_profile, mirroring how
# apply_ssf_profile() falls back to _generic_ssf for decoding them.
# ─────────────────────────────────────────────────────────────────────────────

_ENCODER_WIRING: Dict[Tuple[str, str], Callable] = {
    ('esophagus', 'SSF1'): _ESO_PETCT_MAP.encode,
    ('esophagus', 'SSF2'): _ESO_MIE_MAP.encode,
    ('esophagus', 'SSF3'): _ESO_PATH_RESPONSE_MAP.encode,
    ('esophagus', 'SSF4'): _ESO_RT_RESPONSE_MAP.encode,
    **{('esophagus', f'SSF{i}'): _NOT_COLLECTED_MAP.encode for i in range(5, 11)},

    ('pancreas', 'SSF1'): _enc.encode_cea_lab_value,
    ('pancreas', 'SSF2'): _CEA_NORMAL_MAP.encode,
    ('pancreas', 'SSF3'): _enc.encode_ca19_9,
    ('pancreas', 'SSF4'): _enc.encode_ki67,
    ('pancreas', 'SSF5'): _enc.encode_mitotic_count,
    ('pancreas', 'SSF6'): _enc.encode_hba1c,
    **{('pancreas', f'SSF{i}'): _NOT_COLLECTED_MAP.encode for i in range(7, 11)},

    ('ovary', 'SSF1'): lambda s: _enc.encode_ca125(s, 'pre'),
    ('ovary', 'SSF2'): lambda s: _enc.encode_ca125(s, 'post'),
    ('ovary', 'SSF3'): _OVARY_RESIDUAL_MAP.encode,
    **{('ovary', f'SSF{i}'): _NOT_COLLECTED_MAP.encode for i in range(4, 11)},

    ('bladder', 'SSF1'): _BLADDER_GRADE_MAP.encode,
    ('bladder', 'SSF2'): _BLADDER_ENE_MAP.encode,
    ('bladder', 'SSF3'): _BLADDER_MUSCULARIS_MAP.encode,
    **{('bladder', f'SSF{i}'): _NOT_COLLECTED_MAP.encode for i in range(4, 11)},
    ('head_neck', 'SSF1'):  _enc.encode_hn_node_size,
    ('head_neck', 'SSF2'):  _HN_ECE_MAP.encode,
    ('head_neck', 'SSF3'):  lambda s: _enc.encode_hn_levels(s, 'SSF3'),
    ('head_neck', 'SSF4'):  lambda s: _enc.encode_hn_levels(s, 'SSF4'),
    ('head_neck', 'SSF5'):  lambda s: _enc.encode_hn_levels(s, 'SSF5'),
    ('head_neck', 'SSF6'):  lambda s: _enc.encode_hn_levels(s, 'SSF6'),
    ('head_neck', 'SSF7'):  _enc.encode_hn_tumor_depth,
    ('head_neck', 'SSF8'):  _enc.encode_hn_margin,
    ('head_neck', 'SSF9'):  _enc.encode_hn_ene_clinical,
    ('head_neck', 'SSF10'): _enc.encode_hn_ene_pathological,
    # Fields a site does not collect: the only legal code is 988.
    **{('thyroid', f'SSF{i}'): _NOT_COLLECTED_MAP.encode for i in range(1, 11)},
    **{('cervix', f'SSF{i}'): _NOT_COLLECTED_MAP.encode for i in range(3, 11)},
    **{('stomach', f'SSF{i}'): _NOT_COLLECTED_MAP.encode for i in range(6, 11)},
    **{('liver', f'SSF{i}'): _NOT_COLLECTED_MAP.encode for i in (9, 10)},
    ('lung', 'SSF10'): _NOT_COLLECTED_MAP.encode,

    ('cervix', 'SSF1'): _enc.encode_scc_lab_value,
    ('stomach', 'SSF4'): _enc.encode_tumor_depth,
    ('colorectum', 'SSF3'): _TUMOR_REGRESSION_MAP.encode,
    ('colorectum', 'SSF4'): _enc.encode_crm,
    ('colorectum', 'SSF5'): _BRAF_MAP.encode,
    ('colorectum', 'SSF7'): _OBSTRUCTION_MAP.encode,
    ('colorectum', 'SSF8'): _PERFORATION_MAP.encode,
    ('colorectum', 'SSF9'): _enc.encode_distance_to_anus,


    ('breast', 'SSF1'):  lambda s: _enc.encode_er_pr(s, 'ER'),
    ('breast', 'SSF2'):  lambda s: _enc.encode_er_pr(s, 'PR'),
    ('breast', 'SSF3'):  _enc.encode_ssf3_neoadj,
    ('breast', 'SSF4'):  lambda s: _enc.encode_sentinel(s, kind='examined'),
    ('breast', 'SSF5'):  lambda s: _enc.encode_sentinel(s, kind='positive'),
    ('breast', 'SSF6'):  _enc.encode_nottingham,
    ('breast', 'SSF7'):  _enc.encode_her2,
    ('breast', 'SSF8'):  _PAGET_MAP.encode,
    ('breast', 'SSF9'):  _LVI_BREAST_MAP.encode,
    ('breast', 'SSF10'): _enc.encode_ki67,

    ('lung', 'SSF1'): _LUNG_SSF1_MAP.encode,
    ('lung', 'SSF2'): _LUNG_SSF2_MAP.encode,
    ('lung', 'SSF3'): _LUNG_SSF3_MAP.encode,
    ('lung', 'SSF4'): _LUNG_SSF4_MAP.encode,
    ('lung', 'SSF5'): _enc.encode_lung_ssf5_mediastinal,
    ('lung', 'SSF6'): _enc.encode_lung_egfr,
    ('lung', 'SSF7'): _LUNG_ALK_MAP.encode,
    ('lung', 'SSF8'): _LUNG_SSF8_MAP.encode,
    ('lung', 'SSF9'): _enc.encode_lung_ssf9_nodules,

    ('colorectum', 'SSF1'):  _enc.encode_cea_lab_value,
    ('colorectum', 'SSF2'):  _CEA_NORMAL_MAP.encode,
    ('colorectum', 'SSF6'):  _enc.encode_ras_mutation,
    ('colorectum', 'SSF10'): _MSI_CRC_MAP.encode,

    ('liver', 'SSF1'): _enc.encode_liver_afp,
    ('liver', 'SSF2'): _LIVER_FIBROSIS_MAP.encode,
    ('liver', 'SSF3'): _CHILD_PUGH_MAP.encode,
    ('liver', 'SSF4'): lambda s: _enc.encode_lab_value_10x(s, 'Creatinine', 'mg/dL'),
    ('liver', 'SSF5'): lambda s: _enc.encode_lab_value_10x(s, 'Total bilirubin', 'mg/dL'),
    ('liver', 'SSF6'): _enc.encode_liver_inr,
    ('liver', 'SSF7'): _HBSAG_MAP.encode,
    ('liver', 'SSF8'): _ANTI_HCV_MAP.encode,

    ('cervix', 'SSF2'): _SCC_ANTIGEN_NORMAL_MAP.encode,

    ('stomach', 'SSF1'): _enc.encode_cea_lab_value,
    ('stomach', 'SSF2'): _CEA_NORMAL_MAP.encode,
    ('stomach', 'SSF3'): _H_PYLORI_MAP.encode,
    ('stomach', 'SSF5'): _STOMACH_LVI_MAP.encode,


    ('prostate', 'SSF1'): _enc.encode_psa,
    ('prostate', 'SSF2'): lambda s: _enc.encode_gleason_patterns(s, 'biopsy'),
    ('prostate', 'SSF3'): lambda s: _enc.encode_gleason_score(s, 'biopsy'),
    ('prostate', 'SSF4'): lambda s: _enc.encode_gleason_patterns(s, 'prostatectomy'),
    ('prostate', 'SSF5'): lambda s: _enc.encode_gleason_score(s, 'prostatectomy'),
    ('prostate', 'SSF6'): lambda s: _enc.encode_biopsy_cores(s, 'examined'),
    ('prostate', 'SSF7'): lambda s: _enc.encode_biopsy_cores(s, 'positive'),
    ('prostate', 'SSF8'): _PROSTATE_T_STAGING_MAP.encode,

    ('endometrium', 'SSF1'):  lambda s: _enc.encode_er_pr(s, 'ER'),
    ('endometrium', 'SSF2'):  lambda s: _enc.encode_er_pr(s, 'PR'),
    ('endometrium', 'SSF3'):  _UTERUS_FIGO_GRADE_MAP.encode,
    ('endometrium', 'SSF4'):  _POLE_MAP.encode,
    ('endometrium', 'SSF5'):  _MSI_UTERUS_MAP.encode,
    ('endometrium', 'SSF6'):  _P53_MAP.encode,
    ('endometrium', 'SSF7'):  _NOT_COLLECTED_MAP.encode,
    ('endometrium', 'SSF8'):  _NOT_COLLECTED_MAP.encode,
    ('endometrium', 'SSF9'):  _NOT_COLLECTED_MAP.encode,
    ('endometrium', 'SSF10'): _NOT_COLLECTED_MAP.encode,
}

for (_grp, _ssf_key), _encoder_fn in _ENCODER_WIRING.items():
    _PROFILES[_grp].fields[_ssf_key].encoder = _encoder_fn


# Reverse lookup: ICD-O-3 prefix → cancer_group
_CODE_TO_GROUP: Dict[str, str] = {}
for _group, _profile in _PROFILES.items():
    if _group == 'generic':
        continue
    for _code in _profile.site_codes:
        _CODE_TO_GROUP[_code.upper()] = _group


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_cancer_group(tcode1: str) -> str:
    """Detect the cancer group from an ICD-O-3 topography code.

    Args:
        tcode1: ICD-O-3 site code, e.g. 'C50.1', 'C34.1', 'C220'

    Returns:
        Cancer group string ('breast', 'lung', 'colorectum', etc.)
        Falls back to 'generic' if code is unknown.

    Examples:
        >>> detect_cancer_group('C50.1')
        'breast'
        >>> detect_cancer_group('C34.0')
        'lung'
        >>> detect_cancer_group('C18.2')
        'colorectum'
    """
    if not tcode1 or pd.isna(tcode1):
        return 'generic'
    code = str(tcode1).strip().upper()
    # Try exact prefix match (C50, C34, etc.)
    prefix = re.match(r'(C\d+)', code)
    if prefix:
        p = prefix.group(1)
        # Try the 4-character form FIRST so a single sub-site can be
        # registered on its own: C76.0 (cervical nodes with an unknown
        # primary) is head & neck, while C76.1-C76.8 are other ill-defined
        # sites with no SSFs at all. The dot is stripped above, so 'C76.0'
        # and 'C760' both arrive here as 'C760'.
        digits = re.sub(r'[^0-9]', '', code)
        c4 = f'C{digits[:3]}' if len(digits) >= 3 else ''
        if c4 in _CODE_TO_GROUP:
            return _CODE_TO_GROUP[c4]
        c3 = p[:3]
        if c3 in _CODE_TO_GROUP:
            return _CODE_TO_GROUP[c3]
    return 'generic'


def detect_cancer_group_from_series(tcode1_series: pd.Series) -> str:
    """Detect the cancer group from a series of ICD-O-3 codes.

    Uses the most common (mode) cancer group across all patients.
    If all patients are the same cancer type (expected for single-cancer registry),
    returns that group. Mixed registries return the dominant group.

    Args:
        tcode1_series: Series of TCODE1 values

    Returns:
        Cancer group string
    """
    groups = tcode1_series.dropna().apply(detect_cancer_group)
    if len(groups) == 0:
        return 'generic'
    mode_group = groups.mode().iloc[0]
    n_mode = (groups == mode_group).sum()
    n_total = len(groups)
    pct = n_mode / n_total * 100

    if pct < 90:
        # Mixed registry warning
        import warnings
        others = groups[groups != mode_group].value_counts().head(3).to_dict()
        warnings.warn(
            f"Mixed cancer registry detected: dominant group '{mode_group}' "
            f"({n_mode}/{n_total}, {pct:.0f}%). Other groups: {others}. "
            f"SSF decoding will use '{mode_group}' profile for ALL patients. "
            f"For mixed registries, decode each cancer group separately.",
            UserWarning, stacklevel=2,
        )
    return mode_group


def get_ssf_profile(cancer_group: str) -> SSFProfile:
    """Get the SSF profile for a cancer group.

    Args:
        cancer_group: Cancer group string (e.g., 'breast', 'lung')

    Returns:
        SSFProfile object with field definitions and decoders

    Raises:
        KeyError: If cancer_group is not in the registry (use 'generic' as fallback)
    """
    return _PROFILES.get(cancer_group, _PROFILES['generic'])


def list_supported_cancers() -> pd.DataFrame:
    """List all supported cancer groups with their site codes.

    Returns:
        DataFrame with columns: Cancer_Group, Site_Label, ICD_O_3_Codes, Notes
    """
    rows = []
    for group, profile in _PROFILES.items():
        rows.append({
            'Cancer_Group': group,
            'Site_Label': profile.site_label,
            'ICD_O_3_Codes': ', '.join(profile.site_codes) if profile.site_codes else '(any)',
            'Custom_Decoders': sum(1 for f in profile.fields.values() if f.decoder is not None),
            'Total_SSF_Fields': len(profile.fields),
            'Notes': profile.notes[:80] + '...' if len(profile.notes) > 80 else profile.notes,
        })
    return pd.DataFrame(rows)


def apply_ssf_profile(df: pd.DataFrame, cancer_group: str) -> pd.DataFrame:
    """Apply the appropriate SSF decoders to a DataFrame.

    This is the main entry point called by core.py during the decode() step.
    It replaces the breast-only SSF decode logic with cancer-aware routing.

    Args:
        df: DataFrame with SSF1_raw … SSF10_raw columns
        cancer_group: Cancer group string from detect_cancer_group()

    Returns:
        DataFrame with decoded SSF columns added (named per SSFFieldDef.column_name)
    """
    df = df.copy()
    profile = get_ssf_profile(cancer_group)

    for ssf_key, field_def in profile.fields.items():
        raw_col = f'{ssf_key}_raw'
        if raw_col not in df.columns:
            continue

        raw_series = df[raw_col].astype(str).replace('nan', '')

        if field_def.decoder is not None:
            # Use custom decoder
            decoded = field_def.decoder(df[raw_col])
        else:
            # Use generic numeric decoder
            decoded = _generic_ssf(df[raw_col], field_name=ssf_key, unit=field_def.unit)

        df[field_def.column_name] = decoded

    return df


def apply_ssf_encode_profile(
    df: pd.DataFrame, cancer_group: str, on_error: str = 'raise',
) -> pd.DataFrame:
    """Apply the appropriate SSF encoders to a DataFrame -- the inverse of
    apply_ssf_profile().

    Args:
        df: DataFrame with decoded SSF columns (named per SSFFieldDef.column_name,
            e.g. 'ER_Status', 'HER2_Status', ...) -- i.e. a TCRDecoder.clean-shaped
            DataFrame for this cancer_group.
        cancer_group: Cancer group string from detect_cancer_group()
        on_error: 'raise' (default) to fail loudly on the first unrecognized
            label (a typo or a label this tool never produced), or 'empty'
            to blank out just that cell and keep going -- used by
            best-effort batch tooling like compare_roundtrip.

    Returns:
        DataFrame with raw SSF1_raw … SSF10_raw code columns added.
    """
    df = df.copy()
    profile = get_ssf_profile(cancer_group)

    for ssf_key, field_def in profile.fields.items():
        col = field_def.column_name
        if col not in df.columns:
            continue

        series = df[col].astype(str).replace('nan', '')
        if field_def.encoder is not None:
            encoded = _enc.batch_encode(field_def.encoder, series, on_error=on_error)
        else:
            encoded = _enc.batch_encode(
                lambda s: _enc.encode_generic_ssf(s, unit=field_def.unit), series, on_error=on_error)

        df[f'{ssf_key}_raw'] = encoded

    return df


def get_ssf_column_names(cancer_group: str) -> Dict[str, str]:
    """Get mapping of SSF field → output column name for a cancer group.

    Useful for updating COLUMN_REGISTRY in data_dictionary.py.

    Returns:
        Dict like {'SSF1': 'ER_Status', 'SSF2': 'PR_Status', ...}
    """
    profile = get_ssf_profile(cancer_group)
    return {k: v.column_name for k, v in profile.fields.items()}


# ─────────────────────────────────────────────────────────────────────────────
# CLI: python -m tcr_decoder.ssf_registry
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print('TCR Decoder — Supported Cancer Groups\n')
    df = list_supported_cancers()
    with pd.option_context('display.max_colwidth', 60, 'display.width', 120):
        print(df.to_string(index=False))

    print('\n\nSSF Field Definitions by Cancer Group:')
    print('=' * 80)
    for group in ['breast', 'lung', 'colorectum', 'liver', 'prostate', 'generic']:
        profile = get_ssf_profile(group)
        print(f'\n── {profile.site_label} ({", ".join(profile.site_codes) or "fallback"}) ──')
        for ssf, fdef in profile.fields.items():
            decoder_tag = '✓ custom' if fdef.decoder else '○ generic'
            print(f'  {ssf:5s}  {fdef.column_name:35s}  {decoder_tag}  {fdef.description[:50]}')
