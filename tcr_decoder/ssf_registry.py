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


def _decode_paget(series: pd.Series) -> pd.Series:
    """SSF8 for breast: Paget disease of the nipple."""
    MAP = {
        0:   'No Paget disease',
        10:  'Paget disease present',
        888: 'Not applicable (conversion)',
        988: 'Not applicable (specimen excludes nipple/areola)',
        999: 'Unknown / not documented',
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Code {iv}')
    return series.apply(_d)


def _decode_lvi(series: pd.Series) -> pd.Series:
    """SSF9 for breast: Lymphovascular invasion (LVI)."""
    MAP = {
        0:   'No lymphovascular invasion',
        10:  'Lymphovascular invasion present',
        888: 'Not applicable (conversion)',
        988: 'Not applicable',
        990: 'No residual tumor (LVI not assessable after neoadjuvant therapy)',
        999: 'Unknown / not documented',
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Code {iv}')
    return series.apply(_d)


# ─────────────────────────────────────────────────────────────────────────────
# Lung-cancer SSF decoders
# ─────────────────────────────────────────────────────────────────────────────

def _decode_lung_ssf1_nodules(series: pd.Series) -> pd.Series:
    """SSF1 for lung: Separate tumor nodules / ipsilateral lung."""
    MAP = {
        0:   'No separate ipsilateral tumor nodules; in situ',
        10:  'Separate nodule(s) — ipsilateral same lobe',
        20:  'Separate nodule(s) — ipsilateral different lobe',
        30:  'Separate nodule(s) — both same and different lobe (ipsilateral)',
        40:  'Separate nodule(s) — ipsilateral, lobe unknown',
        999: 'Unknown / not documented',
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Code {iv}')
    return series.apply(_d)


def _decode_lung_ssf2_vpi(series: pd.Series) -> pd.Series:
    """SSF2 for lung: Visceral pleural invasion (PL0-PL3)."""
    MAP = {
        0:   'PL0 — No visceral pleural invasion (elastic layer not reached)',
        10:  'PL1 — Invasion to elastic layer of visceral pleura',
        20:  'PL2 — Invasion to surface of visceral pleura',
        30:  'PL3 — Invasion through to parietal pleura',
        40:  'Pleural invasion present; PL level not specified',
        988: 'Not applicable (no surgery to primary site)',
        999: 'Unknown / not documented',
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Code {iv}')
    return series.apply(_d)


def _decode_lung_ssf3_ecog(series: pd.Series) -> pd.Series:
    """SSF3 for lung: Performance status (ECOG/KPS) before treatment."""
    MAP = {
        0:   'ECOG 0 — Fully active (KPS 100)',
        1:   'ECOG 1 — Light work only (KPS 80-90)',
        2:   'ECOG 2 — Self-care, up >50% of day (KPS 60-70)',
        3:   'ECOG 3 — Limited self-care, confined >50% of day (KPS 40-50)',
        4:   'ECOG 4 — Completely disabled (KPS 10-30)',
        5:   'ECOG 5 — Death (KPS 0)',
        988: 'Not applicable',
        998: 'Not assessed',
        999: 'Unknown / not documented',
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Code {iv}')
    return series.apply(_d)


def _decode_lung_ssf4_pleural_effusion(series: pd.Series) -> pd.Series:
    """SSF4 for lung: Malignant pleural effusion."""
    MAP = {
        0:   'No malignant pleural effusion (imaging/cytology negative; or non-malignant cause confirmed)',
        11:  'Imaging: effusion present; no cytology; physician considers malignant',
        12:  'Imaging: effusion present; cytology negative/atypical; physician considers malignant',
        13:  'Cytology confirmed malignant pleural effusion',
        14:  'Imaging: effusion present; no cytology; physician does NOT consider malignant',
        15:  'Imaging: effusion present; cytology negative/atypical; physician does NOT consider malignant',
        988: 'Not applicable — M0 case',
        999: 'Unknown / not documented',
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Code {iv}')
    return series.apply(_d)


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


def _decode_lung_egfr(series: pd.Series) -> pd.Series:
    """SSF6 for lung: EGFR gene mutation (3-character alphabetic code).

    The TCR encodes up to 3 concurrent EGFR mutations as a 3-letter string.
    Each letter position represents one mutation:
      A=Exon19del  B=L858R  C=E709  D=G719X  E=Exon20ins
      F=S768I  G=T790M  H=L861  U=other  V=mutated(NOS)
      X=no mutation  Z=uninterpretable
    Sentinels: 999 = unknown / not tested
    """
    LETTER_MAP = {
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
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        s = str(val).strip().upper()
        if s in ('999', ''):
            return 'Unknown / not tested'
        if len(s) == 3:
            mutations = [LETTER_MAP.get(c, f'?({c})') for c in s if c != 'X']
            if not mutations:
                return 'EGFR — No mutation (XXX)'
            return 'EGFR — ' + ' + '.join(mutations)
        # Single letter or numeric fallback
        if s in LETTER_MAP:
            return 'EGFR — ' + LETTER_MAP[s]
        return f'EGFR code: {val}'
    return series.apply(_d)


def _decode_lung_alk(series: pd.Series) -> pd.Series:
    """SSF7 for lung: ALK gene translocation."""
    MAP = {
        10:  'ALK positive — rearrangement/translocation present',
        20:  'ALK negative — no rearrangement',
        30:  'ALK test performed; result uninterpretable',
        999: 'Unknown / not tested / no ALK test ordered',
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Code {iv}')
    return series.apply(_d)


def _decode_lung_ssf8_adeno(series: pd.Series) -> pd.Series:
    """SSF8 for lung: Specific lung adenocarcinoma component (micropapillary/solid/cribriform).

    Codes are additive bitmask: micropapillary=1, solid=2, cribriform=4.
    """
    MAP = {
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
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Code {iv}')
    return series.apply(_d)


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


def _decode_lung_ros1(series: pd.Series) -> pd.Series:
    """ROS1 rearrangement (kept for reference; not in current TCR lung codebook)."""
    MAP = {
        0:   'ROS1 negative',
        1:   'ROS1 positive; rearrangement',
        9:   'ROS1 equivocal',
        988: 'Not applicable',
        999: 'Unknown; not tested',
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Code {iv}')
    return series.apply(_d)


def _decode_pdl1(series: pd.Series) -> pd.Series:
    """PD-L1 expression (kept for reference; not in current TCR lung codebook)."""
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        if iv in (988, 998):
            return 'Not applicable'
        if iv == 999:
            return 'Unknown; not tested'
        if iv == 0:
            return 'PD-L1 negative (<1%)'
        if 1 <= iv <= 100:
            return f'PD-L1 positive ({iv}%)'
        return f'Code {iv}'
    return series.apply(_d)


# ─────────────────────────────────────────────────────────────────────────────
# Colorectal SSF decoders
# ─────────────────────────────────────────────────────────────────────────────

def _decode_cea(series: pd.Series) -> pd.Series:
    """CEA (carcinoembryonic antigen) in ng/mL."""
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        if iv == 0:
            return 'CEA within normal limits (≤3.0 ng/mL)'
        if iv == 988:
            return 'Not applicable'
        if iv == 997:
            return 'Test ordered, results not in chart'
        if iv == 999:
            return 'Unknown; not documented'
        if 1 <= iv <= 980:
            return f'CEA {iv} ng/mL (elevated)'
        return f'Code {iv}'
    return series.apply(_d)


def _decode_msi(series: pd.Series) -> pd.Series:
    """Microsatellite instability (MSI) status."""
    MAP = {
        0:   'MSS — Microsatellite stable',
        1:   'MSI-L — Low instability',
        2:   'MSI-H — High instability',
        8:   'Not applicable (Lynch syndrome excluded by other means)',
        9:   'MSI equivocal / inconclusive',
        988: 'Not applicable',
        999: 'Unknown; not tested',
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Code {iv}')
    return series.apply(_d)


def _decode_kras(series: pd.Series) -> pd.Series:
    """KRAS mutation status."""
    MAP = {
        0:   'KRAS wild-type (no mutation)',
        1:   'KRAS mutated; codon 12',
        2:   'KRAS mutated; codon 13',
        3:   'KRAS mutated; codon 12 and codon 13',
        4:   'KRAS mutated; NOS (codon not specified)',
        9:   'KRAS equivocal',
        988: 'Not applicable',
        999: 'Unknown; not tested',
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Code {iv}')
    return series.apply(_d)


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
        # Alphabetic A-codes (2021+ only): A00-A99 = 1-99 ng/mL
        if len(s) == 3 and s[0] == 'A' and s[1:].isdigit():
            num = int(s[1:])
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


def _decode_liver_fibrosis(series: pd.Series) -> pd.Series:
    """SSF2 for liver: Liver fibrosis grade (Ishak score)."""
    MAP = {
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
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Code {iv}')
    return series.apply(_d)


def _decode_child_pugh(series: pd.Series) -> pd.Series:
    """SSF3 for liver: Child-Pugh class and score.

    3-char code: first digit = class (1=A, 2=B, 3=C),
    next two = score (05-15) or 99 for class only.
    """
    MAP = {
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
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Child-Pugh code {iv}')
    return series.apply(_d)


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


def _decode_hbsag(series: pd.Series) -> pd.Series:
    """SSF7 for liver: HBsAg (hepatitis B surface antigen) with history."""
    MAP = {
        0:   'Not tested; no HBV carrier history',
        1:   'Not tested; HBV carrier history documented',
        10:  'Negative; no HBV carrier history',
        11:  'Negative; HBV carrier history documented',
        20:  'Positive',
        999: 'Unknown / not documented',
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Code {iv}')
    return series.apply(_d)


def _decode_anti_hcv(series: pd.Series) -> pd.Series:
    """SSF8 for liver: Anti-HCV (hepatitis C antibody/antigen/RNA) with history."""
    MAP = {
        0:   'Not tested; no HCV infection history',
        1:   'Not tested; HCV infection history documented',
        10:  'Negative; no HCV infection history',
        11:  'Negative; HCV infection history (treated / SVR)',
        20:  'Positive (Anti-HCV positive and/or HCV RNA positive)',
        999: 'Unknown / not documented',
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Code {iv}')
    return series.apply(_d)


def _decode_hbv_hcv(series: pd.Series, virus: str = 'HBV') -> pd.Series:
    """HBV/HCV infection status (legacy decoder, kept for reference)."""
    MAP = {
        0:   f'{virus} negative; no evidence of infection',
        1:   f'{virus} positive; active infection',
        2:   f'{virus} positive; resolved/carrier state',
        9:   f'{virus} equivocal',
        988: 'Not applicable',
        999: 'Unknown; not tested',
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Code {iv}')
    return series.apply(_d)


# ─────────────────────────────────────────────────────────────────────────────
# Prostate SSF decoders
# ─────────────────────────────────────────────────────────────────────────────

def _decode_psa(series: pd.Series) -> pd.Series:
    """PSA (prostate-specific antigen) in ng/mL."""
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        if iv == 0:
            return 'PSA <0.1 ng/mL (undetectable)'
        if iv == 988:
            return 'Not applicable'
        if iv == 999:
            return 'Unknown; not documented'
        if 1 <= iv <= 980:
            return f'PSA {iv/10:.1f} ng/mL'   # stored as ×10
        return f'Code {iv}'
    return series.apply(_d)


def _decode_gleason(series: pd.Series) -> pd.Series:
    """Gleason score for prostate cancer (sum of primary + secondary)."""
    MAP = {
        2:  'Gleason Score 2 (1+1) — Grade Group 1',
        3:  'Gleason Score 3 (1+2 or 2+1) — Grade Group 1',
        4:  'Gleason Score 4 (2+2) — Grade Group 1',
        5:  'Gleason Score 5 — Grade Group 1',
        6:  'Gleason Score 6 (3+3) — Grade Group 1',
        7:  'Gleason Score 7 (3+4 or 4+3) — Grade Group 2 or 3',
        8:  'Gleason Score 8 (4+4, 3+5, or 5+3) — Grade Group 4',
        9:  'Gleason Score 9 (4+5, 5+4) — Grade Group 5',
        10: 'Gleason Score 10 (5+5) — Grade Group 5',
        88: 'Not applicable',
        99: 'Unknown; not documented',
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Gleason Score {iv}')
    return series.apply(_d)


# ─────────────────────────────────────────────────────────────────────────────
# Thyroid SSF decoders
# ─────────────────────────────────────────────────────────────────────────────

def _decode_thyroid_focality(series: pd.Series) -> pd.Series:
    """Tumor focality for thyroid cancer."""
    MAP = {
        0:   'Unifocal tumor',
        1:   'Multifocal tumor; ipsilateral lobe only',
        2:   'Multifocal tumor; bilateral lobes',
        3:   'Multifocal tumor; isthmus involved',
        8:   'Not applicable (total thyroidectomy not performed)',
        9:   'Unknown; not documented',
        988: 'Not applicable',
        999: 'Unknown',
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Code {iv}')
    return series.apply(_d)


def _decode_extrathyroidal(series: pd.Series) -> pd.Series:
    """Extrathyroidal extension for thyroid cancer."""
    MAP = {
        0:   'No extrathyroidal extension',
        1:   'Minimal/microscopic extrathyroidal extension (T3b)',
        2:   'Gross extrathyroidal extension — strap muscles (T4a)',
        3:   'Gross extrathyroidal extension — major structures (T4b)',
        8:   'Not applicable',
        9:   'Unknown',
        988: 'Not applicable',
        999: 'Unknown',
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Code {iv}')
    return series.apply(_d)


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


def _decode_cea_normal(series: pd.Series) -> pd.Series:
    """CEA vs. normal range (SSF2 for stomach/colorectum/pancreas)."""
    MAP = {
        10:  'CEA positive — above normal range',
        20:  'CEA negative — within normal range',
        30:  'CEA borderline — uncertain positive/negative',
        988: 'Not applicable',
        999: 'Unknown / CEA not tested',
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Code {iv}')
    return series.apply(_d)


def _decode_h_pylori(series: pd.Series) -> pd.Series:
    """SSF3 for stomach: H. pylori infection status and detection method."""
    MAP = {
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
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Code {iv}')
    return series.apply(_d)


def _decode_stomach_lvi(series: pd.Series) -> pd.Series:
    """SSF5 for stomach: Lymphovascular invasion (LVI)."""
    MAP = {
        0:   'No lymphovascular invasion',
        10:  'Lymphovascular invasion present',
        988: 'Not applicable',
        990: 'No residual tumor (LVI not assessable after neoadjuvant)',
        999: 'Unknown / not documented',
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Code {iv}')
    return series.apply(_d)


def _decode_ras_mutation(series: pd.Series) -> pd.Series:
    """SSF6 for colorectum: RAS (KRAS + NRAS) combined 3-character code.

    Position 1 = KRAS: 0=WT, 1=Codon12, 2=Codon13, 3=Codon61, 4=multi-codon,
                        5=non-12/13/61, 6=mutated NOS, 7=uninterpretable, 9=not tested
    Position 2 = NRAS: same scheme as KRAS
    Position 3 = always '8' (filler)
    Special: 988=N/A(GIST/NETs/no external data), 998=not documented/not tested
    """
    KRAS_NRAS = {
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


def _decode_msi_crc(series: pd.Series) -> pd.Series:
    """SSF10 for colorectum: MSI/MMR status."""
    MAP = {
        0:   'MSS / Microsatellite stable; MMR proficient (pMMR)',
        10:  'MSI-L — Low instability',
        20:  'MSI-H — High instability; or MMR deficient (dMMR)',
        988: 'Not applicable (GIST/NETs/high-grade dysplasia or no external data)',
        999: 'Unknown / not tested; MSI indeterminate/equivocal',
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Code {iv}')
    return series.apply(_d)


def _decode_scc_antigen_normal(series: pd.Series) -> pd.Series:
    """SSF2 for cervix: SCC antigen vs. normal range."""
    MAP = {
        10:  'SCC antigen positive — above normal range',
        20:  'SCC antigen negative — within normal range',
        30:  'SCC antigen borderline',
        988: 'Not applicable',
        999: 'Unknown / not tested',
    }
    def _d(val):
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return MAP.get(iv, f'Code {iv}')
    return series.apply(_d)


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
                                 decoder=None),
            'SSF9':  SSFFieldDef('SSF9', 'LVI_SSF', 'Lymphovascular invasion (SSF source)',
                                 decoder=None),
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
            'SSF3':  SSFFieldDef('SSF3', 'Performance_Status',
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
                                 decoder=None),
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
                                 decoder=None),
            'SSF4':  SSFFieldDef('SSF4', 'Circumferential_Resection_Margin',
                                 'Circumferential resection margin',
                                 decoder=None),
            'SSF5':  SSFFieldDef('SSF5', 'BRAF_Mutation',
                                 'BRAF mutation status',
                                 decoder=None),
            'SSF6':  SSFFieldDef('SSF6', 'RAS_Mutation',
                                 'RAS (KRAS+NRAS) combined mutation code (3-char)',
                                 decoder=_decode_ras_mutation),
            'SSF7':  SSFFieldDef('SSF7', 'Intestinal_Obstruction',
                                 'Intestinal obstruction',
                                 decoder=None),
            'SSF8':  SSFFieldDef('SSF8', 'Intestinal_Perforation',
                                 'Intestinal perforation',
                                 decoder=None),
            'SSF9':  SSFFieldDef('SSF9', 'Distance_to_Anus',
                                 'Distance to anus (rectum/rectosigmoid C19/C20 only; colon C18/C21 = 988)',
                                 decoder=None, unit='mm'),
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
                                 decoder=None),
            'SSF10': SSFFieldDef('SSF10', 'SSF10_Liver',
                                 'SSF10 — Not defined in TCR codebook for liver (988 for all)',
                                 decoder=None),
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
                                 decoder=None),
            'SSF2':  SSFFieldDef('SSF2', 'SCC_Antigen_vs_Normal',
                                 'SCC antigen vs. normal range',
                                 decoder=_decode_scc_antigen_normal),
            'SSF3':  SSFFieldDef('SSF3', 'SSF3_Cervix',
                                 'SSF3 — Not defined in TCR codebook for cervix (988 for all)',
                                 decoder=None),
            'SSF4':  SSFFieldDef('SSF4', 'SSF4_Cervix',
                                 'SSF4 — Not defined in TCR codebook for cervix (988 for all)',
                                 decoder=None),
            'SSF5':  SSFFieldDef('SSF5', 'SSF5_Cervix',
                                 'SSF5 — Not defined in TCR codebook for cervix (988 for all)',
                                 decoder=None),
            'SSF6':  SSFFieldDef('SSF6', 'SSF6_Cervix',
                                 'SSF6 — Not defined in TCR codebook for cervix (988 for all)',
                                 decoder=None),
            'SSF7':  SSFFieldDef('SSF7', 'SSF7_Cervix',
                                 'SSF7 — Not defined in TCR codebook for cervix (988 for all)',
                                 decoder=None),
            'SSF8':  SSFFieldDef('SSF8', 'SSF8_Cervix',
                                 'SSF8 — Not defined in TCR codebook for cervix (988 for all)',
                                 decoder=None),
            'SSF9':  SSFFieldDef('SSF9', 'SSF9_Cervix',
                                 'SSF9 — Not defined in TCR codebook for cervix (988 for all)',
                                 decoder=None),
            'SSF10': SSFFieldDef('SSF10', 'SSF10_Cervix',
                                 'SSF10 — Not defined in TCR codebook for cervix (988 for all)',
                                 decoder=None),
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
                                 decoder=None, unit='0.1mm'),
            'SSF5':  SSFFieldDef('SSF5', 'LVI_Stomach',
                                 'Lymphovascular invasion (LVI)',
                                 decoder=_decode_stomach_lvi),
            'SSF6':  SSFFieldDef('SSF6', 'SSF6_Stomach',
                                 'SSF6 — Not defined in TCR codebook for stomach (988 for all)',
                                 decoder=None),
            'SSF7':  SSFFieldDef('SSF7', 'SSF7_Stomach',
                                 'SSF7 — Not defined in TCR codebook for stomach (988 for all)',
                                 decoder=None),
            'SSF8':  SSFFieldDef('SSF8', 'SSF8_Stomach',
                                 'SSF8 — Not defined in TCR codebook for stomach (988 for all)',
                                 decoder=None),
            'SSF9':  SSFFieldDef('SSF9', 'SSF9_Stomach',
                                 'SSF9 — Not defined in TCR codebook for stomach (988 for all)',
                                 decoder=None),
            'SSF10': SSFFieldDef('SSF10', 'SSF10_Stomach',
                                 'SSF10 — Not defined in TCR codebook for stomach (988 for all)',
                                 decoder=None),
        },
        notes='Per 2025 TCR codebook: SSF1=CEA value, SSF2=CEA vs normal, SSF3=H.pylori method, SSF4=tumor depth (0.1mm), SSF5=LVI. SSF6-10 not defined (all 988).',
    )

    # ── THYROID ───────────────────────────────────────────────────────────────
    profiles['thyroid'] = SSFProfile(
        cancer_group='thyroid',
        site_label='Thyroid Cancer',
        site_codes=('C73',),
        fields={
            'SSF1':  SSFFieldDef('SSF1', 'Tumor_Focality',
                                 'Tumor focality (unifocal vs multifocal)',
                                 decoder=_decode_thyroid_focality),
            'SSF2':  SSFFieldDef('SSF2', 'Vascular_Invasion_Thyroid',
                                 'Vascular invasion',
                                 decoder=None),
            'SSF3':  SSFFieldDef('SSF3', 'Extrathyroidal_Extension',
                                 'Extrathyroidal extension',
                                 decoder=_decode_extrathyroidal),
            'SSF4':  SSFFieldDef('SSF4', 'Thyroid_Capsule_Invasion',
                                 'Thyroid capsule invasion',
                                 decoder=None),
            'SSF5':  SSFFieldDef('SSF5', 'Completeness_of_Resection',
                                 'Completeness of thyroid resection',
                                 decoder=None),
            'SSF6':  SSFFieldDef('SSF6', 'BRAF_Thyroid',
                                 'BRAF V600E mutation',
                                 decoder=None),
            'SSF7':  SSFFieldDef('SSF7', 'RAS_Mutation',
                                 'RAS mutation (NRAS/HRAS/KRAS)',
                                 decoder=None),
            'SSF8':  SSFFieldDef('SSF8', 'Postop_Thyroglobulin',
                                 'Post-operative thyroglobulin level',
                                 decoder=None, unit='ng/mL'),
            'SSF9':  SSFFieldDef('SSF9', 'RAI_Response',
                                 'Response to radioactive iodine therapy',
                                 decoder=None),
            'SSF10': SSFFieldDef('SSF10', 'Histologic_Variant',
                                 'Histologic variant (classical/follicular/tall-cell)',
                                 decoder=None),
        },
        notes='Extrathyroidal extension drives T3b/T4a upstaging. BRAF mutation affects prognosis.',
    )

    # ── PROSTATE ──────────────────────────────────────────────────────────────
    profiles['prostate'] = SSFProfile(
        cancer_group='prostate',
        site_label='Prostate Cancer',
        site_codes=('C61',),
        fields={
            'SSF1':  SSFFieldDef('SSF1', 'PSA_Preop',
                                 'Preoperative PSA (×10, e.g., 45 = 4.5 ng/mL)',
                                 decoder=_decode_psa, unit='ng/mL'),
            'SSF2':  SSFFieldDef('SSF2', 'Gleason_Score',
                                 'Gleason score (primary + secondary pattern)',
                                 decoder=_decode_gleason),
            'SSF3':  SSFFieldDef('SSF3', 'Number_Positive_Cores',
                                 'Number of positive biopsy cores',
                                 decoder=None),
            'SSF4':  SSFFieldDef('SSF4', 'Number_Cores_Examined',
                                 'Total biopsy cores examined',
                                 decoder=None),
            'SSF5':  SSFFieldDef('SSF5', 'Prostate_Surgical_Margins',
                                 'Surgical margins (radical prostatectomy)',
                                 decoder=None),
            'SSF6':  SSFFieldDef('SSF6', 'Seminal_Vesicle_Invasion',
                                 'Seminal vesicle invasion',
                                 decoder=None),
            'SSF7':  SSFFieldDef('SSF7', 'Perineural_Invasion_Prostate',
                                 'Perineural invasion on biopsy',
                                 decoder=None),
            'SSF8':  SSFFieldDef('SSF8', 'Extraprostatic_Extension',
                                 'Extraprostatic extension (focal vs established)',
                                 decoder=None),
            'SSF9':  SSFFieldDef('SSF9', 'LVI_Prostate',
                                 'Lymphovascular invasion',
                                 decoder=None),
            'SSF10': SSFFieldDef('SSF10', 'Grade_Group',
                                 'ISUP/WHO 2016 Grade Group (1-5)',
                                 decoder=None),
        },
        notes='PSA stored as ×10 integer. Gleason + PSA + T-stage = NCCN risk stratification.',
    )

    # ── NASOPHARYNX ───────────────────────────────────────────────────────────
    profiles['nasopharynx'] = SSFProfile(
        cancer_group='nasopharynx',
        site_label='Nasopharyngeal Cancer',
        site_codes=('C11',),
        fields={
            'SSF1':  SSFFieldDef('SSF1', 'EBV_VCA_IgA',
                                 'EBV VCA IgA titer',
                                 decoder=None),
            'SSF2':  SSFFieldDef('SSF2', 'EBV_EA_IgA',
                                 'EBV EA IgA titer',
                                 decoder=None),
            'SSF3':  SSFFieldDef('SSF3', 'Plasma_EBV_DNA',
                                 'Plasma EBV DNA (copies/mL)',
                                 decoder=None, unit='copies/mL'),
            'SSF4':  SSFFieldDef('SSF4', 'Cranial_Nerve_Palsy',
                                 'Cranial nerve involvement',
                                 decoder=None),
            'SSF5':  SSFFieldDef('SSF5', 'Parapharyngeal_Extension',
                                 'Parapharyngeal extension',
                                 decoder=None),
            'SSF6':  SSFFieldDef('SSF6', 'Skull_Base_Extension',
                                 'Skull base or intracranial extension',
                                 decoder=None),
            'SSF7':  SSFFieldDef('SSF7', 'Orbital_Extension',
                                 'Orbital or infratemporal fossa extension',
                                 decoder=None),
            'SSF8':  SSFFieldDef('SSF8', 'Node_Level',
                                 'Level of regional lymph node involvement',
                                 decoder=None),
            'SSF9':  SSFFieldDef('SSF9', 'WHO_Histology_Type',
                                 'WHO histologic type (I/II/III)',
                                 decoder=None),
            'SSF10': SSFFieldDef('SSF10', 'NPC_Tumor_Response',
                                 'Tumor response to concurrent chemoradiation',
                                 decoder=None),
        },
        notes='EBV serology drives screening and monitoring in NPC. Plasma EBV DNA is prognostic.',
    )

    # ── ENDOMETRIUM ───────────────────────────────────────────────────────────
    profiles['endometrium'] = SSFProfile(
        cancer_group='endometrium',
        site_label='Endometrial Cancer',
        site_codes=('C54',),
        fields={
            'SSF1':  SSFFieldDef('SSF1', 'ER_Endometrium',
                                 'Estrogen receptor status',
                                 decoder=None),
            'SSF2':  SSFFieldDef('SSF2', 'PR_Endometrium',
                                 'Progesterone receptor status',
                                 decoder=None),
            'SSF3':  SSFFieldDef('SSF3', 'LVSI_Endometrium',
                                 'Lymphovascular space invasion',
                                 decoder=None),
            'SSF4':  SSFFieldDef('SSF4', 'Cervical_Extension',
                                 'Cervical stromal invasion',
                                 decoder=None),
            'SSF5':  SSFFieldDef('SSF5', 'Adnexal_Extension',
                                 'Adnexal or extrauterine extension',
                                 decoder=None),
            'SSF6':  SSFFieldDef('SSF6', 'Peritoneal_Cytology_Uterus',
                                 'Peritoneal washing cytology',
                                 decoder=None),
            'SSF7':  SSFFieldDef('SSF7', 'MMR_Status_Uterus',
                                 'Mismatch repair (MMR) / MSI status',
                                 decoder=_decode_msi),
            'SSF8':  SSFFieldDef('SSF8', 'POLE_Mutation',
                                 'POLE exonuclease domain mutation',
                                 decoder=None),
            'SSF9':  SSFFieldDef('SSF9', 'p53_Status',
                                 'TP53 mutation / p53 IHC status',
                                 decoder=None),
            'SSF10': SSFFieldDef('SSF10', 'FIGO_Molecular_Class',
                                 'FIGO 2023 molecular classification (POLE/MMRd/p53/NSMP)',
                                 decoder=None),
        },
        notes='FIGO 2023 integrates molecular subgroups (POLE/MMRd/p53-abn/NSMP) for risk stratification.',
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
        # Try 3-char prefix first (e.g., C50)
        c3 = p[:3]
        if c3 in _CODE_TO_GROUP:
            return _CODE_TO_GROUP[c3]
        # Try 4-char prefix (for C181, C182 etc.)
        c4 = p[:4]
        if c4 in _CODE_TO_GROUP:
            return _CODE_TO_GROUP[c4]
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
