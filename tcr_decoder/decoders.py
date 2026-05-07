"""
Field-specific decoders for Taiwan Cancer Registry SSF and clinical fields.

Each decoder takes a raw pd.Series and returns a decoded pd.Series
with clean English clinical labels.
"""

import re
import pandas as pd
import numpy as np
from tcr_decoder.utils import strip_float_suffix, _norm


# ─── ER / PR (SSF1, SSF2) ──────────────────────────────────────────

def decode_er_pr(raw_series: pd.Series, receptor: str) -> pd.Series:
    """Decode SSF1 (ER) or SSF2 (PR) codes into clinical receptor status.

    Codebook: Cancer-SSF-Manual (breast), p.127-130
    Codes: 000-100 (%), W/I/S prefix (staining), 110-121 (special), 888/988/999
    """
    def _decode(v):
        v = _norm(v)
        if not v:
            return ''
        special = {
            '988': 'Not applicable (Oncotype/Phyllodes/Sarcoma)',
            '888': f'{receptor} converted Neg→Pos after neoadjuvant therapy',
            '999': 'Unknown',
            '120': f'{receptor} Negative (<1% or not specified)',
            '121': f'{receptor} Negative (post-neoadjuvant value only)',
            '110': f'{receptor} Positive (proportion unclear)',
            '111': f'{receptor} Positive (post-neoadjuvant value only)',
        }
        if v in special:
            return special[v]
        # Letter-prefix staining codes W/I/S
        if len(v) >= 3 and v[0].upper() in 'WIS' and v[1:].isdigit():
            prefix = {'W': 'Weak', 'I': 'Intermediate', 'S': 'Strong'}[v[0].upper()]
            pct = int(v[1:])
            return f'{receptor} Positive ({prefix} staining, {pct}%)'
        # Numeric percentage
        if v.isdigit():
            n = int(v)
            if 0 < n <= 100:
                return f'{receptor} Positive ({n}%)'
            if n == 0:
                return f'{receptor} Negative (0%)'
        return v

    return raw_series.fillna('').astype(str).apply(_decode)


# ─── Ki-67 (SSF10) ──────────────────────────────────────────────────

def decode_ki67(raw_series: pd.Series) -> pd.Series:
    """Decode SSF10 Ki-67 index codes.

    Codebook: Cancer-SSF-Manual (breast), p.150-151
    Codes: 000-100 (%), A00-A09 (sub-1%), 988/998/999
    """
    def _decode(v):
        v = _norm(v)
        if not v:
            return ''
        special = {
            '888': 'Not applicable (conversion)',
            '988': 'Not applicable (Phyllodes/Sarcoma)',
            '998': 'Tested, percentage unknown',
            '999': 'Unknown',
        }
        if v in special:
            return special[v]
        # A00-A09: sub-1% values
        if len(v) == 3 and v[0].upper() == 'A' and v[1:].isdigit():
            pct = int(v[1:]) * 0.1
            return f'{pct:.1f}%'
        if v.isdigit():
            n = int(v)
            if 0 <= n <= 100:
                if n < 14:
                    category = 'Low'
                elif n <= 30:
                    category = 'Intermediate'
                else:
                    category = 'High'
                return f'{n}% ({category})'
        return v

    return raw_series.fillna('').astype(str).apply(_decode)


# ─── HER2 (SSF7) ────────────────────────────────────────────────────

def decode_her2(raw_series: pd.Series) -> pd.Series:
    """Decode SSF7 HER2 combined IHC+ISH codes.

    Codebook: Cancer-SSF-Manual (breast), p.138-145
    Complex 3-digit system: 1st digit=test type, 2nd=IHC, 3rd=ISH
    """
    HER2_MAP = {
        '0':   'IHC 0 — Negative',
        '1':   'IHC 1+ — Negative (Low HER2)',
        '2':   'IHC 2+ — Equivocal',
        '3':   'IHC 3+ — Positive',
        '000': 'IHC 0 — Negative',
        '004': 'IHC 0 Ultralow (0%<staining≤10%) — Negative',
        '100': 'IHC 0 → Negative (no ISH)',
        '101': 'IHC 1+ — Negative (Low HER2)',
        '102': 'IHC 2+ — Equivocal (no ISH)',
        '103': 'IHC 3+ — Positive',
        # Legacy CISH codes (dx yr 100-107 only)
        '200': 'CISH Negative (legacy: dx yr 100-107 only)',
        '201': 'CISH Positive (legacy: dx yr 100-107 only)',
        '202': 'CISH Equivocal (legacy: dx yr 100-107 only)',
        # Legacy other-test codes (dx yr 100-107 only)
        '400': 'Other test — HER2 Negative (legacy: dx yr 100-107 only)',
        '401': 'Other test — HER2 Positive (legacy: dx yr 100-107 only)',
        '402': 'Other test — HER2 Equivocal (legacy: dx yr 100-107 only)',
        # Legacy ISH-only codes (dx yr 100-107)
        '300': 'ISH Negative',
        '301': 'ISH Positive',
        '302': 'ISH Equivocal',
        # IHC+ISH combined (dx yr 108+)
        '500': 'IHC 0 + ISH Negative — Negative',
        '501': 'IHC 0 + ISH Positive — Positive',
        '502': 'IHC 0 + ISH Equivocal',
        '510': 'IHC 1+ + ISH Negative — Negative',
        '511': 'IHC 1+ + ISH Positive — Positive',
        '512': 'IHC 1+ + ISH Equivocal',
        '520': 'IHC 2+ + ISH Negative — Negative',
        '521': 'IHC 2+ + ISH Positive — Positive',
        '522': 'IHC 2+ + ISH Equivocal',
        '530': 'IHC 3+ + ISH Negative — Positive (IHC overrides)',
        '531': 'IHC 3+ + ISH Positive — Positive',
        '532': 'IHC 3+ + ISH Equivocal — Positive (IHC overrides)',
        '590': 'IHC unknown + ISH Negative — Negative',
        '591': 'IHC unknown + ISH Positive — Positive',
        '592': 'IHC unknown + ISH Equivocal',
        # Ultralow HER2 + ISH (dx yr 114+)
        '600': 'IHC 0 (staining=0%) + ISH Negative — Negative',
        '601': 'IHC 0 (staining=0%) + ISH Positive — Positive',
        '602': 'IHC 0 (staining=0%) + ISH Equivocal',
        '640': 'IHC 0 Ultralow + ISH Negative — Negative',
        '641': 'IHC 0 Ultralow + ISH Positive — Positive',
        '642': 'IHC 0 Ultralow + ISH Equivocal',
        # Neoadjuvant / Other
        '888': 'HER2 converted Neg→Pos after neoadjuvant therapy',
        '900': 'HER2 Negative (other/unknown test method)',
        '901': 'HER2 Positive (other/unknown test method)',
        '902': 'HER2 Equivocal (other/unknown test method)',
        '988': 'Not applicable (Phyllodes/Sarcoma)',
        '999': 'Unknown',
    }

    def _decode(v):
        v = _norm(v)
        if not v:
            return ''
        if v in HER2_MAP:
            return HER2_MAP[v]
        v3 = v.zfill(3)
        if v3 in HER2_MAP:
            return HER2_MAP[v3]
        return v

    return raw_series.fillna('').astype(str).apply(_decode)


# ─── Nottingham Grade (SSF6) ────────────────────────────────────────

def decode_nottingham(raw_series: pd.Series) -> pd.Series:
    """Decode SSF6 Nottingham/Bloom-Richardson score and grade.

    Codebook: Cancer-SSF-Manual (breast), p.140-141
    Codes: 030-090 (score×10), 110-130 (grade only), 988/999
    """
    def _decode(v):
        v = _norm(v)
        if not v:
            return ''
        special = {'888': 'Not applicable (conversion)', '988': 'Not applicable', '999': 'Unknown'}
        if v in special:
            return special[v]
        if v.isdigit():
            n = int(v)
            # Score codes: 30-90 (÷10), or 3-9 (raw), or 13-19 (alt format)
            if n in (3, 4, 5, 13, 14, 15, 30, 40, 50):
                score = n if n <= 9 else (n % 10 if n < 20 else n // 10)
                return f'Score {score} → Grade 1 (Well differentiated)'
            if n in (6, 7, 16, 17, 60, 70):
                score = n if n <= 9 else (n % 10 if n < 20 else n // 10)
                return f'Score {score} → Grade 2 (Moderately differentiated)'
            if n in (8, 9, 18, 19, 80, 90):
                score = n if n <= 9 else (n % 10 if n < 20 else n // 10)
                return f'Score {score} → Grade 3 (Poorly differentiated)'
            # Grade-only codes
            if n == 110:
                return 'Grade 1 (Well differentiated)'
            if n == 120:
                return 'Grade 2 (Moderately differentiated)'
            if n == 130:
                return 'Grade 3 (Poorly differentiated)'
        return v

    return raw_series.fillna('').astype(str).apply(_decode)


# ─── Neoadjuvant Response (SSF3) ────────────────────────────────────

def decode_ssf3_neoadj(raw_series: pd.Series) -> pd.Series:
    """Decode SSF3 neoadjuvant therapy response codes.

    Codebook: Cancer-SSF-Manual (breast), p.135-136
    Codes: 010 (cCR), 011 (pCR), 020 (PR), 030 (SD), 040 (PD), 988/990/999
    """
    MAP = {
        '10':  'cCR — Clinical complete response',
        '010': 'cCR — Clinical complete response',
        '11':  'pCR — Pathologic complete response (no residual in breast + nodes)',
        '011': 'pCR — Pathologic complete response (no residual in breast + nodes)',
        '20':  'Partial response / Moderate response',
        '020': 'Partial response / Moderate response',
        '30':  'Stable disease / Minimal response',
        '030': 'Stable disease / Minimal response',
        '40':  'Progressive disease / No response',
        '040': 'Progressive disease / No response',
        '888': 'Not applicable (conversion / neoadjuvant outcome not assessed)',
        '988': 'Not applicable (no neoadjuvant therapy)',
        '990': 'Post-treatment shrinkage, degree not specified',
        '999': 'Unknown',
    }

    def _decode(v):
        v = _norm(v)
        if not v:
            return ''
        if v in MAP:
            return MAP[v]
        v2 = v.zfill(3)
        if v2 in MAP:
            return MAP[v2]
        return v

    return raw_series.fillna('').astype(str).apply(_decode)


# ─── EBRT Technique (additive coding) ───────────────────────────────

def decode_ebrt_additive(raw_series: pd.Series) -> pd.Series:
    """Decode EBRT technique using additive coding system.

    Codebook: Longform-Manual, p.240-241
    Base codes: 1=2D, 2=3D-CRT, 4=IMRT, 8=VMAT/Tomo, 16=Mixed, 32=IGRT, 64=Resp
    Final code = sum of all techniques used across treatment phases.
    """
    COMPONENTS = {
        1: '2D/Simple', 2: '3D-CRT', 4: 'IMRT', 8: 'VMAT/Tomotherapy',
        16: 'Mixed Photon+Particle', 32: 'IGRT', 64: 'Respiratory Control',
    }

    def _decode(v):
        v = _norm(v)
        if not v:
            return ''
        if v in ('-9', '999'):
            return 'Unknown'
        if v == '-1':
            return 'EBRT NOS'
        if v == '0':
            return 'No EBRT'
        try:
            n = int(v)
        except ValueError:
            return v
        if n < 0:
            return 'Unknown'
        parts = []
        for bit in sorted(COMPONENTS.keys(), reverse=True):
            if n >= bit:
                parts.append(COMPONENTS[bit])
                n -= bit
        if n > 0:
            parts.append(f'code-{n}')
        return ' + '.join(reversed(parts)) if parts else v

    return raw_series.fillna('').astype(str).apply(_decode)


# ─── Sentinel LN (SSF4, SSF5) ───────────────────────────────────────

def decode_sentinel(raw_series: pd.Series, kind: str) -> pd.Series:
    """Decode SSF4 (SLN examined) or SSF5 (SLN positive).

    kind: 'examined' or 'positive'
    """
    def _decode(v):
        v = _norm(v)
        if not v:
            return ''
        special = {
            '888': 'Not applicable (conversion)',
            '988': 'Not applicable (no SLN biopsy)',
            '996': 'Sentinel LN biopsy performed; no lymph node tissue found or count unknown',
            '999': 'Unknown',
        }
        if v in special:
            return special[v]
        if v == '0':
            return f'None {kind}' if kind == 'positive' else 'None examined'
        if v.isdigit():
            n = int(v)
            if 1 <= n <= 89:
                return f'{n} node(s) {kind}'
        return v

    return raw_series.fillna('').astype(str).apply(_decode)


# ─── LN Positive ────────────────────────────────────────────────────

def decode_lnpositive(raw_series: pd.Series) -> pd.Series:
    """Decode LN_POSITI field with sentinel codes."""
    def _decode(v):
        v = _norm(v)
        if not v:
            return ''
        special = {
            '95': 'Positive LN, count not applicable',
            '97': 'Positive LN, count not specified',
            '98': 'Positive LN, count not applicable',
            '99': 'Unknown',
        }
        if v in special:
            return special[v]
        if v.isdigit():
            return v  # Actual count
        return v

    return raw_series.fillna('').astype(str).apply(_decode)


# ─── Cause of Death ─────────────────────────────────────────────────

def decode_cause_of_death(series: pd.Series) -> pd.Series:
    """Decode DIECAUSE ICD-O-3/ICD-10 cause of death codes."""
    from tcr_decoder.mappings import CODE_MAPPINGS
    diecause_map = CODE_MAPPINGS.get('DIECAUSE', {})

    def _decode(v):
        v = _norm(v)
        if not v:
            return ''
        # Try direct lookup
        from tcr_decoder.utils import clean_text
        if v in diecause_map:
            return clean_text(diecause_map[v])
        # For non-0 codes, use generic pattern
        if v == '0' or v == '0.0':
            return 'Non-cancer / Not applicable'
        return clean_text(v)

    return series.fillna('').astype(str).apply(_decode)


# ─── Smoking Triplet ────────────────────────────────────────────────

def decode_smoking_triplet(series: pd.Series) -> pd.Series:
    """Decode raw smoking/betelnut/alcohol triplet format (XX,XX,XX).

    Format: pack-years, betelnut-years, alcohol-years
    00 = No, 88 = N/A, 99 = Unknown, others = actual years
    """
    def _decode(v):
        v = str(v).strip()
        if not v or v.lower() == 'nan':
            return ''
        parts = v.split(',')
        if len(parts) != 3:
            return v
        labels = ['Smoking', 'Betelnut', 'Alcohol']
        results = []
        for label, p in zip(labels, parts):
            p = p.strip()
            if p == '00':
                results.append(f'{label}: No')
            elif p == '88':
                results.append(f'{label}: N/A')
            elif p == '99':
                results.append(f'{label}: Unknown')
            elif p.isdigit():
                unit = 'pack-yr' if label == 'Smoking' else 'yr'
                results.append(f'{label}: {int(p)} {unit}')
            else:
                results.append(f'{label}: {p}')
        return '; '.join(results)

    return series.fillna('').astype(str).apply(_decode)
