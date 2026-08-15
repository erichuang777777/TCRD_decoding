"""
Field-specific decoders for Taiwan Cancer Registry SSF and clinical fields.

Each decoder takes a raw pd.Series and returns a decoded pd.Series
with clean English clinical labels.
"""

import re
import pandas as pd
import numpy as np
from tcr_decoder.utils import strip_float_suffix, _norm


# Codes shaped like a TCR sentinel (888, 900-902, 988, 990, 996-999) that are
# NOT part of a given field's official code range. Letting one fall through a
# decoder's raw passthrough would put a bare '888' into a clinical column,
# where it reads as a measurement rather than a code; each breast decoder
# below labels it explicitly instead. The label is machine-reversible, so the
# encode direction still reproduces the original code exactly.
_SENTINEL_LIKE = frozenset({
    '888', '900', '901', '902', '988', '990', '996', '997', '998', '999',
})


def _unlisted_code(v: str) -> str:
    """Label for a sentinel-shaped code outside this field's official range."""
    return f'Unlisted code {v}' if v in _SENTINEL_LIKE else v


# ─── ER / PR (SSF1, SSF2) ──────────────────────────────────────────

def decode_er_pr(raw_series: pd.Series, receptor: str) -> pd.Series:
    """Decode SSF1 (ER) or SSF2 (PR) codes into clinical receptor status.

    Codebook: Cancer-SSF-Manual (breast), p.127-130
    Codes: 000-100 (%), W/I/S prefix (staining), 110-121 (special), 888/988/999

    Note -- Allred score coverage: the codebook (p.121-123) actually defines
    TWO independent encoding schemes for this field, chosen per how the
    pathology report describes the result. The scheme this function
    literally implements ("letter + literal percentage", e.g. 'S70' = Strong
    staining, 70%) is scheme A. Scheme B, Allred score (intensity 0/W/I/S
    + a 2-digit PROPORTION SCORE 00/06/22/49/84 for 0/1-10/11-33/34-66/>=67%
    positive cells), is not separately implemented -- but its proportion-
    score codes happen to equal the mean %-positive of their range, so this
    function's existing logic decodes them correctly by coincidence (e.g.
    'S84' -> "Strong staining, 84%", a faithful reading of Allred 3+5). See
    tests/test_encoders.py::TestEncodeErPr::test_allred_score_codes_decode_and_roundtrip
    for the pinned cases.

    The one cell previously flagged as unconfirmed -- Allred proportion
    score 1 (<1% positive cells) -- is settled: the manual's own table
    (p.123, codebook_md/ssf_chunk_007) gives that row the shared code 120
    ("陰性: ER 反應的比例<1%，不論染色強度") rather than a letter code, which
    is exactly what the '120' entry above already decodes. No special case
    is needed. See coding_rules/breast_coding_spec.md for the full table.
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
        # Letter-prefix staining codes W/I/S. Per the Cancer-SSF-Manual
        # (breast SSF1, p.121), the field is 3 chars: intensity letter +
        # 2-digit proportion, and the proportion carries the staining % with
        # 100% encoded as '00' (manual examples: strong 100% -> 'S00';
        # weak 1% -> 'W01'). Treating '00' as a literal 0% would misread a
        # strongly-positive tumour as negative, so map 0 -> 100.
        if len(v) >= 3 and v[0].upper() in 'WIS' and v[1:].isdigit():
            prefix = {'W': 'Weak', 'I': 'Intermediate', 'S': 'Strong'}[v[0].upper()]
            pct = int(v[1:])
            if pct == 0:
                pct = 100
            return f'{receptor} Positive ({prefix} staining, {pct}%)'
        # Numeric percentage
        if v.isdigit():
            n = int(v)
            if 0 < n <= 100:
                return f'{receptor} Positive ({n}%)'
            if n == 0:
                return f'{receptor} Negative (0%)'
        return _unlisted_code(v)

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
        # Official code range (Cancer-SSF-Manual, breast SSF10, p.150-151):
        # A00-A09, 000-100, 988, 998, 999. 888 is NOT part of this field's
        # range -- it is only defined for SSF1/SSF2 (ER/PR "converted after
        # neoadjuvant therapy") and SSF7 -- so it is deliberately absent here
        # and falls through to the raw passthrough below.
        special = {
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
        return _unlisted_code(v)

    return raw_series.fillna('').astype(str).apply(_decode)


# ─── HER2 (SSF7) ────────────────────────────────────────────────────

HER2_MAP = {
    # Legacy 1-digit forms. The codebook's valid range starts at 000/004/100,
    # and a bare 1-digit code carries no staining-percentage information, so
    # each one is given the same text as its IHC-only 3-digit equivalent
    # (100-103) and encode_her2() canonicalizes it to that official code.
    # In particular '0' means "IHC 0, staining % not described" = 100, NOT
    # 000 (which specifically asserts staining = 0%, a 114-dx-year code).
    # p.142 note 1: an IHC 2+ with no ISH follow-up is coded 102.
    '0':   'IHC 0 → Negative (no ISH)',
    '1':   'IHC 1+ — Negative (Low HER2)',
    '2':   'IHC 2+ — Equivocal (no ISH)',
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
    '300': 'FISH Negative (legacy: dx yr 100-107 only)',
    '301': 'FISH Positive (legacy: dx yr 100-107 only)',
    '302': 'FISH Equivocal (legacy: dx yr 100-107 only)',
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


def decode_her2(raw_series: pd.Series) -> pd.Series:
    """Decode SSF7 HER2 combined IHC+ISH codes.

    Codebook: Cancer-SSF-Manual (breast), p.138-145
    Complex 3-digit system: 1st digit=test type, 2nd=IHC, 3rd=ISH
    """
    def _decode(v):
        v = _norm(v)
        if not v:
            return ''
        if v in HER2_MAP:
            return HER2_MAP[v]
        v3 = v.zfill(3)
        if v3 in HER2_MAP:
            return HER2_MAP[v3]
        return _unlisted_code(v)

    return raw_series.fillna('').astype(str).apply(_decode)


# ─── Nottingham Grade (SSF6) ────────────────────────────────────────

def decode_nottingham(raw_series: pd.Series) -> pd.Series:
    """Decode SSF6 Nottingham/Bloom-Richardson score and grade.

    Codebook: Cancer-SSF-Manual (breast), p.140-141.
    Official code range: 030,040,050,060,070,080,090 (score x10),
    110/120/130 (grade only), 988, 999. 888 is not part of this field's
    range and is left to fall through as an unrecognized raw code.

    Bare single-digit scores 3-9 are also accepted: they are not codebook
    codes, but a spreadsheet export that dropped the leading zeros turns
    '030' into '30' and, in some hospital extracts, into '3'. Both are
    decoded to the same "Score N" text; encode_nottingham() always writes
    back the official 3-digit code (030-090).
    """
    def _decode(v):
        v = _norm(v)
        if not v:
            return ''
        special = {'988': 'Not applicable', '999': 'Unknown'}
        if v in special:
            return special[v]
        if v.isdigit():
            n = int(v)
            # Score codes: 30-90 (official, x10) or 3-9 (leading zeros lost)
            if n in (3, 4, 5, 30, 40, 50):
                score = n if n <= 9 else n // 10
                return f'Score {score} → Grade 1 (Well differentiated)'
            if n in (6, 7, 60, 70):
                score = n if n <= 9 else n // 10
                return f'Score {score} → Grade 2 (Moderately differentiated)'
            if n in (8, 9, 80, 90):
                score = n if n <= 9 else n // 10
                return f'Score {score} → Grade 3 (Poorly differentiated)'
            # Grade-only codes
            if n == 110:
                return 'Grade 1 (Well differentiated)'
            if n == 120:
                return 'Grade 2 (Moderately differentiated)'
            if n == 130:
                return 'Grade 3 (Poorly differentiated)'
        return _unlisted_code(v)

    return raw_series.fillna('').astype(str).apply(_decode)


# ─── Neoadjuvant Response (SSF3) ────────────────────────────────────

SSF3_NEOADJ_MAP = {
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
    # 888 is not in this field's official range (010,011,020,030,040,988,
    # 990,999 -- Cancer-SSF-Manual breast SSF3, p.135-136); an 888 here is
    # left untouched as an unrecognized raw code.
    '988': 'Not applicable (no neoadjuvant therapy / no surgery)',
    '990': 'Post-treatment shrinkage, degree not specified',
    '999': 'Unknown',
}


def decode_ssf3_neoadj(raw_series: pd.Series) -> pd.Series:
    """Decode SSF3 neoadjuvant therapy response codes.

    Codebook: Cancer-SSF-Manual (breast), p.135-136
    Codes: 010 (cCR), 011 (pCR), 020 (PR), 030 (SD), 040 (PD), 988/990/999
    """
    def _decode(v):
        v = _norm(v)
        if not v:
            return ''
        if v in SSF3_NEOADJ_MAP:
            return SSF3_NEOADJ_MAP[v]
        v2 = v.zfill(3)
        if v2 in SSF3_NEOADJ_MAP:
            return SSF3_NEOADJ_MAP[v2]
        return _unlisted_code(v)

    return raw_series.fillna('').astype(str).apply(_decode)


# ─── EBRT Technique (additive coding) ───────────────────────────────

EBRT_COMPONENTS = {
    1: '2D/Simple', 2: '3D-CRT', 4: 'IMRT', 8: 'VMAT/Tomotherapy',
    16: 'Mixed Photon+Particle', 32: 'IGRT', 64: 'Respiratory Control',
}


def decode_ebrt_additive(raw_series: pd.Series) -> pd.Series:
    """Decode EBRT technique using additive coding system.

    Codebook: Longform-Manual, p.240-241
    Base codes: 1=2D, 2=3D-CRT, 4=IMRT, 8=VMAT/Tomo, 16=Mixed, 32=IGRT, 64=Resp
    Final code = sum of all techniques used across treatment phases.
    """
    COMPONENTS = EBRT_COMPONENTS

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

    Codebook: Cancer-SSF-Manual (breast) p.137-139. Official range for both
    fields: 000-089, 988, 996, 999. The two fields give code 000 different
    meanings -- SSF4 000 = no sentinel-node surgery at all, SSF5 000 = no
    involved node (including ITC-only involvement) -- so the decoded text is
    per-field, not shared. 888 is not in either field's range and falls
    through as an unrecognized raw code.
    """
    def _decode(v):
        v = _norm(v)
        if not v:
            return ''
        special = {
            '988': ('Not applicable (no SLN surgery, post-neoadjuvant SLN, '
                    'or unknown whether performed)'),
            '996': ('Sentinel LN biopsy performed; count unknown or no lymph '
                    'node tissue found in the specimen'),
            '999': 'Unknown',
        }
        if v in special:
            return special[v]
        if v.isdigit():
            n = int(v)
            # Compare numerically: the field is 3 characters, so the zero
            # count arrives as '000' in a real file (and as '0' from an
            # export that dropped the padding).
            if n == 0:
                return ('None positive (or ITC-only involvement)' if kind == 'positive'
                        else 'No sentinel LN surgery performed')
            if 1 <= n <= 89:
                return f'{n} node(s) {kind}'
        return _unlisted_code(v)

    return raw_series.fillna('').astype(str).apply(_decode)


# ─── LN Positive ────────────────────────────────────────────────────

def decode_lnpositive(raw_series: pd.Series) -> pd.Series:
    """Decode LN_POSITI (區域淋巴結侵犯數目) with its sentinel codes.

    Codebook: Longform-Manual p.130-131. Official range 00-90, 95, 97-99.
    95, 97 and 98 are three DIFFERENT situations -- an earlier version of
    this decoder gave 95 and 98 the same text, which both lost the
    distinction and made 98 re-encode as 95.
    """
    def _decode(v):
        v = _norm(v)
        if not v:
            return ''
        special = {
            '95': 'Positive by aspiration/core biopsy only (nodes not surgically removed)',
            '97': 'Positive LN present, count not specified',
            '98': 'No nodes removed/examined, or no lymph node tissue found (clinical assessment only)',
            '99': 'Unknown / not applicable / not documented',
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
