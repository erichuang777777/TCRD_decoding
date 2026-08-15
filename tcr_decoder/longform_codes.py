# -*- coding: utf-8 -*-
"""Code tables for the Longform's own structural fields.

These are the fields the registry defines itself, as opposed to the
site-specific factors (ssf_registry.py) and the per-site surgery codes
(surgery_codes.py). Each one is transcribed from the printed 編碼 table, and
its 編碼範圍 and 欄位長度 are recorded in code_ranges.LONGFORM so
test_codebook_conformance.py can enumerate the whole legal range.

Field numbers below are the manual's own 癌登欄位序號, verified against
longform_fields.py.
"""

import pandas as pd

from tcr_decoder.codemap import CodeMap

# ─────────────────────────────────────────────────────────────────────────────
# 側性 Laterality (#2.7, p.87-88)
# ─────────────────────────────────────────────────────────────────────────────

LATERALITY_MAP = CodeMap({
    0: 'Not a paired organ',
    1: 'Origin in the right side',
    2: 'Origin in the left side',
    3: 'Paired organ, one side involved but the side of origin is not stated',
    4: 'Paired organ, bilateral involvement with the side of origin unknown, '
       'recorded as a single primary',
    5: 'Midline tumour of a paired site (C70.0, C71.0-C71.4, C72.2-C72.5, '
       'C44.3-C44.5)',
    9: 'Paired organ, laterality not stated',
}, width=1)


# ─────────────────────────────────────────────────────────────────────────────
# 性態碼 Behavior (#2.9, p.91-92)
# ─────────────────────────────────────────────────────────────────────────────

BEHAVIOR_MAP = CodeMap({
    2: 'In situ (non-invasive, intraepithelial, no stromal invasion)',
    3: 'Invasive or microinvasive',
}, width=1)


# ─────────────────────────────────────────────────────────────────────────────
# 癌症確診方式 Diagnostic Confirmation (#2.11, pp.102-104)
#
# TWO tables. Code 3 exists only for the haematolymphoid morphologies
# (M9590-9993), and code 5 is worded differently in each: for a solid tumour
# it is a lab/marker result alone, for a haematolymphoid one it also covers
# immunophenotyping and genetic testing. Decoding therefore needs MCODE.
# ─────────────────────────────────────────────────────────────────────────────

_CONFER_SHARED = {
    1: 'Positive histology',
    2: 'Positive cytology',
    4: 'Microscopically confirmed, method not stated',
    6: 'Direct visualisation at surgery or endoscopy, not microscopically '
       'confirmed',
    7: 'Radiography or other imaging, not microscopically confirmed',
    8: 'Clinical diagnosis only (excluding 5, 6 and 7)',
    9: 'Unknown whether microscopically confirmed',
}

CONFIRMATION_SOLID_MAP = CodeMap({
    **_CONFER_SHARED,
    5: 'Positive laboratory test or tumour marker, not microscopically '
       'confirmed',
}, width=1)

CONFIRMATION_HAEM_MAP = CodeMap({
    **_CONFER_SHARED,
    3: 'Positive histology plus positive immunophenotyping and/or genetic '
       'testing',
    5: 'Positive laboratory test or tumour marker, or positive '
       'immunophenotyping and/or genetic testing',
}, width=1)

# ICD-O-3 morphologies that use the haematolymphoid table (manual p.104).
_HAEM_MORPHOLOGY = (9590, 9993)


def _is_haematolymphoid(morphology) -> bool:
    from tcr_decoder.ssf_registry import _parse_morphology

    m = _parse_morphology(morphology)
    return bool(m) and _HAEM_MORPHOLOGY[0] <= m <= _HAEM_MORPHOLOGY[1]


def decode_confirmation(raw_series: pd.Series,
                        morphology_series: pd.Series = None) -> pd.Series:
    """Decode CONFER (#2.11), choosing the table by morphology.

    Without a morphology column the solid-tumour table is used, which is
    right for every site except M9590-9993 and is stated as the assumption
    rather than silently applied: code 3 then decodes as an unlisted code
    instead of being given the haematolymphoid meaning by accident.
    """
    if morphology_series is None:
        return CONFIRMATION_SOLID_MAP.decode(raw_series)

    morph = morphology_series.reindex(raw_series.index)
    return pd.Series(
        [(CONFIRMATION_HAEM_MAP if _is_haematolymphoid(m)
          else CONFIRMATION_SOLID_MAP).decode_one(v)
         for v, m in zip(raw_series, morph)],
        index=raw_series.index)


def encode_confirmation(series: pd.Series,
                        morphology_series: pd.Series = None) -> pd.Series:
    """Inverse of decode_confirmation()."""
    if morphology_series is None:
        return CONFIRMATION_SOLID_MAP.encode(series)

    morph = morphology_series.reindex(series.index)
    return pd.Series(
        [(CONFIRMATION_HAEM_MAP if _is_haematolymphoid(m)
          else CONFIRMATION_SOLID_MAP).encode_one(v)
         for v, m in zip(series, morph)],
        index=series.index)


# ─────────────────────────────────────────────────────────────────────────────
# 神經侵襲 Perineural Invasion (#2.13.1, pp.114-115)
# 淋巴管或血管侵犯 Lymph-vascular Invasion (#2.13.2, pp.117-118)
#
# Identical code shape, different subject. Kept as two maps: sharing one
# would make 'no invasion' ambiguous between the two fields, which is the
# same defect that made breast SSF4 and SSF5 un-encodable.
# ─────────────────────────────────────────────────────────────────────────────

def _invasion_map(subject: str, na_extra: str = '') -> CodeMap:
    not_applicable = (
        'Not applicable: GIST, NET, high-grade dysplasia or carcinoma in '
        'situ; lymphoma or leukemia; plasma cell myeloma; central nervous '
        'system malignancy; unknown primary; or no pathology performed on '
        'the primary site'
    )
    return CodeMap({
        0: f'No {subject}',
        1: f'{subject.capitalize()} present',
        7: f'{subject.capitalize()} cannot be assessed: reported as NA, '
           f'specimen too small, insufficient sample, no primary-site report '
           f'describes it, or the report found no malignant cells',
        8: not_applicable,
        9: 'Not documented in the medical record',
    }, width=1)


PERINEURAL_INVASION_MAP = _invasion_map('perineural invasion')
LVI_MAP = _invasion_map('lymph-vascular invasion')


# Field tag -> (CodeMap, 癌登欄位序號). Fields whose decoding needs a second
# column (CONFER) are handled by their own function above.
LONGFORM_CODE_MAPS = {
    'LAT95':  (LATERALITY_MAP, '2.7'),
    'MCODE5': (BEHAVIOR_MAP, '2.9'),
    'PNI':    (PERINEURAL_INVASION_MAP, '2.13.1'),
    'LVI':    (LVI_MAP, '2.13.2'),
}
