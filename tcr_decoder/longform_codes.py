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


# ─────────────────────────────────────────────────────────────────────────────
# 首次療程的全身性治療 (#4.3.x, #4.4, #4.5.1, pp.268-303)
#
# Chemotherapy, hormone/steroid, immunotherapy and targeted therapy are each
# reported twice: once for the outside hospital and once for the reporting
# hospital. Both halves of a pair share the modality codes; only the reporting
# hospital has the 8x block, because only it knows why a planned treatment was
# not given.
#
# The modality codes are NOT interchangeable between therapies -- 02 is
# "systemic chemotherapy, single agent" for chemo but "regional hormone/steroid
# therapy" for hormones -- so each therapy has its own table.
# ─────────────────────────────────────────────────────────────────────────────

def _therapy_map(therapy, modality, trial, this_hospital, extra=None):
    """One therapy field's table: modality + trial + (8x when this hospital)."""
    cap = therapy[0].upper() + therapy[1:]
    codes = {
        0: f'No {therapy}; not part of the first course, or the cancer was '
           f'found only at autopsy',
        **modality,
        **trial,
    }
    if this_hospital:
        codes.update(extra or {})
        codes.update({
            82: f'{cap} not advised or given because of a contraindication or '
                f'another patient risk factor (comorbidity, advanced age)',
            83: f'{cap} not advised or given because the disease progressed',
            85: f'{cap} was part of the planned first course, but the patient '
                f'died or was discharged critically ill before it started',
            86: f'{cap} was part of the planned first course and was not '
                f'given, with no reason recorded; or it was given at another '
                f'hospital',
            87: f'{cap} was part of the planned first course but the patient '
                f'or family refused it',
            88: f'{cap} was part of the planned first course but had not '
                f'started when the case was abstracted',
        })
    codes[99] = (f'Not documented, so it is unknown whether {therapy} was '
                 f'advised or given; or the cancer is known only from a death '
                 f'certificate')
    return CodeMap(codes, width=2)


_CHEMO_MODALITY = {
    1:  'Systemic chemotherapy',
    2:  'Systemic chemotherapy, single agent (diagnosis year 2017 or earlier)',
    3:  'Systemic chemotherapy, more than one agent (diagnosis year 2017 or '
        'earlier)',
    4:  'Transarterial chemoembolisation (TACE) of the primary site only',
    5:  'TACE of the primary site plus systemic chemotherapy',
    6:  'TACE of the primary site plus other regional chemotherapy',
    7:  'TACE of the primary site plus other regional and systemic chemotherapy',
    8:  'Regional chemotherapy only, excluding TACE (intrapleural, '
        'intrapericardial, intraperitoneal, intravesical, intrathecal, or '
        'another regional route such as a BCNU wafer implant)',
    9:  'Systemic and regional chemotherapy, excluding TACE',
    10: 'TACE for liver metastases',
    11: 'TACE for liver metastases plus systemic chemotherapy',
    12: 'TACE for liver metastases plus other regional chemotherapy',
    13: 'TACE for liver metastases plus other regional and systemic '
        'chemotherapy',
}

_HORMONE_MODALITY = {
    1: 'Systemic hormone/steroid therapy in the first course (for a '
       'haematolymphoid malignancy, systemic steroids from the date of '
       'diagnosis, with or without chemotherapy)',
    2: 'Regional hormone/steroid therapy (for a haematolymphoid malignancy, '
       'regional steroids from the date of diagnosis, with or without '
       'chemotherapy)',
    3: 'Systemic and regional hormone/steroid therapy',
}

_IMMUNO_MODALITY = {
    1: 'Systemic immunotherapy drug',
    2: 'Regional immunotherapy drug',
    3: 'Systemic and regional immunotherapy drugs',
    4: 'Cellular immunotherapy only',
    5: 'Cellular immunotherapy plus a systemic immunotherapy drug',
    6: 'Cellular immunotherapy plus a regional immunotherapy drug',
    7: 'Cellular immunotherapy plus systemic and regional immunotherapy drugs',
}

_TARGETED_MODALITY = {
    1: 'Targeted therapy given in the first course',
}


def _trial_codes(therapy, cellular=False):
    cap = therapy[0].upper() + therapy[1:]
    codes = {
        20: f'Clinical-trial {therapy} only',
        21: f'{cap} plus clinical-trial {therapy}',
        30: f'Double-blind-trial {therapy} only',
        31: f'{cap} plus double-blind-trial {therapy}',
    }
    if cellular:
        codes.update({
            22: f'Cellular {therapy} plus clinical-trial {therapy}',
            23: f'Cellular {therapy} with systemic and/or regional drugs, plus '
                f'clinical-trial {therapy}',
            32: f'Cellular {therapy} plus double-blind-trial {therapy}',
            33: f'Cellular {therapy} with systemic and/or regional drugs, plus '
                f'double-blind-trial {therapy}',
            40: f'Clinical-trial cellular {therapy} only',
            41: f'Systemic and/or regional drugs plus clinical-trial cellular '
                f'{therapy}',
        })
    return codes


CHEMO_OTHER_MAP = _therapy_map(
    'chemotherapy', _CHEMO_MODALITY, _trial_codes('chemotherapy'), False)
CHEMO_THIS_MAP = _therapy_map(
    'chemotherapy', _CHEMO_MODALITY, _trial_codes('chemotherapy'), True,
    {81: 'Chemotherapy was the planned first course but was not advised or '
         'given because of a genetic test result'})

HORMONE_OTHER_MAP = _therapy_map(
    'hormone/steroid therapy', _HORMONE_MODALITY,
    _trial_codes('hormone/steroid therapy'), False)
HORMONE_THIS_MAP = _therapy_map(
    'hormone/steroid therapy', _HORMONE_MODALITY,
    _trial_codes('hormone/steroid therapy'), True)

IMMUNO_OTHER_MAP = _therapy_map(
    'immunotherapy', _IMMUNO_MODALITY,
    _trial_codes('immunotherapy', cellular=True), False)
IMMUNO_THIS_MAP = _therapy_map(
    'immunotherapy', _IMMUNO_MODALITY,
    _trial_codes('immunotherapy', cellular=True), True)

TARGETED_OTHER_MAP = _therapy_map(
    'targeted therapy', _TARGETED_MODALITY, _trial_codes('targeted therapy'),
    False)
TARGETED_THIS_MAP = _therapy_map(
    'targeted therapy', _TARGETED_MODALITY, _trial_codes('targeted therapy'),
    True)


# ─────────────────────────────────────────────────────────────────────────────
# 其他治療 Other Treatment (#4.5.1, p.301) -- a single field, not a pair
# ─────────────────────────────────────────────────────────────────────────────

OTHER_TREATMENT_MAP = CodeMap({
    0:  'No other treatment; other treatment was not part of the first course',
    1:  'Other treatment in the first course at the reporting hospital',
    2:  'Other treatment in the first course at another hospital',
    3:  'Other treatment in the first course at both the reporting hospital '
        'and another hospital',
    99: 'Not documented, so it is unknown whether other treatment was advised '
        'or given',
}, width=2)


# ─────────────────────────────────────────────────────────────────────────────
# 申報醫院緩和照護 Palliative Care (#4.4, pp.298-300)
# ─────────────────────────────────────────────────────────────────────────────

PALLIATIVE_CARE_MAP = CodeMap({
    0: 'No palliative care',
    1: 'Surgery to relieve symptoms only, not for diagnosis, staging or '
       'treatment (may include a bypass procedure)',
    2: 'Radiotherapy to relieve symptoms only',
    3: 'Regional or systemic drug therapy to relieve symptoms only',
    4: 'Pain management or referral for it, with no other palliative care',
    5: 'Two or more of codes 1, 2 and 3, without code 4',
    6: 'One or more of codes 1, 2 and 3, together with code 4',
    7: 'Palliative care given or referred, but the record does not say what '
       'kind',
    9: 'Unknown whether palliative care was given or referred; not documented',
}, width=1)


LONGFORM_CODE_MAPS.update({
    'PREC':   (CHEMO_OTHER_MAP, '4.3.2'),
    'C':      (CHEMO_THIS_MAP, '4.3.3'),
    'PREH':   (HORMONE_OTHER_MAP, '4.3.5'),
    'H':      (HORMONE_THIS_MAP, '4.3.6'),
    'PREI':   (IMMUNO_OTHER_MAP, '4.3.8'),
    'I':      (IMMUNO_THIS_MAP, '4.3.9'),
    'PRETAR': (TARGETED_OTHER_MAP, '4.3.13'),
    'TAR':    (TARGETED_THIS_MAP, '4.3.14'),
    'OTH':    (OTHER_TREATMENT_MAP, '4.5.1'),
    'PREP':   (PALLIATIVE_CARE_MAP, '4.4'),
})
