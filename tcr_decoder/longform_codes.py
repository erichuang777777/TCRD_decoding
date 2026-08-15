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


# ─────────────────────────────────────────────────────────────────────────────
# 放射治療 Radiation therapy (#4.2.1.x, #4.2.2.x, pp.207-233)
#
# Four of these fields are ADDITIVE bitmasks, the same shape as EBRT
# (decoders.decode_ebrt_additive): the code is the SUM of every technique or
# phase used, not a single choice from a list. Enumerating "every legal code"
# for these means every subset sum, not a flat table -- listing 0-63 by hand
# would silently accept combinations the manual never defines and reject none
# of them, so they build their component table the same way EBRT does.
# ─────────────────────────────────────────────────────────────────────────────

def _additive_map(components: dict, zero_label: str, unknown_label: str,
                  nos_label: str, width: int) -> CodeMap:
    """A field whose legal codes are -9, -1, and every subset-sum of components."""
    codes = {-9: unknown_label, -1: nos_label, 0: zero_label}
    for r in range(1, len(components) + 1):
        import itertools
        for combo in itertools.combinations(components, r):
            total = sum(combo)
            label = ' + '.join(components[c] for c in sorted(combo))
            codes[total] = label
    return CodeMap(codes, width=width)


# 放射治療臨床標靶體積摘要 (#4.2.1.1), 最高/較低放射劑量臨床標靶體積
# (#4.2.2.2.1 / #4.2.2.3.1). Same target-volume vocabulary in all three --
# what differs between HTAR and LTAR is which dose band it describes, not
# what the codes mean, so one component table serves all three fields.
_TARGET_VOLUME_COMPONENTS = {
    1:  'Primary tumour (T)',
    2:  'Regional lymph nodes (N)',
    4:  'Distant metastasis (M)',
    8:  'Extended lymphoid region (mini-mantle/mantle/inverted-Y, or total '
        'lymphoid irradiation; Hodgkin and non-Hodgkin lymphoma only)',
    16: 'Total body / total bone marrow',
    32: 'Total skin (Kaposi sarcoma, primary cutaneous lymphoma, or another '
        'disease needing total-skin electron-beam therapy)',
}

RTAR_MAP = _additive_map(
    _TARGET_VOLUME_COMPONENTS,
    zero_label='No radiation therapy',
    unknown_label='Unknown whether radiation therapy was given',
    nos_label='Radiation therapy given, target volume not specified '
              '(includes RT as an endocrine procedure)',
    width=2)

_EBRT_TARGET_ZERO = ('No external beam radiotherapy, or EBRT given with no '
                     'clinical target volume')
HTAR_MAP = _additive_map(
    _TARGET_VOLUME_COMPONENTS,
    zero_label=_EBRT_TARGET_ZERO,
    unknown_label='Unknown whether external beam radiotherapy was given',
    nos_label='EBRT given, target volume not specified (includes EBRT as an '
              'endocrine procedure)',
    width=2)
LTAR_MAP = HTAR_MAP  # identical vocabulary; HTAR/LTAR differ by dose band,
                      # a fact carried by the column name, not the code.

# 放射治療儀器 (#4.2.1.2)
_RT_MODALITY_COMPONENTS = {
    1:  'External beam radiation therapy (cobalt unit, linear-accelerator '
        'photon or electron beam, tomotherapy)',
    2:  'Radiosurgery (Gamma Knife, Linac-based, CyberKnife, Zap-X, or SBRT/'
        'SABR delivered in <=6 fractions at >=800 cGy/fraction)',
    4:  'Brachytherapy (interstitial implants, moulds, seeds, needles, or '
        'intracavitary applicators of radioactive material)',
    8:  'Radioisotopes (injected radioactive material, e.g. I-131, Sr-89)',
    16: 'Proton therapy',
    32: 'Other charged-particle or neutron therapy',
    64: 'Boron neutron capture therapy (BNCT)',
}
RMOD_MAP = _additive_map(
    _RT_MODALITY_COMPONENTS,
    zero_label='No radiation therapy',
    unknown_label='Unknown whether radiation therapy was given',
    nos_label='Radiation therapy given, modality not specified',
    width=3)

# 放射治療與手術順序 (#4.2.1.5)
_SEQRS_COMPONENTS = {
    1: 'Pre-operative radiation therapy',
    2: 'Intra-operative radiation therapy (IORT)',
    4: 'Post-operative radiation therapy',
}
SEQRS_MAP = CodeMap({
    -9: 'Unknown whether the case had surgery and/or radiation therapy',
    -8: 'No surgery of the primary site or regional nodes (radiation therapy '
        'only, including radiation to a distant site)',
    -7: 'Not comparable: nodal lymphoma, a haematologic malignancy, distant '
        'metastasis, or a first course including surgery and radiation aimed '
        'at different sites',
    -6: 'More than two primary-site or regional-node surgeries with '
        'radiation therapy in between, so the sequence cannot be determined',
    -1: 'Sequence unknown: the first course had both surgery and radiation '
        'therapy, but the order is not documented',
    **{sum(c): ' + '.join(_SEQRS_COMPONENTS[x] for x in sorted(c))
       for r in (1, 2, 3) for c in __import__('itertools').combinations(
           _SEQRS_COMPONENTS, r)},
    0: 'No radiation therapy in the first course (with or without surgery)',
}, width=2)

# 區域治療與全身性治療順序 (#4.2.1.6)
_SEQLS_COMPONENTS = {
    1: 'Induction / neoadjuvant systemic therapy (before locoregional therapy, '
       'or before radiation when radiation is the main locoregional therapy)',
    2: 'Concurrent / concomitant systemic and radiation therapy (CSRT), or '
       'perioperative systemic therapy given around surgery',
    4: 'Adjuvant systemic therapy (after locoregional therapy)',
}
SEQLS_MAP = CodeMap({
    -9: 'Unknown whether the case had locoregional and/or systemic therapy',
    -8: 'No locoregional therapy: the first course had systemic drug therapy '
        'only (regional drug therapy such as TACE may still have been given)',
    -7: 'The first course had only regional drug therapy (TACE, '
        'intraperitoneal, intrapleural, intrathecal, intravesical, '
        'intraocular or intratumoral), with no systemic drug therapy',
    -1: 'Sequence unknown, or systemic therapy is the main treatment modality '
        '(nodal lymphoma, haematologic malignancy, or distant metastasis)',
    **{sum(c): ' + '.join(_SEQLS_COMPONENTS[x] for x in sorted(c))
       for r in (1, 2, 3) for c in __import__('itertools').combinations(
           _SEQLS_COMPONENTS, r)},
    0: 'No systemic drug therapy in the first course (chemotherapy, hormone/'
       'steroid, immunotherapy or targeted therapy), whether or not '
       'locoregional therapy was given',
}, width=2)


# ─────────────────────────────────────────────────────────────────────────────
# 放射治療執行狀態 RT Status (#4.2.1.8, pp.236-238)
#
# Best-evidence match for the 'R' column (documented as "radiation therapy
# performed, this hospital"): no field is numbered #4.2 in the manual, and
# this is the field that actually answers where/whether radiation was given.
# See docs/codebook_conformance_findings.md.
# ─────────────────────────────────────────────────────────────────────────────

RT_STATUS_MAP = CodeMap({
    0:  'Radiation therapy given in the first course, at the reporting '
        'hospital only',
    1:  'Radiation therapy was not part of the planned first course',
    2:  'Not advised or given because of a contraindication or another '
        'patient risk factor (comorbidity, advanced age)',
    3:  'Not advised or given because the disease progressed',
    4:  'Given at the reporting hospital but not completed, for a personal '
        'reason (comorbidity, poor performance status, side effects, death)',
    5:  'Part of the planned first course, but the patient died or was '
        'discharged critically ill before it started',
    6:  'Part of the planned first course, not given, with no reason '
        'recorded',
    7:  'Part of the planned first course, but the patient or family refused '
        'it',
    8:  'Part of the planned first course, but had not started when the case '
        'was abstracted',
    9:  'Radiation therapy given in the first course, at another hospital '
        'only',
    10: 'Radiation therapy given in the first course, at both the reporting '
        'hospital and another hospital',
    99: 'Not documented, so it is unknown whether radiation therapy was '
        'advised or given',
}, width=2)


# ─────────────────────────────────────────────────────────────────────────────
# 微創手術 Minimally Invasive Surgery (#4.1.4.1, pp.181-183)
# ─────────────────────────────────────────────────────────────────────────────

MINIMALLY_INVASIVE_MAP = CodeMap({
    0: 'Open surgery only; no minimally invasive or robotic-assisted surgery',
    1: 'Endoscopic surgery (entry through a natural body opening: e.g. '
       'gastroscopy, colonoscopy, bronchoscopy, hysteroscopy, colposcopy, '
       'cystoscopy)',
    2: 'Thoracoscopic, laparoscopic, or a similar minimally invasive surgery '
       '(percutaneous entry, or a natural opening reopened, into a body '
       'cavity)',
    3: 'Robotic-assisted surgery',
    4: 'Minimally invasive or robotic-assisted surgery combined with, or '
       'converted to, open surgery',
    8: 'Not applicable: no primary-site surgery; primary-site surgery coded '
       '100-190; prostate cancer primary-site surgery coded 210-270; primary-'
       'site surgery performed at another hospital; diagnosis year 2017 or '
       'earlier; or a haematopoietic/reticuloendothelial/immunoproliferative/'
       'myeloproliferative neoplasm',
    9: 'Not documented',
}, width=1)


LONGFORM_CODE_MAPS.update({
    'RTAR':  (RTAR_MAP, '4.2.1.1'),
    'RMOD':  (RMOD_MAP, '4.2.1.2'),
    'SEQRS': (SEQRS_MAP, '4.2.1.5'),
    'SEQLS': (SEQLS_MAP, '4.2.1.6'),
    'R':     (RT_STATUS_MAP, '4.2.1.8'),
    'HTAR':  (HTAR_MAP, '4.2.2.2.1'),
    'LTAR':  (LTAR_MAP, '4.2.2.3.1'),
    'MINS':  (MINIMALLY_INVASIVE_MAP, '4.1.4.1'),
})


# ─────────────────────────────────────────────────────────────────────────────
# 性別 Sex (#1.5, p.65)
# ─────────────────────────────────────────────────────────────────────────────

SEX_MAP = CodeMap({
    1: 'Male',
    2: 'Female',
    3: 'Other (e.g. intersex)',
    4: 'Transsexual',
    9: 'Unknown or not documented',
}, width=1)


# ─────────────────────────────────────────────────────────────────────────────
# 個案分類 Class of Case (#2.3, pp.28-29)
# 診斷狀態分類 Class of Diagnosis Status (#2.3.1, p.82)
# 治療狀態分類 Class of Treatment Status (#2.3.2, pp.83-88)
#
# 2.3 is a summary the registrar assigns; 2.3.1 and 2.3.2 are the two inputs
# the manual says must jointly determine it (p.73). All three are kept as
# separate tables because they answer different questions -- diagnosis status
# is "where/how was this found", treatment status is "what happened to the
# planned treatment", and class of case is the analysability verdict.
# ─────────────────────────────────────────────────────────────────────────────

CLASS_OF_CASE_MAP = CodeMap({
    0: 'Diagnosed at the reporting hospital, but the first course of therapy '
       'was not given there (the patient chose another hospital, was '
       'referred, refused treatment, or died/was discharged critically ill '
       'before treatment started)',
    1: 'Diagnosed at the reporting hospital, and all or part of the first '
       'course of therapy was given there',
    2: 'Diagnosed elsewhere, and all or part of the first course of therapy '
       'was given at the reporting hospital',
    3: 'Diagnosed elsewhere with no first-course therapy at the reporting '
       'hospital; presented here for recurrent or persistent disease',
    5: 'Diagnosed only at autopsy',
    7: 'Pathology report only; the patient was never seen at the reporting '
       'hospital for diagnosis or treatment (excludes autopsy diagnosis) -- '
       'not reportable',
    8: 'Known only from a death certificate (DCO) -- not reportable',
    9: 'Unknown; the record does not document enough to classify the case',
}, width=1)

CLASS_OF_DIAGNOSIS_MAP = CodeMap({
    1: 'Diagnosed at the reporting hospital',
    2: 'Diagnosed elsewhere; presented at the reporting hospital within the '
       'first course of therapy, with no recurrence, or recurrence status '
       'unknown',
    3: 'Diagnosed elsewhere; presented at the reporting hospital after '
       'recurrence or progression',
    5: 'Diagnosed only at autopsy',
    7: 'Pathology report only; not seen at the reporting hospital for '
       'diagnosis or treatment (excludes autopsy diagnosis)',
    8: 'Known only from a death certificate',
}, width=1)

CLASS_OF_TREATMENT_MAP = CodeMap({
    0: 'Died at the reporting hospital without receiving any treatment there '
       '(diagnosed here and died or discharged critically ill, or diagnosed '
       'elsewhere and transferred here before dying)',
    1: 'The entire first course of therapy was given at the reporting '
       'hospital, with none at another hospital (includes a case whose only '
       'first-course therapy is one of the registry\'s defined "other '
       'treatments")',
    2: 'No first-course therapy at the reporting hospital: the whole first '
       'course was given elsewhere, or the case came for a second opinion '
       'with no treatment here, or the main treatment plan was set by '
       'another hospital (e.g. maintenance therapy continued here, or a '
       'referred-out partial radiotherapy course)',
    3: 'Part of the first course of therapy at the reporting hospital, part '
       'at another hospital',
    4: 'The first course of therapy was watchful observation, or only '
       'non-tumour-directed palliative surgery, pain control, supportive '
       'care, or a hospice referral',
    5: 'The first course of therapy was alternative therapy only',
    6: 'The patient refused treatment in the first course',
    7: 'Diagnosed and treated elsewhere; presented at the reporting hospital '
       'because of a complication of the cancer or its treatment',
    8: 'Diagnosed elsewhere; presented at the reporting hospital for an '
       'unrelated condition -- not reportable',
    9: 'First-course treatment status unknown, and refusal cannot be '
       'confirmed either (e.g. lost to follow-up after the initial '
       'diagnosis at the reporting hospital)',
}, width=1)


# ─────────────────────────────────────────────────────────────────────────────
# 生存狀態 Vital Status (#5.4, p.234)
# ─────────────────────────────────────────────────────────────────────────────

VITAL_STATUS_MAP = CodeMap({
    0: 'Dead',
    1: 'Alive',
}, width=1)


# ─────────────────────────────────────────────────────────────────────────────
# 首次復發型式 Recurrence Type (#5.2, pp.317-319)
# ─────────────────────────────────────────────────────────────────────────────

RECURRENCE_TYPE_MAP = CodeMap({
    0:  'Disease-free after treatment; no recurrence',
    4:  'Recurrence of an invasive cancer, presenting as carcinoma in situ',
    6:  'Recurrence of a carcinoma in situ, presenting as carcinoma in situ',
    10: 'Local recurrence, not specific enough for 13-17 (in the residual '
        'primary organ, the primary organ, an anastomosis, or the scar of '
        'the resected organ)',
    13: 'Invasive cancer, local recurrence',
    14: 'Invasive cancer, recurrence at a trocar/port site (along the port '
        'track or a prior surgical entrance site)',
    15: 'Invasive cancer, local recurrence AND trocar/port-site recurrence '
        '(codes 13 and 14 combined)',
    16: 'Carcinoma in situ, local recurrence, not otherwise specified',
    17: 'Carcinoma in situ, local AND trocar/port-site recurrence',
    20: 'Regional recurrence, not specific enough for 21-27',
    21: 'Invasive cancer, recurrence in adjacent tissue or organs only',
    22: 'Invasive cancer, recurrence in regional lymph nodes only',
    25: 'Invasive cancer, recurrence in adjacent tissue/organs AND regional '
        'lymph nodes (codes 21 and 22 combined)',
    26: 'Carcinoma in situ, regional recurrence, not otherwise specified',
    27: 'Carcinoma in situ, recurrence in adjacent tissue/organs AND '
        'regional lymph nodes',
    30: 'Invasive cancer, one recurrence type from each of two groups: '
        '(adjacent tissue/organ or regional node recurrence, codes 20-25) '
        'combined with (local, prior port-site, or surgical-entrance-site '
        'recurrence, codes 10, 13-15)',
    36: 'Carcinoma in situ, one recurrence type from each of two groups: '
        '(regional recurrence, codes 26-27) combined with (local or '
        'port-site recurrence, codes 16-17)',
    40: 'Distant recurrence, not specific enough for 46-62',
    46: 'Carcinoma in situ, distant recurrence',
    51: 'Invasive cancer, distant recurrence in the peritoneum only, or '
        'malignant cells in ascites',
    52: 'Invasive cancer, distant recurrence in the lung only (including '
        'visceral pleura)',
    53: 'Invasive cancer, distant recurrence in the pleura only, or '
        'malignant cells in pleural effusion',
    54: 'Invasive cancer, distant recurrence in the liver only',
    55: 'Invasive cancer, distant recurrence in bone only, excluding bone at '
        'the primary site',
    56: 'Invasive cancer, distant recurrence in the CNS only (brain and '
        'spinal cord; excludes the external eye)',
    57: 'Invasive cancer, distant recurrence in the skin only, excluding '
        'skin at the primary site',
    58: 'Invasive cancer, distant recurrence in distant lymph nodes only '
        '(per each cancer site\'s own definition of distant nodes)',
    59: 'Invasive cancer, systemic distant recurrence only (lymphoma, '
        'leukemia, bone marrow metastasis, carcinomatosis, or generalised '
        'disease)',
    60: 'Invasive cancer, one recurrence type from each of two groups: '
        '(one or more distant-site recurrences) combined with (local or '
        'regional recurrence, codes 10-15, 20-25 or 30)',
    62: 'Invasive cancer, recurrence at multiple distant sites',
    70: 'Never disease-free since diagnosis (already had distant metastasis '
        'at diagnosis, systemic disease at diagnosis, an unknown primary, or '
        'disease too limited to have been treated)',
    88: 'Recurrence occurred, but the type is not documented',
    99: 'Unknown whether the case ever recurred or was ever disease-free '
        '(usually class of case 3)',
}, width=2)


# ─────────────────────────────────────────────────────────────────────────────
# 首次治療前生活功能狀態評估 Performance Status: KPS + ECOG (#7.6, pp.335-337)
#
# The manual states the composite rule explicitly (p.337): characters 1-2
# carry the KPS decile (00, 10, 20, ..., 100), character 3 the ECOG grade
# (0-5). When only ECOG was assessed, KPS is coded '00' -- but 000-004 never
# collide with a genuine KPS=0 combination, because KPS=0 only ever pairs
# with ECOG=5 (both mean death before treatment), which is code 005.
# ─────────────────────────────────────────────────────────────────────────────

PERFORMANCE_STATUS_MAP = CodeMap({
    0:   'ECOG PS 0 only (KPS not assessed): normal activity, unaffected by '
         'disease',
    1:   'ECOG PS 1 only (KPS not assessed): unable to do strenuous activity, '
         'but ambulatory and able to do light or sedentary work',
    2:   'ECOG PS 2 only (KPS not assessed): ambulatory and fully '
         'self-caring but unable to work, up to about half of waking hours '
         'out of bed',
    3:   'ECOG PS 3 only (KPS not assessed): limited self-care, confined to '
         'bed or chair more than half of waking hours',
    4:   'ECOG PS 4 only (KPS not assessed): completely disabled, no '
         'self-care, totally confined to bed or chair',
    5:   'Died before treatment (KPS=0 and/or ECOG PS=5)',
    100: 'KPS 100, ECOG PS 0: normal, no complaints, no evidence of disease',
    104: 'KPS 10, ECOG PS 4: moribund, fatal processes progressing rapidly',
    204: 'KPS 20, ECOG PS 4: very sick, urgent hospitalisation needed',
    209: 'KPS 20 (ECOG not assessed): very sick, urgent hospitalisation '
         'needed',
    303: 'KPS 30, ECOG PS 3: severely disabled, hospitalisation indicated, '
         'death not imminent',
    304: 'KPS 30, ECOG PS 4: severely disabled, hospitalisation indicated, '
         'death not imminent',
    309: 'KPS 30 (ECOG not assessed): severely disabled, hospitalisation '
         'indicated, death not imminent',
    403: 'KPS 40, ECOG PS 3: disabled, requires special care and assistance',
    409: 'KPS 40 (ECOG not assessed): disabled, requires special care and '
         'assistance',
    502: 'KPS 50, ECOG PS 2: requires considerable assistance and frequent '
         'medical care',
    503: 'KPS 50, ECOG PS 3: requires considerable assistance and frequent '
         'medical care',
    509: 'KPS 50 (ECOG not assessed): requires considerable assistance and '
         'frequent medical care',
    602: 'KPS 60, ECOG PS 2: requires occasional assistance but can care for '
         'most needs',
    609: 'KPS 60 (ECOG not assessed): requires occasional assistance but can '
         'care for most needs',
    701: 'KPS 70, ECOG PS 1: cares for self, unable to carry on normal '
         'activity',
    702: 'KPS 70, ECOG PS 2: cares for self, unable to carry on normal '
         'activity',
    709: 'KPS 70 (ECOG not assessed): cares for self, unable to carry on '
         'normal activity',
    801: 'KPS 80, ECOG PS 1: normal activity with effort, some disease '
         'symptoms',
    809: 'KPS 80 (ECOG not assessed): normal activity with effort, some '
         'disease symptoms',
    900: 'KPS 90, ECOG PS 0: able to carry on normal activity, minor disease '
         'symptoms',
    901: 'KPS 90, ECOG PS 1: able to carry on normal activity, minor disease '
         'symptoms',
    909: 'KPS 90 (ECOG not assessed): able to carry on normal activity, '
         'minor disease symptoms',
    988: 'Not applicable: the first course of therapy was given entirely at '
         'another hospital, and a pre-treatment functional-status assessment '
         'could not be obtained',
    999: 'Not documented, or unknown',
}, width=3)


LONGFORM_CODE_MAPS.update({
    'SEX':          (SEX_MAP, '1.5'),
    'CLASS95':      (CLASS_OF_CASE_MAP, '2.3'),
    'CLASSOFDIAG':  (CLASS_OF_DIAGNOSIS_MAP, '2.3.1'),
    'CLASSOFTREAT': (CLASS_OF_TREATMENT_MAP, '2.3.2'),
    'VSTA':         (VITAL_STATUS_MAP, '5.4'),
    'RETYPE95':     (RECURRENCE_TYPE_MAP, '5.2'),
    'KPSECOG':      (PERFORMANCE_STATUS_MAP, '7.6'),
})
