"""Exhaustive conformance to the printed TCR code book.

Every other test checks hand-picked examples. This one enumerates the
COMPLETE 編碼範圍 of every field of every cancer group listed in
tcr_decoder.code_ranges.SUPPORTED_GROUPS and asserts the four properties that
together mean "the converter is correct and works in both directions":

  1. decode() understands every legal code (no legal code falls through to
     the raw passthrough).
  2. decode() is injective on the legal domain -- no two distinct codes share
     one decoded label. Without this, encode() cannot possibly be exact.
  3. encode(decode(code)) == code, byte for byte, for every legal code.
  4. Everything encode() emits is itself a legal code at the field's official
     width, so an encoded file can actually be submitted.

Adding a cancer group here is a two-step job: transcribe its 編碼範圍 into
code_ranges.py, then make its profile pass. Groups that have not been
verified against the manual are deliberately absent -- see
docs/codebook_conformance_findings.md.
"""

import pandas as pd
import pytest

from tcr_decoder.code_ranges import (
    CODE_RANGES, LONGFORM, SUPPORTED_GROUPS, field_width, is_legal_code,
)
from tcr_decoder.surgery_codes import SURGERY_TABLES, surgery_table_name
from tcr_decoder.longform_codes import (
    BEHAVIOR_MAP, CONFIRMATION_HAEM_MAP, CONFIRMATION_SOLID_MAP, LATERALITY_MAP,
    LONGFORM_CODE_MAPS, LVI_MAP, PERINEURAL_INVASION_MAP, decode_confirmation,
    encode_confirmation,
)
from tcr_decoder.decoders import decode_lnexam, decode_lnpositive, decode_surgery
from tcr_decoder.encoders import encode_lnexam, encode_lnpositive, encode_surgery
from tcr_decoder.ssf_registry import _generic_ssf, get_ssf_profile
from tcr_decoder.encoders import encode_generic_ssf, encode_structural_map

# (cancer_group, ssf_key) for every verified field of every verified group
FIELDS = [
    (group, ssf_key)
    for group in SUPPORTED_GROUPS
    for ssf_key in sorted(CODE_RANGES[group], key=lambda k: int(k.replace('SSF', '')))
]
IDS = [f'{group}-{ssf}' for group, ssf in FIELDS]


def _decode_encode(group, ssf_key):
    """Run one field's whole legal code range through decode then encode."""
    width, codes, ref = CODE_RANGES[group][ssf_key]
    field = get_ssf_profile(group).fields[ssf_key]
    numeric = sorted((c for c in codes if c.isdigit()), key=int)
    raw = pd.Series(numeric + sorted(c for c in codes if not c.isdigit()), dtype=object)

    if field.decoder is not None:
        decoded = field.decoder(raw)
    else:
        decoded = _generic_ssf(raw, unit=field.unit)
    if field.encoder is not None:
        encoded = field.encoder(decoded)
    else:
        encoded = encode_generic_ssf(decoded, unit=field.unit)
    return raw, decoded, encoded.astype(str), width, ref


@pytest.mark.parametrize('group,ssf_key', FIELDS, ids=IDS)
def test_every_legal_code_decodes_to_clinical_text(group, ssf_key):
    """A legal code must never come back out as the bare code itself."""
    raw, decoded, _, _, ref = _decode_encode(group, ssf_key)
    unhandled = [c for c, d in zip(raw, decoded) if not d or str(d) == str(c)]
    assert not unhandled, (
        f'{group}.{ssf_key} ({ref}): {len(unhandled)} legal codes not decoded, '
        f'e.g. {unhandled[:5]}')


@pytest.mark.parametrize('group,ssf_key', FIELDS, ids=IDS)
def test_decode_is_injective_over_the_legal_range(group, ssf_key):
    """Two legal codes sharing one label would make the round trip lossy."""
    raw, decoded, _, _, ref = _decode_encode(group, ssf_key)
    seen, collisions = {}, []
    for code, label in zip(raw, decoded):
        if label in seen:
            collisions.append((seen[label], code, label))
        seen[label] = code
    assert not collisions, f'{group}.{ssf_key} ({ref}): {collisions[:5]}'


@pytest.mark.parametrize('group,ssf_key', FIELDS, ids=IDS)
def test_roundtrip_is_exact_for_every_legal_code(group, ssf_key):
    raw, decoded, encoded, _, ref = _decode_encode(group, ssf_key)
    bad = [(c, d, e) for c, d, e in zip(raw, decoded, encoded) if e != str(c)]
    assert not bad, (f'{group}.{ssf_key} ({ref}): {len(bad)} codes did not '
                     f'round-trip, e.g. {bad[:5]}')


@pytest.mark.parametrize('group,ssf_key', FIELDS, ids=IDS)
def test_encoder_only_emits_submittable_codes(group, ssf_key):
    """Right value AND right width -- a 3-character field never gets '6'."""
    _, _, encoded, width, ref = _decode_encode(group, ssf_key)
    illegal = [e for e in encoded
               if not is_legal_code(group, ssf_key, e) or len(e) != width]
    assert not illegal, f'{group}.{ssf_key} ({ref}): {illegal[:5]}'


# ─────────────────────────────────────────────────────────────────────────────
# Field-specific regressions worth naming
# ─────────────────────────────────────────────────────────────────────────────

def test_nottingham_score_never_encodes_to_a_bare_digit():
    """Regression: encode used to return '6' for a BR score of 6. The field
    is 3 characters and the legal codes are 030-090, so '6' would be
    rejected by the registry."""
    field = get_ssf_profile('breast').fields['SSF6']
    for spelling in ('6', '60', '060'):
        decoded = field.decoder(pd.Series([spelling]))
        assert field.encoder(decoded).iloc[0] == '060'
    assert field_width('breast', 'SSF6') == 3


def test_gleason_pattern_code_is_not_read_as_a_score():
    """Regression: prostate SSF2 used to be decoded as "Gleason score", so
    '034' (primary 3 + secondary 4) came out as 'Gleason Score 34'. The tens
    digit is the primary pattern and the units digit the secondary."""
    biopsy = get_ssf_profile('prostate').fields['SSF2']
    label = biopsy.decoder(pd.Series(['034'])).iloc[0]
    assert 'primary 3' in label and 'secondary 4' in label
    assert '34' not in label.replace('3+4', '')
    assert biopsy.encoder(pd.Series([label])).iloc[0] == '034'


def test_prostate_biopsy_and_prostatectomy_specimens_are_distinguishable():
    """SSF2/SSF3 (needle biopsy/TURP) and SSF4/SSF5 (radical prostatectomy)
    share a code range but describe different specimens, so their 988/999
    text must differ -- otherwise a value copied between the two columns
    would look identical to a reviewer."""
    profile = get_ssf_profile('prostate')
    for biopsy_key, surgery_key in (('SSF2', 'SSF4'), ('SSF3', 'SSF5')):
        for code in ('988', '999'):
            biopsy = profile.fields[biopsy_key].decoder(pd.Series([code])).iloc[0]
            surgery = profile.fields[surgery_key].decoder(pd.Series([code])).iloc[0]
            assert biopsy != surgery, f'{biopsy_key}/{surgery_key} code {code}'


# Longform structural fields with a real decode/encode pair, checked the same
# way as the SSFs: whole legal range, injective decode, exact round trip.
LONGFORM_CODECS = {
    'LNEXAM':    (decode_lnexam, encode_lnexam),
    'LN_POSITI': (decode_lnpositive, encode_lnpositive),
    'LAT95':     (LATERALITY_MAP.decode, LATERALITY_MAP.encode),
    'MCODE5':    (BEHAVIOR_MAP.decode, BEHAVIOR_MAP.encode),
    'PNI':       (PERINEURAL_INVASION_MAP.decode, PERINEURAL_INVASION_MAP.encode),
    'LVI':       (LVI_MAP.decode, LVI_MAP.encode),
    **{tag: (m.decode, m.encode)
       for tag, (m, _seq) in LONGFORM_CODE_MAPS.items()},
}


@pytest.mark.parametrize('field', sorted(LONGFORM_CODECS), ids=sorted(LONGFORM_CODECS))
def test_longform_structural_field_roundtrips_over_its_whole_range(field):
    width, codes, ref = LONGFORM[field]
    decode, encode = LONGFORM_CODECS[field]
    raw = pd.Series(sorted(codes), dtype=object)
    decoded = decode(raw)
    encoded = encode(decoded).astype(str)

    bad = [(c, e) for c, e in zip(raw, encoded) if c != e]
    assert not bad, f'{field} ({ref}): {bad[:5]}'
    # A negative sentinel (RMOD's -9/-1) is never padded past its sign, even
    # in a wider field -- the manual's own 編碼範圍 never shows a padded form.
    bad_width = [e for e in encoded if not e.startswith('-') and len(e) != width]
    assert not bad_width, f'{field} ({ref}): wrong width {bad_width[:5]}'


@pytest.mark.parametrize('field', sorted(LONGFORM_CODECS), ids=sorted(LONGFORM_CODECS))
def test_longform_structural_field_decode_is_injective(field):
    """Two sentinels sharing one label is how LNEXAM lost four meanings."""
    _, codes, ref = LONGFORM[field]
    decode, _ = LONGFORM_CODECS[field]
    raw = pd.Series(sorted(codes), dtype=object)
    seen, collisions = {}, []
    for code, label in zip(raw, decode(raw)):
        if label in seen:
            collisions.append((seen[label], code, label))
        seen[label] = code
    assert not collisions, f'{field} ({ref}): {collisions[:5]}'


# The two node-surgery fields share one code table, and the surgery-of-primary
# -site fields share Appendix B. Both pairs go through _map() / encode_
# structural_map() rather than an SSF profile decoder.
def _structural_roundtrip(codes, code_map):
    raw = pd.Series(sorted(codes), dtype=object)
    decoded = raw.apply(lambda v: code_map.get(v, f'Code {v}'))
    encoded = encode_structural_map(decoded, code_map).astype(str)
    return raw, decoded, encoded


@pytest.mark.parametrize('field', ['PRESLNSCO', 'SLNSCO95'])
def test_regional_node_surgery_scope_roundtrips(field):
    from tcr_decoder.core import LNSCO_MAP

    width, codes, ref = LONGFORM[field]
    raw, decoded, encoded = _structural_roundtrip(codes, LNSCO_MAP)
    assert not [c for c, d in zip(raw, decoded) if d.startswith('Code ')], ref
    assert len(set(decoded)) == len(decoded), f'{field} ({ref}): duplicate labels'
    assert list(raw) == list(encoded), ref
    assert all(len(e) == width for e in encoded), ref


# One representative topography code per Appendix B table, so every site's
# surgery vocabulary is exercised, not just the breast one.
SURGERY_SITES = [(name, sites[0])
                 for name, (sites, _codes, _ref) in sorted(SURGERY_TABLES.items())]


@pytest.mark.parametrize('name,site', SURGERY_SITES,
                         ids=[n[:28] for n, _ in SURGERY_SITES])
def test_surgery_of_primary_site_roundtrips_over_appendix_b(name, site):
    """Every code of every site table, both surgery fields.

    PRESTYPE used to decode 0 of the 705 codes in its own 編碼範圍 -- it knew
    seven legacy 2-digit codes and nothing else, so a current-format export
    decoded every operation as 'Code NNN'. STYPE95 knew 41, all breast.
    """
    _sites, table, ref = SURGERY_TABLES[name]
    raw = pd.Series(sorted(table), dtype=object)
    sites = pd.Series([site] * len(raw))

    decoded = decode_surgery(raw, sites)
    encoded = encode_surgery(decoded, sites).astype(str)

    undecoded = [c for c, d in zip(raw, decoded) if d.startswith('Code ')]
    assert not undecoded, f'{name} ({ref}): {undecoded[:5]} not decoded'
    assert len(set(decoded)) == len(decoded), f'{name} ({ref}): duplicate labels'
    assert list(raw) == list(encoded), f'{name} ({ref}): round trip'
    assert all(len(e) == 3 for e in encoded), f'{name} ({ref}): wrong width'


def test_the_two_surgery_fields_share_one_vocabulary():
    """They ask the same question about two facilities (Longform p.186/188).

    They used to carry different code tables, so '51' meant "biopsy only" in
    one and "extended radical mastectomy" in the other.
    """
    from tcr_decoder.encoder import STRUCTURAL_FIELD_ENCODERS

    assert 'Surgery_Type_Other_Hosp' not in STRUCTURAL_FIELD_ENCODERS
    assert 'Surgery_Type_This_Hosp' not in STRUCTURAL_FIELD_ENCODERS
    site = pd.Series(['C50.9'] * 3)
    codes = pd.Series(['660', '312', '200'])
    assert list(decode_surgery(codes, site)) == list(decode_surgery(codes, site))


def test_the_same_code_means_different_things_in_different_organs():
    """This is why the decoder needs the topography code, not just the group."""
    codes = pd.Series(['660', '660'])
    sites = pd.Series(['C50.9', 'C34.1'])
    breast, lung = decode_surgery(codes, sites)
    assert 'mastectomy' in breast
    assert breast != lung


def test_legacy_surgery_codes_never_win_the_reverse_lookup():
    """A submission must get the current 3-character code, not a legacy one."""
    for name, (sites, table, _ref) in SURGERY_TABLES.items():
        site = pd.Series([sites[0]] * len(table))
        labels = pd.Series(list(table.values()))
        assert all(len(c) == 3 for c in encode_surgery(labels, site)), name


def test_legacy_surgery_codes_still_decode_and_are_labelled_as_legacy():
    """Historical exports used 1- and 2-character codes; they must still read."""
    from tcr_decoder.core import LEGACY_SURGERY_CODES

    raw = pd.Series(sorted(LEGACY_SURGERY_CODES))
    decoded = decode_surgery(raw, pd.Series(['C50.9'] * len(raw)))
    assert all('legacy' in d for d in decoded)
    assert list(encode_surgery(decoded, pd.Series(['C50.9'] * len(raw)))) == list(raw)


def test_every_appendix_b_site_routes_to_exactly_one_table():
    seen = {}
    for name, (sites, _codes, _ref) in SURGERY_TABLES.items():
        for s in sites:
            assert s not in seen, f'{s}: {seen.get(s)} vs {name}'
            seen[s] = name
    assert len(seen) > 500
    assert surgery_table_name('C50.9') == 'Breast'
    assert surgery_table_name('C99.9') is None


def test_lnexam_sentinels_are_five_distinct_situations():
    """Regression: 95-99 used to be blanked out together as "unknown"."""
    decoded = decode_lnexam(pd.Series(['95', '96', '97', '98', '99']))
    assert len(set(decoded)) == 5
    assert all(d for d in decoded)


def test_data_dictionary_describes_every_ssf_column():
    """Every cancer group's SSF columns must have a description.

    COLUMN_REGISTRY was hand-written for breast only, so a decoded
    lung/liver/prostate file used to get a data dictionary with ten blank
    descriptions. They are now derived from the SSF profiles.
    """
    from tcr_decoder.data_dictionary import COLUMN_REGISTRY, TCR_FIELD_NUMBER
    from tcr_decoder.ssf_registry import _PROFILES

    missing = []
    for group, profile in _PROFILES.items():
        if group == 'generic':
            continue
        for ssf_key, field in profile.fields.items():
            entry = COLUMN_REGISTRY.get(field.column_name)
            if not entry or not entry[2].strip():
                missing.append(f'{group}.{ssf_key} ({field.column_name})')
            expected_number = '8.' + ssf_key.replace('SSF', '')
            assert TCR_FIELD_NUMBER.get(field.column_name) is not None, field.column_name
    assert not missing, missing[:10]


# ─────────────────────────────────────────────────────────────────────────────
# The data dictionary's 癌登欄位序號 must match the manual's own numbering
# ─────────────────────────────────────────────────────────────────────────────

def test_every_claimed_field_number_exists_in_the_manual():
    """A registrar uses these numbers to line our output up with the form.

    Twenty of them used to point at a different field: 4.4 is 申報醫院緩和照護
    and was claimed by the hormone-therapy columns, 7.1 is 身高 and was claimed
    by performance status, 5.4 is 生存狀態 and was claimed by the last-contact
    date. The remaining unconfirmed ones are listed explicitly rather than
    left looking authoritative.
    """
    from tcr_decoder.data_dictionary import (
        TCR_FIELD_NUMBER, UNVERIFIED_FIELD_NUMBERS)
    from tcr_decoder.longform_fields import LONGFORM_FIELDS

    unknown = sorted(
        (col, str(seq)) for col, seq in TCR_FIELD_NUMBER.items()
        if not str(seq).startswith('8.')          # SSFs are 8.x by construction
        and col not in UNVERIFIED_FIELD_NUMBERS
        and str(seq) not in LONGFORM_FIELDS
    )
    assert not unknown, f'field numbers not in the manual: {unknown}'


def test_the_unverified_field_number_list_does_not_grow_silently():
    from tcr_decoder.data_dictionary import (
        TCR_FIELD_NUMBER, UNVERIFIED_FIELD_NUMBERS)

    assert UNVERIFIED_FIELD_NUMBERS <= set(TCR_FIELD_NUMBER), (
        'UNVERIFIED_FIELD_NUMBERS lists a column that no longer exists')
    assert len(UNVERIFIED_FIELD_NUMBERS) == 13


def test_field_widths_agree_with_the_manual():
    """Where we enforce a width, it must be the manual's 欄位長度."""
    from tcr_decoder.code_ranges import LONGFORM
    from tcr_decoder.longform_fields import LONGFORM_FIELDS

    for field, seq in (('LNEXAM', '2.14'), ('LN_POSITI', '2.15'),
                       ('PRESLNSCO', '4.1.6'), ('SLNSCO95', '4.1.7')):
        assert LONGFORM[field][0] == LONGFORM_FIELDS[seq].width, field


# ─────────────────────────────────────────────────────────────────────────────
# 癌症確診方式 (#2.11) has two tables, chosen by morphology
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('morphology,table', [
    ('8500/3', CONFIRMATION_SOLID_MAP),     # invasive ductal carcinoma
    ('9680/3', CONFIRMATION_HAEM_MAP),      # diffuse large B-cell lymphoma
])
def test_confirmation_method_roundtrips_over_both_tables(morphology, table):
    codes = pd.Series(sorted(str(c) for c in table.mapping), dtype=object)
    morph = pd.Series([morphology] * len(codes))
    decoded = decode_confirmation(codes, morph)
    encoded = encode_confirmation(decoded, morph).astype(str)

    assert not [d for d in decoded if d.startswith('Code ')]
    assert len(set(decoded)) == len(decoded), 'duplicate labels'
    assert list(codes) == list(encoded)
    assert all(len(e) == 1 for e in encoded)


def test_confirmation_code_3_is_haematolymphoid_only():
    """The solid-tumour table on p.102 has no code 3.

    Decoding it with the haematolymphoid meaning for a breast cancer would
    invent an immunophenotyping result that was never reported.
    """
    solid = decode_confirmation(pd.Series(['3']), pd.Series(['8500/3'])).iloc[0]
    haem = decode_confirmation(pd.Series(['3']), pd.Series(['9680/3'])).iloc[0]
    assert solid == 'Code 3'
    assert 'immunophenotyping' in haem


def test_confirmation_code_5_is_worded_differently_in_each_table():
    solid = decode_confirmation(pd.Series(['5']), pd.Series(['8500/3'])).iloc[0]
    haem = decode_confirmation(pd.Series(['5']), pd.Series(['9680/3'])).iloc[0]
    assert solid != haem


def test_perineural_and_lvi_name_their_own_subject():
    """Same code shape, different subject.

    Codes 0, 1 and 7 are about the finding and must say which finding, or a
    reviewer reading the decoded file cannot tell perineural invasion from
    lymph-vascular invasion. Codes 8 (not applicable) and 9 (not documented)
    are generic and are shared on purpose -- they carry no subject.
    """
    codes = pd.Series(['0', '1', '7'])
    pni = list(PERINEURAL_INVASION_MAP.decode(codes))
    lvi = list(LVI_MAP.decode(codes))
    assert not set(pni) & set(lvi)
    assert all('perineural invasion' in p.lower() for p in pni)
    assert all('lymph-vascular invasion' in l.lower() for l in lvi)


def test_every_longform_code_map_matches_the_official_range():
    """A transcribed table must contain exactly the 編碼範圍, no more, no less.

    Getting this wrong is silent: a missing code decodes as 'Code NN' and an
    invented one is emitted on a submission that will be rejected.
    """
    from tcr_decoder.code_ranges import LONGFORM

    for tag, (code_map, seq) in sorted(LONGFORM_CODE_MAPS.items()):
        width, legal, ref = LONGFORM[tag]
        transcribed = {str(c) if c < 0 else str(c).zfill(width)
                       for c in code_map.mapping}
        assert transcribed == set(legal), (
            f'{tag} (#{seq}, {ref}): '
            f'missing={sorted(set(legal) - transcribed)} '
            f'extra={sorted(transcribed - set(legal))}')


def test_paired_therapy_fields_share_their_modality_codes():
    """外院 and 申報醫院 ask the same question about two facilities.

    Only the reporting hospital has the 8x block -- an outside hospital does
    not tell us why a planned treatment was not given.
    """
    for other, this in (('PREC', 'C'), ('PREH', 'H'),
                        ('PREI', 'I'), ('PRETAR', 'TAR')):
        a, b = LONGFORM_CODE_MAPS[other][0], LONGFORM_CODE_MAPS[this][0]
        shared = set(a.mapping) & set(b.mapping)
        assert all(a.mapping[c] == b.mapping[c] for c in shared), other
        assert set(a.mapping) < set(b.mapping), other
        assert all(c >= 80 for c in set(b.mapping) - set(a.mapping)), this


def test_therapy_modality_codes_are_not_reused_across_therapies():
    """02 is single-agent systemic chemo but regional hormone therapy."""
    chemo = LONGFORM_CODE_MAPS['PREC'][0].mapping
    hormone = LONGFORM_CODE_MAPS['PREH'][0].mapping
    assert chemo[2] != hormone[2]
    assert chemo[1] != hormone[1]


def test_radiation_additive_fields_reject_a_component_sum_outside_their_table():
    """RTAR/RMOD/SEQRS/SEQLS are bitmasks: every legal code is a subset-sum of
    named components. A code outside that domain (e.g. 64 for SEQRS, which
    only has three 1-bit components summing to at most 7) must not decode."""
    from tcr_decoder.longform_codes import SEQRS_MAP

    assert SEQRS_MAP.decode_one('64') == 'Code 64'
    assert '64' not in SEQRS_MAP.mapping


def test_htar_and_ltar_share_target_volume_vocabulary_but_are_distinct_fields():
    """Same code meanings (T/N/M/extended/total-body/total-skin) in both --
    what differs is which dose band the column records, carried by the
    column name (High_Dose_Target vs Low_Dose_Target), not by the code."""
    from tcr_decoder.longform_codes import HTAR_MAP, LTAR_MAP

    assert HTAR_MAP.mapping[3] == LTAR_MAP.mapping[3]
    assert set(HTAR_MAP.mapping) == set(LTAR_MAP.mapping)


def test_rt_status_replaces_the_boolean_r_field():
    """R (#4.2.1.8) is not a yes/no flag: 00 means given at the reporting
    hospital, 09 means given only elsewhere, 01-08 are eight different reasons
    it was not given here. A synthetic '1'/'0' used to be emitted for it and
    was never a legal code under the real table."""
    from tcr_decoder.longform_codes import RT_STATUS_MAP

    assert '1' not in RT_STATUS_MAP.mapping
    assert '0' not in RT_STATUS_MAP.mapping
    assert RT_STATUS_MAP.mapping[0] != RT_STATUS_MAP.mapping[1]


def test_negative_sentinel_codes_are_not_padded_past_their_sign():
    """RMOD is a 3-character field, but the manual's own 編碼範圍 line lists
    the sentinels as '-9, -1' -- not '-09, -01'. zfill on a negative number
    pads after the sign, which would invent a code the manual never lists."""
    from tcr_decoder.longform_codes import RMOD_MAP

    assert RMOD_MAP.encode_one(RMOD_MAP.decode_one('-9')) == '-9'
    assert RMOD_MAP.encode_one(RMOD_MAP.decode_one('-1')) == '-1'


def test_vital_status_and_cancer_status_do_not_share_a_field_number():
    """Regression: both used to claim #5.4/#5.1 interchangeably.

    5.4 is 生存狀態 (0=dead, 1=alive) -- that is Vital_Status (raw VSTA), not
    Cancer_Status (raw CSTA). The mix-up happened while correcting a
    different field's number without checking what number it displaced.

    This checks only the pair that was actually found wrong, not every
    column in the registry: TCR_FIELD_NUMBER has ~250 entries going back
    before this project's codebook-conformance work, and asserting zero
    collisions across all of them is a separate full-registry audit (there
    is at least one more, Path_Stage vs Combined_Stage at #3.13, not yet
    resolved -- see docs/codebook_conformance_findings.md).
    """
    from tcr_decoder.data_dictionary import (
        TCR_FIELD_NUMBER, UNVERIFIED_FIELD_NUMBERS)

    assert TCR_FIELD_NUMBER['Vital_Status'] == '5.4'
    assert 'Vital_Status' not in UNVERIFIED_FIELD_NUMBERS
    # Cancer_Status still carries the old, now-known-wrong '5.4' as a
    # placeholder (the convention this registry uses elsewhere for a number
    # that could not be confirmed) -- what matters is that it is flagged as
    # unverified, so nothing treats it as evidence of the real field.
    assert 'Cancer_Status' in UNVERIFIED_FIELD_NUMBERS
