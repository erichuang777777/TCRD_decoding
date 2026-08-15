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
from tcr_decoder.decoders import decode_lnexam, decode_lnpositive
from tcr_decoder.encoders import encode_lnexam, encode_lnpositive
from tcr_decoder.ssf_registry import _generic_ssf, get_ssf_profile
from tcr_decoder.encoders import encode_generic_ssf

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
    assert all(len(e) == width for e in encoded), f'{field} ({ref}): wrong width'


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
