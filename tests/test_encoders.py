"""
Tests for the encode direction: CodeMap, bespoke encoders (tcr_decoder.encoders),
SSF profile encoding (apply_ssf_encode_profile), TCREncoder, and compare_roundtrip.

All tests exercise decode() -> encode() round trips: for a lossless field,
encode(decode(x)) must recover x exactly. Where the TCR codebook itself
renders two different raw codes with identical decoded text (a handful of
documented cases -- see KNOWN_LABEL_COLLISIONS below), encode() is only
required to return SOME code that decodes back to the same label, not
necessarily the original one. That's not a bug in this package; it's an
ambiguity already baked into the codebook, and every test here treats it as
a known, allowlisted exception rather than silently overlooking it.
"""

import re

import pandas as pd
import pytest

from tcr_decoder.codemap import CodeMap
from tcr_decoder.decoders import (
    decode_er_pr, decode_ki67, decode_her2, decode_nottingham, decode_ssf3_neoadj,
    decode_ebrt_additive, decode_sentinel, decode_lnpositive,
)
from tcr_decoder.encoders import (
    encode_er_pr, encode_ki67, encode_her2, encode_nottingham, encode_ssf3_neoadj,
    encode_ebrt_additive, encode_sentinel, encode_lnpositive,
    encode_lung_ssf5_mediastinal, encode_lung_egfr, encode_lung_ssf9_nodules,
    encode_cea_lab_value, encode_ras_mutation, encode_liver_afp,
    encode_lab_value_10x, encode_liver_inr, encode_psa, encode_generic_ssf,
    encode_structural_map, batch_encode,
)
from tcr_decoder.ssf_registry import (
    _decode_lung_ssf5_mediastinal, _decode_lung_egfr, _decode_lung_ssf9_nodules,
    _decode_cea_lab_value, _decode_ras_mutation, _decode_liver_afp,
    _decode_lab_value_10x, _decode_liver_inr, _decode_psa, _generic_ssf,
    apply_ssf_profile, apply_ssf_encode_profile, get_ssf_profile,
    list_supported_cancers,
)
from tcr_decoder import TCRDecoder, TCREncoder
from tcr_decoder.roundtrip import compare_roundtrip


def _roundtrip(codes, decode_fn, encode_fn):
    s = pd.Series([str(c) for c in codes])
    decoded = decode_fn(s)
    return decoded, encode_fn(decoded)


# ─────────────────────────────────────────────────────────────────────────────
# CodeMap primitive
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeMap:
    def test_decode_matches_map_decode_semantics(self):
        cm = CodeMap({0: 'Negative', 1: 'Positive', 999: 'Unknown'})
        s = pd.Series(['0', '1', '999', '2', '', 'nan', None])
        decoded = cm.decode(s)
        assert list(decoded) == ['Negative', 'Positive', 'Unknown', 'Code 2', '', '', '']

    def test_encode_recovers_original_code(self):
        cm = CodeMap({0: 'Negative', 1: 'Positive', 999: 'Unknown'})
        s = pd.Series(['Negative', 'Positive', 'Unknown'])
        assert list(cm.encode(s)) == ['0', '1', '999']

    def test_encode_recovers_fallback_prefixed_code(self):
        cm = CodeMap({0: 'Negative'}, fallback='Code')
        assert cm.encode_one('Code 42') == '42'

    def test_encode_blank_for_empty(self):
        cm = CodeMap({0: 'Negative'})
        assert cm.encode(pd.Series(['', None, float('nan')])).tolist() == ['', '', '']

    def test_encode_raises_on_unrecognized_label(self):
        cm = CodeMap({0: 'Negative'})
        with pytest.raises(KeyError):
            cm.encode_one('some made up label')

    def test_encode_on_error_empty_blanks_instead_of_raising(self):
        cm = CodeMap({0: 'Negative'})
        result = cm.encode(pd.Series(['Negative', 'bogus']), on_error='empty')
        assert list(result) == ['0', '']

    def test_duplicate_label_canonicalizes_to_first_listed_code(self):
        # Mirrors the TCR codebook's own 888-vs-988 "Not applicable" pattern.
        cm = CodeMap({888: 'Not applicable', 988: 'Not applicable'})
        assert cm.encode_one('Not applicable') == '888'


# ─────────────────────────────────────────────────────────────────────────────
# Bespoke encoders -- breast
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodeErPr:
    @pytest.mark.parametrize('code', [0, 1, 50, 100, 110, 120, 121, 888, 988, 999])
    def test_roundtrip(self, code):
        """SSF1/SSF2 are 3-character fields, so encode always writes the
        official zero-padded code (1% -> '001', not '1')."""
        dec, enc = _roundtrip([code], lambda s: decode_er_pr(s, 'ER'),
                              lambda s: encode_er_pr(s, 'ER'))
        assert enc.iloc[0] == str(code).zfill(3)

    # Valid 3-char staining codes per the Cancer-SSF-Manual (breast SSF1,
    # p.121): intensity letter + 2-digit proportion, 100% stored as '00'.
    @pytest.mark.parametrize('code', ['W15', 'I50', 'S70', 'W01', 'S99', 'S00'])
    def test_roundtrip_staining_codes(self, code):
        dec, enc = _roundtrip([code], lambda s: decode_er_pr(s, 'ER'),
                              lambda s: encode_er_pr(s, 'ER'))
        assert enc.iloc[0] == code

    def test_staining_00_means_100_percent(self):
        """Manual (breast SSF1, p.121): staining proportion '00' == 100%.
        decode must read S00/W00/I00 as 100% (not 0%, which would misread a
        strongly-positive tumour as negative), and encode must round-trip
        100% staining back to the canonical '00' code."""
        for letter, intensity in [('S', 'Strong'), ('W', 'Weak'), ('I', 'Intermediate')]:
            dec = decode_er_pr(pd.Series([f'{letter}00']), 'ER')
            assert dec.iloc[0] == f'ER Positive ({intensity} staining, 100%)'
            enc = encode_er_pr(dec, 'ER')
            assert enc.iloc[0] == f'{letter}00'

    def test_staining_single_digit_percent_zero_padded(self):
        """5% weak staining round-trips to the canonical 'W05' (2-digit
        proportion), matching the manual's 'W01' for 1%."""
        enc = encode_er_pr(pd.Series(['ER Positive (Weak staining, 5%)']), 'ER')
        assert enc.iloc[0] == 'W05'
        assert decode_er_pr(pd.Series(['W05']), 'ER').iloc[0] == 'ER Positive (Weak staining, 5%)'

    def test_out_of_range_percent_passes_through(self):
        """decode_er_pr() itself falls back to bare passthrough for n>100;
        encode must mirror that, not raise."""
        dec, enc = _roundtrip([150], lambda s: decode_er_pr(s, 'ER'),
                              lambda s: encode_er_pr(s, 'ER'))
        assert dec.iloc[0] == '150'
        assert enc.iloc[0] == '150'

    def test_out_of_range_staining_percent_not_silently_wrapped(self):
        """Regression: decode_er_pr()'s staining regex has no upper bound on
        the digit run, so an already-malformed 'S150' decodes leniently to
        "Strong staining, 150%". encode_er_pr used to compute `pct % 100`
        when reconstructing the code, which silently WRAPPED this into a
        different, wrong value ('S150' -> 'S50') instead of preserving it --
        found during code review. It must now round-trip exactly."""
        dec, enc = _roundtrip(['S150'], lambda s: decode_er_pr(s, 'ER'),
                              lambda s: encode_er_pr(s, 'ER'))
        assert dec.iloc[0] == 'ER Positive (Strong staining, 150%)'
        assert enc.iloc[0] == 'S150'

    def test_pr_uses_pr_vocabulary(self):
        dec, enc = _roundtrip([70], lambda s: decode_er_pr(s, 'PR'),
                              lambda s: encode_er_pr(s, 'PR'))
        assert 'PR' in dec.iloc[0]
        assert enc.iloc[0] == '070'

    @pytest.mark.parametrize('code,expected_label', [
        # Allred score (Cancer-SSF-Manual, breast SSF1, p.123): first digit/
        # letter is intensity (0/W/I/S), remaining 2 digits are a proportion
        # SCORE (0-5) whose code happens to equal the mean of that score's
        # positive-cell-% range (00->0%, 06->~6%, 22->~22%, 49->~49%,
        # 84->~67%+ mean ~84%). This coincides exactly with the "letter +
        # literal percentage" scheme decode_er_pr already implements, so
        # Allred-encoded values are decoded correctly by accident of design
        # -- pinned here so nobody "fixes" decode_er_pr in a way that breaks
        # Allred without realizing it's covered. See fable-opinion.md 工作5
        # for the full analysis, including the one still-uncertain cell
        # (Allred proportion score 1, <1%) that needs human PDF verification.
        ('000', 'ER Negative (0%)'),
        ('S84', 'ER Positive (Strong staining, 84%)'),
        ('S06', 'ER Positive (Strong staining, 6%)'),
        ('I22', 'ER Positive (Intermediate staining, 22%)'),
        ('S49', 'ER Positive (Strong staining, 49%)'),
        ('120', 'ER Negative (<1% or not specified)'),
        ('110', 'ER Positive (proportion unclear)'),
    ])
    def test_allred_score_codes_decode_and_roundtrip(self, code, expected_label):
        dec = decode_er_pr(pd.Series([code]), 'ER')
        assert dec.iloc[0] == expected_label
        enc = encode_er_pr(dec, 'ER')
        # Every code round-trips byte-identical now that encode emits the
        # official 3-character field width.
        assert enc.iloc[0] == code


class TestEncodeKi67:
    @pytest.mark.parametrize('code', [0, 1, 13, 14, 30, 31, 100, 888, 988, 998, 999,
                                      'A00', 'A05', 'A55', 'A99'])
    def test_roundtrip(self, code):
        """SSF10 is a 3-character field, so a percentage encodes zero-padded
        ('025'). 888 is not in this field's official range (A00-A09,
        000-100, 988, 998, 999), so it decodes to the explicit
        'Unlisted code 888' marker and still recovers exactly."""
        dec, enc = _roundtrip([code], decode_ki67, encode_ki67)
        expected = str(code).zfill(3) if str(code).isdigit() else str(code)
        assert enc.iloc[0] == expected

    def test_888_is_not_in_this_fields_range(self):
        assert decode_ki67(pd.Series(['888'])).iloc[0] == 'Unlisted code 888'


class TestEncodeHER2:
    @pytest.mark.parametrize('code', [
        '100', '101', '102', '103', '500', '501', '510', '511', '520', '521',
        '530', '531', '532', '590', '591', '600', '601', '640', '641',
        '888', '900', '901', '902', '988', '999',
    ])
    def test_roundtrip(self, code):
        dec, enc = _roundtrip([code], decode_her2, encode_her2)
        assert enc.iloc[0] == code

    @pytest.mark.parametrize('legacy,official', [
        ('0', '100'), ('1', '101'), ('2', '102'), ('3', '103')])
    def test_short_code_canonicalizes_to_official_3digit(self, legacy, official):
        """The codebook's valid range starts at 000/004/100; a bare 1-digit
        code carries no staining-percentage information, so it maps to the
        IHC-only 3-digit code (100-103). '0' must NOT become '000', which
        specifically asserts staining = 0%."""
        dec, enc = _roundtrip([legacy], decode_her2, encode_her2)
        assert enc.iloc[0] == official


class TestEncodeNottingham:
    @pytest.mark.parametrize('code,expected', [
        # A bare score is not a codebook code: the official range is
        # 030-090 (score x10), 110/120/130, 988, 999 (p.140-141), so encode
        # must emit the official code or the submission would be rejected.
        (3, '030'), (4, '040'), (5, '050'), (6, '060'), (7, '070'),
        (8, '080'), (9, '090'),
        (30, '030'), (60, '060'), (90, '090'),
        (110, '110'), (120, '120'), (130, '130'),
        (988, '988'), (999, '999'),
        (888, '888'),   # not in this field's range -> 'Unlisted code 888'
    ])
    def test_roundtrip(self, code, expected):
        dec, enc = _roundtrip([code], decode_nottingham, encode_nottingham)
        assert enc.iloc[0] == expected

    @pytest.mark.parametrize('code', [3, 30, '030'])
    def test_all_score_spellings_canonicalize_to_official_code(self, code):
        """'3', '30' and '030' all decode to the same 'Score 3 -> ...' text
        (the first two are what a spreadsheet export that dropped leading
        zeros produces); encode always writes back the official '030'."""
        dec, enc = _roundtrip([code], decode_nottingham, encode_nottingham)
        assert dec.iloc[0] == 'Score 3 → Grade 1 (Well differentiated)'
        assert enc.iloc[0] == '030'


class TestEncodeSSF3Neoadj:
    @pytest.mark.parametrize('code', ['010', '011', '020', '030', '040', '888', '988', '990', '999'])
    def test_roundtrip(self, code):
        dec, enc = _roundtrip([code], decode_ssf3_neoadj, encode_ssf3_neoadj)
        assert enc.iloc[0] == code


class TestEncodeSentinel:
    @pytest.mark.parametrize('code', [0, 1, 5, 89, 888, 988, 996, 999])
    @pytest.mark.parametrize('kind', ['examined', 'positive'])
    def test_roundtrip(self, code, kind):
        """3-character field (000-089, 988, 996, 999): counts encode
        zero-padded; 888 is outside the range and recovers via the
        'Unlisted code' marker."""
        dec, enc = _roundtrip(
            [code], lambda s: decode_sentinel(s, kind=kind), lambda s: encode_sentinel(s, kind=kind))
        assert enc.iloc[0] == str(code).zfill(3)

    def test_code_000_means_different_things_per_field(self):
        """Codebook p.137/138: SSF4 000 = no sentinel-node surgery at all,
        SSF5 000 = no involved node (ITC-only counts as none). The two must
        not share one label."""
        examined = decode_sentinel(pd.Series(['000']), kind='examined').iloc[0]
        positive = decode_sentinel(pd.Series(['000']), kind='positive').iloc[0]
        assert examined != positive
        assert encode_sentinel(pd.Series([examined]), kind='examined').iloc[0] == '000'
        assert encode_sentinel(pd.Series([positive]), kind='positive').iloc[0] == '000'


class TestEncodeLNPositive:
    @pytest.mark.parametrize('code', [0, 1, 5, 95, 97, 98, 99])
    def test_roundtrip(self, code):
        """LN_POSITI is a 2-character field (00-90, 95, 97-99)."""
        dec, enc = _roundtrip([code], decode_lnpositive, encode_lnpositive)
        assert enc.iloc[0] == str(code).zfill(2)

    def test_95_97_98_are_three_distinct_situations(self):
        """Longform-Manual p.130-131: 95 = positive by aspiration/core
        biopsy with no surgical removal, 97 = positive nodes present but
        count unknown, 98 = no nodes removed/examined (or no lymph node
        tissue found in the specimen). An earlier version decoded 95 and 98
        to identical text, which both lost the distinction and silently
        re-encoded 98 as 95."""
        labels = {c: decode_lnpositive(pd.Series([c])).iloc[0] for c in ('95', '97', '98')}
        assert len(set(labels.values())) == 3
        for code, label in labels.items():
            assert encode_lnpositive(pd.Series([label])).iloc[0] == code


class TestEncodeEBRT:
    @pytest.mark.parametrize('code', [0, 1, 2, 3, 4, 5, 7, 8, 16, 32, 64, 127])
    def test_roundtrip(self, code):
        dec, enc = _roundtrip([code], decode_ebrt_additive, encode_ebrt_additive)
        assert enc.iloc[0] == str(code)

    def test_negative_sentinels_canonicalize_to_999(self):
        # -1 -> 'EBRT NOS' -> '-1'; -9/999 both -> 'Unknown' -> canonical '999'
        assert encode_ebrt_additive(decode_ebrt_additive(pd.Series(['-1']))).iloc[0] == '-1'
        assert encode_ebrt_additive(decode_ebrt_additive(pd.Series(['-9']))).iloc[0] == '999'
        assert encode_ebrt_additive(decode_ebrt_additive(pd.Series(['999']))).iloc[0] == '999'


# ─────────────────────────────────────────────────────────────────────────────
# Bespoke encoders -- lung / colorectum / liver / prostate
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodeLungFields:
    @pytest.mark.parametrize('code', [0, 1, 4, 8, 988, 999])
    def test_ssf5_mediastinal_roundtrip(self, code):
        dec, enc = _roundtrip([code], _decode_lung_ssf5_mediastinal, encode_lung_ssf5_mediastinal)
        assert enc.iloc[0] == str(code).zfill(3)

    @pytest.mark.parametrize('code', ['XXX', 'AXX', 'ABX', 'ADX', 'AEX', '999', ''])
    def test_egfr_roundtrip(self, code):
        dec, enc = _roundtrip([code], _decode_lung_egfr, encode_lung_egfr)
        assert enc.iloc[0] == code

    def test_egfr_position_is_not_meaningful(self):
        """'AXB' and 'ABX' decode identically (position among the 3 chars
        carries no clinical meaning -- it's a set of mutations); both
        canonicalize to the same re-encoded form."""
        a = encode_lung_egfr(_decode_lung_egfr(pd.Series(['AXB']))).iloc[0]
        b = encode_lung_egfr(_decode_lung_egfr(pd.Series(['ABX']))).iloc[0]
        assert a == b == 'ABX'

    @pytest.mark.parametrize('code', [2, 10, 20, 21, 988, 999])
    def test_ssf9_nodules_roundtrip(self, code):
        dec, enc = _roundtrip([code], _decode_lung_ssf9_nodules, encode_lung_ssf9_nodules)
        assert enc.iloc[0] == str(code).zfill(3)


class TestEncodeColorectumFields:
    @pytest.mark.parametrize('code', [1, 2, 500, 986, 987, 988, 999])
    def test_cea_lab_value_roundtrip(self, code):
        dec, enc = _roundtrip([code], _decode_cea_lab_value, encode_cea_lab_value)
        assert enc.iloc[0] == str(code).zfill(3)

    @pytest.mark.parametrize('code', ['008', '118', '228', '338', '988', '998'])
    def test_ras_mutation_roundtrip(self, code):
        dec, enc = _roundtrip([code], _decode_ras_mutation, encode_ras_mutation)
        assert enc.iloc[0] == code

    def test_filler_digit_9_collides_with_8_known_not_a_bug(self):
        """_decode_ras_mutation's own leniency accepts either '8' or '9' as
        the 3rd (filler) digit (`s[2] in ('8', '9')`), but the decoded label
        never records which one it was -- decode('119') and decode('118')
        produce identical text (found during code review). encode_ras_mutation
        always reconstructs the codebook-documented filler '8', so a raw
        code ending in '9' does not round-trip byte-identical; this is
        expected label-collision behavior, not a bug, and is pinned here
        instead of surfacing as an unexplained compare_roundtrip mismatch."""
        label_118 = _decode_ras_mutation(pd.Series(['118'])).iloc[0]
        label_119 = _decode_ras_mutation(pd.Series(['119'])).iloc[0]
        assert label_118 == label_119
        assert encode_ras_mutation(pd.Series([label_119])).iloc[0] == '118'


class TestEncodeLiverFields:
    @pytest.mark.parametrize('code', ['A00', 'A01', 'A05', 'A99', 0, 5, 10, 50, 987, 991, 992, 993, 988, 999])
    def test_afp_roundtrip(self, code):
        dec, enc = _roundtrip([code], _decode_liver_afp, encode_liver_afp)
        assert enc.iloc[0] == str(code).zfill(3)

    def test_a00_means_under_1_not_zero(self):
        """Per Cancer-SSF-Manual (liver SSF1, p.81): an actual AFP value of
        0.91 ng/mL is coded A00 -- A00 means "<1 ng/mL", not literally zero/
        undetectable. Same category of bug as the ER/PR S00-means-100% fix:
        a floor/sentinel code misread as a literal value."""
        dec = _decode_liver_afp(pd.Series(['A00']))
        assert dec.iloc[0] == 'AFP <1 ng/mL (A-code, 2021+ scheme)'
        assert encode_liver_afp(dec).iloc[0] == 'A00'

    @pytest.mark.parametrize('code', [1, 2, 500, 986, 987, 988, 999])
    def test_lab_value_10x_roundtrip(self, code):
        dec, enc = _roundtrip(
            [code],
            lambda s: _decode_lab_value_10x(s, 'Creatinine', 'mg/dL'),
            lambda s: encode_lab_value_10x(s, 'Creatinine', 'mg/dL'),
        )
        assert enc.iloc[0] == str(code).zfill(3)

    @pytest.mark.parametrize('code', [1, 10, 60, 988, 997, 999])
    def test_inr_roundtrip(self, code):
        dec, enc = _roundtrip([code], _decode_liver_inr, encode_liver_inr)
        assert enc.iloc[0] == str(code).zfill(3)


class TestEncodePSA:
    @pytest.mark.parametrize('code', [
        1, 45, 500, 979, 980, 981, 982, 983, 984, 985, 986, 987,
        989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 988, 999,
    ])
    def test_roundtrip(self, code):
        """Covers the full PSA code table (Cancer-SSF-Manual, prostate SSF1,
        p.183-184), including the high-value tiers (981-998, PSA >=98 ng/mL)
        that the decoder originally didn't handle at all -- those values are
        common in advanced/metastatic prostate cancer and were previously
        falling through to a meaningless 'Code 991'-style fallback. The
        field is 3 characters, so encode emits the zero-padded code."""
        dec, enc = _roundtrip([code], _decode_psa, encode_psa)
        assert enc.iloc[0] == str(code).zfill(3)

    def test_high_value_tiers_are_clinically_meaningful_not_raw_fallback(self):
        for code in (981, 991, 998):
            label = _decode_psa(pd.Series([str(code)])).iloc[0]
            assert not label.startswith('Code '), (
                f'PSA code {code} should decode to a clinical value, not fall through: {label!r}')


class TestEncodeGenericSSF:
    @pytest.mark.parametrize('code', [0, 5, -1, 888, 900, 901, 902, 988, 998, 999])
    def test_roundtrip(self, code):
        dec, enc = _roundtrip(
            [code], lambda s: _generic_ssf(s, 'SSF1'), lambda s: encode_generic_ssf(s))
        assert enc.iloc[0] == str(code)


# ─────────────────────────────────────────────────────────────────────────────
# Structural fields (AJCC/PRESTYPE/STYPE95/PRESLNSCO/SLNSCO95) -- every key
# of TCRDecoder._map()'s code tables round-trips through encode_structural_map.
# ─────────────────────────────────────────────────────────────────────────────

def _map_decode_one(code_map: dict, v: str) -> str:
    """Mirrors TCRDecoder._map()'s exact per-cell decode logic."""
    v = str(v).strip()
    return code_map.get(v, code_map.get(v.lstrip('0'), f'Code {v}'))


class TestEncodeStructuralFields:
    # PRESTYPE / STYPE95 are no longer flat dicts: Appendix B defines their
    # codes per primary site, so they are covered by the per-site round trip
    # in test_codebook_conformance.py instead.
    @pytest.mark.parametrize('code_map_name', ['AJCC_MAP', 'LNSCO_MAP'])
    def test_every_code_roundtrips(self, code_map_name):
        from tcr_decoder import core
        code_map = getattr(core, code_map_name)
        for code in code_map:
            label = _map_decode_one(code_map, code)
            encoded = encode_structural_map(pd.Series([label]), code_map).iloc[0]
            # Two different raw codes can share identical label text (e.g.
            # legacy 2-digit '20' and 3-digit '020'-style aliases for the
            # same real-world meaning) -- encode() then returns whichever
            # code is canonical (first-listed) for that label, which is not
            # necessarily this exact `code`. What must hold is that the
            # canonical code decodes to the SAME label, i.e. round-tripping
            # is label-preserving even when it isn't code-identical.
            assert _map_decode_one(code_map, encoded) == label, (
                f'{code_map_name}[{code!r}] -> {label!r} -> {encoded!r} '
                f'does not decode back to the same label')


# ─────────────────────────────────────────────────────────────────────────────
# apply_ssf_profile <-> apply_ssf_encode_profile, across all cancer groups
# ─────────────────────────────────────────────────────────────────────────────

# One realistic, field-appropriate raw code per SSF slot for each cancer
# group -- chosen to hit a real (non-sentinel-only) value per profile so the
# round trip actually exercises each field's real decoder, not just 988/999.
SSF_SAMPLES = {
    # Official, codebook-valid codes at their declared 3-character width.
    'breast':      ['070', '000', '010', '005', '002', '070', '531', '000', '010', '025'],
    'lung':        ['010', '010', '002', '013', '003', 'AXX', '010', '005', '010', '988'],
    'colorectum':  ['050', '010', '020', '030', '001', '118', '010', '010', '030', '020'],
    'liver':       ['A05', '003', '105', '050', '300', '015', '010', '020', '988', '988'],
    'cervix':      ['005', '010', '988', '988', '988', '988', '988', '988', '988', '988'],
    'stomach':     ['050', '010', '001', '030', '010', '988', '988', '988', '988', '988'],
    # Thyroid collects no SSFs at all (manual p.1) -- every field is 988.
    'thyroid':     ['988'] * 10,
    'prostate':    ['045', '034', '007', '043', '007', '012', '003', '030', '988', '988'],
    'head_neck':   ['015', '005', '110', '000', '000', '000', '030', '025', '110', '115'],
    'endometrium': ['070', '050', '002', '020', '020', '010', '988', '988', '988', '988'],
    'esophagus':   ['020', '010', '002', '020', '988', '988', '988', '988', '988', '988'],
    'pancreas':    ['045', '010', '350', '014', '120', '158', '988', '988', '988', '988'],
    'ovary':       ['350', '045', '010', '988', '988', '988', '988', '988', '988', '988'],
    'bladder':     ['020', '010', '010', '988', '988', '988', '988', '988', '988', '988'],
    # SSF10 is a composite in both: lymphoma = ESR + IPS, leukemia = months
    # since diagnosis + log reduction.
    'lymphoma':    ['001', '010', '002', '988', '001', '001', '010', '000', '001', '253'],
    'leukemia':    ['001', '008', '001', '011', '000', '001', '010', '000', '001', '124'],
    'generic':     ['5', '5', '5', '5', '5', '5', '5', '5', '5', '5'],
}


@pytest.mark.parametrize('cancer_group', list(SSF_SAMPLES))
def test_ssf_profile_full_roundtrip(cancer_group):
    vals = SSF_SAMPLES[cancer_group]
    df = pd.DataFrame({f'SSF{i}_raw': [vals[i - 1]] for i in range(1, 11)})
    decoded = apply_ssf_profile(df, cancer_group)
    profile = get_ssf_profile(cancer_group)
    col_names = [f.column_name for f in profile.fields.values()]
    encoded = apply_ssf_encode_profile(decoded[col_names], cancer_group)
    raw_cols = [f'SSF{i}_raw' for i in range(1, 11)]
    assert df[raw_cols].iloc[0].tolist() == encoded[raw_cols].iloc[0].tolist()


def test_all_supported_cancer_groups_covered_by_samples():
    supported = set(list_supported_cancers()['Cancer_Group'])
    assert supported == set(SSF_SAMPLES)


# ─────────────────────────────────────────────────────────────────────────────
# Known, documented label collisions (NOT bugs -- see module docstring)
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_LABEL_COLLISIONS = [
    # (cancer_group, ssf_key, code_a, code_b) -- both decode to the same text.
    # The 888-vs-988 collisions that used to exist in colorectum/stomach/
    # cervix SSF2 are gone: those fields no longer inherit the generic
    # sentinel block, so 888 is not a legal code for them at all.
    ('breast', 'SSF6', '60', '6'),
]


@pytest.mark.parametrize('cancer_group,ssf_key,code_a,code_b', KNOWN_LABEL_COLLISIONS)
def test_known_collision_pairs_decode_identically(cancer_group, ssf_key, code_a, code_b):
    """Guards the assumption compare_roundtrip's KNOWN_LABEL_COLLISION_NOTE
    depends on: these code pairs really do decode to identical text. If this
    ever starts failing, someone disambiguated the decoder text -- update
    both this list and the roundtrip module's note."""
    df = pd.DataFrame({
        f'SSF{i}_raw': (['888'] * 2 if i != int(ssf_key.replace('SSF', ''))
                         else [code_a, code_b])
        for i in range(1, 11)
    })
    decoded = apply_ssf_profile(df, cancer_group)
    profile = get_ssf_profile(cancer_group)
    col_name = profile.fields[ssf_key].column_name
    assert decoded[col_name].iloc[0] == decoded[col_name].iloc[1]


# ─────────────────────────────────────────────────────────────────────────────
# TCREncoder (full clinical-facts DataFrame -> raw fields)
# ─────────────────────────────────────────────────────────────────────────────

class TestTCREncoder:
    def test_breast_encode_runs_without_raising(self, breast_clean_df):
        enc = TCREncoder(breast_clean_df)
        raw = enc.encode(on_error='raise')
        assert enc.cancer_group == 'breast'
        assert 'SSF1_raw' in raw.columns
        assert len(raw) == len(breast_clean_df)

    def test_breast_ssf8_ssf9_now_round_trip(self, breast_clean_df):
        """TCRDecoder.decode() used to overwrite Pagets_Disease/LVI_SSF with
        text from the input file's own SSF8_decoded/SSF9_decoded columns,
        which left them the only two breast SSF fields TCREncoder could not
        invert. They now come from the profile decoder like every other SSF
        field, so all ten breast SSF fields encode."""
        enc = TCREncoder(breast_clean_df)
        raw = enc.encode(on_error='raise')
        assert 'Pagets_Disease' not in enc.unencoded_columns
        assert 'LVI_SSF' not in enc.unencoded_columns
        assert {'SSF8_raw', 'SSF9_raw'} <= set(raw.columns)

    def test_lung_generic_performance_status_now_has_a_codetable(self, lung_clean_df):
        """'Performance_Status' (#7.6 KPSECOG, all cancer types) used to be
        decoded by trusting the input file's own pre-existing column, with no
        code table this package owned -- it stayed unencoded regardless of
        cancer group. It now has one (longform_codes.PERFORMANCE_STATUS_MAP)
        and round-trips. This used to collide in name with lung's SSF3
        (ECOG/KPS) column; SSF3 has its own column ('Performance_Status_SSF3')
        and round-trips independently -- see the next test."""
        enc = TCREncoder(lung_clean_df)
        raw = enc.encode(on_error='raise')
        assert enc.cancer_group == 'lung'
        assert 'Performance_Status' not in enc.unencoded_columns
        assert 'Performance_Status_SSF3' not in enc.unencoded_columns
        assert 'KPSECOG_raw' in raw.columns
        assert 'SSF3_raw' in raw.columns

    def test_colorectum_encode_runs_without_raising(self, colorectum_clean_df):
        enc = TCREncoder(colorectum_clean_df)
        raw = enc.encode(on_error='raise')
        assert enc.cancer_group == 'colorectum'
        assert len(raw) == len(colorectum_clean_df)

    def test_forced_cancer_group_overrides_autodetect(self, breast_clean_df):
        enc = TCREncoder(breast_clean_df, cancer_group='generic')
        enc.encode(on_error='empty')
        assert enc.cancer_group == 'generic'

    def test_structural_fields_encoded(self, breast_clean_df):
        enc = TCREncoder(breast_clean_df)
        raw = enc.encode(on_error='raise')
        for col in ('AJCC_raw', 'STYPE95_raw', 'PRESTYPE_raw'):
            assert col in raw.columns


# ─────────────────────────────────────────────────────────────────────────────
# compare_roundtrip / export_roundtrip_report
# ─────────────────────────────────────────────────────────────────────────────

class TestCompareRoundtrip:
    def test_breast_roundtrip_is_lossless(self, breast_raw_df, tmp_path):
        """A breast file made of codebook-legal codes must survive
        decode -> encode with ZERO mismatches. Breast is the one cancer
        group whose profile has been verified field-by-field against the
        printed manual (see tcr_decoder/code_ranges.py), so nothing here is
        allowed to be explained away as a known collision."""
        xlsx = tmp_path / 'breast.xlsx'
        with pd.ExcelWriter(str(xlsx), engine='openpyxl') as w:
            breast_raw_df.to_excel(w, sheet_name='All_Fields_Decoded', index=False)
        mismatches = compare_roundtrip(str(xlsx))
        assert len(mismatches) == 0, mismatches.head(10).to_string()

    def test_breast_synthetic_codes_are_all_codebook_legal(self, breast_raw_df):
        """The generator must not produce codes the registry would reject --
        otherwise the round-trip tests above only exercise the decoders'
        leniency paths."""
        from tcr_decoder.code_ranges import is_legal_code
        illegal = {
            f'SSF{i}': sorted({v for v in breast_raw_df[f'SSF{i}_raw'].astype(str)
                               if not is_legal_code('breast', f'SSF{i}', v)})
            for i in range(1, 11)
        }
        assert not any(illegal.values()), {k: v for k, v in illegal.items() if v}

    def test_colorectum_mismatches_are_only_known_888_988_collision(
            self, colorectum_raw_df, tmp_path):
        xlsx = tmp_path / 'crc.xlsx'
        with pd.ExcelWriter(str(xlsx), engine='openpyxl') as w:
            colorectum_raw_df.to_excel(w, sheet_name='All_Fields_Decoded', index=False)
        mismatches = compare_roundtrip(str(xlsx))
        for _, row in mismatches.iterrows():
            assert {row['Original_Code'], row['Roundtrip_Code']} <= {'888', '988'}

    def test_all_zero_code_is_not_a_false_positive_mismatch(self):
        """Regression: '000' and '0' both decode to the same structural label
        (e.g. STYPE95's 'No surgery'), and encode canonicalizes back to '0'.
        The old `lstrip('0')` comparison excluded the all-zero case (both
        sides strip to '') via a `!= ''` guard meant to stop blank cells
        from matching non-blank ones, so '000' vs '0' was reported as a
        mismatch even though it's a harmless leading-zero difference --
        found during code review."""
        from tcr_decoder.roundtrip import _numerically_equal
        assert _numerically_equal('000', '0') is True
        assert _numerically_equal('020', '20') is True
        # must NOT swallow the genuinely different Nottingham-style
        # collision (different *values*, not just leading-zero padding)
        assert _numerically_equal('60', '6') is False
        # must NOT match a blank cell against a non-blank one
        assert _numerically_equal('', '0') is False

    def test_export_roundtrip_report_encodes_only_once(self, breast_raw_df, tmp_path, monkeypatch):
        """Regression: export_roundtrip_report used to call TCREncoder.encode()
        once for `.unencoded_columns`, then call compare_roundtrip() which
        built a SECOND TCREncoder and ran the full encode pipeline again --
        found during code review. Assert TCREncoder.encode is now invoked
        exactly once per report."""
        from tcr_decoder.roundtrip import export_roundtrip_report
        from tcr_decoder.encoder import TCREncoder

        call_count = 0
        original_encode = TCREncoder.encode

        def counting_encode(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_encode(self, *args, **kwargs)

        monkeypatch.setattr(TCREncoder, 'encode', counting_encode)

        src = tmp_path / 'breast.xlsx'
        with pd.ExcelWriter(str(src), engine='openpyxl') as w:
            breast_raw_df.to_excel(w, sheet_name='All_Fields_Decoded', index=False)
        export_roundtrip_report(str(src), str(tmp_path / 'report.xlsx'))

        assert call_count == 1

    def test_export_roundtrip_report_writes_expected_sheets(self, breast_raw_df, tmp_path):
        from tcr_decoder.roundtrip import export_roundtrip_report
        src = tmp_path / 'breast.xlsx'
        with pd.ExcelWriter(str(src), engine='openpyxl') as w:
            breast_raw_df.to_excel(w, sheet_name='All_Fields_Decoded', index=False)
        out = tmp_path / 'report.xlsx'
        export_roundtrip_report(str(src), str(out))
        assert out.exists()
        xl = pd.ExcelFile(str(out))
        assert set(xl.sheet_names) == {
            'Mismatches', 'Field_Summary', 'Unencoded_Columns', 'Notes'}
