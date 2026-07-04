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
        dec, enc = _roundtrip([code], lambda s: decode_er_pr(s, 'ER'),
                              lambda s: encode_er_pr(s, 'ER'))
        assert enc.iloc[0] == str(code)

    @pytest.mark.parametrize('code', ['W15', 'I50', 'S70', 'W1', 'S100'])
    def test_roundtrip_staining_codes(self, code):
        dec, enc = _roundtrip([code], lambda s: decode_er_pr(s, 'ER'),
                              lambda s: encode_er_pr(s, 'ER'))
        assert enc.iloc[0] == code

    def test_out_of_range_percent_passes_through(self):
        """decode_er_pr() itself falls back to bare passthrough for n>100;
        encode must mirror that, not raise."""
        dec, enc = _roundtrip([150], lambda s: decode_er_pr(s, 'ER'),
                              lambda s: encode_er_pr(s, 'ER'))
        assert dec.iloc[0] == '150'
        assert enc.iloc[0] == '150'

    def test_pr_uses_pr_vocabulary(self):
        dec, enc = _roundtrip([70], lambda s: decode_er_pr(s, 'PR'),
                              lambda s: encode_er_pr(s, 'PR'))
        assert 'PR' in dec.iloc[0]
        assert enc.iloc[0] == '70'


class TestEncodeKi67:
    @pytest.mark.parametrize('code', [0, 1, 13, 14, 30, 31, 100, 888, 988, 998, 999,
                                      'A00', 'A05', 'A55', 'A99'])
    def test_roundtrip(self, code):
        dec, enc = _roundtrip([code], decode_ki67, encode_ki67)
        assert enc.iloc[0] == str(code)


class TestEncodeHER2:
    @pytest.mark.parametrize('code', [
        '100', '101', '102', '103', '500', '501', '510', '511', '520', '521',
        '530', '531', '532', '590', '591', '600', '601', '640', '641',
        '888', '900', '901', '902', '988', '999',
    ])
    def test_roundtrip(self, code):
        dec, enc = _roundtrip([code], decode_her2, encode_her2)
        assert enc.iloc[0] == code

    def test_short_code_canonicalizes_to_3digit(self):
        """'0' and '000' both decode to 'IHC 0 — Negative'; encode always
        returns the canonical 3-digit form."""
        dec, enc = _roundtrip(['0'], decode_her2, encode_her2)
        assert enc.iloc[0] == '000'


class TestEncodeNottingham:
    @pytest.mark.parametrize('code', [3, 4, 5, 6, 7, 8, 9, 110, 120, 130, 888, 988, 999])
    def test_roundtrip(self, code):
        dec, enc = _roundtrip([code], decode_nottingham, encode_nottingham)
        assert enc.iloc[0] == str(code)

    @pytest.mark.parametrize('code,canonical_score', [(30, '3'), (60, '6'), (90, '9')])
    def test_tens_format_canonicalizes_to_bare_score(self, code, canonical_score):
        """Codes '30' and '3' both decode to the identical 'Score 3 -> ...'
        text (per the ORIGINAL decode_nottingham logic) -- a documented,
        pre-existing ambiguity in the codebook, not introduced by encoding.
        encode() canonicalizes to the bare score per breast_coding_spec.md."""
        dec, enc = _roundtrip([code], decode_nottingham, encode_nottingham)
        assert enc.iloc[0] == canonical_score


class TestEncodeSSF3Neoadj:
    @pytest.mark.parametrize('code', ['010', '011', '020', '030', '040', '888', '988', '990', '999'])
    def test_roundtrip(self, code):
        dec, enc = _roundtrip([code], decode_ssf3_neoadj, encode_ssf3_neoadj)
        assert enc.iloc[0] == code


class TestEncodeSentinel:
    @pytest.mark.parametrize('code', [0, 1, 5, 89, 888, 988, 996, 999])
    @pytest.mark.parametrize('kind', ['examined', 'positive'])
    def test_roundtrip(self, code, kind):
        dec, enc = _roundtrip(
            [code], lambda s: decode_sentinel(s, kind=kind), lambda s: encode_sentinel(s, kind=kind))
        assert enc.iloc[0] == str(code)


class TestEncodeLNPositive:
    @pytest.mark.parametrize('code', [0, 1, 5, 95, 97, 99])
    def test_roundtrip(self, code):
        dec, enc = _roundtrip([code], decode_lnpositive, encode_lnpositive)
        assert enc.iloc[0] == str(code)


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
        assert enc.iloc[0] == str(code)

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
        assert enc.iloc[0] == str(code)


class TestEncodeColorectumFields:
    @pytest.mark.parametrize('code', [1, 2, 500, 986, 987, 988, 999])
    def test_cea_lab_value_roundtrip(self, code):
        dec, enc = _roundtrip([code], _decode_cea_lab_value, encode_cea_lab_value)
        assert enc.iloc[0] == str(code)

    @pytest.mark.parametrize('code', ['008', '118', '228', '338', '988', '998'])
    def test_ras_mutation_roundtrip(self, code):
        dec, enc = _roundtrip([code], _decode_ras_mutation, encode_ras_mutation)
        assert enc.iloc[0] == code


class TestEncodeLiverFields:
    @pytest.mark.parametrize('code', ['A05', 'A99', 0, 5, 10, 50, 987, 991, 992, 993, 988, 999])
    def test_afp_roundtrip(self, code):
        dec, enc = _roundtrip([code], _decode_liver_afp, encode_liver_afp)
        assert enc.iloc[0] == str(code)

    @pytest.mark.parametrize('code', [1, 2, 500, 986, 987, 988, 999])
    def test_lab_value_10x_roundtrip(self, code):
        dec, enc = _roundtrip(
            [code],
            lambda s: _decode_lab_value_10x(s, 'Creatinine', 'mg/dL'),
            lambda s: encode_lab_value_10x(s, 'Creatinine', 'mg/dL'),
        )
        assert enc.iloc[0] == str(code)

    @pytest.mark.parametrize('code', [1, 10, 60, 988, 997, 999])
    def test_inr_roundtrip(self, code):
        dec, enc = _roundtrip([code], _decode_liver_inr, encode_liver_inr)
        assert enc.iloc[0] == str(code)


class TestEncodePSA:
    @pytest.mark.parametrize('code', [0, 1, 45, 980, 988, 999])
    def test_roundtrip(self, code):
        dec, enc = _roundtrip([code], _decode_psa, encode_psa)
        assert enc.iloc[0] == str(code)


class TestEncodeGenericSSF:
    @pytest.mark.parametrize('code', [0, 5, -1, 888, 900, 901, 902, 988, 998, 999])
    def test_roundtrip(self, code):
        dec, enc = _roundtrip(
            [code], lambda s: _generic_ssf(s, 'SSF1'), lambda s: encode_generic_ssf(s))
        assert enc.iloc[0] == str(code)


# ─────────────────────────────────────────────────────────────────────────────
# apply_ssf_profile <-> apply_ssf_encode_profile, across all cancer groups
# ─────────────────────────────────────────────────────────────────────────────

# One realistic, field-appropriate raw code per SSF slot for each cancer
# group -- chosen to hit a real (non-sentinel-only) value per profile so the
# round trip actually exercises each field's real decoder, not just 988/999.
SSF_SAMPLES = {
    'breast':      ['70', '0', '010', '5', '2', '7', '531', '0', '10', '25'],
    'lung':        ['10', '10', '2', '13', '3', 'AXX', '10', '5', '10', '988'],
    'colorectum':  ['50', '10', '988', '988', '988', '118', '988', '988', '988', '20'],
    'liver':       ['A05', '3', '105', '50', '300', '15', '10', '20', '988', '988'],
    'cervix':      ['5', '10', '988', '988', '988', '988', '988', '988', '988', '988'],
    'stomach':     ['50', '10', '1', '988', '10', '988', '988', '988', '988', '988'],
    'thyroid':     ['1', '988', '2', '988', '988', '988', '988', '988', '988', '988'],
    'prostate':    ['45', '7', '988', '988', '988', '988', '988', '988', '988', '988'],
    'nasopharynx': ['5', '5', '988', '988', '988', '988', '988', '988', '988', '988'],
    'endometrium': ['988', '988', '988', '988', '988', '988', '1', '988', '988', '988'],
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
    ('colorectum', 'SSF2', '888', '988'),
    ('stomach', 'SSF2', '888', '988'),
    ('cervix', 'SSF2', '888', '988'),
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

    def test_breast_pipeline_override_columns_are_flagged_not_crashed(self, breast_clean_df):
        enc = TCREncoder(breast_clean_df)
        enc.encode(on_error='raise')
        assert 'Pagets_Disease' in enc.unencoded_columns
        assert 'LVI_SSF' in enc.unencoded_columns

    def test_lung_performance_status_collision_is_flagged(self, lung_clean_df):
        enc = TCREncoder(lung_clean_df)
        enc.encode(on_error='raise')
        assert enc.cancer_group == 'lung'
        assert 'Performance_Status' in enc.unencoded_columns

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
    def test_breast_mismatches_are_only_known_nottingham_collision(self, breast_raw_df, tmp_path):
        xlsx = tmp_path / 'breast.xlsx'
        with pd.ExcelWriter(str(xlsx), engine='openpyxl') as w:
            breast_raw_df.to_excel(w, sheet_name='All_Fields_Decoded', index=False)
        mismatches = compare_roundtrip(str(xlsx))
        if len(mismatches):
            assert set(mismatches['Field'].unique()) <= {'SSF6_raw'}
            # every mismatch must be a tens-vs-ones-digit rendering of the
            # SAME score (e.g. '60' vs '6'), never an unrelated value
            for _, row in mismatches.iterrows():
                orig, rt = row['Original_Code'], row['Roundtrip_Code']
                assert orig.rstrip('0') == rt or orig == rt + '0'

    def test_colorectum_mismatches_are_only_known_888_988_collision(
            self, colorectum_raw_df, tmp_path):
        xlsx = tmp_path / 'crc.xlsx'
        with pd.ExcelWriter(str(xlsx), engine='openpyxl') as w:
            colorectum_raw_df.to_excel(w, sheet_name='All_Fields_Decoded', index=False)
        mismatches = compare_roundtrip(str(xlsx))
        for _, row in mismatches.iterrows():
            assert {row['Original_Code'], row['Roundtrip_Code']} <= {'888', '988'}

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
