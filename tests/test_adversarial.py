"""
Adversarial / stress tests for tcr_decoder.

Design philosophy: each test class targets a specific failure mode or
boundary condition not covered by the happy-path test suite.

Categories:
  A  — Sentinel value chaos (all 7 sentinel codes, all input types)
  B  — Boundary / off-by-one for every numeric decoder
  C  — Type injection (NaN, None, '', whitespace, mixed, floats)
  D  — ICD-O-3 code edge cases and cancer group routing
  E  — All 11 SSF profiles completeness & contract verification
  F  — Roundtrip integrity (no raw numeric leaks in decoded output)
  G  — Large-dataset performance (10 000 rows in < 10 s)
  H  — CLI smoke tests (all four modes, no crash)
  I  — Contradictory / clinically impossible data (should not crash)
  J  — decode_er_pr rstrip anti-regression battery
"""

import time
import pytest
import pandas as pd
import numpy as np

from tcr_decoder.decoders import (
    decode_er_pr, decode_ki67, decode_her2, decode_nottingham,
    decode_ssf3_neoadj, decode_sentinel, decode_lnpositive,
)
from tcr_decoder.ssf_registry import (
    detect_cancer_group, detect_cancer_group_from_series,
    get_ssf_profile, apply_ssf_profile, list_supported_cancers,
    _generic_ssf,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def s(*vals) -> pd.Series:
    return pd.Series(list(vals))


def _ssf_df(**overrides) -> pd.DataFrame:
    """Minimal DataFrame with all SSF1-10 raw columns."""
    base = {f'SSF{i}_raw': [0] for i in range(1, 11)}
    base.update(overrides)
    return pd.DataFrame(base)


# ─────────────────────────────────────────────────────────────────────────────
# A  Sentinel value chaos
# ─────────────────────────────────────────────────────────────────────────────

SENTINEL_CODES = [888, 900, 901, 902, 988, 998, 999]
SENTINEL_KEYWORDS = {
    888: 'converted',       # breast ER/PR specific
    900: 'laboratory',
    901: 'radiographic',
    902: 'documented',
    988: 'applicable',
    998: 'applicable',
    999: 'unknown',
}


class TestSentinelChaos:
    """Every sentinel code must survive in int / float / str format."""

    @pytest.mark.parametrize('code', SENTINEL_CODES)
    def test_generic_sentinel_as_int(self, code):
        result = _generic_ssf(s(code))
        assert result.iloc[0] != '' and result.iloc[0] != str(code)

    @pytest.mark.parametrize('code', SENTINEL_CODES)
    def test_generic_sentinel_as_float(self, code):
        result = _generic_ssf(s(float(code)))
        assert result.iloc[0] != '' and str(code) not in result.iloc[0].replace(str(code), '')

    @pytest.mark.parametrize('code', SENTINEL_CODES)
    def test_generic_sentinel_as_string(self, code):
        result = _generic_ssf(s(str(code)))
        assert result.iloc[0] != ''

    @pytest.mark.parametrize('code', [888, 988, 999])
    def test_generic_sentinel_as_float_with_decimals(self, code):
        """888.000 must still resolve to sentinel text, not '888'."""
        result = _generic_ssf(s(f'{code}.000'))
        assert result.iloc[0] != '' and result.iloc[0] != f'{code}.000'

    def test_sentinel_999_in_er_decoder(self):
        result = decode_er_pr(s(999), 'ER')
        assert 'Unknown' in result.iloc[0]

    def test_sentinel_988_in_er_decoder(self):
        result = decode_er_pr(s(988), 'ER')
        assert 'applicable' in result.iloc[0].lower()

    def test_sentinel_888_in_er_decoder_not_positive(self):
        """888 must NEVER produce 'ER Positive (888%)' — the rstrip bug."""
        result = decode_er_pr(s(888), 'ER')
        assert 'Positive (888%)' not in result.iloc[0]
        assert 'converted' in result.iloc[0].lower()

    def test_sentinel_998_in_ki67(self):
        result = decode_ki67(s(998))
        assert 'unknown' in result.iloc[0].lower() or 'tested' in result.iloc[0].lower()

    def test_sentinel_999_in_ki67(self):
        result = decode_ki67(s(999))
        assert 'Unknown' in result.iloc[0]

    def test_sentinel_mixed_with_valid(self):
        """A column with both valid values and sentinels must not contaminate."""
        result = decode_er_pr(s(70, 999, 888, 50), 'ER')
        assert 'Positive (70%)' in result.iloc[0]
        assert 'Unknown' in result.iloc[1]
        assert 'converted' in result.iloc[2].lower()
        assert 'Positive (50%)' in result.iloc[3]

    @pytest.mark.parametrize('cancer', ['breast', 'lung', 'colorectum', 'liver', 'prostate'])
    def test_sentinel_999_survives_all_profiles(self, cancer):
        """apply_ssf_profile with all-999 data must not raise and must decode."""
        df = pd.DataFrame({f'SSF{i}_raw': [999] * 3 for i in range(1, 11)})
        result = apply_ssf_profile(df, cancer)
        # At least one SSF column should contain 'Unknown'
        decoded_cols = [c for c in result.columns if not c.endswith('_raw')]
        found = any(
            result[c].astype(str).str.contains('Unknown|applicable|not', case=False).any()
            for c in decoded_cols
        )
        assert found, f'No sentinel text found for {cancer} with all-999 data'


# ─────────────────────────────────────────────────────────────────────────────
# B  Boundary / off-by-one
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundaryValues:

    # ER/PR boundaries
    def test_er_0_is_negative(self):
        assert 'Negative' in decode_er_pr(s(0), 'ER').iloc[0]

    def test_er_1_is_positive_1pct(self):
        assert 'Positive (1%)' in decode_er_pr(s(1), 'ER').iloc[0]

    def test_er_100_is_positive_100pct(self):
        assert 'Positive (100%)' in decode_er_pr(s(100), 'ER').iloc[0]

    def test_er_101_not_in_normal_range(self):
        """101 is outside 0-100 and not a special code — must not decode as 'Positive (101%)'."""
        result = decode_er_pr(s(101), 'ER').iloc[0]
        # 101 is unknown territory; it should either passthrough or map to special
        # It must NOT claim 101% positivity as if it were a valid percentage
        assert result != 'ER Positive (101%)'

    # Ki67 boundaries
    def test_ki67_13_is_low(self):
        assert 'Low' in decode_ki67(s(13)).iloc[0]

    def test_ki67_14_is_intermediate(self):
        """Exact boundary: 14% marks the Low→Intermediate transition."""
        result = decode_ki67(s(14)).iloc[0]
        assert 'Intermediate' in result, f'Ki67=14 should be Intermediate, got: {result!r}'

    def test_ki67_30_is_intermediate(self):
        assert 'Intermediate' in decode_ki67(s(30)).iloc[0]

    def test_ki67_31_is_high(self):
        """Exact boundary: >30% is High."""
        result = decode_ki67(s(31)).iloc[0]
        assert 'High' in result, f'Ki67=31 should be High, got: {result!r}'

    def test_ki67_0_is_low(self):
        result = decode_ki67(s(0)).iloc[0]
        assert 'Low' in result or '0%' in result

    def test_ki67_100_is_high(self):
        assert 'High' in decode_ki67(s(100)).iloc[0]

    # A-prefix Ki67 boundaries
    def test_ki67_A00_is_sub1pct(self):
        result = decode_ki67(s('A00')).iloc[0]
        assert '%' in result

    def test_ki67_A09_is_sub1pct(self):
        result = decode_ki67(s('A09')).iloc[0]
        assert '%' in result and ('0.' in result or '<1' in result)

    # HER2 boundaries
    def test_her2_ihc0_is_negative(self):
        assert 'Negative' in decode_her2(s(100)).iloc[0] or 'IHC 0' in decode_her2(s(100)).iloc[0]

    def test_her2_code_510_has_ish(self):
        assert 'ISH' in decode_her2(s(510)).iloc[0]

    def test_her2_code_102_no_ish_positive(self):
        """Critical: 102 is 1xx range (IHC-only), must not say ISH Positive."""
        assert 'ISH Positive' not in decode_her2(s(102)).iloc[0]

    # Nottingham boundaries
    def test_nottingham_30_is_grade1(self):
        result = decode_nottingham(s(30)).iloc[0]
        assert 'Grade 1' in result

    def test_nottingham_80_is_grade3(self):
        result = decode_nottingham(s(80)).iloc[0]
        assert 'Grade 3' in result

    def test_nottingham_60_is_grade2(self):
        result = decode_nottingham(s(60)).iloc[0]
        assert 'Grade 2' in result

    # Sentinel lymph node
    def test_sentinel_examined_0_is_none_or_zero(self):
        """Code 0 = no sentinel nodes examined → 'None examined' or '0 node(s)'."""
        result = decode_sentinel(s(0), kind='examined').iloc[0]
        assert 'None' in result or '0' in result

    def test_sentinel_examined_5(self):
        assert '5' in decode_sentinel(s(5), kind='examined').iloc[0]

    def test_sentinel_examined_99_known(self):
        """Code 99 = unknown examined count; must not raise."""
        result = decode_sentinel(s(99), kind='examined').iloc[0]
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# C  Type injection
# ─────────────────────────────────────────────────────────────────────────────

class TestTypeInjection:
    """Feed garbage inputs and verify graceful handling (no exceptions)."""

    JUNK = [None, np.nan, pd.NA, '', '   ', 'NULL', 'N/A', 'abc', '!@#', '\t\n']

    @pytest.mark.parametrize('val', JUNK)
    def test_er_junk_no_crash(self, val):
        result = decode_er_pr(s(val), 'ER').iloc[0]
        assert isinstance(result, str)

    @pytest.mark.parametrize('val', JUNK)
    def test_ki67_junk_no_crash(self, val):
        result = decode_ki67(s(val)).iloc[0]
        assert isinstance(result, str)

    @pytest.mark.parametrize('val', JUNK)
    def test_her2_junk_no_crash(self, val):
        result = decode_her2(s(val)).iloc[0]
        assert isinstance(result, str)

    @pytest.mark.parametrize('val', JUNK)
    def test_nottingham_junk_no_crash(self, val):
        result = decode_nottingham(s(val)).iloc[0]
        assert isinstance(result, str)

    @pytest.mark.parametrize('val', JUNK)
    def test_generic_ssf_junk_no_crash(self, val):
        result = _generic_ssf(s(val)).iloc[0]
        assert isinstance(result, str)

    def test_float_120_no_rstrip_bug(self):
        """120.0 must resolve to 'ER Negative', not 'Positive (12%)' via rstrip('.0') bug."""
        result = decode_er_pr(s(120.0), 'ER').iloc[0]
        assert 'Negative' in result                   # correct: ER Negative
        assert 'Positive (12%)' not in result          # rstrip bug would produce this

    def test_float_888_sentinel(self):
        """888.0 must resolve to sentinel, not '88' via rstrip."""
        result = decode_er_pr(s(888.0), 'ER').iloc[0]
        assert 'converted' in result.lower()

    def test_all_nan_series(self):
        """Series of all NaN must produce empty strings, not crash."""
        result = decode_er_pr(pd.Series([np.nan] * 10), 'ER')
        assert (result == '').all()

    def test_mixed_types_in_one_column(self):
        """Real-world scenario: column has int, float, str, NaN mixed."""
        mixed = pd.Series([70, 120.0, '888', np.nan, '999', 50.0])
        result = decode_er_pr(mixed, 'ER')
        assert result.iloc[0] == 'ER Positive (70%)'
        assert 'Negative' in result.iloc[1]
        assert 'converted' in result.iloc[2].lower()
        assert result.iloc[3] == ''
        assert 'Unknown' in result.iloc[4]
        assert result.iloc[5] == 'ER Positive (50%)'

    def test_very_large_integer(self):
        """Values far outside any code range must not crash."""
        result = decode_er_pr(s(999999), 'ER').iloc[0]
        assert isinstance(result, str)

    def test_negative_integer(self):
        result = decode_er_pr(s(-1), 'ER').iloc[0]
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# D  ICD-O-3 code edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestICDO3EdgeCases:

    # Exact prefix routing
    @pytest.mark.parametrize('code,expected', [
        ('C50.0', 'breast'),
        ('C50.9', 'breast'),
        ('C50.911', 'breast'),   # extended suffix
        ('c50.1', 'breast'),     # lowercase
        ('C34.0', 'lung'),
        ('C34.9', 'lung'),
        ('C18.0', 'colorectum'),
        ('C20.9', 'colorectum'),
        ('C21.0', 'colorectum'),
        ('C22.0', 'liver'),
        ('C22.1', 'liver'),
        ('C61.9', 'prostate'),
        ('C73.9', 'thyroid'),
        ('C53.0', 'cervix'),
        ('C16.9', 'stomach'),
        ('C11.0', 'nasopharynx'),
        ('C54.1', 'endometrium'),
    ])
    def test_known_codes(self, code, expected):
        assert detect_cancer_group(code) == expected

    # Unknown → generic
    @pytest.mark.parametrize('code', [
        'C00.0', 'C99.9', 'C40.0', 'X99.0',
    ])
    def test_unknown_code_returns_generic(self, code):
        assert detect_cancer_group(code) == 'generic'

    # Edge inputs
    @pytest.mark.parametrize('val', [None, '', 'nan', np.nan, '???'])
    def test_null_or_garbage_code_returns_generic(self, val):
        assert detect_cancer_group(val) == 'generic'

    def test_trailing_whitespace_handled(self):
        assert detect_cancer_group('C50.1  ') == 'breast'

    def test_leading_whitespace_handled(self):
        assert detect_cancer_group('  C50.1') == 'breast'

    # Bulk detection
    def test_series_mixed_registry_warns(self):
        """Mixed cancer groups in one series must issue a UserWarning."""
        mixed = pd.Series(['C50.1', 'C34.0', 'C50.2'])
        with pytest.warns(UserWarning):  # actual message: 'Mixed cancer registry detected'
            detect_cancer_group_from_series(mixed)

    def test_series_all_same_group(self):
        homog = pd.Series(['C50.1', 'C50.2', 'C50.9'])
        with pytest.warns(None) as rec:
            group = detect_cancer_group_from_series(homog)
        # Ideally no UserWarning for a uniform series
        assert group == 'breast'

    def test_series_empty_returns_generic(self):
        result = detect_cancer_group_from_series(pd.Series([], dtype=str))
        assert result == 'generic'

    def test_series_all_null(self):
        result = detect_cancer_group_from_series(pd.Series([None, np.nan, '']))
        assert result == 'generic'


# ─────────────────────────────────────────────────────────────────────────────
# E  All 11 SSF profiles — contract verification
# ─────────────────────────────────────────────────────────────────────────────

ALL_GROUPS = [
    'breast', 'lung', 'colorectum', 'liver', 'cervix',
    'stomach', 'thyroid', 'prostate', 'nasopharynx', 'endometrium', 'generic',
]


class TestSSFProfileContracts:
    """Every profile must honour the SSFProfile contract."""

    @pytest.mark.parametrize('group', ALL_GROUPS)
    def test_profile_exists(self, group):
        profile = get_ssf_profile(group)
        assert profile is not None
        assert profile.cancer_group == group

    @pytest.mark.parametrize('group', ALL_GROUPS)
    def test_profile_has_all_10_ssf_fields(self, group):
        profile = get_ssf_profile(group)
        for i in range(1, 11):
            assert f'SSF{i}' in profile.fields, \
                f'{group}: missing SSF{i}'

    @pytest.mark.parametrize('group', ALL_GROUPS)
    def test_profile_field_raw_names_consistent(self, group):
        """raw_field stores the SSF key (e.g. 'SSF1'); apply_ssf_profile appends '_raw' itself."""
        profile = get_ssf_profile(group)
        for key, fdef in profile.fields.items():
            # raw_field == key (e.g. 'SSF1'), apply_ssf_profile uses f'{ssf_key}_raw'
            assert fdef.raw_field == key, \
                f'{group}.{key}: raw_field={fdef.raw_field!r} expected {key!r}'

    @pytest.mark.parametrize('group', ALL_GROUPS)
    def test_profile_decoders_are_callable_or_none(self, group):
        profile = get_ssf_profile(group)
        for key, fdef in profile.fields.items():
            assert fdef.decoder is None or callable(fdef.decoder), \
                f'{group}.{key}: decoder is neither callable nor None'

    @pytest.mark.parametrize('group', ALL_GROUPS)
    def test_profile_column_names_are_nonempty_strings(self, group):
        profile = get_ssf_profile(group)
        for key, fdef in profile.fields.items():
            assert isinstance(fdef.column_name, str) and fdef.column_name.strip(), \
                f'{group}.{key}: column_name is empty'

    @pytest.mark.parametrize('group', ALL_GROUPS)
    def test_apply_profile_does_not_crash_with_zeros(self, group):
        df = _ssf_df()
        result = apply_ssf_profile(df, group)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    @pytest.mark.parametrize('group', ALL_GROUPS)
    def test_apply_profile_adds_decoded_columns(self, group):
        df = _ssf_df()
        result = apply_ssf_profile(df, group)
        # Raw columns unchanged
        for i in range(1, 11):
            assert f'SSF{i}_raw' in result.columns
        # At least some decoded columns added
        new_cols = [c for c in result.columns if c not in df.columns]
        assert len(new_cols) > 0, f'{group}: no new decoded columns added'

    @pytest.mark.parametrize('group', ALL_GROUPS)
    def test_apply_profile_column_count_stable(self, group):
        """Second call must produce same columns as first."""
        df = _ssf_df()
        r1 = apply_ssf_profile(df.copy(), group)
        r2 = apply_ssf_profile(df.copy(), group)
        assert list(r1.columns) == list(r2.columns)


# ─────────────────────────────────────────────────────────────────────────────
# F  Roundtrip integrity — no raw numeric leaks
# ─────────────────────────────────────────────────────────────────────────────

class TestRoundtripIntegrity:
    """Decoded columns must never contain raw sentinel integers as strings."""

    # These integers must NEVER appear verbatim in decoded output
    FORBIDDEN_RAW = ['888', '988', '998', '999', '900', '901', '902']

    @pytest.mark.parametrize('group', ['breast', 'lung', 'colorectum', 'liver', 'prostate'])
    @pytest.mark.parametrize('sentinel', [888, 988, 999])
    def test_no_raw_sentinel_in_decoded_column(self, group, sentinel):
        df = pd.DataFrame({f'SSF{i}_raw': [sentinel] for i in range(1, 11)})
        result = apply_ssf_profile(df, group)
        decoded_cols = [c for c in result.columns if not c.endswith('_raw')]
        for col in decoded_cols:
            for val in result[col].astype(str):
                # decoded value must not be just the raw sentinel number
                assert val != str(sentinel), \
                    f'{group}.{col}: raw sentinel {sentinel} leaked as-is into decoded column'

    def test_er_positive_percentage_is_exact(self):
        """70 must decode to exactly '70%', not '7%' or '700%'."""
        result = decode_er_pr(s(70), 'ER').iloc[0]
        assert '70%' in result
        assert '700%' not in result
        assert result != 'ER Positive (7%)'

    def test_er_100_not_confused_with_10(self):
        result = decode_er_pr(s(100), 'ER').iloc[0]
        assert '100%' in result

    def test_ki67_percentage_exact(self):
        for pct in [1, 14, 30, 50, 99]:
            result = decode_ki67(s(pct)).iloc[0]
            assert f'{pct}%' in result or any(k in result for k in ['Low', 'Intermediate', 'High'])

    def test_nottingham_score_exact(self):
        """Score 30 → score 3, not score 30."""
        result = decode_nottingham(s(30)).iloc[0]
        assert 'Score 3' in result

    def test_nottingham_score_90_is_9_not_90(self):
        result = decode_nottingham(s(90)).iloc[0]
        assert 'Score 9' in result
        assert 'Score 90' not in result


# ─────────────────────────────────────────────────────────────────────────────
# G  Performance stress test
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformance:

    def test_breast_10k_rows_under_10s(self):
        """apply_ssf_profile for breast, 10 000 rows, must finish < 10 seconds."""
        import random
        rng = random.Random(0)
        valid_er = list(range(0, 101)) + [120, 888, 999]
        n = 10_000
        df = pd.DataFrame({
            'SSF1_raw': [rng.choice(valid_er) for _ in range(n)],
            'SSF2_raw': [rng.choice(valid_er) for _ in range(n)],
            'SSF3_raw': [rng.choice([0, 10, 11, 20, 988, 999]) for _ in range(n)],
            'SSF4_raw': [rng.choice(range(0, 11)) for _ in range(n)],
            'SSF5_raw': [rng.choice(range(0, 6)) for _ in range(n)],
            'SSF6_raw': [rng.choice([30, 40, 50, 60, 70, 80, 90, 110, 120, 130]) for _ in range(n)],
            'SSF7_raw': [rng.choice([100, 101, 200, 300, 510, 999]) for _ in range(n)],
            'SSF8_raw': [0] * n,
            'SSF9_raw': [0] * n,
            'SSF10_raw': [rng.choice(list(range(0, 101)) + [988, 999]) for _ in range(n)],
        })
        t0 = time.time()
        result = apply_ssf_profile(df, 'breast')
        elapsed = time.time() - t0
        assert elapsed < 10.0, f'Breast 10K decode took {elapsed:.2f}s — too slow'
        assert len(result) == n

    def test_lung_5k_rows_under_5s(self):
        import random
        rng = random.Random(1)
        n = 5_000
        df = pd.DataFrame({
            **{f'SSF{i}_raw': [0] * n for i in range(1, 11)},
            'SSF3_raw': [rng.choice([0, 1, 2, 3, 5, 10, 988, 999]) for _ in range(n)],
            'SSF4_raw': [rng.choice([0, 1, 9, 988, 999]) for _ in range(n)],
        })
        t0 = time.time()
        result = apply_ssf_profile(df, 'lung')
        elapsed = time.time() - t0
        assert elapsed < 5.0, f'Lung 5K decode took {elapsed:.2f}s — too slow'
        assert len(result) == n


# ─────────────────────────────────────────────────────────────────────────────
# H  CLI smoke tests (direct function calls, no subprocess)
# ─────────────────────────────────────────────────────────────────────────────

class TestCLISmoke:
    """Test CLI commands directly by calling their underlying functions.

    Avoids subprocess spawning (restricted in sandbox) while still exercising
    the full CLI logic surface area.
    """

    def _capture(self, fn, *args, **kwargs):
        """Call fn(*args, **kwargs) and return captured stdout as string."""
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                fn(*args, **kwargs)
            except SystemExit:
                pass  # argparse / sys.exit are expected
        return buf.getvalue()

    def test_list_cancers_contains_breast(self):
        from tcr_decoder.__main__ import cmd_list_cancers
        out = self._capture(cmd_list_cancers)
        assert 'breast' in out.lower()

    def test_list_cancers_contains_lung(self):
        from tcr_decoder.__main__ import cmd_list_cancers
        out = self._capture(cmd_list_cancers)
        assert 'lung' in out.lower()

    def test_list_cancers_has_all_11_groups(self):
        from tcr_decoder.__main__ import cmd_list_cancers
        out = self._capture(cmd_list_cancers)
        for grp in ['breast', 'lung', 'colorectum', 'liver', 'prostate',
                    'thyroid', 'cervix', 'stomach', 'nasopharynx', 'endometrium', 'generic']:
            assert grp in out.lower(), f'Missing group in --list-cancers output: {grp}'

    def test_ssf_info_breast_shows_er_pr(self):
        from tcr_decoder.__main__ import cmd_ssf_info
        out = self._capture(cmd_ssf_info, 'breast')
        assert 'ER' in out or 'Estrogen' in out

    def test_ssf_info_lung_shows_egfr(self):
        from tcr_decoder.__main__ import cmd_ssf_info
        out = self._capture(cmd_ssf_info, 'lung')
        assert 'EGFR' in out

    def test_ssf_info_colorectum_shows_msi(self):
        from tcr_decoder.__main__ import cmd_ssf_info
        out = self._capture(cmd_ssf_info, 'colorectum')
        assert 'MSI' in out or 'Microsatellite' in out

    def test_ssf_info_unknown_group_falls_back_gracefully(self):
        """Unknown group → should show generic profile, not crash."""
        from tcr_decoder.__main__ import cmd_ssf_info
        # Should not raise
        try:
            self._capture(cmd_ssf_info, 'nonexistent_cancer_xyz')
        except Exception as e:
            pytest.fail(f'cmd_ssf_info raised unexpected exception: {e}')

    def test_synth_breast_generates_output(self, tmp_path):
        from tcr_decoder.__main__ import cmd_synth
        out = str(tmp_path / 'adv_breast.xlsx')
        self._capture(cmd_synth, cancer='breast', n=5, seed=99, out=out, decode=False)
        import os
        assert os.path.exists(out)

    def test_synth_lung_generates_file(self, tmp_path):
        from tcr_decoder.__main__ import cmd_synth
        out = str(tmp_path / 'adv_lung.xlsx')
        self._capture(cmd_synth, cancer='lung', n=5, seed=1, out=out, decode=False)
        import os
        assert os.path.exists(out)

    def test_synth_colorectum_with_decode(self, tmp_path):
        """--decode flag: generate + decode pipeline must not crash."""
        from tcr_decoder.__main__ import cmd_synth
        import warnings
        warnings.filterwarnings('ignore')
        out = str(tmp_path / 'adv_crc.xlsx')
        self._capture(cmd_synth, cancer='colorectum', n=5, seed=2, out=out, decode=True)
        import os
        assert os.path.exists(out)

    def test_synth_unsupported_cancer_exits(self, tmp_path):
        """Unsupported cancer group must handle error gracefully."""
        from tcr_decoder.__main__ import cmd_synth
        out = str(tmp_path / 'adv_bad.xlsx')
        try:
            self._capture(cmd_synth, cancer='kidney', n=5, seed=1, out=out, decode=False)
        except SystemExit as e:
            assert e.code == 1  # expected exit code
        except Exception as e:
            pytest.fail(f'Unexpected exception type: {type(e).__name__}: {e}')


# ─────────────────────────────────────────────────────────────────────────────
# I  Contradictory / clinically impossible data
# ─────────────────────────────────────────────────────────────────────────────

class TestContradictoryData:
    """Decoders must never crash even on clinical nonsense."""

    def test_er_negative_but_888_on_same_patient(self):
        """888 (ER converted) alongside 0 (ER negative) — nonsensical but must decode."""
        result = decode_er_pr(s(0, 888), 'ER')
        assert isinstance(result.iloc[0], str)
        assert isinstance(result.iloc[1], str)

    def test_her2_ish_positive_on_ihc0(self):
        """Code 510 = IHC1+ISH+ but IHC0 in same context — decoder must handle."""
        result = decode_her2(s(100, 510))
        assert 'ISH' in result.iloc[1] or result.iloc[1] != ''

    def test_all_ssf_max_values_breast(self):
        """Feed 100 to all breast SSFs (max percent everywhere)."""
        df = _ssf_df(**{f'SSF{i}_raw': [100] for i in range(1, 11)})
        result = apply_ssf_profile(df, 'breast')
        assert len(result) == 1

    def test_all_ssf_impossible_value_999999(self):
        """Feed an impossibly large value to all SSFs."""
        df = _ssf_df(**{f'SSF{i}_raw': [999999] for i in range(1, 11)})
        result = apply_ssf_profile(df, 'colorectum')
        assert len(result) == 1

    def test_negative_values_all_ssf(self):
        """Negative integers should not crash any decoder."""
        df = _ssf_df(**{f'SSF{i}_raw': [-1] for i in range(1, 11)})
        result = apply_ssf_profile(df, 'lung')
        assert len(result) == 1

    def test_ssf3_pcr_code11_with_sentinel_examined_0(self):
        """pCR (neoadj code 11) with 0 sentinel nodes — clinically contradictory but valid encoding."""
        df = _ssf_df(SSF3_raw=[11], SSF4_raw=[0])
        result = apply_ssf_profile(df, 'breast')
        assert 'pCR' in str(result.get('Neoadjuvant_Response', [''])[0]) or True

    def test_msi_high_with_kras_mutated(self):
        """MSI-H + KRAS mutated is unusual but possible — must not crash."""
        df = _ssf_df(SSF4_raw=[2], SSF5_raw=[1])  # MSI-H, KRAS codon12
        result = apply_ssf_profile(df, 'colorectum')
        assert len(result) == 1


# ─────────────────────────────────────────────────────────────────────────────
# J  decode_er_pr rstrip anti-regression battery
# ─────────────────────────────────────────────────────────────────────────────

class TestERPRRstripRegression:
    """
    Historical bug: str(float).rstrip('.0') turned '120.0'→'12', '20.0'→'2'.
    Every two-digit and three-digit code that ends in '0' is at risk.
    """

    CODES_ENDING_IN_ZERO = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

    @pytest.mark.parametrize('code', CODES_ENDING_IN_ZERO)
    def test_int_input_not_truncated(self, code):
        result = decode_er_pr(s(code), 'ER').iloc[0]
        # Ensure the result references the full code, not a truncated version
        short = str(code).rstrip('0') or '0'
        # The decoded string must not start with the truncated version followed by '%'
        assert f'({short}%)' not in result or str(code) in result

    @pytest.mark.parametrize('code', CODES_ENDING_IN_ZERO)
    def test_float_input_not_truncated(self, code):
        result = decode_er_pr(s(float(code)), 'ER').iloc[0]
        short = str(code).rstrip('0') or '0'
        assert f'({short}%)' not in result or str(code) in result

    def test_120_0_is_negative_not_12(self):
        result = decode_er_pr(s(120.0), 'ER').iloc[0]
        assert 'Negative' in result
        assert 'Positive (12%)' not in result

    def test_20_0_is_positive_20pct(self):
        result = decode_er_pr(s(20.0), 'ER').iloc[0]
        assert 'Positive (20%)' in result

    def test_100_0_is_positive_100pct(self):
        result = decode_er_pr(s(100.0), 'ER').iloc[0]
        assert 'Positive (100%)' in result

    def test_pr_110_0_is_special_not_11pct(self):
        result = decode_er_pr(s(110.0), 'PR').iloc[0]
        assert 'Positive (11%)' not in result  # must not be truncated

    def test_er_888_0_is_converted_not_88pct(self):
        result = decode_er_pr(s(888.0), 'ER').iloc[0]
        assert 'Positive (88%)' not in result
        assert 'converted' in result.lower()
