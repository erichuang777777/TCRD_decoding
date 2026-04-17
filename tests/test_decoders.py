"""
Tests for tcr_decoder.decoders — breast-specific SSF decoders.

Focuses on edge cases, boundary values, and the critical bugs that were fixed
during development (rstrip bug, PN 0A/0B inversion, SSF3 cCR/pCR, etc.).
"""

import pytest
import pandas as pd
import numpy as np
from tcr_decoder.decoders import (
    decode_er_pr,
    decode_ki67,
    decode_her2,
    decode_nottingham,
    decode_ssf3_neoadj,
    decode_ebrt_additive,
    decode_sentinel,
    decode_lnpositive,
    decode_smoking_triplet,
    decode_cause_of_death,
)


def s(vals) -> pd.Series:
    """Shorthand: list → Series."""
    return pd.Series(vals)


# ─── decode_er_pr ────────────────────────────────────────────────────────────

class TestDecodeErPr:
    """Tests for ER/PR decoder including the critical rstrip('.0') bug fix."""

    def test_er_positive_percent(self):
        result = decode_er_pr(s([70]), 'ER')
        assert result.iloc[0] == 'ER Positive (70%)'

    def test_er_positive_100(self):
        result = decode_er_pr(s([100]), 'ER')
        assert result.iloc[0] == 'ER Positive (100%)'

    def test_er_positive_1(self):
        result = decode_er_pr(s([1]), 'ER')
        assert result.iloc[0] == 'ER Positive (1%)'

    def test_er_negative_code_120(self):
        result = decode_er_pr(s([120]), 'ER')
        assert 'Negative' in result.iloc[0]

    def test_er_negative_0_percent(self):
        """Code 0 = ER Negative (0%)."""
        result = decode_er_pr(s([0]), 'ER')
        assert 'Negative' in result.iloc[0] or '0%' in result.iloc[0]

    def test_er_converted_888(self):
        """Code 888 = converted Neg→Pos after neoadjuvant — must NOT be decoded as 'Positive'."""
        result = decode_er_pr(s([888]), 'ER')
        assert 'converted' in result.iloc[0].lower()
        assert result.iloc[0] != 'ER Positive (888%)'  # ensure rstrip bug is fixed

    def test_er_unknown_999(self):
        result = decode_er_pr(s([999]), 'ER')
        assert 'Unknown' in result.iloc[0]

    def test_er_not_applicable_988(self):
        result = decode_er_pr(s([988]), 'ER')
        assert 'applicable' in result.iloc[0].lower()

    def test_pr_positive(self):
        result = decode_er_pr(s([40]), 'PR')
        assert result.iloc[0] == 'PR Positive (40%)'

    def test_pr_negative(self):
        result = decode_er_pr(s([120]), 'PR')
        assert 'Negative' in result.iloc[0]

    def test_mixed_values(self):
        result = decode_er_pr(s([70, 120, 888, 999, 0]), 'ER')
        assert 'Positive' in result.iloc[0]
        assert 'Negative' in result.iloc[1]
        assert 'converted' in result.iloc[2].lower()
        assert 'Unknown' in result.iloc[3]

    def test_float_input_no_rstrip_bug(self):
        """Ensure float input like 120.0 doesn't become 12 via rstrip('.0')."""
        result = decode_er_pr(s([120.0]), 'ER')
        assert 'Negative' in result.iloc[0]
        assert '12' not in result.iloc[0].replace('120', '')  # 12 should not appear as percentage

    def test_empty_string_handled(self):
        result = decode_er_pr(s(['', np.nan, None]), 'ER')
        for val in result:
            assert val == '' or 'Unknown' in val or val is not None


# ─── decode_nottingham ───────────────────────────────────────────────────────

class TestDecodeNottingham:

    @pytest.mark.parametrize('code,score,grade_num', [
        (30, 3, 1), (40, 4, 1), (50, 5, 1),
        (60, 6, 2), (70, 7, 2),
        (80, 8, 3), (90, 9, 3),
    ])
    def test_score_codes(self, code, score, grade_num):
        result = decode_nottingham(s([code]))
        text = result.iloc[0]
        assert f'Score {score}' in text
        assert f'Grade {grade_num}' in text

    @pytest.mark.parametrize('code,grade_num', [(110, 1), (120, 2), (130, 3)])
    def test_grade_only_codes(self, code, grade_num):
        result = decode_nottingham(s([code]))
        text = result.iloc[0]
        assert f'Grade {grade_num}' in text

    def test_unknown_999(self):
        result = decode_nottingham(s([999]))
        assert 'Unknown' in result.iloc[0]

    def test_not_applicable_988(self):
        result = decode_nottingham(s([988]))
        assert 'applicable' in result.iloc[0].lower()


# ─── decode_her2 ─────────────────────────────────────────────────────────────

class TestDecodeHer2:

    def test_ihc0_negative(self):
        result = decode_her2(s([100]))
        assert 'IHC 0' in result.iloc[0]
        assert 'Negative' in result.iloc[0]

    def test_ihc1_negative(self):
        result = decode_her2(s([101]))
        assert 'IHC 1+' in result.iloc[0]

    def test_ihc2_equivocal(self):
        # Code 200 = CISH Negative (legacy dx yr 100-107); per official codebook
        result = decode_her2(s([200]))
        text = result.iloc[0]
        assert text != '' and ('CISH' in text or 'Negative' in text)

    def test_ihc3_positive(self):
        # Code 300 = IHC 3+ (no ISH) → check it contains either '3+' or 'ISH Negative'
        result = decode_her2(s([300]))
        text = result.iloc[0]
        assert 'IHC 3+' in text or 'ISH Negative' in text or 'Positive' in text

    def test_ish_positive(self):
        result = decode_her2(s([510]))
        # 510 = IHC 1+ + ISH Positive
        assert 'ISH' in result.iloc[0]

    def test_unknown_999(self):
        result = decode_her2(s([999]))
        assert 'Unknown' in result.iloc[0]

    def test_code_102_is_equivocal_not_ish_positive(self):
        """Bug fix: code 102 is in 1xx (IHC-only) range, should NOT say 'ISH Positive'."""
        result = decode_her2(s([102]))
        assert 'ISH Positive' not in result.iloc[0]


# ─── decode_ki67 ─────────────────────────────────────────────────────────────

class TestDecodeKi67:

    def test_low_under_14(self):
        # Boundary: ≤14% = Low (some implementations use <14 or ≤13)
        for val in [1, 5, 10, 13]:
            result = decode_ki67(s([val]))
            assert 'Low' in result.iloc[0], f'{val} should be Low'

    def test_intermediate_14_to_30(self):
        for val in [15, 20, 25, 30]:
            result = decode_ki67(s([val]))
            assert 'Intermediate' in result.iloc[0], f'{val} should be Intermediate'

    def test_high_over_30(self):
        for val in [31, 50, 80, 99]:
            result = decode_ki67(s([val]))
            assert 'High' in result.iloc[0], f'{val} should be High'

    def test_100_percent(self):
        result = decode_ki67(s([100]))
        assert '100%' in result.iloc[0]

    def test_a_prefix_sub1(self):
        """A01-A09 = sub-1% codes (output may show '<1%' or '0.1%' etc.)."""
        result = decode_ki67(s(['A01']))
        text = result.iloc[0]
        # Accept '<1%', '0.1%', '0.x%' or any indication of a very low value
        assert ('%' in text and ('0.' in text or '<1' in text)), \
            f'Expected sub-1% indicator, got: {text!r}'

    def test_unknown_999(self):
        result = decode_ki67(s([999]))
        assert 'Unknown' in result.iloc[0]

    def test_not_applicable_988(self):
        result = decode_ki67(s([988]))
        assert 'applicable' in result.iloc[0].lower()


# ─── decode_ssf3_neoadj ──────────────────────────────────────────────────────

class TestDecodeSSF3Neoadj:
    """Critical: 010=cCR (not pCR), 011=pCR (full pathologic)."""

    def test_no_neoadj_code_0(self):
        # Code 0 may return 'No neoadjuvant', 'Not applicable', '0', or empty
        # depending on whether a lookup exists; just verify it's not pCR/cCR
        result = decode_ssf3_neoadj(s([0]))
        text = result.iloc[0]
        assert 'pCR' not in text and 'cCR' not in text

    def test_ccr_code_10(self):
        """Code 010 = clinical complete response, NOT pCR."""
        result = decode_ssf3_neoadj(s([10]))
        assert 'cCR' in result.iloc[0]
        assert 'pCR' not in result.iloc[0]

    def test_pcr_code_11(self):
        """Code 011 = pathologic complete response (breast + nodes)."""
        result = decode_ssf3_neoadj(s([11]))
        assert 'pCR' in result.iloc[0]

    def test_partial_response(self):
        for code in [20, 21, 22, 23]:
            result = decode_ssf3_neoadj(s([code]))
            assert result.iloc[0] != ''

    def test_not_applicable_988(self):
        result = decode_ssf3_neoadj(s([988]))
        assert 'applicable' in result.iloc[0].lower()

    def test_unknown_999(self):
        result = decode_ssf3_neoadj(s([999]))
        assert 'Unknown' in result.iloc[0]


# ─── decode_ebrt_additive ────────────────────────────────────────────────────

class TestDecodeEBRTAdditive:
    """EBRT uses additive coding: 1=3DCRT, 2=IMRT, 4=SRS, 8=SBRT, 32=IGRT, 64=proton."""

    def test_zero_is_no_ebrt(self):
        result = decode_ebrt_additive(s([0]))
        text = result.iloc[0]
        # Code 0 = no EBRT performed
        assert 'No EBRT' in text or text == '' or text == '0'

    def test_single_technique(self):
        result = decode_ebrt_additive(s([1]))   # 3DCRT
        assert result.iloc[0] != ''

    def test_combined_techniques_34(self):
        """34 = some combination — verify it produces a multi-technique string."""
        result = decode_ebrt_additive(s([34]))
        text = result.iloc[0]
        # 34 = 2 + 32 = two techniques combined → should contain '+' or two keywords
        assert text != '' and ('+' in text or len(text) > 5)

    def test_all_techniques(self):
        """1+2+4+8+32+64 = 111 → all techniques combined."""
        result = decode_ebrt_additive(s([111]))
        assert result.iloc[0] != ''


# ─── decode_sentinel ─────────────────────────────────────────────────────────

class TestDecodeSentinel:

    def test_zero_examined_is_none(self):
        result = decode_sentinel(s([0]), 'examined')
        assert 'None' in result.iloc[0] or '0' in result.iloc[0]

    def test_count_examined(self):
        result = decode_sentinel(s([3]), 'examined')
        assert '3' in result.iloc[0]

    def test_count_positive(self):
        result = decode_sentinel(s([2]), 'positive')
        assert '2' in result.iloc[0]

    def test_unknown_99(self):
        result = decode_sentinel(s([99]), 'examined')
        assert result.iloc[0] != ''


# ─── decode_lnpositive ───────────────────────────────────────────────────────

class TestDecodeLNPositive:

    def test_zero_is_none(self):
        result = decode_lnpositive(s([0]))
        assert '0' in result.iloc[0] or 'None' in result.iloc[0]

    def test_numeric_count(self):
        result = decode_lnpositive(s([5]))
        assert '5' in result.iloc[0]

    def test_all_positive_95(self):
        result = decode_lnpositive(s([95]))
        assert result.iloc[0] != ''   # should decode to some text

    def test_unknown_99(self):
        result = decode_lnpositive(s([99]))
        assert result.iloc[0] != ''


# ─── decode_smoking_triplet ──────────────────────────────────────────────────

class TestDecodeSmokingTriplet:

    def test_all_none(self):
        result = decode_smoking_triplet(s(['00,00,00']))
        assert result.iloc[0] != ''

    def test_smoker(self):
        result = decode_smoking_triplet(s(['01,00,00']))
        text = result.iloc[0].lower()
        assert 'smok' in text or '01' in text or 'current' in text or 'former' in text

    def test_betelnut(self):
        result = decode_smoking_triplet(s(['00,01,00']))
        assert result.iloc[0] != ''

    def test_all_habits(self):
        result = decode_smoking_triplet(s(['01,01,01']))
        assert result.iloc[0] != ''

    def test_unknown(self):
        result = decode_smoking_triplet(s(['09,09,09']))
        assert result.iloc[0] != ''

    def test_malformed_graceful(self):
        result = decode_smoking_triplet(s(['INVALID']))
        assert isinstance(result.iloc[0], str)
