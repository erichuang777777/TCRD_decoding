# -*- coding: utf-8 -*-
"""The bidirectional validation data sets must stay clean.

tcr_decoder/validation.py builds the evidence that every legal code and every
two-field combination of a cancer group survives raw code -> clinical meaning
-> raw code. These tests run that builder for every verified group and assert
the result is perfect, so the claim can never quietly rot.
"""

import pandas as pd
import pytest

from tcr_decoder.code_ranges import CODE_RANGES, SUPPORTED_GROUPS
from tcr_decoder.ssf_registry import get_ssf_profile
from tcr_decoder.validation import (
    EQUIVALENCE_CLASSES, SSF_KEYS, build_case_combinations, build_case_dataset,
    build_field_coverage, export_validation_workbook, pairwise_coverage,
    validate_cases, zh_definition,
)

GROUPS = list(SUPPORTED_GROUPS)

_field_cache = {}
_case_cache = {}


@pytest.fixture(scope='module')
def field_coverage():
    def _get(group):
        if group not in _field_cache:
            _field_cache[group] = build_field_coverage(group)
        return _field_cache[group]
    return _get


@pytest.fixture(scope='module')
def case_results():
    def _get(group):
        if group not in _case_cache:
            cases = build_case_combinations(group)
            per_field, clinical_view, _dec = validate_cases(
                build_case_dataset(cases, cancer_group=group), cancer_group=group)
            _case_cache[group] = (cases, per_field, clinical_view)
        return _case_cache[group]
    return _get


@pytest.mark.parametrize('group', GROUPS)
class TestFieldLevel:
    def test_covers_every_legal_code_of_every_field(self, group, field_coverage):
        coverage = field_coverage(group)
        for ssf_key in SSF_KEYS:
            expected = CODE_RANGES[group][ssf_key][1]
            got = set(coverage.loc[coverage['SSF'] == ssf_key, '原始代碼'])
            assert got == set(expected), f'{group}.{ssf_key}: {set(expected) ^ got}'

    def test_every_code_roundtrips_exactly(self, group, field_coverage):
        bad = field_coverage(group)
        bad = bad[~bad['往返一致']]
        assert len(bad) == 0, bad.head(10).to_string()

    def test_every_encoded_code_has_the_official_width(self, group, field_coverage):
        bad = field_coverage(group)
        bad = bad[~bad['寬度正確']]
        assert len(bad) == 0, bad.head(10).to_string()

    def test_every_code_has_a_chinese_codebook_definition(self, group, field_coverage):
        from tcr_decoder.validation import ZH_TRANSCRIBED
        if group not in ZH_TRANSCRIBED:
            pytest.skip(f'{group}: 中文定義尚未轉錄 (see docs/codebook_conformance_findings.md)')
        coverage = field_coverage(group)
        # The placeholder for a code with no transcribed definition STARTS
        # with 未定義; a legitimate definition may still mention the word
        # (e.g. endometrium SSF3 987, which the manual itself lists in the
        # 編碼範圍 but leaves out of the code table).
        missing = coverage[
            coverage['碼冊中文定義'].str.startswith('未定義')
            | coverage['碼冊中文定義'].str.contains('尚未轉錄')
            | (coverage['碼冊中文定義'].str.len() < 3)
        ]
        assert len(missing) == 0, missing.head(10).to_string()

    def test_clinical_meaning_is_unambiguous_within_a_field(self, group, field_coverage):
        """Two codes of the same field must not share one clinical meaning --
        that is what makes the reverse direction well-defined."""
        coverage = field_coverage(group)
        for ssf_key in SSF_KEYS:
            subset = coverage[coverage['SSF'] == ssf_key]
            dup = subset[subset.duplicated('解碼臨床意義 (EN)', keep=False)]
            assert len(dup) == 0, f'{group}.{ssf_key}:\n{dup.head(6).to_string()}'


class TestChineseDefinitions:
    @pytest.mark.parametrize('group,ssf_key,code,fragment', [
        ('breast', 'SSF1', '000', 'ER 0%'),
        ('breast', 'SSF1', 'S00', '強染'),
        ('breast', 'SSF1', '888', '前導性治療後的數值由陰性轉為陽性'),
        ('breast', 'SSF6', '060', 'BR score 6 分'),
        ('breast', 'SSF7', '000', '染色比例為 0%'),
        ('breast', 'SSF7', '100', '未描述染色比例'),
        ('breast', 'SSF7', '521', 'IHC 2+'),
        ('breast', 'SSF9', '990', '無殘餘腫瘤'),
        ('breast', 'SSF10', 'A05', '0.5%'),
        ('prostate', 'SSF1', '001', 'PSA ≦0.1 ng/ml'),
        ('prostate', 'SSF1', '045', 'PSA 4.5 ng/ml'),
        ('prostate', 'SSF1', '991', '1000-1999'),
        ('prostate', 'SSF2', '034', '主要級數3，次要級數4'),
        ('prostate', 'SSF2', '039', '次要級數不明'),
        ('prostate', 'SSF3', '007', "Gleason's Score 7 分"),
        ('prostate', 'SSF6', '012', '檢查 12 切片條數'),
        ('prostate', 'SSF7', '000', '均為陰性'),
        ('prostate', 'SSF8', '030', '肛門指檢與 TRUS'),
        ('prostate', 'SSF9', '988', '未收錄此欄位'),
    ])
    def test_matches_the_manual(self, group, ssf_key, code, fragment):
        assert fragment in zh_definition(group, ssf_key, code)


@pytest.mark.parametrize('group', GROUPS)
class TestCaseLevel:
    def test_pairwise_coverage_is_complete(self, group, case_results):
        cases, _, _ = case_results(group)
        covered, total = pairwise_coverage(cases, group)
        assert covered == total, f'{group}: {total - covered} pairs uncovered'

    def test_every_class_representative_appears(self, group, case_results):
        cases, _, _ = case_results(group)
        for ssf_key in SSF_KEYS:
            assert set(cases[ssf_key]) == set(EQUIVALENCE_CLASSES[group][ssf_key])

    def test_every_case_field_roundtrips_through_the_full_pipeline(self, group, case_results):
        _, per_field, _ = case_results(group)
        bad = per_field[~per_field['往返一致']]
        assert len(bad) == 0, bad.head(10).to_string()

    def test_clinical_view_has_a_meaning_for_every_case_and_field(self, group, case_results):
        """Every SSF column must carry clinical text for every case.

        The derived score columns (Molecular_Subtype, NPI_*) are deliberately
        excluded: they are allowed to be blank when the combination genuinely
        cannot support a conclusion (e.g. HER2 unknown), which is a correct
        clinical answer, not a decoding gap."""
        _, _, clinical_view = case_results(group)
        ssf_cols = [get_ssf_profile(group).fields[k].column_name for k in SSF_KEYS]
        for ssf_col in ssf_cols:
            blank = clinical_view[clinical_view[ssf_col].astype(str).str.strip() == '']
            assert len(blank) == 0, f'{group}.{ssf_col}: {len(blank)} blank'


class TestLeadingZeroIntegrity:
    def test_her2_000_survives_an_excel_round_trip(self, tmp_path):
        """Regression: pandas inferred a column of digit strings as int64, so
        HER2 '000' (IHC 0 with staining 0%) arrived at the decoder as '0' and
        re-encoded to '100' (IHC 0, staining % not described) -- a different
        code book entry. Found by the case-level validation data set."""
        from tcr_decoder.core import TCRDecoder
        from tcr_decoder.encoder import TCREncoder

        raw = build_case_dataset(build_case_combinations('breast').head(20))
        raw.loc[:, 'SSF7_raw'] = '000'
        xlsx = tmp_path / 'zeros.xlsx'
        with pd.ExcelWriter(str(xlsx), engine='openpyxl') as writer:
            raw.drop(columns=['Case_ID']).to_excel(
                writer, sheet_name='All_Fields_Decoded', index=False)

        dec = TCRDecoder(str(xlsx))
        dec.load(skip_input_check=True).decode()
        assert (dec._raw_df['SSF7_raw'].astype(str) == '000').all()
        encoded = TCREncoder(dec.clean, cancer_group='breast').encode(on_error='raise')
        assert (encoded['SSF7_raw'].astype(str) == '000').all()


@pytest.mark.parametrize('group', GROUPS)
def test_export_validation_workbook_reports_zero_failures(group, tmp_path):
    out = tmp_path / f'validation_{group}.xlsx'
    summary = export_validation_workbook(out, cancer_group=group)
    assert out.exists()
    assert summary['欄位往返失敗筆數'] == 0
    assert summary['病例往返失敗筆數'] == 0
    assert summary['欄位寬度錯誤筆數'] == 0
    assert summary['兩兩組合覆蓋率'] == '100.0%'
    sheets = pd.ExcelFile(str(out)).sheet_names
    assert {'說明', 'Summary', 'Field_Coverage', 'Case_Combinations',
            'Case_Roundtrip', 'Clinical_View', 'Failures'} <= set(sheets)


def test_unverified_group_is_refused_rather_than_half_built():
    """A group with no transcribed code ranges must fail loudly instead of
    producing a validation file that looks authoritative but isn't."""
    from tcr_decoder.validation import build_field_coverage as build
    with pytest.raises(ValueError, match='code ranges'):
        build('nasopharynx')
