"""
Round 3 adversarial regression tests — common script-level data-pipeline bugs.

These tests lock in fixes for bugs that Round 1 (basic correctness) and
Round 2 (clinical applicability gates) did NOT catch:

  K1 — Excel round-trip integrity (leading-zero patient IDs)
  K2 — Friendly error on missing / wrong sheet name
  K3 — Decimal ER/PR percent extraction (silent wrong answer)
  K4 — ScoreRegistry idempotent registration (notebook reload safety)
  K5 — Column-header whitespace normalisation
  K6 — Empty DataFrame / header-only input
  K7 — Corrupted / non-Excel input file
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tcr_decoder.core import TCRDecoder
from tcr_decoder.data_dictionary import generate_data_dictionary
from tcr_decoder.derived import add_er_pr_percent
from tcr_decoder.input_validator import validate_input
from tcr_decoder.scores.base import ScoreRegistry
from tcr_decoder.validators import (
    validate_egfr_without_targeted,
    validate_msi_immunotherapy,
)


# ─────────────────────────────────────────────────────────────────────────────
# K1 — Excel round-trip: leading-zero patient IDs must survive read
# ─────────────────────────────────────────────────────────────────────────────

def _write_minimal_tcr(tmp_path: Path, pk_values: list[str],
                        sheet_name: str = 'All_Fields_Decoded') -> Path:
    """Write a tiny TCR-shaped Excel file with just PK and TCODE1."""
    df = pd.DataFrame({
        'PK_raw': pk_values,
        'SEX_raw': ['1'] * len(pk_values),
        'SEX_decoded': ['Female'] * len(pk_values),
        'TCODE1_raw': ['C509'] * len(pk_values),
        'TCODE1_decoded': ['Breast NOS'] * len(pk_values),
        'MCODE_raw': ['8500'] * len(pk_values),
    })
    path = tmp_path / 'mini.xlsx'
    with pd.ExcelWriter(path, engine='openpyxl') as w:
        df.to_excel(w, sheet_name=sheet_name, index=False)
    return path


class TestK1_LeadingZeroPatientID:
    """Leading zeros in Patient_ID must NOT be silently stripped on Excel read."""

    def test_leading_zero_pk_preserved_as_string(self, tmp_path):
        path = _write_minimal_tcr(tmp_path, ['0001234', '0005678', '0099999'])
        dec = TCRDecoder(path).load(skip_input_check=True)

        pk = dec._raw_df['PK_raw']
        # Must be string dtype — not Int64 or object-of-ints
        assert pk.dtype == object
        assert pk.tolist() == ['0001234', '0005678', '0099999']

    def test_mixed_numeric_and_zero_padded_pk(self, tmp_path):
        path = _write_minimal_tcr(tmp_path, ['0001', '1234', '9999999'])
        dec = TCRDecoder(path).load(skip_input_check=True)
        pk = dec._raw_df['PK_raw'].tolist()
        assert pk == ['0001', '1234', '9999999']
        # The '0001' case is the one that would break without dtype=str
        assert pk[0] != '1'
        assert pk[0] != 1


# ─────────────────────────────────────────────────────────────────────────────
# K2 — Friendly errors for missing / wrong sheet name
# ─────────────────────────────────────────────────────────────────────────────

class TestK2_FriendlySheetNameError:
    def test_missing_sheet_lists_available_sheets(self, tmp_path):
        path = _write_minimal_tcr(tmp_path, ['001'], sheet_name='Data')
        dec = TCRDecoder(path, sheet_name='All_Fields_Decoded')
        with pytest.raises(ValueError) as exc_info:
            dec.load(skip_input_check=True)
        msg = str(exc_info.value)
        assert "'All_Fields_Decoded'" in msg
        assert 'Data' in msg  # lists available sheets
        assert 'Available sheets' in msg

    def test_missing_file_raises_filenotfounderror(self, tmp_path):
        missing = tmp_path / 'does_not_exist.xlsx'
        dec = TCRDecoder(missing)
        with pytest.raises(FileNotFoundError):
            dec.load(skip_input_check=True)

    def test_corrupted_excel_file_clear_error(self, tmp_path):
        # Write plain text with .xlsx extension → not a valid workbook
        bad = tmp_path / 'broken.xlsx'
        bad.write_text('this is not a valid excel file', encoding='utf-8')
        dec = TCRDecoder(bad)
        with pytest.raises(ValueError) as exc_info:
            dec.load(skip_input_check=True)
        assert 'not a valid' in str(exc_info.value).lower() or \
               'failed' in str(exc_info.value).lower()


# ─────────────────────────────────────────────────────────────────────────────
# K3 — Decimal ER/PR percent extraction bug
# ─────────────────────────────────────────────────────────────────────────────

class TestK3_DecimalERPRExtraction:
    """The regex `(\\d+)%` silently extracted only the last digit group of a
    decimal: '15.3%' → 3 (wrong), '10.5%' → 5 (wrong). The fix uses
    `([0-9]+(?:\\.[0-9]+)?)%` to capture the full number."""

    def test_integer_percent(self):
        df = pd.DataFrame({'ER_Status': ['ER Positive (90%)']})
        out = add_er_pr_percent(df)
        assert out['ER_Percent'].iloc[0] == 90.0

    def test_decimal_percent_preserved(self):
        df = pd.DataFrame({'ER_Status': [
            'ER Positive (10.5%)',
            'ER Positive (15.3%)',
            'ER Positive (90.5%)',
            'ER Positive (99.9%)',
        ]})
        out = add_er_pr_percent(df)
        assert out['ER_Percent'].tolist() == [10.5, 15.3, 90.5, 99.9]

    def test_regression_15_3_is_not_extracted_as_3(self):
        """The original bug: '15.3%' → 3 (off by 10x)."""
        df = pd.DataFrame({'ER_Status': ['ER Positive (15.3%)']})
        out = add_er_pr_percent(df)
        assert out['ER_Percent'].iloc[0] != 3.0
        assert out['ER_Percent'].iloc[0] == 15.3

    def test_lt1_convention(self):
        df = pd.DataFrame({'ER_Status': ['ER Negative (<1%)']})
        out = add_er_pr_percent(df)
        assert out['ER_Percent'].iloc[0] == 0.5

    def test_zero_percent_negative(self):
        df = pd.DataFrame({'ER_Status': ['ER Negative (0%)']})
        out = add_er_pr_percent(df)
        assert out['ER_Percent'].iloc[0] == 0.0

    def test_out_of_range_clipped_to_nan(self):
        # 120% is biologically impossible → NaN
        df = pd.DataFrame({'ER_Status': ['ER Positive (120%)']})
        out = add_er_pr_percent(df)
        assert pd.isna(out['ER_Percent'].iloc[0])

    def test_empty_and_nan(self):
        df = pd.DataFrame({'ER_Status': ['', None, 'Unknown']})
        out = add_er_pr_percent(df)
        assert out['ER_Percent'].isna().all()

    def test_pr_percent_also_fixed(self):
        df = pd.DataFrame({'PR_Status': ['PR Positive (25.7%)']})
        out = add_er_pr_percent(df)
        assert out['PR_Percent'].iloc[0] == 25.7


# ─────────────────────────────────────────────────────────────────────────────
# K4 — ScoreRegistry idempotent registration
# ─────────────────────────────────────────────────────────────────────────────

class TestK4_IdempotentScoreRegistry:
    """Reloading a score module must not double-register."""

    def test_reload_npi_does_not_duplicate(self):
        import tcr_decoder.scores.npi as npi_mod
        n_before = len(ScoreRegistry._scores)
        names_before = [s.NAME for s in ScoreRegistry._scores]

        importlib.reload(npi_mod)

        n_after = len(ScoreRegistry._scores)
        names_after = [s.NAME for s in ScoreRegistry._scores]
        assert n_after == n_before, \
            f'Registry grew on reload: {names_before} → {names_after}'
        assert len(names_after) == len(set(names_after)), \
            f'Duplicate score names after reload: {names_after}'

    def test_manual_duplicate_registration_is_noop(self):
        from tcr_decoder.scores.npi import NPIScore
        n_before = len(ScoreRegistry._scores)
        # Attempt to register the same class again
        ScoreRegistry.register(NPIScore)
        n_after = len(ScoreRegistry._scores)
        assert n_after == n_before

    def test_registry_names_are_unique(self):
        names = [s.NAME for s in ScoreRegistry._scores]
        assert len(names) == len(set(names)), \
            f'Registry contains duplicate score NAMEs: {names}'


# ─────────────────────────────────────────────────────────────────────────────
# K5 — Column header whitespace normalisation
# ─────────────────────────────────────────────────────────────────────────────

class TestK5_ColumnWhitespace:
    def test_stripped_column_headers_on_load(self, tmp_path):
        # Write a file where column headers have leading/trailing whitespace
        df = pd.DataFrame({
            ' PK_raw ': ['001', '002'],
            'SEX_raw': ['1', '2'],
            'SEX_decoded': ['Female', 'Male'],
            'TCODE1_raw': ['C509', 'C509'],
            'TCODE1_decoded': ['Breast', 'Breast'],
            'MCODE_raw ': ['8500', '8500'],  # trailing space
        })
        path = tmp_path / 'whitespace_headers.xlsx'
        with pd.ExcelWriter(path, engine='openpyxl') as w:
            df.to_excel(w, sheet_name='All_Fields_Decoded', index=False)

        dec = TCRDecoder(path).load(skip_input_check=True)
        cols = list(dec._raw_df.columns)
        assert 'PK_raw' in cols
        assert 'MCODE_raw' in cols
        assert ' PK_raw ' not in cols
        assert 'MCODE_raw ' not in cols

    def test_validator_warns_on_whitespace_headers(self):
        df = pd.DataFrame({
            ' PK_raw ': ['1'],
            'SEX_raw': ['1'],
        })
        result = validate_input(df)
        warnings_text = ' '.join(w['Check'] + w['Detail'] for w in result.warnings)
        assert 'whitespace' in warnings_text.lower() or \
               'Whitespace' in warnings_text

    def test_validator_errors_on_even_one_missing_required_raw_field(self):
        from tcr_decoder.input_validator import REQUIRED_RAW_FIELDS
        df = pd.DataFrame({
            f'{field}_raw': ['1']
            for field in REQUIRED_RAW_FIELDS
            if field != 'AGE'
        })
        result = validate_input(df)
        assert not result.is_ok
        assert any(e['Check'] == 'Missing columns' and 'AGE' in e['Detail']
                   for e in result.errors)


# ─────────────────────────────────────────────────────────────────────────────
# K6 — Empty DataFrame / header-only file handling
# ─────────────────────────────────────────────────────────────────────────────

class TestK6_EmptyInput:
    def test_empty_dataframe_validator_errors(self):
        result = validate_input(pd.DataFrame())
        assert not result.is_ok
        assert any('empty' in e['Check'].lower() or 'empty' in e['Detail'].lower()
                   for e in result.errors)

    def test_header_only_excel_clear_error(self, tmp_path):
        # Create an Excel file with headers but zero data rows
        df = pd.DataFrame(columns=['PK_raw', 'SEX_raw'])
        path = tmp_path / 'headers_only.xlsx'
        with pd.ExcelWriter(path, engine='openpyxl') as w:
            df.to_excel(w, sheet_name='All_Fields_Decoded', index=False)

        dec = TCRDecoder(path)
        with pytest.raises(ValueError) as exc_info:
            dec.load(skip_input_check=True)
        assert '0 data rows' in str(exc_info.value) or \
               'contains 0' in str(exc_info.value)


class TestK8_ValidationAndReporting:
    def test_load_blocks_input_validation_errors_by_default(self, tmp_path):
        path = _write_minimal_tcr(tmp_path, ['001'])
        dec = TCRDecoder(path)
        with pytest.raises(ValueError, match='Input validation failed'):
            dec.load()

    def test_egfr_mutation_validator_matches_decoded_lung_text(self):
        df = pd.DataFrame({
            'Patient_ID': ['L001', 'L002', 'L003'],
            'EGFR_Mutation': [
                'EGFR - Exon 19 deletion',
                'EGFR - No mutation (XXX)',
                'Unknown / not tested',
            ],
            'Targeted_This_Hosp': [
                'No targeted therapy',
                'No targeted therapy',
                'No targeted therapy',
            ],
        })
        flags = validate_egfr_without_targeted(df)
        assert len(flags) == 1
        assert flags[0]['Patient_ID'] == 'L001'

    def test_egfr_validator_does_not_flag_malformed_code_as_mutation(self):
        df = pd.DataFrame({
            'Patient_ID': ['L001'],
            'EGFR_Mutation': ['EGFR code: QQQ'],
            'Targeted_This_Hosp': ['No targeted therapy'],
        })
        assert validate_egfr_without_targeted(df) == []

    def test_msi_validator_uses_crc_msi_mmr_column(self):
        df = pd.DataFrame({
            'Patient_ID': ['C001', 'C002'],
            'MSI_MMR_Status': [
                'MSI-H - High instability; or MMR deficient (dMMR)',
                'MSS / Microsatellite stable; MMR proficient (pMMR)',
            ],
            'Immuno_This_Hosp': ['No immunotherapy', 'No immunotherapy'],
        })
        flags = validate_msi_immunotherapy(df)
        assert len(flags) == 1
        assert flags[0]['Patient_ID'] == 'C001'

    def test_data_dictionary_counts_unknown_and_not_applicable_as_missing(self):
        df = pd.DataFrame({
            'ER_Status': [
                'ER Positive (70%)',
                'Unknown / not stated',
                'Not applicable',
                '',
            ],
        })
        dd = generate_data_dictionary(df)
        row = dd.iloc[0]
        assert row['N_Filled'] == 1
        assert row['N_Missing'] == 3
        assert row['Completeness_%'] == 25.0

    def test_data_dictionary_handles_zero_rows(self):
        dd = generate_data_dictionary(pd.DataFrame({'ER_Status': []}))
        row = dd.iloc[0]
        assert row['N_Filled'] == 0
        assert row['N_Missing'] == 0
        assert row['Completeness_%'] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# K7 — End-to-end pipeline still works after fixes
# ─────────────────────────────────────────────────────────────────────────────

class TestK7_EndToEndSmoke:
    def test_pipeline_still_runs_after_round3_fixes(self, breast_clean_df):
        """Smoke test: Round 3 fixes must not break the full pipeline."""
        assert breast_clean_df is not None
        assert len(breast_clean_df) > 0
        # ER_Percent must be numeric (not str)
        if 'ER_Percent' in breast_clean_df.columns:
            assert pd.api.types.is_numeric_dtype(breast_clean_df['ER_Percent'])
