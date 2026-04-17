"""
Tests for tcr_decoder.synth — synthetic data generator.
"""

import pytest
import pandas as pd
import numpy as np
from tcr_decoder.synth import SyntheticTCRGenerator


class TestSyntheticGeneratorBasic:

    def test_supported_cancers(self):
        assert 'breast' in SyntheticTCRGenerator.SUPPORTED
        assert 'lung' in SyntheticTCRGenerator.SUPPORTED
        assert 'colorectum' in SyntheticTCRGenerator.SUPPORTED

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match='not supported'):
            SyntheticTCRGenerator(cancer_group='unknown')

    def test_generates_correct_row_count(self):
        for cancer in SyntheticTCRGenerator.SUPPORTED:
            gen = SyntheticTCRGenerator(cancer_group=cancer, n=20, seed=1)
            df = gen.generate()
            assert len(df) == 20, f'{cancer}: expected 20 rows, got {len(df)}'

    def test_reproducible_with_same_seed(self):
        gen1 = SyntheticTCRGenerator(cancer_group='breast', n=10, seed=42)
        gen2 = SyntheticTCRGenerator(cancer_group='breast', n=10, seed=42)
        df1, df2 = gen1.generate(), gen2.generate()
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        gen1 = SyntheticTCRGenerator(cancer_group='breast', n=20, seed=42)
        gen2 = SyntheticTCRGenerator(cancer_group='breast', n=20, seed=99)
        df1, df2 = gen1.generate(), gen2.generate()
        assert not df1.equals(df2)

    def test_required_raw_columns_present(self):
        from tcr_decoder.input_validator import REQUIRED_RAW_FIELDS
        gen = SyntheticTCRGenerator(cancer_group='breast', n=5, seed=1)
        df = gen.generate()
        for field in REQUIRED_RAW_FIELDS:
            col = f'{field}_raw'
            assert col in df.columns, f'Missing column: {col}'

    def test_pk_unique(self):
        for cancer in SyntheticTCRGenerator.SUPPORTED:
            gen = SyntheticTCRGenerator(cancer_group=cancer, n=50, seed=7)
            df = gen.generate()
            assert df['PK_raw'].nunique() == 50, f'{cancer}: duplicate PKs'

    def test_tcode1_matches_cancer_group(self):
        """TCODE1 should match the expected ICD-O-3 prefix."""
        from tcr_decoder.ssf_registry import detect_cancer_group
        mapping = {
            'breast': 'C50',
            'lung': 'C34',
            'colorectum': ('C18', 'C19', 'C20', 'C21'),
        }
        for cancer, prefixes in mapping.items():
            gen = SyntheticTCRGenerator(cancer_group=cancer, n=20, seed=5)
            df = gen.generate()
            for code in df['TCODE1_raw']:
                if isinstance(prefixes, str):
                    assert code.startswith(prefixes), \
                        f'{cancer}: unexpected TCODE1={code}'
                else:
                    assert any(code.startswith(p) for p in prefixes), \
                        f'{cancer}: unexpected TCODE1={code}'

    def test_age_in_plausible_range(self):
        gen = SyntheticTCRGenerator(cancer_group='breast', n=100, seed=1)
        df = gen.generate()
        ages = pd.to_numeric(df['AGE_raw'], errors='coerce').dropna()
        assert (ages >= 18).all()
        assert (ages <= 95).all()

    def test_sex_mostly_female_for_breast(self):
        gen = SyntheticTCRGenerator(cancer_group='breast', n=200, seed=1)
        df = gen.generate()
        pct_female = (df['SEX_raw'] == '2').mean()
        assert pct_female >= 0.95, f'Expected ≥95% female for breast, got {pct_female:.0%}'

    def test_ssf1_range_breast(self):
        gen = SyntheticTCRGenerator(cancer_group='breast', n=100, seed=1)
        df = gen.generate()
        vals = pd.to_numeric(df['SSF1_raw'], errors='coerce').dropna()
        valid = set(range(0, 101)) | {120, 888, 999}
        invalid = set(vals.astype(int).unique()) - valid
        assert len(invalid) == 0, f'Invalid SSF1 values: {invalid}'

    def test_ssf3_lung_performance_status_codes(self):
        """Lung SSF3 = Performance Status per 2025 TCR codebook (p.111).
        Valid codes: 0=PS0, 1=PS1, 2=PS2, 3=PS3, 4=PS4, 988=N/A, 999=unknown.
        """
        gen = SyntheticTCRGenerator(cancer_group='lung', n=100, seed=2)
        df = gen.generate()
        vals = pd.to_numeric(df['SSF3_raw'], errors='coerce').dropna().astype(int)
        valid = {0, 1, 2, 3, 4, 988, 999}
        invalid = set(vals.unique()) - valid
        assert len(invalid) == 0, f'Invalid Performance_Status codes: {invalid}'

    def test_ssf6_lung_egfr_alpha_codes(self):
        """Lung SSF6 = EGFR mutation (3-char alpha code) per 2025 TCR codebook (p.117).
        Valid codes: 'XXX'=WT, 'AXX'=exon19del, 'BXX'=L858R, 'VVV'=mutated NOS, '999'=unknown, etc.
        """
        gen = SyntheticTCRGenerator(cancer_group='lung', n=100, seed=2)
        df = gen.generate()
        vals = df['SSF6_raw'].dropna().astype(str)
        # All valid codes are 3-char alphabetic or '999'
        invalid = [v for v in vals if not (len(v) == 3 and v.isalpha()) and v != '999']
        assert len(invalid) == 0, f'Invalid EGFR alpha codes: {set(invalid)}'

    def test_ssf7_lung_alk_codes(self):
        """Lung SSF7 = ALK translocation per 2025 TCR codebook (p.119).
        Valid codes: 10=positive, 20=negative, 30=uninterpretable, 999=unknown.
        """
        gen = SyntheticTCRGenerator(cancer_group='lung', n=100, seed=2)
        df = gen.generate()
        vals = pd.to_numeric(df['SSF7_raw'], errors='coerce').dropna().astype(int)
        valid = {10, 20, 30, 999}
        invalid = set(vals.unique()) - valid
        assert len(invalid) == 0, f'Invalid ALK codes: {invalid}'

    def test_ssf4_colorectum_crm_codes(self):
        """CRC SSF4 = Circumferential Resection Margin per 2025 TCR codebook.
        Valid codes: 0-980 (in 0.1mm units), 988=N/A, 999=unknown.
        """
        gen = SyntheticTCRGenerator(cancer_group='colorectum', n=100, seed=3)
        df = gen.generate()
        vals = pd.to_numeric(df['SSF4_raw'], errors='coerce').dropna().astype(int)
        # All valid values should be 0-980 or 988/999
        invalid = set(v for v in vals.unique() if not (0 <= v <= 980 or v in (988, 999)))
        assert len(invalid) == 0, f'Invalid CRM codes: {invalid}'

    def test_ssf10_colorectum_msi_codes(self):
        """CRC SSF10 = MSI/MMR status per 2025 TCR codebook (p.87).
        Valid codes: 0=MSS, 10=MSI-L, 20=MSI-H/dMMR, 988=N/A, 999=not tested.
        """
        gen = SyntheticTCRGenerator(cancer_group='colorectum', n=100, seed=3)
        df = gen.generate()
        vals = pd.to_numeric(df['SSF10_raw'], errors='coerce').dropna().astype(int)
        valid = {0, 10, 20, 988, 999}
        invalid = set(vals.unique()) - valid
        assert len(invalid) == 0, f'Invalid MSI codes: {invalid}'

    def test_ssf6_colorectum_ras_codes(self):
        """CRC SSF6 = RAS mutation (3-char code) per 2025 TCR codebook (p.79).
        Format: pos1=KRAS code, pos2=NRAS code, pos3='8' filler.
        """
        gen = SyntheticTCRGenerator(cancer_group='colorectum', n=100, seed=3)
        df = gen.generate()
        vals = df['SSF6_raw'].dropna().astype(str)
        # Each code must be 3 characters and end with '8'
        invalid = [v for v in vals if len(v) != 3 or v[2] != '8']
        assert len(invalid) == 0, f'Invalid RAS codes: {set(invalid)}'


class TestSyntheticExcel:

    def test_to_excel_creates_file(self, tmp_path):
        gen = SyntheticTCRGenerator(cancer_group='breast', n=10, seed=1)
        gen.generate()
        out = tmp_path / 'test.xlsx'
        gen.to_excel(str(out))
        assert out.exists()
        assert out.stat().st_size > 5_000

    def test_excel_has_correct_sheet(self, tmp_path):
        gen = SyntheticTCRGenerator(cancer_group='lung', n=10, seed=1)
        gen.generate()
        out = tmp_path / 'lung.xlsx'
        gen.to_excel(str(out))
        sheets = pd.ExcelFile(str(out)).sheet_names
        assert 'All_Fields_Decoded' in sheets

    def test_excel_readable_by_tcrdecoder(self, tmp_path):
        """Generated Excel must be loadable by TCRDecoder without crash."""
        import warnings
        warnings.filterwarnings('ignore')

        gen = SyntheticTCRGenerator(cancer_group='breast', n=15, seed=7)
        gen.generate()
        out = tmp_path / 'breast.xlsx'
        gen.to_excel(str(out))

        from tcr_decoder import TCRDecoder
        dec = TCRDecoder(str(out))
        dec.load(skip_input_check=True).decode()
        assert len(dec.clean) == 15

    def test_summary_returns_string(self):
        gen = SyntheticTCRGenerator(cancer_group='breast', n=20, seed=1)
        gen.generate()
        summary = gen.summary()
        assert isinstance(summary, str)
        assert 'breast' in summary.lower()
        assert '20' in summary

    def test_summary_before_generate_returns_message(self):
        gen = SyntheticTCRGenerator(cancer_group='breast', n=5, seed=1)
        summary = gen.summary()
        assert 'generate' in summary.lower()
