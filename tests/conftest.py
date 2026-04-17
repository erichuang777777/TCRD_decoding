"""
Shared pytest fixtures for tcr_decoder tests.

All fixtures produce DataFrames in the _raw/_decoded format expected by TCRDecoder.
No real patient data is used — all data is synthetically generated.
"""

import pytest
import pandas as pd
import numpy as np
from tcr_decoder.synth import SyntheticTCRGenerator


@pytest.fixture(scope='session')
def breast_raw_df():
    """50 synthetic breast cancer patients in raw TCR format."""
    gen = SyntheticTCRGenerator(cancer_group='breast', n=50, seed=42)
    return gen.generate()


@pytest.fixture(scope='session')
def lung_raw_df():
    """40 synthetic lung cancer patients in raw TCR format."""
    gen = SyntheticTCRGenerator(cancer_group='lung', n=40, seed=99)
    return gen.generate()


@pytest.fixture(scope='session')
def colorectum_raw_df():
    """30 synthetic colorectal cancer patients in raw TCR format."""
    gen = SyntheticTCRGenerator(cancer_group='colorectum', n=30, seed=7)
    return gen.generate()


@pytest.fixture(scope='session')
def breast_clean_df(tmp_path_factory, breast_raw_df):
    """Decoded breast cancer DataFrame with prognostic scores (run once per session).

    Represents the full Module 1 + Module 2 pipeline output:
        decode() → structural derived variables
        decode_with_scores() → NPI, PEPI, IHC4, CTS5, Molecular Subtype, PREDICT
    """
    import warnings
    warnings.filterwarnings('ignore')
    tmp = tmp_path_factory.mktemp('breast')
    xlsx = tmp / 'breast.xlsx'
    with pd.ExcelWriter(str(xlsx), engine='openpyxl') as w:
        breast_raw_df.to_excel(w, sheet_name='All_Fields_Decoded', index=False)
    from tcr_decoder import TCRDecoder
    dec = TCRDecoder(str(xlsx))
    dec.load(skip_input_check=True).decode().decode_with_scores()
    return dec.clean


@pytest.fixture(scope='session')
def lung_clean_df(tmp_path_factory, lung_raw_df):
    """Decoded lung cancer DataFrame."""
    import warnings
    warnings.filterwarnings('ignore')
    tmp = tmp_path_factory.mktemp('lung')
    xlsx = tmp / 'lung.xlsx'
    with pd.ExcelWriter(str(xlsx), engine='openpyxl') as w:
        lung_raw_df.to_excel(w, sheet_name='All_Fields_Decoded', index=False)
    from tcr_decoder import TCRDecoder
    dec = TCRDecoder(str(xlsx))
    dec.load(skip_input_check=True).decode()
    return dec.clean


@pytest.fixture(scope='session')
def colorectum_clean_df(tmp_path_factory, colorectum_raw_df):
    """Decoded colorectal cancer DataFrame."""
    import warnings
    warnings.filterwarnings('ignore')
    tmp = tmp_path_factory.mktemp('crc')
    xlsx = tmp / 'crc.xlsx'
    with pd.ExcelWriter(str(xlsx), engine='openpyxl') as w:
        colorectum_raw_df.to_excel(w, sheet_name='All_Fields_Decoded', index=False)
    from tcr_decoder import TCRDecoder
    dec = TCRDecoder(str(xlsx))
    dec.load(skip_input_check=True).decode()
    return dec.clean
