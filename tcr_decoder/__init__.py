"""
Taiwan Cancer Registry (TCR) Decoder
=====================================
A modular tool for decoding Taiwan Cancer Registry data into
clean, English clinical databases with comprehensive QA flags.

Supports ALL cancer types in the Taiwan Cancer Registry.
SSF1-10 are automatically decoded using the correct cancer-specific
interpretation based on the ICD-O-3 topography code (TCODE1).

Quick Start:
    from tcr_decoder import TCRDecoder

    # Auto-detects cancer type from data
    TCRDecoder('raw_data.xlsx').run('Clinical_Clean.xlsx')

    # Force a specific cancer group
    TCRDecoder('raw_data.xlsx', cancer_group='lung').run('Lung_Clean.xlsx')

Supported cancer groups:
    breast, lung, colorectum, liver, cervix, stomach,
    thyroid, prostate, nasopharynx, endometrium, generic

Utility functions:
    from tcr_decoder import list_supported_cancers, detect_cancer_group
    print(list_supported_cancers())
    detect_cancer_group('C50.1')   # → 'breast'
    detect_cancer_group('C34.0')   # → 'lung'
"""

from tcr_decoder.core import TCRDecoder
from tcr_decoder.pipeline import TCRPipeline
from tcr_decoder.scores.engine import ClinicalScoreEngine
from tcr_decoder.ssf_registry import (
    list_supported_cancers,
    detect_cancer_group,
    detect_cancer_group_from_series,
    get_ssf_profile,
    apply_ssf_profile,
)

__version__ = '2.0.0'
__all__ = [
    'TCRDecoder',
    'TCRPipeline',
    'ClinicalScoreEngine',
    'list_supported_cancers',
    'detect_cancer_group',
    'detect_cancer_group_from_series',
    'get_ssf_profile',
    'apply_ssf_profile',
]
