"""
Taiwan Cancer Registry (TCR) Decoder
=====================================
A modular tool for decoding Taiwan Cancer Registry data into
clean, English clinical databases with comprehensive QA flags.

Supports ALL cancer types in the Taiwan Cancer Registry.
SSF1-10 are automatically decoded using the correct cancer-specific
interpretation based on the ICD-O-3 topography code (TCODE1).

Coding Standard Versions
-------------------------
This package decodes according to the following official TCR codebooks:

    Codebook (Longform):
        Longform-Manual_Official-version_20251224_W-1.pdf
        (Taiwan Cancer Registry Official Coding Manual, 454 pp.)
        Effective date: 2025-12-24

    SSF Manual:
        Cancer-SSF-Manual_Official-version_20251204_W.pdf
        (Site-Specific Factor Definitions, 256 pp.)
        Effective date: 2025-12-04

    Classification systems referenced:
        ICD-O-3       (International Classification of Diseases for Oncology, 3rd ed.)
        AJCC          (7th and 8th editions, auto-detected from registry AJCC field)
        TNM           (UICC 8th edition for staging fields)
        WHO 2022      (Breast tumour classification, Nottingham grade)
        St. Gallen    (2013/2015 consensus, molecular subtype)
        ASCO/CAP 2010 (ER/PR/HER2 positivity thresholds)

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

# ── Coding standard versions ──────────────────────────────────────────────────
# These constants identify the EXACT codebook editions that the decoders
# were written against.  If the TCR updates its codebook, bump these
# strings and update the affected mappings.

TCR_CODEBOOK_VERSION   = 'Longform-Manual_Official-version_20251224_W-1'
TCR_SSF_MANUAL_VERSION = 'Cancer-SSF-Manual_Official-version_20251204_W'
TCR_CODEBOOK_DATE      = '2025-12-24'   # effective date of Longform Manual
TCR_SSF_MANUAL_DATE    = '2025-12-04'   # effective date of SSF Manual

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
