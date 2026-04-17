"""
tcr_decoder.scores — Breast cancer prognostic score calculators.

All calculators inherit from BaseScore and are registered with ScoreRegistry.
Import order determines execution order in apply_all().

Usage:
    from tcr_decoder.scores import ScoreRegistry

    # Apply all scores to a decoded DataFrame
    df = ScoreRegistry.apply_all(df)

    # List available calculators
    for info in ScoreRegistry.list_scores():
        print(info['name'], '—', info['citation'])

    # Apply a single named score
    df = ScoreRegistry.apply_one('NPI Score', df)

Available calculators (in application order):
    1. NPI Score               — Galea 1992 (Nottingham Prognostic Index)
    2. PEPI Score              — Ellis 2008 (Preoperative Endocrine PI)
    3. IHC4 Score              — Cuzick 2011
    4. CTS5 Score              — Sestak 2018 (late recurrence post-5yr ET)
    5. Molecular Subtype       — St. Gallen 2013 surrogate IHC classification
    6. PREDICT Breast v3.0     — Jenkins 2023 (competing-risks Cox model)
"""

from tcr_decoder.scores.base import BaseScore, ScoreRegistry, extract_ki67_numeric, her2_binary
from tcr_decoder.scores.engine import ClinicalScoreEngine

# Import each module to trigger @ScoreRegistry.register decorators
from tcr_decoder.scores import npi           # noqa: F401
from tcr_decoder.scores import pepi          # noqa: F401
from tcr_decoder.scores import ihc4         # noqa: F401
from tcr_decoder.scores import cts5          # noqa: F401
from tcr_decoder.scores import molecular_subtype  # noqa: F401
from tcr_decoder.scores import predict           # noqa: F401

__all__ = [
    'BaseScore',
    'ScoreRegistry',
    'extract_ki67_numeric',
    'her2_binary',
    'ClinicalScoreEngine',
]
