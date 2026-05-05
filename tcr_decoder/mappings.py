"""
Taiwan Cancer Registry code mappings.

This is a refactored version of cancer_registry_mapping.py,
organized by category for maintainability.

All mappings are sourced from:
- Longform-Manual_Official-version_20251224_W-1.pdf (454 pages)
- Cancer-SSF-Manual_Official-version_20251204_W.pdf (256 pages)
"""

# Older development trees stored the generated mappings in a top-level
# cancer_registry_mapping.py file. The packaged project must still import
# cleanly when that generated file is not present.
try:
    from cancer_registry_mapping import CODE_MAPPINGS, FIELD_NAMES
except ModuleNotFoundError as exc:
    if exc.name != 'cancer_registry_mapping':
        raise
    CODE_MAPPINGS = {}
    FIELD_NAMES = {}

__all__ = ['CODE_MAPPINGS', 'FIELD_NAMES']
