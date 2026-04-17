"""
Taiwan Cancer Registry code mappings.

This is a refactored version of cancer_registry_mapping.py,
organized by category for maintainability.

All mappings are sourced from:
- Longform-Manual_Official-version_20251224_W-1.pdf (454 pages)
- Cancer-SSF-Manual_Official-version_20251204_W.pdf (256 pages)
"""

# Import the full mapping from the existing file
import sys
import os

# Add parent directory to path to import the original mapping
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from cancer_registry_mapping import CODE_MAPPINGS, FIELD_NAMES

__all__ = ['CODE_MAPPINGS', 'FIELD_NAMES']
