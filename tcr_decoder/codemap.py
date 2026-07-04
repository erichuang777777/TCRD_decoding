"""
Bidirectional code <-> label maps: single source of truth for TCR field mappings.

Every simple lookup-table field in the TCR codebook (MSI status, KRAS mutation,
HBsAg, Gleason score, ...) is defined ONCE as a CodeMap. `.decode()` walks
code -> label (used by TCRDecoder); `.encode()` walks label -> code (used by
TCREncoder). Because both directions read the same dict, decode and encode
can never drift apart -- there is exactly one place to fix a wrong code or
add a missing one.

`_map_decode()` in utils.py is reimplemented on top of CodeMap so every
existing call site keeps its exact decode behavior unchanged.
"""

from typing import Callable, Dict, Optional, Union

import pandas as pd


class CodeMap:
    """Bidirectional integer-code <-> clinical-label map.

    Parameters
    ----------
    mapping : dict
        Raw TCR code (int) -> decoded clinical label (str).
    fallback : str
        Prefix used by decode() for an unmapped code: ``'{fallback} {code}'``.
        encode() recognizes and reverses this exact fallback format too, so
        round-tripping an unmapped code's decoded text still recovers it.
    """

    def __init__(self, mapping: Dict[int, str], fallback: str = 'Code'):
        self.mapping: Dict[int, str] = dict(mapping)
        self.fallback = fallback
        # label -> code. First occurrence wins when two codes share a label
        # (list the canonical/lowest code first in the source mapping).
        self.reverse: Dict[str, int] = {}
        for k, v in self.mapping.items():
            self.reverse.setdefault(v, k)

    def decode_one(self, val) -> str:
        if pd.isna(val) or str(val).strip() in ('', 'nan'):
            return ''
        try:
            iv = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return str(val).strip()
        return self.mapping.get(iv, f'{self.fallback} {iv}')

    def encode_one(self, label) -> str:
        """Reverse-lookup: clinical label -> raw TCR code (as a string).

        Returns '' for blank/NaN input. Raises KeyError for a label that
        isn't a recognized decoded value, so a typo or unmodeled label
        variant surfaces as an error instead of a silently blank code.
        """
        if label is None or (isinstance(label, float) and pd.isna(label)):
            return ''
        s = str(label).strip()
        if not s or s.lower() == 'nan':
            return ''
        if s in self.reverse:
            return str(self.reverse[s])
        prefix = f'{self.fallback} '
        if s.startswith(prefix):
            rest = s[len(prefix):]
            try:
                int(rest)
                return rest
            except ValueError:
                pass
        raise KeyError(f'Unrecognized label for reverse lookup: {s!r}')

    def decode(self, series: pd.Series) -> pd.Series:
        return series.apply(self.decode_one)

    def encode(self, series: pd.Series, on_error: str = 'raise') -> pd.Series:
        """Vectorized encode.

        on_error : 'raise' (default) to fail loudly on an unrecognized
            label, or 'empty' to blank it out and keep going (used by
            best-effort batch tooling like compare_roundtrip).
        """
        def _safe(v):
            try:
                return self.encode_one(v)
            except KeyError:
                if on_error == 'empty':
                    return ''
                raise
        return series.apply(_safe)

    def as_decoder(self) -> Callable[[pd.Series], pd.Series]:
        """Plain function form, drop-in compatible with the old _map_decode()."""
        return self.decode
