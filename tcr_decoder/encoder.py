"""
TCREncoder -- inverse of TCRDecoder: clinical facts -> raw TCR Longform codes.

Usage:
    from tcr_decoder import TCREncoder

    enc = TCREncoder(clinical_df)          # clinical_df shaped like TCRDecoder(...).clean
    raw = enc.encode()                     # DataFrame of {FIELD}_raw columns
    print(enc.cancer_group, enc.unencoded_columns)

SCOPE -- read this before assuming full coverage of a registry file:

    TCRDecoder.decode() produces ~100 output columns. Most of them
    (Sex, Primary_Site, Histology, Vital_Status, every treatment-given flag,
    TNM stage text, Cause_of_Death, ...) are decoded by trusting a
    `{FIELD}_decoded` column that is ALREADY PRESENT in the registry export
    (produced by the hospital's own TCR submission tooling), or -- for a
    couple of fields such as Cause_of_Death -- by looking a code up in
    `cancer_registry_mapping.py`, an external, large, non-public file that is
    NOT part of this repository (see tcr_decoder/mappings.py).

    There is no code table inside this package to invert for those fields.
    TCREncoder does not guess one: fabricating a plausible-looking code
    table would risk silently emitting wrong TCR registry codes, which is
    strictly worse than declining to encode the field at all.

    What TCREncoder DOES reconstruct, with the same fidelity as the decode
    side (same dict, both directions -- see tcr_decoder.codemap.CodeMap):
      - SSF1-10 (the site-specific factors) for all 11 cancer profiles --
        ER/PR, HER2, Ki-67, Nottingham grade, AFP, PSA, Gleason, MSI/MMR,
        EGFR, RAS, Child-Pugh, and every other biomarker this codebase
        defines a decoder for.
      - AJCC edition, surgery-type codes (PRESTYPE/STYPE95), regional-LN
        surgery codes (PRESLNSCO/SLNSCO95), EBRT technique bitmask, and
        LN_POSITI -- the structural fields whose code table lives in
        tcr_decoder.core / tcr_decoder.decoders.

    `TCREncoder.encode()` returns exactly the `{FIELD}_raw` columns it could
    reconstruct; `unencoded_columns` (populated after encode()) lists every
    input column it recognized but could not turn back into a raw code and
    why, so a partial result is never mistaken for a complete one.
"""

import logging
from typing import Dict, Optional, Tuple

import pandas as pd

from tcr_decoder.core import AJCC_MAP, PRESTYPE_MAP, STYPE95_MAP, LNSCO_MAP
from tcr_decoder.encoders import (
    batch_encode, encode_structural_map, encode_ebrt_additive, encode_lnpositive,
)
from tcr_decoder.ssf_registry import (
    apply_ssf_encode_profile, detect_cancer_group_from_series, get_ssf_profile,
)

logger = logging.getLogger('tcr_decoder')


# clean-column-name -> (raw TCR field name, encoder callable)
STRUCTURAL_FIELD_ENCODERS: Dict[str, Tuple[str, callable]] = {
    'AJCC_Edition':              ('AJCC',      lambda s: encode_structural_map(s, AJCC_MAP)),
    'Surgery_Type_Other_Hosp':   ('PRESTYPE',  lambda s: encode_structural_map(s, PRESTYPE_MAP)),
    'Surgery_Type_This_Hosp':    ('STYPE95',   lambda s: encode_structural_map(s, STYPE95_MAP)),
    'Regional_LN_Surgery_Other': ('PRESLNSCO', lambda s: encode_structural_map(s, LNSCO_MAP)),
    'Regional_LN_Surgery_This':  ('SLNSCO95',  lambda s: encode_structural_map(s, LNSCO_MAP)),
    'EBRT_Technique':            ('EBRT',      encode_ebrt_additive),
    'LN_Positive':                ('LN_POSITI', encode_lnpositive),
}

# (cancer_group, clean_column) pairs where TCRDecoder.decode()'s full
# pipeline OVERWRITES the SSF-profile decoder's output with its own
# post-processing that trusts the input file's pre-existing {FIELD}_decoded
# column (see core.py: the breast Pagets_Disease/LVI_SSF block). The
# profile's decoder/encoder pair is internally consistent when used directly
# via apply_ssf_profile()/apply_ssf_encode_profile(), but a `clean` DataFrame
# coming out of the FULL TCRDecoder pipeline holds that other, unstructured
# text instead -- there is no fixed vocabulary to invert, so TCREncoder
# reports these as unencoded rather than attempting (and failing) to parse
# them as if they were normal SSF-profile output.
_SSF_PIPELINE_OVERRIDDEN = {
    ('breast', 'Pagets_Disease'): (
        "TCRDecoder.decode() overwrites this column with cleaned-up text "
        "from the input file's own SSF8_decoded column, not the SSF profile's "
        "decoder -- there is no fixed code table to invert here."
    ),
    ('breast', 'LVI_SSF'): (
        "TCRDecoder.decode() overwrites this column with cleaned-up text "
        "from the input file's own SSF9_decoded column, not the SSF profile's "
        "decoder -- there is no fixed code table to invert here."
    ),
}


class TCREncoder:
    """Reconstruct raw TCR Longform codes from a clinical-facts DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Shaped like ``TCRDecoder(...).clean`` (or any DataFrame using the
        same column-name / decoded-label conventions).
    cancer_group : str, optional
        Override auto-detection. Auto-detection reads the 'Primary_Site_Code'
        column (the raw ICD-O-3 TCODE1 value, preserved unchanged by
        TCRDecoder.decode()) the same way TCRDecoder itself detects it.
    """

    def __init__(self, df: pd.DataFrame, cancer_group: Optional[str] = None):
        self.df = df
        self._forced_cancer_group = cancer_group
        self._detected_cancer_group: Optional[str] = None
        self._log: list = []
        self.unencoded_columns: Dict[str, str] = {}

    @property
    def cancer_group(self) -> Optional[str]:
        return self._forced_cancer_group or self._detected_cancer_group

    def _log_msg(self, msg: str):
        self._log.append(msg)
        logger.info(msg)

    def encode(self, on_error: str = 'raise') -> pd.DataFrame:
        """Run the full encode pipeline.

        Parameters
        ----------
        on_error : str
            'raise' (default) to fail loudly the first time a cell holds a
            label this tool doesn't recognize (a typo, a hand-edited value,
            output from a different tool version). 'empty' blanks out just
            that cell, records it, and keeps going -- used by
            compare_roundtrip for best-effort batch diagnostics.

        Returns
        -------
        pd.DataFrame
            One column per successfully-reconstructed raw TCR field, named
            '{FIELD}_raw' (e.g. 'SSF1_raw', 'AJCC_raw'), same row index as
            the input.
        """
        df = self.df
        out = pd.DataFrame(index=df.index)
        self.unencoded_columns = {}

        if self._forced_cancer_group:
            self._detected_cancer_group = self._forced_cancer_group
            self._log_msg(f'Cancer group: {self._forced_cancer_group} (forced)')
        elif 'Primary_Site_Code' in df.columns:
            self._detected_cancer_group = detect_cancer_group_from_series(
                df['Primary_Site_Code'])
            self._log_msg(f'Cancer group: {self._detected_cancer_group} (auto-detected)')
        else:
            self._detected_cancer_group = 'generic'
            self._log_msg("Cancer group: generic ('Primary_Site_Code' column not found)")

        profile = get_ssf_profile(self.cancer_group)
        overridden_cols = {
            col for (grp, col) in _SSF_PIPELINE_OVERRIDDEN if grp == self.cancer_group
        }
        for col in overridden_cols:
            if col in df.columns:
                self.unencoded_columns[col] = _SSF_PIPELINE_OVERRIDDEN[(self.cancer_group, col)]

        ssf_input_cols = [f.column_name for f in profile.fields.values()
                          if f.column_name in df.columns and f.column_name not in overridden_cols]
        if ssf_input_cols:
            self._log_msg(f'Encoding {len(ssf_input_cols)} SSF field(s)...')
            ssf_encoded = apply_ssf_encode_profile(
                df[ssf_input_cols], self.cancer_group, on_error=on_error)
            for i in range(1, 11):
                col = f'SSF{i}_raw'
                if col in ssf_encoded.columns:
                    out[col] = ssf_encoded[col]

        for clean_col, (raw_field, encoder_fn) in STRUCTURAL_FIELD_ENCODERS.items():
            if clean_col not in df.columns:
                continue
            self._log_msg(f'Encoding {clean_col} -> {raw_field}_raw')
            series = df[clean_col].astype(str).replace('nan', '')
            out[f'{raw_field}_raw'] = batch_encode(encoder_fn, series, on_error=on_error)

        handled_clean_cols = set(ssf_input_cols) | set(STRUCTURAL_FIELD_ENCODERS)
        for col in df.columns:
            if (col not in handled_clean_cols and col != 'Primary_Site_Code'
                    and col not in self.unencoded_columns):
                self.unencoded_columns[col] = (
                    'No TCR code table in this package for this field '
                    '(decode trusts a pre-existing {FIELD}_decoded column, '
                    'or an external cancer_registry_mapping.py, for this one)'
                )

        self._log_msg(
            f'Encoded {len(out.columns)} raw field(s); '
            f'{len(self.unencoded_columns)} input column(s) left un-encoded '
            f'(see .unencoded_columns)')
        return out
