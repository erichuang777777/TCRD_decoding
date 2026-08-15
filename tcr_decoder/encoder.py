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

from tcr_decoder.core import AJCC_MAP, LNSCO_MAP
from tcr_decoder.longform_codes import (
    BEHAVIOR_MAP, LATERALITY_MAP, LONGFORM_CODE_MAPS, LVI_MAP,
    PERINEURAL_INVASION_MAP, encode_confirmation,
)
from tcr_decoder.encoders import (
    batch_encode, encode_structural_map, encode_ebrt_additive, encode_lnpositive,
    encode_lnexam, encode_surgery,
)
from tcr_decoder.ssf_registry import (
    apply_ssf_encode_profile, detect_cancer_group_from_series, get_ssf_profile,
)

logger = logging.getLogger('tcr_decoder')


# clean-column-name -> (raw TCR field name, encoder callable)
STRUCTURAL_FIELD_ENCODERS: Dict[str, Tuple[str, callable]] = {
    'AJCC_Edition':              ('AJCC',      lambda s: encode_structural_map(s, AJCC_MAP)),
    'Regional_LN_Surgery_Other': ('PRESLNSCO', lambda s: encode_structural_map(s, LNSCO_MAP)),
    'Regional_LN_Surgery_This':  ('SLNSCO95',  lambda s: encode_structural_map(s, LNSCO_MAP)),
    'EBRT_Technique':            ('EBRT',      encode_ebrt_additive),
    'LN_Positive':               ('LN_POSITI', encode_lnpositive),
    'LN_Examined_Status':        ('LNEXAM',    encode_lnexam),
    'Laterality':                ('LAT95',     LATERALITY_MAP.encode),
    'Behavior':                  ('MCODE5',    BEHAVIOR_MAP.encode),
    'Perineural_Invasion':       ('PNI',       PERINEURAL_INVASION_MAP.encode),
    'LVI':                       ('LVI',       LVI_MAP.encode),
    'Chemo_Other_Hosp':          ('PREC',      LONGFORM_CODE_MAPS['PREC'][0].encode),
    'Chemo_This_Hosp':           ('C',         LONGFORM_CODE_MAPS['C'][0].encode),
    'Hormone_Other_Hosp':        ('PREH',      LONGFORM_CODE_MAPS['PREH'][0].encode),
    'Hormone_This_Hosp':         ('H',         LONGFORM_CODE_MAPS['H'][0].encode),
    'Immuno_Other_Hosp':         ('PREI',      LONGFORM_CODE_MAPS['PREI'][0].encode),
    'Immuno_This_Hosp':          ('I',         LONGFORM_CODE_MAPS['I'][0].encode),
    'Targeted_Other_Hosp':       ('PRETAR',    LONGFORM_CODE_MAPS['PRETAR'][0].encode),
    'Targeted_This_Hosp':        ('TAR',       LONGFORM_CODE_MAPS['TAR'][0].encode),
    'Other_Treatment':           ('OTH',       LONGFORM_CODE_MAPS['OTH'][0].encode),
    'Palliative_Care':           ('PREP',      LONGFORM_CODE_MAPS['PREP'][0].encode),
    'Radiation_Performed':       ('R',          LONGFORM_CODE_MAPS['R'][0].encode),
    'RT_Target_Summary':         ('RTAR',       LONGFORM_CODE_MAPS['RTAR'][0].encode),
    'RT_Modality':               ('RMOD',       LONGFORM_CODE_MAPS['RMOD'][0].encode),
    'High_Dose_Target':          ('HTAR',       LONGFORM_CODE_MAPS['HTAR'][0].encode),
    'Low_Dose_Target':           ('LTAR',       LONGFORM_CODE_MAPS['LTAR'][0].encode),
    'RT_Seq_Surgery':            ('SEQRS',      LONGFORM_CODE_MAPS['SEQRS'][0].encode),
    'RT_vs_Systemic_Seq':        ('SEQLS',      LONGFORM_CODE_MAPS['SEQLS'][0].encode),
    'Minimally_Invasive':        ('MINS',       LONGFORM_CODE_MAPS['MINS'][0].encode),
}

# (cancer_group, clean_column) pairs whose value in a `clean` DataFrame does
# NOT come from the SSF profile decoder, and therefore has no fixed
# vocabulary for TCREncoder to invert. Kept as an extension point: breast
# SSF8/SSF9 used to be listed here because TCRDecoder.decode() overwrote
# them with text from the input file's own {FIELD}_decoded columns; they now
# come from the profile like every other SSF field, so this is empty and all
# ten breast SSF fields round-trip.
_SSF_PIPELINE_OVERRIDDEN: Dict[Tuple[str, str], str] = {}


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
            # Histology_Code carries MCODE, which is what selects the lymphoma
            # and leukemia profiles (Cancer-SSF-Manual pp.194, 207).
            histology = (df['Histology_Code'] if 'Histology_Code' in df.columns
                         else None)
            self._detected_cancer_group = detect_cancer_group_from_series(
                df['Primary_Site_Code'], histology)
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

        # CONFER's table depends on the morphology (manual p.102/104).
        histology = df.get('Histology_Code')
        if 'Confirmation_Method' in df.columns:
            if histology is None:
                self.unencoded_columns['Confirmation_Method'] = (
                    'Code 3 exists only for M9590-9993, and this DataFrame '
                    'has no Histology_Code column to tell the tables apart')
            else:
                self._log_msg('Encoding Confirmation_Method -> CONFER_raw')
                series = df['Confirmation_Method'].astype(str).replace('nan', '')
                out['CONFER_raw'] = batch_encode(
                    lambda s: encode_confirmation(s, histology), series,
                    on_error=on_error)

        # Surgery of primary site is site-specific (Appendix B), so its
        # encoder needs the topography code alongside the label.
        site = df.get('Primary_Site_Code')
        for clean_col, raw_field in (('Surgery_Type_Other_Hosp', 'PRESTYPE'),
                                     ('Surgery_Type_This_Hosp', 'STYPE95')):
            if clean_col not in df.columns:
                continue
            if site is None:
                self.unencoded_columns[clean_col] = (
                    'Surgery codes are defined per primary site (Longform '
                    '附錄B), and this DataFrame has no Primary_Site_Code column'
                )
                continue
            self._log_msg(f'Encoding {clean_col} -> {raw_field}_raw')
            series = df[clean_col].astype(str).replace('nan', '')
            out[f'{raw_field}_raw'] = batch_encode(
                lambda s: encode_surgery(s, site), series, on_error=on_error)

        handled_clean_cols = (set(ssf_input_cols) | set(STRUCTURAL_FIELD_ENCODERS)
                              | {'Surgery_Type_Other_Hosp',
                                 'Surgery_Type_This_Hosp',
                                 'Confirmation_Method'})
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
