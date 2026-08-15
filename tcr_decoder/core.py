"""
TCRDecoder — Main orchestrator for Taiwan Cancer Registry decoding pipeline.

Supports ANY cancer type registered in the Taiwan Cancer Registry (TCR).
SSF1-10 are automatically decoded using the correct cancer-specific
interpretation, routed by the ICD-O-3 topography code (TCODE1) -- except for
lymphoma and leukemia, which the code book routes by MORPHOLOGY (MCODE).

Supported cancer groups with full custom SSF decoders:
    breast      (C50) — ER/PR/HER2/Ki67/Nottingham/sentinel nodes
    lung        (C34) — nodules/VPI/ECOG/effusion/mediastinal LN/EGFR/ALK
    colorectum  (C18-C21) — CEA/CRM/BRAF/RAS/obstruction/perforation/MSI
    liver       (C22) — AFP/HBV/HCV/Child-Pugh/Ishak fibrosis
    cervix      (C53) — SCC antigen value/status
    stomach     (C16) — CEA/H.pylori/tumour depth/LVI
    thyroid     (C73) — not an SSF-collecting site: every field is 988
    prostate    (C61) — PSA/Gleason patterns and score/biopsy cores/cT method
    endometrium (C54) — ER/PR/FIGO grade/POLE/MSI/p53
    head_neck   (C00-C14, C30-C32, C76.0) — node size/ECE/levels/depth/ENE
    esophagus   (C15) — PET-CT/MIE/tumour regression grade
    pancreas    (C25) — CEA/CA 19-9/Ki-67/mitotic count/HbA1c
    ovary       (C56) — CA-125 pre/post therapy/residual tumour
    bladder     (C67) — WHO-ISUP grade/nodal ENE/muscularis propria
    lymphoma    (by MCODE) — HIV/B symptoms/IPI/FLIPI/HTLV-1/CMV/HBV/HCV/ESR-IPS
    leukemia    (by MCODE) — karyotype/molecular/induction response/GVHD/MRD
    generic     (any) — numeric passthrough for unknown sites

Usage:
    from tcr_decoder import TCRDecoder

    # Auto-detects cancer type from TCODE1 data
    decoder = TCRDecoder('raw_data.xlsx')
    decoder.decode()
    decoder.validate()
    decoder.export('Clinical_Clean.xlsx')

    # Force a specific cancer group (override auto-detection)
    decoder = TCRDecoder('raw_data.xlsx', cancer_group='lung')
    decoder.run('Lung_Clinical_Clean.xlsx')

    # One-liner (auto-detect):
    TCRDecoder('raw_data.xlsx').run('Clinical_Clean.xlsx')
"""

import sys
import re
import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import numpy as np

from tcr_decoder.utils import (
    strip_float_suffix, clean_text, en, clean_date,
    clean_numeric, clean_tnm, shorten_tnm,
)
from tcr_decoder.decoders import (
    decode_er_pr, decode_ki67, decode_her2, decode_nottingham,
    decode_ssf3_neoadj, decode_ebrt_additive, decode_sentinel,
    decode_lnexam, decode_lnpositive, decode_cause_of_death, decode_smoking_triplet,
    decode_surgery,
)
from tcr_decoder.ssf_registry import (
    detect_cancer_group_from_series, apply_ssf_profile,
    get_ssf_profile, list_supported_cancers,
)
from tcr_decoder.longform_codes import (
    BEHAVIOR_MAP, LATERALITY_MAP, LONGFORM_CODE_MAPS, LVI_MAP,
    PERINEURAL_INVASION_MAP, decode_confirmation,
)
from tcr_decoder.validators import run_all_validators
from tcr_decoder.input_validator import validate_input
from tcr_decoder.derived import add_structural_derived
from tcr_decoder.data_dictionary import generate_data_dictionary, apply_tcr_labels
from tcr_decoder.report import generate_summary_report, _format_summary_sheet

logger = logging.getLogger('tcr_decoder')


# ─────────────────────────────────────────────────────────────────────────────
# Structural code tables (module-level so TCREncoder can invert them too --
# see tcr_decoder.encoder). Decode behavior is unchanged: these are the exact
# same dict literals that used to live inline inside TCRDecoder.decode().
# ─────────────────────────────────────────────────────────────────────────────

AJCC_MAP = {
    # Single-digit (standard format)
    '6':     'AJCC 6th Edition (2002)',
    '7':     'AJCC 7th Edition (2010)',
    '8':     'AJCC 8th Edition (2018)',
    '8048':  'AJCC 8th Edition — Prognostic Stage (Breast)',
    '99':    'Unknown AJCC edition',
    '99999': 'Not applicable',
    # Zero-padded (TCR export may use '06', '07', '08')
    '06':    'AJCC 6th Edition (2002)',
    '07':    'AJCC 7th Edition (2010)',
    '08':    'AJCC 8th Edition (2018)',
}

# Surgery of primary site (外院 PRESTYPE #4.1.3, 申報醫院 STYPE95 #4.1.4).
#
# Appendix B defines these codes PER PRIMARY SITE -- 660 is an implant
# reconstruction in the breast and a hemicolectomy elsewhere -- so there is no
# single dict to hand _map(). Both fields go through decode_surgery(), which
# picks the table from TCODE1. See tcr_decoder/surgery_codes.py (generated
# from the manual by scripts/generate_surgery_codes.py).
#
# Pre-2025 exports used 1- and 2-character codes. They are kept so historical
# files still decode, but they are NOT in the official 編碼範圍 and each label
# says so, which also stops encode() from ever emitting one.
LEGACY_SURGERY_CODES = {
    '0':  'No surgery (legacy 1-digit code)',
    '00': 'No surgery (legacy 2-digit code)',
    '20': 'Partial mastectomy / lumpectomy (legacy 2-digit code)',
    '22': 'Modified radical mastectomy (legacy 2-digit code)',
    '24': 'Total / simple mastectomy (legacy 2-digit code)',
    '30': 'Partial surgical removal of primary site (legacy 2-digit code)',
    '40': 'Total surgical removal of primary site (legacy 2-digit code)',
    '41': 'Local excision — margins positive or NOS (legacy 2-digit code)',
    '44': 'Sentinel LN biopsy only (legacy 2-digit code)',
    '45': 'Sentinel LN biopsy + axillary LN dissection (legacy 2-digit code)',
    '50': 'Radical mastectomy (legacy 2-digit code)',
    '51': 'Extended radical mastectomy (legacy 2-digit code)',
    '54': 'Subcutaneous mastectomy (legacy 2-digit code)',
    '55': 'Skin-sparing mastectomy (legacy 2-digit code)',
    '60': 'Other surgery (legacy 2-digit code)',
    '70': 'Radical surgery with organ resection in continuity (legacy 2-digit code)',
    '80': 'Surgery, NOS (legacy 2-digit code)',
    '99': 'Unknown (legacy 2-digit code)',
}


LNSCO_MAP = {
    '0': 'No regional LN procedure performed',
    '1': 'Diagnostic biopsy or aspiration of regional LN only (incisional/excisional/core biopsy or aspiration)',
    '2': 'Sentinel LN biopsy (SLNB) only',
    '3': 'Regional LN dissection — number/extent not specified',
    '4': '1–3 regional LN removed (therapeutic dissection, NOT SLNB)',
    '5': '4 or more regional LN removed (therapeutic dissection, NOT SLNB)',
    '6': 'SLNB + regional LN dissection (same surgery or timing unrecorded)',
    '7': 'SLNB first, then regional LN dissection (separate surgeries)',
    '9': 'Unknown',
}


class TCRDecoder:
    """Main pipeline: load → decode → validate → export.

    Automatically detects cancer type from TCODE1 (ICD-O-3) and routes SSF1-10
    decoding to the appropriate cancer-specific interpreter.
    Use `cancer_group` to override auto-detection.
    """

    def __init__(self, input_path: Union[str, Path],
                 sheet_name: str = 'All_Fields_Decoded',
                 cancer_group: Optional[str] = None):
        """Initialize with path to the decoded registry Excel file.

        Args:
            input_path: Path to the Excel file containing raw + decoded columns
            sheet_name: Sheet name to read from
            cancer_group: Override cancer type auto-detection. One of:
                'breast', 'lung', 'colorectum', 'liver', 'cervix', 'stomach',
                'thyroid', 'prostate', 'nasopharynx', 'endometrium', 'generic'.
                If None (default), auto-detects from TCODE1.
        """
        self.input_path = Path(input_path)
        self.sheet_name = sheet_name
        self._forced_cancer_group = cancer_group
        self._detected_cancer_group: Optional[str] = None
        self._raw_df: Optional[pd.DataFrame] = None
        self._clean_df: Optional[pd.DataFrame] = None
        self._flags_df: Optional[pd.DataFrame] = None
        self._log: list = []

    @property
    def cancer_group(self) -> Optional[str]:
        """The active cancer group (forced or detected)."""
        return self._forced_cancer_group or self._detected_cancer_group

    def _log_msg(self, msg: str):
        self._log.append(msg)
        logger.info(msg)
        print(msg)

    def load(self, skip_input_check: bool = False) -> 'TCRDecoder':
        """Load the input Excel file and run pre-decode validation.

        Args:
            skip_input_check: If True, skip input validation (for advanced users)

        Raises
        ------
        FileNotFoundError
            If the input file does not exist.
        ValueError
            If the requested sheet is not present, with the list of available
            sheets so the user can diagnose the problem.
        """
        self._log_msg(f'Loading {self.input_path.name} [{self.sheet_name}]...')

        if not self.input_path.exists():
            raise FileNotFoundError(
                f'Input file not found: {self.input_path!s}'
            )

        # Sheet-existence check BEFORE read_excel so the user gets a clear
        # error instead of pandas' cryptic 'Worksheet named X not found'.
        try:
            xl = pd.ExcelFile(str(self.input_path))
            available = xl.sheet_names
        except Exception as exc:
            raise ValueError(
                f'Failed to open Excel file {self.input_path!s}: '
                f'{type(exc).__name__}: {exc}.  The file may be corrupted '
                f'or not a valid .xlsx/.xls workbook.'
            ) from exc
        if self.sheet_name not in available:
            raise ValueError(
                f'Sheet {self.sheet_name!r} not found in '
                f'{self.input_path.name}.  Available sheets: {available}.  '
                f'Pass sheet_name=... to TCRDecoder to select a different sheet.'
            )

        # Read every code column as text. TCR codes are FIXED-WIDTH and their
        # leading zeros carry meaning -- HER2 '000' (IHC 0, staining 0%) is a
        # different code from the legacy 1-digit '0' (staining % not
        # described, i.e. 100), and '004' (Ultralow) is not '4' at all. Left
        # to pandas' type inference a column of digit strings becomes int64
        # and every leading zero is lost before any decoder sees it. The same
        # applies to PK/patient IDs ('0001234' -> 1234).
        #
        # The column names aren't known until the header is read, so read the
        # header alone first and build the dtype map from it.
        _text_dtypes = {}
        try:
            _header = pd.read_excel(
                str(self.input_path), sheet_name=self.sheet_name,
                engine='openpyxl', nrows=0,
            )
            _text_dtypes = {
                col: str for col in _header.columns
                if isinstance(col, str)
                and (col.endswith('_raw') or col.endswith('_decoded')
                     or col in ('PK', 'IDNUM'))
            }
        except Exception:  # pragma: no cover - fall back to inference
            _text_dtypes = {
                col: str
                for col in ('PK_raw', 'PK_decoded', 'PK',
                            'IDNUM_raw', 'IDNUM_decoded', 'IDNUM')
            }
        try:
            self._raw_df = pd.read_excel(
                str(self.input_path),
                sheet_name=self.sheet_name,
                engine='openpyxl',
                dtype=_text_dtypes,
            )
        except TypeError:
            # Fallback for engines that reject the dtype argument
            self._raw_df = pd.read_excel(
                str(self.input_path),
                sheet_name=self.sheet_name,
                engine='openpyxl',
            )

        # Normalise column headers: strip whitespace so users who have
        # ' PK_raw ' (trailing/leading space) are not silently served the
        # fallback empty series for every call site that looks up 'PK_raw'.
        original_cols = list(self._raw_df.columns)
        self._raw_df.columns = [
            str(c).strip() if isinstance(c, str) else c for c in original_cols
        ]
        renamed = [
            (o, n)
            for o, n in zip(original_cols, self._raw_df.columns)
            if o != n
        ]
        if renamed:
            self._log_msg(
                f'  Stripped whitespace from {len(renamed)} column header(s)'
            )

        if self._raw_df.shape[0] == 0:
            raise ValueError(
                f'Input file {self.input_path.name} (sheet {self.sheet_name!r}) '
                f'contains 0 data rows.  Nothing to decode.'
            )

        self._log_msg(f'  Loaded {len(self._raw_df)} patients × {len(self._raw_df.columns)} columns')

        if not skip_input_check:
            self._log_msg('Pre-decode input validation...')
            self._input_result = validate_input(self._raw_df)
            for e in self._input_result.errors:
                self._log_msg(f'  [ERROR] {e["Check"]}: {e["Detail"]}')
            for w in self._input_result.warnings:
                self._log_msg(f'  [WARN]  {w["Check"]}: {w["Detail"]}')
            if not self._input_result.is_ok:
                detail = '; '.join(
                    f'{e["Check"]}: {e["Detail"]}'
                    for e in self._input_result.errors[:5]
                )
                raise ValueError(
                    'Input validation failed. Decode was stopped to avoid '
                    f'producing incomplete clinical output. {detail}. '
                    'Pass skip_input_check=True only if you intentionally '
                    'want permissive decoding.'
                )
            else:
                self._log_msg(f'  Input OK ({len(self._input_result.warnings)} warnings)')
        return self

    def _raw(self, col: str) -> pd.Series:
        """Get raw column, fallback to empty series."""
        return self._raw_df.get(col + '_raw', pd.Series([''] * len(self._raw_df)))

    def _dec(self, col: str) -> pd.Series:
        """Get decoded column, fallback to raw."""
        return self._raw_df.get(col + '_decoded', self._raw(col))

    def _map(self, col: str, mapping: dict) -> pd.Series:
        """Apply a custom code mapping using the RAW field value (bypasses _decoded column)."""
        raw = self._raw(col)
        def _apply(v):
            v = strip_float_suffix(str(v).strip())
            if not v or v.lower() in ('nan', ''):
                return ''
            return mapping.get(v, mapping.get(v.lstrip('0'), f'Code {v}'))
        return raw.apply(_apply)

    def decode(self) -> 'TCRDecoder':
        """Run the full decode pipeline, producing the clean clinical DataFrame.

        Auto-detects cancer type from TCODE1 (unless cancer_group was forced).
        SSF1-10 are decoded using the cancer-specific interpreter.
        """
        if self._raw_df is None:
            self.load()

        df = self._raw_df
        out = pd.DataFrame()
        self._log_msg('Decoding...')

        # ── Cancer type detection ─────────────────────────────────────────────
        if self._forced_cancer_group:
            self._detected_cancer_group = self._forced_cancer_group
            self._log_msg(f'  Cancer group: {self._forced_cancer_group} (forced by user)')
        else:
            tcode1_col = 'TCODE1_raw' if 'TCODE1_raw' in df.columns else None
            if tcode1_col:
                # Lymphoma and leukemia are defined by histology, not by site
                # (Cancer-SSF-Manual pp.194, 207), so MCODE has to travel with
                # TCODE1 or a nodal lymphoma would be read as its site's cancer.
                mcode = df['MCODE_raw'] if 'MCODE_raw' in df.columns else None
                self._detected_cancer_group = detect_cancer_group_from_series(
                    df[tcode1_col], mcode)
            else:
                self._detected_cancer_group = 'generic'
            ssf_profile = get_ssf_profile(self._detected_cancer_group)
            self._log_msg(f'  Cancer group: {self._detected_cancer_group} '
                          f'({ssf_profile.site_label}) [auto-detected]')

        # ── Demographics ──────────────────────────────────
        out['Patient_ID']          = self._raw('PK')
        out['Sex']                     = LONGFORM_CODE_MAPS['SEX'][0].decode(
            self._raw('SEX'))
        out['Age_at_Diagnosis']    = clean_numeric(self._raw('AGE'), unknown_vals={'999', '9999'})
        out['Diagnosis_Year']      = self._raw('DX_YEAR')
        out['Date_of_Diagnosis']   = clean_date(self._raw('DXDATE'))
        out['Date_of_First_Visit'] = clean_date(self._raw('VISTDATE'))
        out['Smoking']             = decode_smoking_triplet(self._raw('SMOKING'))
        out['Total_Primaries']     = en(self._dec('SEQ1'))
        out['Cancer_Sequence']     = en(self._dec('SEQ2'))

        # ── Tumour Characteristics ────────────────────────
        out['Primary_Site_Code']   = self._raw('TCODE1')
        out['Primary_Site']        = en(self._dec('TCODE1'))
        out['Laterality']          = LATERALITY_MAP.decode(self._raw('LAT95'))
        out['Histology_Code']      = self._raw('MCODE')
        out['Histology']           = self._dec('MCODE').apply(
            lambda v: clean_text(re.sub(r'^\d+:\s*', '', str(v))))
        out['Behavior']            = BEHAVIOR_MAP.decode(self._raw('MCODE5'))
        out['Grade_Pathologic']    = en(self._dec('MCODE6'))
        out['Grade_Clinical']      = en(self._dec('MCODE6C'))
        # CONFER has two tables and code 3 exists only for M9590-9993, so
        # the morphology has to travel with it (manual p.102/104).
        out['Confirmation_Method'] = decode_confirmation(
            self._raw('CONFER'), self._raw('MCODE'))
        out['Tumor_Size_mm']       = clean_numeric(
            self._raw('CSIZE95'), unknown_vals={'999', '9999', '888', '8888'})
        out['Perineural_Invasion'] = PERINEURAL_INVASION_MAP.decode(self._raw('PNI'))
        out['LVI']                 = LVI_MAP.decode(self._raw('LVI'))
        # 95-99 are five distinct situations (Longform-Manual p.129), not one
        # "unknown": treating them as such lost four of them and made the
        # field un-encodable. The text column keeps them; the numeric column
        # holds the count when there is one.
        _ln_exam_decoded = decode_lnexam(self._raw('LNEXAM'))
        out['LN_Examined_Status']  = _ln_exam_decoded
        out['LN_Examined']         = pd.to_numeric(
            _ln_exam_decoded.where(_ln_exam_decoded.str.isdigit()),
            errors='coerce').astype('Int64')
        _ln_pos_decoded = decode_lnpositive(self._raw('LN_POSITI'))
        out['LN_Positive']         = _ln_pos_decoded
        out['LN_Positive_Count']   = pd.to_numeric(
            _ln_pos_decoded.where(_ln_pos_decoded.str.isdigit()), errors='coerce').astype('Int64')

        # ── Staging ───────────────────────────────────────
        out['AJCC_Edition'] = self._map('AJCC', AJCC_MAP)

        def fix_m0i(series):
            return series.str.replace(
                'M0(i+) - Bone marrow micrometastasis',
                'M0(i+) - ITC in bone marrow [NOT distant metastasis]',
                regex=False)

        def fix_n0_codes(series):
            return (series
                .str.replace('N0(i+) - (ITC) Isolated tumor cells',
                              'N0(i-)/N0a - No isolated tumor cells', regex=False)
                .str.replace('N0(i+) - Isolated tumor cells',
                              'N0(i-)/N0a - No isolated tumor cells', regex=False)
                .str.replace('N0(mol+) - Molecular positive',
                              'N0(i+) - Isolated tumor cells (ITC found)', regex=False)
                .str.replace('N0(mol+)',
                              'N0(i+) - Isolated tumor cells (ITC found)', regex=False))

        _NEOADJ = 'Not applicable (post-neoadjuvant staging)'

        def fix_neoadj(series):
            return series.str.replace(
                r'T888.*|N88.*|M88.*|Stage 888.*', _NEOADJ, regex=True)

        out['Clinical_T']    = shorten_tnm(clean_tnm(en(self._dec('CT'))))
        out['Clinical_N']    = shorten_tnm(fix_n0_codes(clean_tnm(en(self._dec('CN')))))
        out['Clinical_M']    = shorten_tnm(fix_m0i(en(self._dec('CM'))))
        out['Clinical_Stage'] = en(self._dec('CSTG'))
        out['Path_T']        = shorten_tnm(fix_neoadj(clean_tnm(en(self._dec('PT')))))
        out['Path_N']        = shorten_tnm(fix_neoadj(fix_n0_codes(clean_tnm(en(self._dec('PN'))))))
        out['Path_M']        = shorten_tnm(fix_neoadj(fix_m0i(en(self._dec('PM'))).apply(
            lambda x: re.sub(r'^(M\w+(?:\(i\+\))?)\s+\1\b', r'\1', x) if x else x)))
        out['Path_Stage']    = fix_neoadj(
            self._dec('PSTG').apply(
                lambda v: '' if str(v) in ('888', '8888', 'nan') else clean_text(str(v))))
        out['Combined_Stage'] = en(self._dec('SUMSTG'))
        out['Other_Staging_System'] = en(self._dec('OSTG'))

        # OCSTG: code 0 = 'Not applicable' when OSTG=0 (no other staging system)
        _ocstg_raw = self._raw('OCSTG')
        _ostg_raw = self._raw('OSTG')
        _ocstg_dec = en(self._dec('OCSTG'))
        out['Other_Clinical_Stage'] = pd.Series([
            'Not applicable (no other staging system)'
            if (str(_ostg_raw.iloc[i]).strip() in ('0', '0.0') and
                str(_ocstg_raw.iloc[i]).strip() in ('0', '0.0'))
            else _ocstg_dec.iloc[i]
            for i in range(len(df))
        ])

        _opstg_raw = self._raw('OPSTG')
        _opstg_dec = en(self._dec('OPSTG'))
        out['Other_Path_Stage'] = pd.Series([
            'Not applicable (no other staging system)'
            if (str(_ostg_raw.iloc[i]).strip() in ('0', '0.0') and
                str(_opstg_raw.iloc[i]).strip() in ('0', '0.0'))
            else _opstg_dec.iloc[i]
            for i in range(len(df))
        ])

        # Metastasis sites
        for n in (1, 2, 3):
            out[f'Metastasis_Site_{n}'] = en(self._dec(f'META{n}'))

        # ── Surgery ───────────────────────────────────────
        out['Surgery_Performed'] = en(self._dec('S'))
        out['Surgery_Date']      = clean_date(self._raw('FSDATE'))
        _tcode1 = self._raw('TCODE1')
        out['Surgery_Type_Other_Hosp'] = decode_surgery(
            self._raw('PRESTYPE'), _tcode1)
        out['Surgery_Type_This_Hosp'] = decode_surgery(
            self._raw('STYPE95'), _tcode1)
        _surg_this = ~out['Surgery_Type_This_Hosp'].str.contains(
            'No surgery|Unknown|autopsy ONLY|death certificate',
            na=True, case=False)
        _surg_other = ~out['Surgery_Type_Other_Hosp'].str.contains(
            'No outside|Unknown|No surgery|autopsy ONLY|death certificate',
            na=True, case=False)
        out['Any_Surgery'] = np.where(_surg_this | _surg_other, 'Yes', 'No')
        out['Minimally_Invasive']      = LONGFORM_CODE_MAPS['MINS'][0].decode(
            self._raw('MINS'))
        out['Surgical_Margin']     = en(self._dec('MARG95'))
        out['Surgical_Margin_mm']  = clean_numeric(
            self._raw('MARGDIS'), unknown_vals={'990', '999', '988', '9999'})
        out['Regional_LN_Surgery_Other'] = self._map('PRESLNSCO', LNSCO_MAP)
        out['Regional_LN_Surgery_This']  = self._map('SLNSCO95', LNSCO_MAP)
        # Note: Sentinel LN (SSF4/SSF5) is decoded in the SSF section below,
        # using the cancer-specific profile (breast→Sentinel_LN_Examined/Positive;
        # other cancers→ cancer-specific column names).

        # ── Radiation ─────────────────────────────────────
        out['Radiation_Performed']     = LONGFORM_CODE_MAPS['R'][0].decode(
            self._raw('R'))
        out['RT_Target_Summary']       = LONGFORM_CODE_MAPS['RTAR'][0].decode(
            self._raw('RTAR'))
        out['RT_Modality']             = LONGFORM_CODE_MAPS['RMOD'][0].decode(
            self._raw('RMOD'))
        out['EBRT_Technique']      = decode_ebrt_additive(self._raw('EBRT'))
        out['High_Dose_Target']        = LONGFORM_CODE_MAPS['HTAR'][0].decode(
            self._raw('HTAR'))
        out['High_Dose_cGy']       = clean_numeric(self._raw('HDOSE'), unknown_vals={'0', '99999'})
        out['High_Dose_Fractions'] = clean_numeric(self._raw('HNO'), unknown_vals={'0', '99'})
        out['Low_Dose_Target']         = LONGFORM_CODE_MAPS['LTAR'][0].decode(
            self._raw('LTAR'))
        out['Low_Dose_cGy']        = clean_numeric(self._raw('LDOSE'), unknown_vals={'0', '99999'})
        out['Low_Dose_Fractions']  = clean_numeric(self._raw('LNO'), unknown_vals={'0', '99'})
        out['RT_Seq_Surgery']          = LONGFORM_CODE_MAPS['SEQRS'][0].decode(
            self._raw('SEQRS'))
        out['RT_vs_Systemic_Seq']      = LONGFORM_CODE_MAPS['SEQLS'][0].decode(
            self._raw('SEQLS'))

        # ── Systemic Therapy ──────────────────────────────
        out['Chemo_Other_Hosp']        = LONGFORM_CODE_MAPS['PREC'][0].decode(
            self._raw('PREC'))
        out['Chemo_This_Hosp']         = LONGFORM_CODE_MAPS['C'][0].decode(
            self._raw('C'))
        out['Hormone_Other_Hosp']      = LONGFORM_CODE_MAPS['PREH'][0].decode(
            self._raw('PREH'))
        out['Hormone_This_Hosp']       = LONGFORM_CODE_MAPS['H'][0].decode(
            self._raw('H'))
        out['Immuno_Other_Hosp']       = LONGFORM_CODE_MAPS['PREI'][0].decode(
            self._raw('PREI'))
        out['Immuno_This_Hosp']        = LONGFORM_CODE_MAPS['I'][0].decode(
            self._raw('I'))
        out['Stem_Cell_Other_Hosp'] = en(self._dec('PREB'))
        out['Stem_Cell_This_Hosp'] = en(self._dec('B'))
        out['Targeted_Other_Hosp']     = LONGFORM_CODE_MAPS['PRETAR'][0].decode(
            self._raw('PRETAR'))
        out['Targeted_This_Hosp']      = LONGFORM_CODE_MAPS['TAR'][0].decode(
            self._raw('TAR'))
        out['Other_Treatment']         = LONGFORM_CODE_MAPS['OTH'][0].decode(
            self._raw('OTH'))
        out['Palliative_Care']         = LONGFORM_CODE_MAPS['PREP'][0].decode(
            self._raw('PREP'))
        out['Active_Surveillance'] = en(self._dec('WATCHWAITING'))

        # ── Biomarkers (SSF1-10) — cancer-type-aware ─────────────────────────
        # For breast cancer: ER/PR/HER2/Ki67/Nottingham (full custom decoders)
        # For other cancers: EGFR/ALK/AFP/PSA/etc. (cancer-specific decoders)
        # For unknown cancers: generic numeric passthrough
        _active_group = self.cancer_group or 'generic'
        self._log_msg(f'  SSF decoding profile: {_active_group}')

        # Build a temporary df with just the SSF raw columns for apply_ssf_profile
        _ssf_raw_cols = {f'SSF{i}_raw': self._raw(f'SSF{i}')
                         for i in range(1, 11)}
        _ssf_df = pd.DataFrame(_ssf_raw_cols)
        _ssf_decoded = apply_ssf_profile(_ssf_df, _active_group)

        # Add decoded SSF columns to output
        ssf_profile = get_ssf_profile(_active_group)
        _ssf_col_map = {}  # ssf_key → output_column_name
        for ssf_key, field_def in ssf_profile.fields.items():
            col_name = field_def.column_name
            _ssf_col_map[ssf_key] = col_name
            if col_name in _ssf_decoded.columns:
                out[col_name] = _ssf_decoded[col_name]

        # SSF8/SSF9 for breast used to be overwritten here with cleaned-up
        # text from the input file's own SSF8_decoded/SSF9_decoded columns.
        # That made those two columns the only SSF fields NOT produced by the
        # profile decoder, so they had no fixed vocabulary to invert and
        # TCREncoder had to report them as un-encodable. The profile's
        # _PAGET_MAP/_LVI_BREAST_MAP decode the same raw codes straight from
        # the codebook, so the profile output is now used for all 10 SSF
        # fields and the breast round trip is complete.

        # Sentinel LN: for non-breast cancers that don't use SSF4/SSF5 for sentinel LN,
        # these will already be in out under their cancer-specific column names.
        # For breast, overwrite with the specific sentinel decoder (already applied via profile).

        # ── Outcomes ──────────────────────────────────────
        out['Vital_Status']            = LONGFORM_CODE_MAPS['VSTA'][0].decode(
            self._raw('VSTA'))
        out['Cancer_Status']        = en(self._dec('CSTA'))
        out['Last_Contact_Date']    = clean_date(self._raw('LCD'))
        out['Recurrence_Date']      = clean_date(self._raw('REDATE'))
        out['Recurrence_Type']         = LONGFORM_CODE_MAPS['RETYPE95'][0].decode(
            self._raw('RETYPE95'))
        out['Cause_of_Death']       = self._decode_cod(self._dec('DIECAUSE'), out['Vital_Status'])
        out['Vital_Status_Extended'] = en(self._dec('VSTA6'))
        out['Last_Contact_Extended'] = clean_date(self._raw('LCD6'))

        # Recalculate survival from dates (fix stale SURVY6)
        # Impute day-15 for partial dates like '2009/03/99' → '2009-03-15'
        _dx_str = self._raw('DXDATE').astype(str).str.replace(
            r'(\d{4})[/-](\d{2})[/-]99', r'\1-\2-15', regex=True)
        _dx_dt = pd.to_datetime(_dx_str, format='mixed', errors='coerce')
        _lcd6_str = self._raw('LCD6').astype(str).str.replace(
            r'(\d{4})[/-](\d{2})[/-]99', r'\1-\2-15', regex=True)
        _lcd6_dt = pd.to_datetime(_lcd6_str, format='mixed', errors='coerce')
        _surv_calc = ((_lcd6_dt - _dx_dt).dt.days / 365.25).round(1)
        _surv_raw = pd.to_numeric(self._raw('SURVY6'), errors='coerce').where(lambda x: x < 99)
        out['Survival_Years'] = _surv_calc.where(_surv_calc > 0, _surv_raw)

        out['Recurrence_Date_Extended'] = clean_date(self._raw('REDATE6'))
        out['Recurrence_Type_Extended'] = en(self._dec('RETYPE6'))
        out['Cause_of_Death_Extended']  = self._decode_cod(
            self._dec('DIECAUSE6'), out['Vital_Status_Extended'])

        # ── Misc ──────────────────────────────────────────
        out['Height_cm']           = clean_numeric(self._raw('HEIGHT'), unknown_vals={'999', '9999'})
        out['Weight_kg']           = clean_numeric(self._raw('WEIGHT'), unknown_vals={'999', '9999'})
        out['Performance_Status']      = LONGFORM_CODE_MAPS['KPSECOG'][0].decode(
            self._raw('KPSECOG'))
        out['Class_of_Case']           = LONGFORM_CODE_MAPS['CLASS95'][0].decode(
            self._raw('CLASS95'))
        out['Diag_at_Hosp']            = LONGFORM_CODE_MAPS['CLASSOFDIAG'][0].decode(
            self._raw('CLASSOFDIAG'))
        out['Treat_at_Hosp']           = LONGFORM_CODE_MAPS['CLASSOFTREAT'][0].decode(
            self._raw('CLASSOFTREAT'))
        # Flag patients where treatment data is incomplete (Dx & Tx elsewhere)
        out['Treatment_Data_Incomplete'] = out['Class_of_Case'].str.contains(
            'all Tx elsewhere|Tx elsewhere', na=False, case=False)

        # ── Derived Clinical Variables ─────────────────────
        # NOTE: Molecular_Subtype and all prognostic scores are computed by
        # Module 2 (ClinicalScoreEngine / MolecularSubtype calculator).
        # Call decode_with_scores() or use TCRPipeline to add them.
        self._log_msg('  Adding structural derived variables...')
        out = add_structural_derived(out)

        self._clean_df = out
        self._log_msg(f'  Decoded: {len(out)} patients × {len(out.columns)} columns')
        return self

    @staticmethod
    def _decode_cod(series: pd.Series, vital_series: pd.Series = None) -> pd.Series:
        """Decode cause of death — use en() on already-decoded values, with '0' → Non-cancer."""
        decoded = en(series)
        # Raw code '0' means not dead / non-cancer
        result = decoded.apply(
            lambda v: 'Non-cancer / Not applicable' if v in ('0', '') or
            'Not dead' in v or v.lower() == 'nan' else v)
        return result

    def decode_with_scores(
        self,
        scores: Optional[list[str]] = None,
    ) -> 'TCRDecoder':
        """Decode then add clinical scores (Module 1 + Module 2 combined).

        Equivalent to decode() followed by ClinicalScoreEngine.compute().
        Use this when you want scores but do not need the TCRPipeline wrapper.

        Parameters
        ----------
        scores : list[str] or None
            Restrict to specific score calculators by name, or None for all.
        """
        if self._clean_df is None:
            self.decode()
        from tcr_decoder.scores.engine import ClinicalScoreEngine
        self._log_msg('  Computing clinical scores (Module 2)...')
        self._clean_df = ClinicalScoreEngine().compute(self._clean_df, scores=scores)
        return self

    def validate(self) -> 'TCRDecoder':
        """Run all clinical validation rules."""
        if self._clean_df is None:
            raise RuntimeError('Call decode() before validate()')
        self._log_msg('Validating...')
        self._flags_df = run_all_validators(self._clean_df)
        n_flags = len(self._flags_df)
        by_sev = self._flags_df['Severity'].value_counts().to_dict()
        self._log_msg(f'  {n_flags} flags: {by_sev}')
        return self

    def export(self, output_path: Union[str, Path]) -> Path:
        """Export to multi-sheet Excel workbook."""
        if self._clean_df is None:
            raise RuntimeError('Call decode() before export()')
        if self._flags_df is None:
            self.validate()

        out_path = Path(output_path)
        self._log_msg(f'Exporting to {out_path.name}...')

        with pd.ExcelWriter(str(out_path), engine='openpyxl') as writer:
            # Sheet 1: Clean clinical data (column headers prefixed with TCR field numbers)
            apply_tcr_labels(self._clean_df).to_excel(writer, sheet_name='Clinical_Clean', index=False)

            # Sheet 2: Clinical Flags (data-quality warnings for analysts)
            self._flags_df.to_excel(writer, sheet_name='Clinical_Flags', index=False)

            # Sheet 3: Statistical Summary — demographics, stage, treatment, survival
            _grp = self.cancer_group or 'generic'
            _summary = generate_summary_report(self._clean_df, cancer_group=_grp)
            _summary.to_excel(writer, sheet_name='Statistical_Summary', index=False)
            _format_summary_sheet(writer.sheets['Statistical_Summary'])

            # Sheet 4: Subtype / Biomarker Summary (cancer-type aware)
            _grp = self.cancer_group or 'generic'
            sub_col = 'Molecular_Subtype'
            if sub_col in self._clean_df.columns:
                sub = self._clean_df[sub_col].value_counts().reset_index()
                sub.columns = [sub_col, 'Count']
                sub['Percent'] = (100 * sub['Count'] / len(self._clean_df)).round(1)
                sub.to_excel(writer, sheet_name='Subtype_Summary', index=False)

            # Sheet 4: Data Dictionary — one row per column, analyst-friendly.
            # Includes: Description, Data_Type, Completeness_%, N_Filled,
            # N_Missing, N_Unique, Sample_Values.
            # SSF_Field_Map (raw SSF → column name mapping) and the separate
            # Data_Quality sheet have been removed — their content is fully
            # covered here.
            dd = generate_data_dictionary(self._clean_df)
            dd.to_excel(writer, sheet_name='Data_Dictionary', index=False)

            # Sheet 5: Input Validation (if available)
            if hasattr(self, '_input_result'):
                self._input_result.to_dataframe().to_excel(
                    writer, sheet_name='Input_Validation', index=False)

            # Sheet 6: Pipeline Log
            pd.DataFrame({'Log': self._log}).to_excel(
                writer, sheet_name='Pipeline_Log', index=False)

        import os
        kb = os.path.getsize(str(out_path)) / 1024
        self._log_msg(f'  SAVED: {out_path.name} ({kb:.0f} KB)')
        self._log_msg(f'  Cancer group: {self.cancer_group or "generic"}')
        self._log_msg(f'  Sheets: Clinical_Clean, Clinical_Flags, '
                      f'Statistical_Summary, Subtype_Summary, Data_Dictionary, Pipeline_Log')
        return out_path

    def export_html(self, output_path: Union[str, Path]) -> Path:
        """Generate an interactive HTML data browser (Bootstrap + DataTables).

        Parameters
        ----------
        output_path : str | Path
            Destination path for the HTML file.

        Returns
        -------
        Path
            Resolved path of the saved HTML file.

        Notes
        -----
        Requires internet access to load Bootstrap 5 and DataTables from CDN.
        """
        if self._clean_df is None:
            raise RuntimeError('Call decode() before export_html()')
        from tcr_decoder.html_report import generate_html_report
        _grp = self.cancer_group or 'generic'
        out = generate_html_report(self._clean_df, output_path, cancer_group=_grp)
        self._log_msg(f'  HTML browser: {out.name}')
        return out

    def export_eda(self, output_path: Union[str, Path]) -> Path:
        """Generate an interactive HTML EDA report (requires dataprep).

        Parameters
        ----------
        output_path : str | Path
            Destination path for the HTML report.

        Returns
        -------
        Path
            Resolved path of the saved HTML file.

        Notes
        -----
        Install dataprep with: pip install dataprep
        """
        if self._clean_df is None:
            raise RuntimeError('Call decode() before export_eda()')
        from tcr_decoder.eda import generate_eda_report
        _grp = self.cancer_group or 'generic'
        out = generate_eda_report(self._clean_df, output_path, cancer_group=_grp)
        self._log_msg(f'  EDA report: {out.name}')
        return out

    def run(self, output_path: Union[str, Path], scores: bool = True,
            html: bool = False, eda: bool = False) -> Path:
        """One-liner: load → decode → [score] → validate → export → [html/eda].

        Parameters
        ----------
        scores : bool
            If True (default), also compute clinical prognostic scores via
            ClinicalScoreEngine (Module 2).  Set False to decode only.
        html : bool
            If True, also generate an interactive HTML data browser alongside
            the Excel output.  Saved next to ``output_path`` with ``.html``
            extension.  Requires internet access for CDN assets.
        eda : bool
            If True, also generate a dataprep EDA profiling report (requires
            ``pip install dataprep``).  Saved with ``_eda.html`` suffix.
        """
        self.load().decode()
        if scores:
            self.decode_with_scores()
        self.validate()
        out = self.export(output_path)
        if html:
            self.export_html(Path(output_path).with_suffix('.html'))
        if eda:
            self.export_eda(Path(output_path).with_name(
                Path(output_path).stem + '_eda.html'))
        return out

    @property
    def clean(self) -> pd.DataFrame:
        """Access the decoded clean DataFrame."""
        if self._clean_df is None:
            raise RuntimeError('Call decode() first')
        return self._clean_df

    @property
    def flags(self) -> pd.DataFrame:
        """Access the validation flags DataFrame."""
        if self._flags_df is None:
            raise RuntimeError('Call validate() first')
        return self._flags_df
