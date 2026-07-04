"""
compare_roundtrip -- decode a TCR registry file, re-encode it, and diff the
result against the original raw codes.

This is the concrete "convert both ways and compare" tool: point it at a
registry Longform export, and it tells you exactly which fields/patients
the decode -> encode round trip reproduces exactly, which ones land on a
different-but-equivalent code (a documented ambiguity already baked into
the TCR codebook -- see KNOWN_LABEL_COLLISIONS below), and which columns
have no code table to check at all.

Usage:
    from tcr_decoder.roundtrip import compare_roundtrip, export_roundtrip_report

    mismatches = compare_roundtrip('registry.xlsx')
    print(mismatches)

    export_roundtrip_report('registry.xlsx', 'roundtrip_report.xlsx')
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from tcr_decoder.core import TCRDecoder
from tcr_decoder.encoder import TCREncoder
from tcr_decoder.utils import strip_float_suffix

# Raw-code pairs that the TCR codebook itself renders with IDENTICAL decoded
# text (e.g. both 888 and 988 mean "Not applicable" for several SSF fields).
# encode() always returns the same canonical code for such a label -- see
# each field's CodeMap/encoder docstring -- so a mismatch against the
# OTHER member of the pair is expected, not a defect. compare_roundtrip()
# does not attempt to special-case these away (doing so would require
# per-field knowledge this module shouldn't need); they simply show up
# as expected, low-volume, non-clinically-meaningful mismatches.
KNOWN_LABEL_COLLISION_NOTE = (
    'A handful of TCR codes decode to identical text (e.g. 888 vs 988 both '
    '"Not applicable" for several SSF fields, or a Nottingham score entered '
    'as "60" vs "6"). Re-encoding such a label returns the canonical code, '
    'which may differ from the ORIGINAL code even though the clinical '
    'meaning is unchanged. These show up here as mismatches; they are a '
    'property of the TCR codebook, not a decoding error.'
)


def _numerically_equal(o: str, g: str) -> bool:
    """True if two raw-code strings represent the same integer value.

    Leading-zero padding isn't clinically meaningful (e.g. STYPE95's '000'
    and '0' both mean "No surgery"), so '000' vs '0' should count as a
    match, not a reported mismatch. A plain `lstrip('0')` comparison gets
    this wrong for all-zero codes specifically: '000'.lstrip('0') and
    '0'.lstrip('0') both give '', so a naive `stripped != ''` guard (added
    to avoid matching blank-vs-non-blank cells) also excludes this
    legitimate case. Comparing as integers handles it correctly while still
    requiring both sides to actually be digit strings (so blank cells or
    genuinely different alphanumeric codes never match here).
    """
    return o.isdigit() and g.isdigit() and int(o) == int(g)


def compare_roundtrip(
    source: Union[str, Path, TCRDecoder],
    cancer_group: Optional[str] = None,
    sheet_name: str = 'All_Fields_Decoded',
    on_error: str = 'empty',
) -> pd.DataFrame:
    """Decode a registry file, re-encode it, and diff against the original.

    Parameters
    ----------
    source : str, Path, or TCRDecoder
        Path to a TCR Longform registry .xlsx file, or an already-decoded
        TCRDecoder instance (its .decode() is called if not run yet).
    cancer_group : str, optional
        Override auto-detection for both decode and encode.
    sheet_name : str
        Sheet to read, if `source` is a path.
    on_error : str
        Passed to TCREncoder.encode(). 'empty' (default) blanks out any
        cell whose label this tool can't parse, so one bad cell doesn't
        stop the whole comparison; 'raise' fails on the first one.

    Returns
    -------
    pd.DataFrame
        One row per (patient, field) genuine round-trip mismatch, with
        columns: Patient_ID, Field, Original_Code, Roundtrip_Code. Cells
        TCREncoder could not encode at all are excluded (nothing to
        compare) -- see `unencoded_columns` on the TCREncoder this
        function constructs internally if you need that list too.
    """
    dec, _enc, reencoded = _decode_and_encode(source, cancer_group, sheet_name, on_error)
    return _diff_mismatches(dec, reencoded)


def _decode_and_encode(
    source: Union[str, Path, TCRDecoder],
    cancer_group: Optional[str],
    sheet_name: str,
    on_error: str,
):
    """Shared decode -> encode step for compare_roundtrip/export_roundtrip_report.

    Factored out so a single call site (export_roundtrip_report) that needs
    both the mismatch diff AND the TCREncoder's `unencoded_columns` doesn't
    have to run the whole encode pipeline twice.
    """
    if isinstance(source, TCRDecoder):
        dec = source
        if dec._clean_df is None:
            dec.decode()
    else:
        dec = TCRDecoder(source, sheet_name=sheet_name, cancer_group=cancer_group)
        dec.load(skip_input_check=True).decode()

    clean = dec.clean
    enc = TCREncoder(clean, cancer_group=cancer_group or dec.cancer_group)
    reencoded = enc.encode(on_error=on_error)
    return dec, enc, reencoded


def _diff_mismatches(dec: TCRDecoder, reencoded: pd.DataFrame) -> pd.DataFrame:
    clean = dec.clean
    patient_ids = (dec._raw_df['PK_raw'] if 'PK_raw' in dec._raw_df.columns
                   else pd.Series(range(len(clean)), index=clean.index))

    rows = []
    for col in reencoded.columns:
        if col not in dec._raw_df.columns:
            continue
        orig = dec._raw_df[col].astype(str).str.strip().apply(strip_float_suffix)
        got = reencoded[col].astype(str).str.strip()
        for idx in clean.index:
            o, g = orig.loc[idx], got.loc[idx]
            if g == '':
                continue  # unencodable this cell -- nothing to compare
            if o == g or _numerically_equal(o, g):
                continue
            rows.append({
                'Patient_ID': patient_ids.loc[idx],
                'Field': col,
                'Original_Code': o,
                'Roundtrip_Code': g,
            })

    return pd.DataFrame(rows, columns=['Patient_ID', 'Field', 'Original_Code', 'Roundtrip_Code'])


def export_roundtrip_report(
    source: Union[str, Path, TCRDecoder],
    output_path: Union[str, Path],
    cancer_group: Optional[str] = None,
    sheet_name: str = 'All_Fields_Decoded',
) -> Path:
    """Run compare_roundtrip() and save a multi-sheet Excel report.

    Sheets:
        Mismatches         -- one row per (patient, field) discrepancy
        Field_Summary       -- mismatch count and rate per field
        Unencoded_Columns   -- clinical columns with no TCR code table, and why
    """
    # Encode once and reuse the result for both the mismatch diff and
    # `unencoded_columns` -- this used to call compare_roundtrip() (which
    # runs its own TCREncoder.encode() internally) on top of an encode()
    # already run just above, silently running the full SSF+structural
    # encode pipeline twice per report.
    dec, enc, reencoded = _decode_and_encode(
        source, cancer_group, sheet_name, on_error='empty')
    clean = dec.clean
    mismatches = _diff_mismatches(dec, reencoded)

    if len(mismatches):
        field_summary = (
            mismatches.groupby('Field').size().rename('Mismatch_Count').reset_index()
            .sort_values('Mismatch_Count', ascending=False)
        )
        field_summary['Mismatch_Rate_%'] = (
            100 * field_summary['Mismatch_Count'] / len(clean)
        ).round(1)
    else:
        field_summary = pd.DataFrame(columns=['Field', 'Mismatch_Count', 'Mismatch_Rate_%'])

    unencoded = pd.DataFrame(
        [{'Column': col, 'Reason': reason} for col, reason in enc.unencoded_columns.items()]
    )

    out_path = Path(output_path)
    with pd.ExcelWriter(str(out_path), engine='openpyxl') as writer:
        mismatches.to_excel(writer, sheet_name='Mismatches', index=False)
        field_summary.to_excel(writer, sheet_name='Field_Summary', index=False)
        unencoded.to_excel(writer, sheet_name='Unencoded_Columns', index=False)
        pd.DataFrame({'Note': [KNOWN_LABEL_COLLISION_NOTE]}).to_excel(
            writer, sheet_name='Notes', index=False)

    return out_path
