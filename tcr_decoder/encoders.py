"""
Bespoke encoders: clinical label -> raw TCR code.

Each function here is the deliberate inverse of a decoder in decoders.py or
ssf_registry.py. They accept the EXACT decoded-label text that the matching
decoder produces (i.e. what a `Clinical_Clean` sheet from TCRDecoder already
contains) and reconstruct the raw TCR Longform code -- this is what makes
decode() and encode() genuinely mutual inverses and lets `compare_roundtrip`
verify the tool's fidelity against a real registry export.

Fields backed by a plain lookup table (MSI, KRAS, HBsAg, Gleason, ...) don't
need a function here at all -- their CodeMap instance in ssf_registry.py
already provides `.encode()` for free. This module only covers fields whose
decode() logic is genuinely more than a table lookup: composite/scaled
values, bitmasks, and multi-character combinatorial codes.

Where two different raw codes decode to identical label text (a handful of
cases inherited from the TCR codebook itself -- see the KNOWN_LABEL_COLLISIONS
list in tests), encode() returns the canonical (first-listed / modern-width)
code. This is a deliberate, documented choice, not a bug: the ambiguity is
already baked into the decoded text and no amount of parsing recovers it.

Several decode_*/_decode_* functions fall back to passing an unrecognized
raw code through UNCHANGED (e.g. an out-of-range percentage, a garbled
multi-char code) rather than erroring -- decode() never crashes on bad
input. The matching encoder here mirrors that exact leniency: if a value
doesn't match any known label pattern, it is returned as-is (assumed to
already be a raw code that slipped through decode unchanged), not raised.
Only a value that looks like a genuinely malformed instance of a KNOWN
label shape (e.g. a composite label with one unrecognized component) raises
ValueError/KeyError, since decode() itself never produces that shape from
an unrecognized raw code.
"""

import re
from typing import Optional

import pandas as pd

from tcr_decoder.decoders import HER2_MAP, SSF3_NEOADJ_MAP, EBRT_COMPONENTS

# _LUNG_EGFR_LETTER_MAP / _RAS_KRAS_NRAS_MAP live in ssf_registry.py, which in
# turn imports encode_lung_egfr/encode_ras_mutation from this module -- so
# these two are imported lazily (inside the functions that use them) to
# avoid a circular import at module load time.


def _clean(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ''
    s = str(v).strip()
    return '' if s.lower() == 'nan' else s


def batch_encode(enc_fn, series: pd.Series, on_error: str = 'raise') -> pd.Series:
    """Run enc_fn(series) with uniform on_error semantics.

    on_error: 'raise' (default) propagates the first ValueError/KeyError
        from enc_fn -- a real data problem should not pass silently.
        'empty' blanks out just the offending cell(s) and keeps going;
        used by best-effort batch tooling like compare_roundtrip.
    """
    if on_error != 'empty':
        return enc_fn(series)
    try:
        return enc_fn(series)
    except (ValueError, KeyError):
        pass
    out = []
    for v in series:
        try:
            out.append(enc_fn(pd.Series([v])).iloc[0])
        except (ValueError, KeyError):
            out.append('')
    return pd.Series(out, index=series.index)


def encode_structural_map(series: pd.Series, code_map: dict) -> pd.Series:
    """Inverse of TCRDecoder._map(col, code_map): reverse-lookup a decoded
    structural label back to its raw TCR code, using the exact same dict
    that produced it (single source of truth with the decode side).

    Mirrors _map()'s own decode fallback convention: an unmapped code decodes
    to f'Code {v}', so that exact prefix is recognized here too.
    """
    reverse = {}
    for k, v in code_map.items():
        reverse.setdefault(v, k)

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in reverse:
            return reverse[v]
        if v.startswith('Code '):
            return v[len('Code '):]
        raise ValueError(f'Cannot encode structural-code label: {v!r}')

    return series.apply(_encode)


# ─────────────────────────────────────────────────────────────────────────────
# ER / PR (SSF1, SSF2) -- breast
# ─────────────────────────────────────────────────────────────────────────────

def encode_er_pr(series: pd.Series, receptor: str) -> pd.Series:
    """Inverse of decode_er_pr(). Codebook: Cancer-SSF-Manual (breast), p.127-130."""
    special_rev = {
        f'{receptor} converted Neg→Pos after neoadjuvant therapy': '888',
        'Not applicable (Oncotype/Phyllodes/Sarcoma)': '988',
        'Unknown': '999',
        f'{receptor} Negative (<1% or not specified)': '120',
        f'{receptor} Negative (post-neoadjuvant value only)': '121',
        f'{receptor} Positive (proportion unclear)': '110',
        f'{receptor} Positive (post-neoadjuvant value only)': '111',
    }
    intensity_code = {'Weak': 'W', 'Intermediate': 'I', 'Strong': 'S'}
    staining_re = re.compile(
        rf'^{re.escape(receptor)} Positive \((Weak|Intermediate|Strong) staining, (\d+)%\)$')
    positive_re = re.compile(rf'^{re.escape(receptor)} Positive \((\d+)%\)$')
    negative_re = re.compile(rf'^{re.escape(receptor)} Negative \((\d+)%\)$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in special_rev:
            return special_rev[v]
        m = staining_re.match(v)
        if m:
            # Field is intensity letter + 2-digit proportion; 100% is stored
            # as '00' and single digits are zero-padded, per the
            # Cancer-SSF-Manual (breast SSF1, p.121): weak 1% -> 'W01',
            # strong 100% -> 'S00'.
            pct = int(m.group(2))
            return f'{intensity_code[m.group(1)]}{pct % 100:02d}'
        m = positive_re.match(v)
        if m:
            return m.group(1)
        m = negative_re.match(v)
        if m and m.group(1) == '0':
            return '0'
        # decode_er_pr() itself falls back to a bare passthrough for
        # anything it doesn't recognize (e.g. an out-of-range percentage) --
        # mirror that leniency rather than raising.
        return v

    return series.apply(_encode)


# ─────────────────────────────────────────────────────────────────────────────
# Ki-67 (SSF10) -- breast
# ─────────────────────────────────────────────────────────────────────────────

def encode_ki67(series: pd.Series) -> pd.Series:
    """Inverse of decode_ki67(). Codebook: Cancer-SSF-Manual (breast), p.150-151."""
    special_rev = {
        'Not applicable (conversion)': '888',
        'Not applicable (Phyllodes/Sarcoma)': '988',
        'Tested, percentage unknown': '998',
        'Unknown': '999',
    }
    categorized_re = re.compile(r'^(\d+)% \((?:Low|Intermediate|High)\)$')
    fractional_re = re.compile(r'^(\d+\.\d+)%$')  # A-code form, e.g. '0.5%', '5.5%'

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in special_rev:
            return special_rev[v]
        m = categorized_re.match(v)
        if m:
            return m.group(1)
        m = fractional_re.match(v)
        if m:
            pct = float(m.group(1))
            return f'A{round(pct * 10):02d}'
        # decode_ki67() falls back to a bare passthrough for anything else.
        return v

    return series.apply(_encode)


# ─────────────────────────────────────────────────────────────────────────────
# HER2 (SSF7) -- breast
# ─────────────────────────────────────────────────────────────────────────────

_HER2_REVERSE = {}
for _code, _label in HER2_MAP.items():
    # Prefer the modern zero-padded 3-digit code as canonical when a code
    # is spelled two ways (e.g. '0' and '000' both mean IHC 0 -- Negative).
    if _label not in _HER2_REVERSE or (len(_code) == 3 and len(_HER2_REVERSE[_label]) < 3):
        _HER2_REVERSE[_label] = _code


def encode_her2(series: pd.Series) -> pd.Series:
    """Inverse of decode_her2(). Codebook: Cancer-SSF-Manual (breast), p.138-145."""
    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in _HER2_REVERSE:
            return _HER2_REVERSE[v]
        # decode_her2() falls back to a bare passthrough for any code not in
        # HER2_MAP (even after zfill(3)).
        return v

    return series.apply(_encode)


# ─────────────────────────────────────────────────────────────────────────────
# Nottingham Grade (SSF6) -- breast
# ─────────────────────────────────────────────────────────────────────────────

def encode_nottingham(series: pd.Series) -> pd.Series:
    """Inverse of decode_nottingham(). Codebook: Cancer-SSF-Manual (breast), p.140-141.

    Canonicalizes a scored grade to the plain single-digit score (3-9) per
    coding_rules/breast_coding_spec.md, and a grade-only label to 110/120/130.
    """
    special_rev = {
        'Not applicable (conversion)': '888',
        'Not applicable': '988',
        'Unknown': '999',
    }
    scored_re = re.compile(
        r'^Score (\d+) → Grade [123] \((?:Well|Moderately|Poorly) differentiated\)$')
    grade_only_re = re.compile(r'^Grade ([123]) \((?:Well|Moderately|Poorly) differentiated\)$')
    grade_only_code = {'1': '110', '2': '120', '3': '130'}

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in special_rev:
            return special_rev[v]
        m = scored_re.match(v)
        if m:
            return m.group(1)
        m = grade_only_re.match(v)
        if m:
            return grade_only_code[m.group(1)]
        # decode_nottingham() falls back to a bare passthrough otherwise.
        return v

    return series.apply(_encode)


# ─────────────────────────────────────────────────────────────────────────────
# SSF3 Neoadjuvant response -- breast
# ─────────────────────────────────────────────────────────────────────────────

_SSF3_NEOADJ_REVERSE = {}
for _code, _label in SSF3_NEOADJ_MAP.items():
    if _label not in _SSF3_NEOADJ_REVERSE or (
            len(_code) == 3 and len(_SSF3_NEOADJ_REVERSE[_label]) < 3):
        _SSF3_NEOADJ_REVERSE[_label] = _code


def encode_ssf3_neoadj(series: pd.Series) -> pd.Series:
    """Inverse of decode_ssf3_neoadj(). Codebook: Cancer-SSF-Manual (breast), p.135-136."""
    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in _SSF3_NEOADJ_REVERSE:
            return _SSF3_NEOADJ_REVERSE[v]
        # decode_ssf3_neoadj() falls back to a bare passthrough otherwise.
        return v

    return series.apply(_encode)


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel LN (SSF4, SSF5) -- breast; LN_POSITI (structural)
# ─────────────────────────────────────────────────────────────────────────────

def encode_sentinel(series: pd.Series, kind: str) -> pd.Series:
    """Inverse of decode_sentinel(). kind: 'examined' or 'positive'."""
    special_rev = {
        'Not applicable (conversion)': '888',
        'Not applicable (no SLN biopsy)': '988',
        'Sentinel LN biopsy performed; no lymph node tissue found or count unknown': '996',
        'Unknown': '999',
    }
    zero_label = 'None positive' if kind == 'positive' else 'None examined'
    node_re = re.compile(rf'^(\d+) node\(s\) {re.escape(kind)}$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in special_rev:
            return special_rev[v]
        if v == zero_label:
            return '0'
        m = node_re.match(v)
        if m:
            return m.group(1)
        # decode_sentinel() falls back to a bare passthrough otherwise
        # (e.g. an out-of-range count like '99').
        return v

    return series.apply(_encode)


def encode_lnpositive(series: pd.Series) -> pd.Series:
    """Inverse of decode_lnpositive()."""
    special_rev = {
        'Positive LN, count not applicable': '95',
        'Positive LN, count not specified': '97',
        'Unknown': '99',
    }

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in special_rev:
            return special_rev[v]
        # decode_lnpositive() passes any other value through unchanged
        # (actual counts, and anything it doesn't otherwise recognize).
        return v

    return series.apply(_encode)


# ─────────────────────────────────────────────────────────────────────────────
# EBRT technique (additive bitmask) -- structural
# ─────────────────────────────────────────────────────────────────────────────

_EBRT_REVERSE = {v: k for k, v in EBRT_COMPONENTS.items()}


def encode_ebrt_additive(series: pd.Series) -> pd.Series:
    """Inverse of decode_ebrt_additive(). Codebook: Longform-Manual, p.240-241."""
    code_part_re = re.compile(r'^code-(\d+)$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v == 'Unknown':
            return '999'
        if v == 'EBRT NOS':
            return '-1'
        if v == 'No EBRT':
            return '0'
        parts = v.split(' + ')
        if len(parts) == 1 and parts[0] not in _EBRT_REVERSE and not code_part_re.match(parts[0]):
            # decode_ebrt_additive() passes non-numeric garbage through
            # unchanged (int(v) parse failure) -- this isn't a composite
            # label at all, so treat it the same way.
            return v
        total = 0
        for part in parts:
            part = part.strip()
            if part in _EBRT_REVERSE:
                total += _EBRT_REVERSE[part]
                continue
            m = code_part_re.match(part)
            if m:
                total += int(m.group(1))
                continue
            # A real decode() output never mixes recognized and
            # unrecognized components in one composite label -- this
            # indicates a hand-edited/corrupted cell, not a passthrough.
            raise ValueError(f'Cannot encode EBRT technique component: {part!r} (in {v!r})')
        return str(total)

    return series.apply(_encode)


# ─────────────────────────────────────────────────────────────────────────────
# Lung: mediastinal LN sampling (SSF5), EGFR (SSF6), nodule count (SSF9)
# ─────────────────────────────────────────────────────────────────────────────

def encode_lung_ssf5_mediastinal(series: pd.Series) -> pd.Series:
    """Inverse of _decode_lung_ssf5_mediastinal()."""
    station_re = re.compile(r'^(\d+) mediastinal LN station\(s\) sampled/dissected$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v == 'Not applicable (small cell lung cancer or no surgery)':
            return '988'
        if v == 'Unknown / not documented; stations dissected but location unclear':
            return '999'
        if v == 'No mediastinal LN sampling or dissection':
            return '0'
        m = station_re.match(v)
        if m:
            return m.group(1)
        m = re.match(r'^Code (\d+)$', v)
        if m:
            return m.group(1)
        # Bare passthrough: decode's own fallback for non-numeric input.
        return v

    return series.apply(_encode)


def encode_lung_egfr(series: pd.Series) -> pd.Series:
    """Inverse of _decode_lung_egfr(). Always emits the modern 3-char, X-padded code.

    Position among the 3 characters is not clinically meaningful (the field
    is a SET of concurrent mutations, per the codebook), so encoding
    canonicalizes to recognized letters first, X-padded at the end --
    e.g. both 'AXB' and 'ABX' decode identically and both re-encode to 'ABX'.
    """
    from tcr_decoder.ssf_registry import _LUNG_EGFR_LETTER_MAP
    letter_reverse = {v: k for k, v in _LUNG_EGFR_LETTER_MAP.items()}
    unknown_letter_re = re.compile(r'^\?\((.)\)$')
    code_fallback_re = re.compile(r'^EGFR code: (.*)$', re.DOTALL)

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v == 'Unknown / not tested':
            return '999'
        if v == 'EGFR — No mutation (XXX)':
            return 'XXX'
        m = code_fallback_re.match(v)
        if m:
            # decode_lung_egfr()'s own fallback for anything it can't
            # otherwise interpret -- recover the original raw value.
            return m.group(1)
        prefix = 'EGFR — '
        if not v.startswith(prefix):
            return v
        mutation_names = v[len(prefix):].split(' + ')
        if len(mutation_names) > 3:
            raise ValueError(
                f'Cannot encode EGFR label {v!r}: more than 3 concurrent mutations')
        letters = []
        for name in mutation_names:
            if name in letter_reverse:
                letters.append(letter_reverse[name])
                continue
            um = unknown_letter_re.match(name)
            if um:
                # decode's own '?(X)' placeholder for an unrecognized single
                # letter within an otherwise-valid 3-char code.
                letters.append(um.group(1))
                continue
            raise ValueError(f'Unknown EGFR mutation name {name!r} in label {v!r}')
        return ''.join(letters).ljust(3, 'X')

    return series.apply(_encode)


def encode_lung_ssf9_nodules(series: pd.Series) -> pd.Series:
    """Inverse of _decode_lung_ssf9_nodules()."""
    count_re = re.compile(r'^(\d+) tumor nodules$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v == 'Not applicable (T0; non-stage 0-2; T3N0; no surgery; single tumor; no external data)':
            return '988'
        if v == 'Unknown / not documented':
            return '999'
        if v == '>20 tumor nodules':
            return '21'
        m = count_re.match(v)
        if m:
            return m.group(1)
        m = re.match(r'^Code (\d+)$', v)
        if m:
            return m.group(1)
        return v

    return series.apply(_encode)


# ─────────────────────────────────────────────────────────────────────────────
# Colorectum/Stomach: CEA lab value (SSF1), RAS mutation (SSF6)
# ─────────────────────────────────────────────────────────────────────────────

def encode_cea_lab_value(series: pd.Series) -> pd.Series:
    """Inverse of _decode_cea_lab_value()."""
    value_re = re.compile(r'^CEA (\d+(?:\.\d+)?) ng/mL$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v == 'Not applicable (GIST/NETs or treated at external hospital)':
            return '988'
        if v == 'CEA unknown / not tested':
            return '999'
        if v == 'CEA <=0.1 ng/mL':
            return '1'
        if v == 'CEA >=98.7 ng/mL':
            return '987'
        m = value_re.match(v)
        if m:
            code = round(float(m.group(1)) * 10)
            if 2 <= code <= 986:
                return str(code)
        m = re.match(r'^CEA code (\d+)$', v)
        if m:
            return m.group(1)
        return v

    return series.apply(_encode)


def encode_ras_mutation(series: pd.Series) -> pd.Series:
    """Inverse of _decode_ras_mutation(). Filler digit is always '8' (per codebook)."""
    from tcr_decoder.ssf_registry import _RAS_KRAS_NRAS_MAP
    kras_nras_reverse = {v: k for k, v in _RAS_KRAS_NRAS_MAP.items()}
    combo_re = re.compile(r'^KRAS: (.+) \| NRAS: (.+)$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v == 'Not applicable (GIST/NETs/high-grade dysplasia or no external data)':
            return '988'
        if v == 'RAS result not documented / not tested':
            return '998'
        m = combo_re.match(v)
        if m:
            k_label, n_label = m.group(1), m.group(2)
            if k_label not in kras_nras_reverse or n_label not in kras_nras_reverse:
                raise ValueError(f'Unknown KRAS/NRAS mutation name in label: {v!r}')
            return f'{kras_nras_reverse[k_label]}{kras_nras_reverse[n_label]}8'
        m = re.match(r'^RAS code: (.*)$', v, re.DOTALL)
        if m:
            return m.group(1)
        return v

    return series.apply(_encode)


# ─────────────────────────────────────────────────────────────────────────────
# Liver: AFP (SSF1), lab values x10 (Creatinine/Bilirubin SSF4/5), INR (SSF6)
# ─────────────────────────────────────────────────────────────────────────────

def encode_liver_afp(series: pd.Series) -> pd.Series:
    """Inverse of _decode_liver_afp(). Codebook: Cancer-SSF-Manual (liver)."""
    a_code_re = re.compile(r'^AFP (\d+) ng/mL \(A-code, 2021\+ scheme\)$')
    tens_10x_re = re.compile(r'^AFP ~(\d+) ng/mL \(pre-2021 code: value /10, rounded\)$')
    hundreds_re = re.compile(r'^AFP (\d+)-(\d+) ng/mL \(100-999 range code\)$')
    thousands_re = re.compile(r'^AFP ~(\d+) ng/mL \(1000-9879 range code\)$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v == 'Not applicable (treated at another hospital, no external data)':
            return '988'
        if v == 'AFP > instrument max (400-6000 ng/mL range)':
            return '991'
        if v == 'AFP > instrument max (6001-9879 ng/mL range)':
            return '992'
        if v == 'AFP >=9880 ng/mL (or instrument max >=9880)':
            return '993'
        if v == 'Unknown / not tested before first treatment':
            return '999'
        if v == 'AFP <1 ng/mL (A-code, 2021+ scheme)':
            return 'A00'
        m = a_code_re.match(v)
        if m:
            return f'A{int(m.group(1)):02d}'
        m = tens_10x_re.match(v)
        if m:
            return str(round(int(m.group(1)) / 10))
        m = hundreds_re.match(v)
        if m:
            return str(round(int(m.group(1)) / 10))
        m = thousands_re.match(v)
        if m:
            return str(round(int(m.group(1)) / 10))
        m = re.match(r'^AFP code (\d+)$', v)
        if m:
            return m.group(1)
        # Bare passthrough: decode's own fallback for non-numeric input.
        return v

    return series.apply(_encode)


def encode_lab_value_10x(series: pd.Series, analyte: str, unit: str) -> pd.Series:
    """Inverse of _decode_lab_value_10x() (Creatinine, Total bilirubin)."""
    value_re = re.compile(rf'^{re.escape(analyte)} (\d+(?:\.\d+)?) {re.escape(unit)}$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v == 'Not applicable (treated at another hospital, no external data)':
            return '988'
        if v == f'{analyte} unknown / not tested':
            return '999'
        if v == f'{analyte} <=0.1 {unit}':
            return '1'
        if v == f'{analyte} >=98.7 {unit}':
            return '987'
        m = value_re.match(v)
        if m:
            code = round(float(m.group(1)) * 10)
            if 2 <= code <= 986:
                return str(code)
        m = re.match(rf'^{re.escape(analyte)} code (\d+)$', v)
        if m:
            return m.group(1)
        # Bare passthrough: decode's own fallback for non-numeric input.
        return v

    return series.apply(_encode)


def encode_liver_inr(series: pd.Series) -> pd.Series:
    """Inverse of _decode_liver_inr()."""
    value_re = re.compile(r'^INR (\d+(?:\.\d+)?)$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v == 'Not applicable (treated at another hospital, no external data)':
            return '988'
        if v == 'INR >6.0 (or above instrument maximum)':
            return '997'
        if v == 'INR unknown / not tested':
            return '999'
        m = value_re.match(v)
        if m:
            code = round(float(m.group(1)) * 10)
            if 1 <= code <= 60:
                return str(code)
        m = re.match(r'^INR code (\d+)$', v)
        if m:
            return m.group(1)
        # Bare passthrough: decode's own fallback for non-numeric input.
        return v

    return series.apply(_encode)


# ─────────────────────────────────────────────────────────────────────────────
# Prostate: PSA (SSF1)
# ─────────────────────────────────────────────────────────────────────────────

def encode_psa(series: pd.Series) -> pd.Series:
    """Inverse of _decode_psa(). Codebook: Cancer-SSF-Manual (prostate), p.177-178."""
    from tcr_decoder.ssf_registry import _PSA_TIER_MAP, _PSA_THOUSANDS_MAP
    tier_reverse = {f'{lo:.1f}-{hi:.1f}': code for code, (lo, hi) in _PSA_TIER_MAP.items()}
    thousands_reverse = {k: code for code, k in _PSA_THOUSANDS_MAP.items()}
    value_re = re.compile(r'^PSA (\d+(?:\.\d+)?) ng/mL$')
    tier_re = re.compile(r'^PSA (\d+\.\d)-(\d+\.\d) ng/mL$')
    thousands_re = re.compile(r'^PSA (\d+)-(\d+) ng/mL$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v == 'PSA <0.1 ng/mL (undetectable)':
            return '0'
        if v == 'Not applicable':
            return '988'
        if v == 'PSA >=8000 ng/mL':
            return '998'
        if v == 'Unknown; not documented':
            return '999'
        if v == 'PSA 98.0 ng/mL (legacy code, dx year 100-104 only)':
            return '980'
        m = tier_re.match(v)
        if m and f'{m.group(1)}-{m.group(2)}' in tier_reverse:
            return str(tier_reverse[f'{m.group(1)}-{m.group(2)}'])
        m = thousands_re.match(v)
        if m:
            k = int(m.group(1)) // 1000
            if k in thousands_reverse:
                return str(thousands_reverse[k])
        m = value_re.match(v)
        if m:
            code = round(float(m.group(1)) * 10)
            if 1 <= code <= 979:
                return str(code)
        m = re.match(r'^Code (\d+)$', v)
        if m:
            return m.group(1)
        # Bare passthrough: decode's own fallback for non-numeric input.
        return v

    return series.apply(_encode)


# ─────────────────────────────────────────────────────────────────────────────
# Generic SSF passthrough (used by the 'generic' cancer-group fallback, and
# by any cancer-specific SSF field left undefined -- decoder=None)
# ─────────────────────────────────────────────────────────────────────────────

_GENERIC_SENTINEL_REVERSE = {
    'Not applicable (conversion)': '888',
    'No laboratory test done (clinical assessment only)': '900',
    'No laboratory test done (radiographic assessment only)': '901',
    'Not documented in medical record': '902',
    'Not applicable': '988',
    'Not applicable - information not collected for this case': '998',
    'Unknown / not stated': '999',
}


def encode_generic_ssf(series: pd.Series, unit: str = '') -> pd.Series:
    """Inverse of _generic_ssf(). Numeric passthrough, with unit suffix stripped."""
    suffix = f' {unit}' if unit else ''
    value_re = re.compile(rf'^(-?\d+){re.escape(suffix)}$') if suffix else re.compile(r'^(-?\d+)$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in _GENERIC_SENTINEL_REVERSE:
            return _GENERIC_SENTINEL_REVERSE[v]
        m = value_re.match(v)
        if m:
            return m.group(1)
        # _generic_ssf() passes non-numeric input through unchanged.
        return v

    return series.apply(_encode)
