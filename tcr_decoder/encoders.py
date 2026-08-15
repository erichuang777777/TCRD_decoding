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


# Inverse of decoders._unlisted_code(): a sentinel-shaped code outside a
# field's official range is decoded as 'Unlisted code 888' rather than being
# left as a bare number in a clinical column, so the encode side recovers the
# original code from that label.
_UNLISTED_RE = re.compile(r'^Unlisted code (\S+)$')


def _reverse_unlisted(v: str) -> str:
    m = _UNLISTED_RE.match(v)
    return m.group(1) if m else v


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
            if 0 <= pct <= 100:
                return f'{intensity_code[m.group(1)]}{pct % 100:02d}'
            # decode_er_pr()'s staining regex has no upper bound on the
            # digit run, so it leniently decodes an already out-of-spec raw
            # code (e.g. a corrupted 'S150') instead of rejecting it. Mirror
            # that leniency here too: `pct % 100` would silently WRAP an
            # out-of-range value into a different, wrong 2-digit code
            # (150 -> 'S50') instead of preserving it -- return the value
            # unmodified so the round trip doesn't fabricate a new number.
            return f'{intensity_code[m.group(1)]}{pct}'
        # SSF1/SSF2 are 3-character fields (欄位長度：3, range 000-100), so a
        # percentage is written back zero-padded: 70% -> '070', 0% -> '000'.
        m = positive_re.match(v)
        if m:
            return m.group(1).zfill(3)
        m = negative_re.match(v)
        if m and m.group(1) == '0':
            return '000'
        # decode_er_pr() itself falls back to a bare passthrough for
        # anything it doesn't recognize (e.g. an out-of-range percentage) --
        # mirror that leniency rather than raising.
        return _reverse_unlisted(v)

    return series.apply(_encode)


# ─────────────────────────────────────────────────────────────────────────────
# Ki-67 (SSF10) -- breast
# ─────────────────────────────────────────────────────────────────────────────

def encode_ki67(series: pd.Series) -> pd.Series:
    """Inverse of decode_ki67(). Codebook: Cancer-SSF-Manual (breast), p.150-151."""
    special_rev = {
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
        # 3-character field (欄位長度：3, range 000-100): 25% -> '025'.
        m = categorized_re.match(v)
        if m:
            return m.group(1).zfill(3)
        m = fractional_re.match(v)
        if m:
            pct = float(m.group(1))
            return f'A{round(pct * 10):02d}'
        # decode_ki67() falls back to a bare passthrough for anything else.
        return _reverse_unlisted(v)

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
        return _reverse_unlisted(v)

    return series.apply(_encode)


# ─────────────────────────────────────────────────────────────────────────────
# Nottingham Grade (SSF6) -- breast
# ─────────────────────────────────────────────────────────────────────────────

def encode_nottingham(series: pd.Series) -> pd.Series:
    """Inverse of decode_nottingham(). Codebook: Cancer-SSF-Manual (breast), p.140-141.

    Emits the official 3-digit code: a BR score N becomes '0N0' (score 6 ->
    '060'), matching the codebook's range 030,040,050,060,070,080,090,
    110,120,130,999. A bare '6' is NOT a valid TCR code -- writing one back
    into a registry submission would be rejected -- so a decoded "Score 6"
    always re-encodes as '060' even if the source file happened to hold '6'.
    A grade-only label encodes to 110/120/130.
    """
    special_rev = {
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
            return f'0{m.group(1)}0'
        m = grade_only_re.match(v)
        if m:
            return grade_only_code[m.group(1)]
        # decode_nottingham() falls back to a bare passthrough otherwise.
        return _reverse_unlisted(v)

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
        return _reverse_unlisted(v)

    return series.apply(_encode)


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel LN (SSF4, SSF5) -- breast; LN_POSITI (structural)
# ─────────────────────────────────────────────────────────────────────────────

def encode_sentinel(series: pd.Series, kind: str) -> pd.Series:
    """Inverse of decode_sentinel(). kind: 'examined' or 'positive'."""
    special_rev = {
        ('Not applicable (no SLN surgery, post-neoadjuvant SLN, '
         'or unknown whether performed)'): '988',
        ('Sentinel LN biopsy performed; count unknown or no lymph '
         'node tissue found in the specimen'): '996',
        'Unknown': '999',
    }
    zero_label = ('None positive (or ITC-only involvement)' if kind == 'positive'
                  else 'No sentinel LN surgery performed')
    node_re = re.compile(rf'^(\d+) node\(s\) {re.escape(kind)}$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in special_rev:
            return special_rev[v]
        # 3-character field (欄位長度：3, range 000-089): 5 nodes -> '005'.
        if v == zero_label:
            return '000'
        m = node_re.match(v)
        if m:
            return m.group(1).zfill(3)
        # decode_sentinel() falls back to a bare passthrough otherwise
        # (e.g. an out-of-range count like '99').
        return _reverse_unlisted(v)

    return series.apply(_encode)


def encode_lnpositive(series: pd.Series) -> pd.Series:
    """Inverse of decode_lnpositive()."""
    special_rev = {
        'Positive by aspiration/core biopsy only (nodes not surgically removed)': '95',
        'Positive LN present, count not specified': '97',
        'No nodes removed/examined, or no lymph node tissue found (clinical assessment only)': '98',
        'Unknown / not applicable / not documented': '99',
    }

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in special_rev:
            return special_rev[v]
        # decode_lnpositive() passes any other value through unchanged
        # (actual counts, and anything it doesn't otherwise recognize).
        # LN_POSITI is a 2-character field (Longform-Manual p.130), so a
        # count is written back zero-padded: 5 -> '05'.
        if v.isdigit() and len(v) < 2:
            return v.zfill(2)
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
            return '000'
        m = station_re.match(v)
        if m:
            return m.group(1).zfill(3)
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
            return '021'
        m = count_re.match(v)
        if m:
            return m.group(1).zfill(3)
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
            return '001'
        if v == 'CEA >=98.7 ng/mL':
            return '987'
        m = value_re.match(v)
        if m:
            code = round(float(m.group(1)) * 10)
            if 2 <= code <= 986:
                return str(code).zfill(3)
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
            return str(round(int(m.group(1)) / 10)).zfill(3)
        m = hundreds_re.match(v)
        if m:
            return str(round(int(m.group(1)) / 10)).zfill(3)
        m = thousands_re.match(v)
        if m:
            return str(round(int(m.group(1)) / 10)).zfill(3)
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
            return '001'
        if v == f'{analyte} >=98.7 {unit}':
            return '987'
        m = value_re.match(v)
        if m:
            code = round(float(m.group(1)) * 10)
            if 2 <= code <= 986:
                return str(code).zfill(3)
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
                return str(code).zfill(3)
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
        # 3-character field (欄位長度：3, range 001-999 with no 000).
        if v == 'PSA <=0.1 ng/mL':
            return '001'
        if v == 'Not applicable (first course at another hospital, no lab value)':
            return '988'
        if v == 'PSA >=8000 ng/mL':
            return '998'
        if v == 'Unknown; not documented':
            return '999'
        if v == 'PSA 98.0 ng/mL (legacy code, dx year 100-104 only)':
            return '980'
        m = tier_re.match(v)
        if m and f'{m.group(1)}-{m.group(2)}' in tier_reverse:
            return str(tier_reverse[f'{m.group(1)}-{m.group(2)}']).zfill(3)
        m = thousands_re.match(v)
        if m:
            k = int(m.group(1)) // 1000
            if k in thousands_reverse:
                return str(thousands_reverse[k]).zfill(3)
        m = value_re.match(v)
        if m:
            code = round(float(m.group(1)) * 10)
            if 2 <= code <= 979:
                return str(code).zfill(3)
        m = re.match(r'^Code (\d+)$', v)
        if m:
            return m.group(1)
        # Bare passthrough: decode's own fallback for non-numeric input.
        return _reverse_unlisted(v)

    return series.apply(_encode)


# ─────────────────────────────────────────────────────────────────────────────
# Head & neck (Cancer-SSF-Manual pp.3-29)
# ─────────────────────────────────────────────────────────────────────────────

def encode_hn_node_size(series: pd.Series) -> pd.Series:
    """Inverse of _decode_hn_node_size() (head & neck SSF1)."""
    from tcr_decoder.ssf_registry import _decode_hn_node_size

    reverse = _reverse_from_decoder(
        _decode_hn_node_size,
        ['000', '987', '988', '990', '991', '992', '993', '994', '995', '996',
         '997', '999'])
    value_re = re.compile(r'^Involved node (\d+) mm$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in reverse:
            return reverse[v]
        m = value_re.match(v)
        if m and 1 <= int(m.group(1)) <= 986:
            return m.group(1).zfill(3)
        m = re.match(r'^Code (\S+)$', v)
        if m:
            return m.group(1)
        return _reverse_unlisted(v)

    return series.apply(_encode)


def encode_hn_levels(series: pd.Series, ssf_key: str) -> pd.Series:
    """Inverse of _decode_hn_levels() (head & neck SSF3-SSF6)."""
    from tcr_decoder.ssf_registry import _HN_LEVEL_REGIONS, _HN_LEVEL_STATUS

    regions = _HN_LEVEL_REGIONS[ssf_key]
    status_code = {label: code for code, label in _HN_LEVEL_STATUS.items()}
    none_label = f'{regions[0]}/{regions[1]}/{regions[2]}: none involved (N0)'
    na_label = ('Not applicable (examined or treated at another hospital '
                'with no data available)')
    unknown_label = ('Nodal status for these regions unknown / not documented / '
                     'not assessable')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v == none_label:
            return '000'
        if v == na_label:
            return '988'
        if v == unknown_label:
            return '999'
        parts = v.split('; ')
        if len(parts) == 3:
            digits = ''
            for region, part in zip(regions, parts):
                prefix = f'{region}: '
                if not part.startswith(prefix):
                    digits = ''
                    break
                code = status_code.get(part[len(prefix):])
                if code is None:
                    digits = ''
                    break
                digits += code
            if len(digits) == 3:
                return digits
        m = re.match(r'^Code (\S+)$', v)
        if m:
            return m.group(1)
        return _reverse_unlisted(v)

    return series.apply(_encode)


def encode_hn_tumor_depth(series: pd.Series) -> pd.Series:
    """Inverse of _decode_hn_tumor_depth() (head & neck SSF7)."""
    from tcr_decoder.ssf_registry import _decode_hn_tumor_depth

    reverse = _reverse_from_decoder(
        _decode_hn_tumor_depth,
        ['000', '980', '987', '988', '990', '997', '998', '999'])
    value_re = re.compile(r'^Tumour depth (\d+\.\d) mm$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in reverse:
            return reverse[v]
        m = value_re.match(v)
        if m:
            code = round(float(m.group(1)) * 10)
            if 1 <= code <= 979:
                return str(code).zfill(3)
        m = re.match(r'^Code (\S+)$', v)
        if m:
            return m.group(1)
        return _reverse_unlisted(v)

    return series.apply(_encode)


def encode_hn_margin(series: pd.Series) -> pd.Series:
    """Inverse of _decode_hn_margin() (head & neck SSF8)."""
    from tcr_decoder.ssf_registry import _decode_hn_margin

    reverse = _reverse_from_decoder(
        _decode_hn_margin, ['000', '980', '987', '988', '990', '998', '999'])
    value_re = re.compile(r'^Margin negative, (\d+\.\d) mm$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in reverse:
            return reverse[v]
        m = value_re.match(v)
        if m:
            code = round(float(m.group(1)) * 10)
            if 1 <= code <= 979:
                return str(code).zfill(3)
        m = re.match(r'^Code (\S+)$', v)
        if m:
            return m.group(1)
        return _reverse_unlisted(v)

    return series.apply(_encode)


def encode_hn_ene_clinical(series: pd.Series) -> pd.Series:
    """Inverse of _decode_hn_ene_clinical() (head & neck SSF9)."""
    from tcr_decoder.ssf_registry import _HN_ENE_DETAIL, _HN_ENE_FINAL

    final_code = {label: code for code, label in _HN_ENE_FINAL.items()}
    detail_code = {label: code for code, label in _HN_ENE_DETAIL.items()}
    composite_re = re.compile(
        r'^Overall: (.+); imaging: (.+); physical exam: (.+)$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v == 'Not applicable (clinical N category is cN0)':
            return '988'
        if v == ('Not applicable (treated at another hospital with no data, '
                 'or diagnosed only after surgery)'):
            return '998'
        m = composite_re.match(v)
        if m:
            first = final_code.get(m.group(1))
            second = detail_code.get(m.group(2))
            third = detail_code.get(m.group(3))
            if None not in (first, second, third):
                return f'{first}{second}{third}'
        m = re.match(r'^Code (\S+)$', v)
        if m:
            return m.group(1)
        return _reverse_unlisted(v)

    return series.apply(_encode)


def encode_hn_ene_pathological(series: pd.Series) -> pd.Series:
    """Inverse of _decode_hn_ene_pathological() (head & neck SSF10)."""
    from tcr_decoder.ssf_registry import _decode_hn_ene_pathological

    reverse = _reverse_from_decoder(
        _decode_hn_ene_pathological,
        ['000', '199', '210', '299', '399', '988', '998', '999'])
    le2_re = re.compile(r'^Pathological ENE <=2 mm, measured (\d+\.\d) mm$')
    gt2_re = re.compile(r'^Pathological ENE >2 mm, measured (\d+\.\d) mm$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in reverse:
            return reverse[v]
        m = le2_re.match(v)
        if m:
            dist = round(float(m.group(1)) * 10)
            if 1 <= dist <= 20:
                return f'1{dist:02d}'
        m = gt2_re.match(v)
        if m:
            dist = round(float(m.group(1)) * 10)
            if 21 <= dist <= 98:
                return f'2{dist:02d}'
        m = re.match(r'^Code (\S+)$', v)
        if m:
            return m.group(1)
        return _reverse_unlisted(v)

    return series.apply(_encode)


# ─────────────────────────────────────────────────────────────────────────────
# Cervix / stomach / colorectum fields with their own code table
# ─────────────────────────────────────────────────────────────────────────────

def _reverse_from_decoder(decoder, codes) -> dict:
    """Build a label -> code table by running the decoder over its own codes.

    Keeps a bespoke encoder from drifting away from its decoder: the mapping
    is derived, never typed out twice.
    """
    labels = decoder(pd.Series(list(codes), dtype=object))
    reverse = {}
    for code, label in zip(codes, labels):
        reverse.setdefault(label, code)
    return reverse


def encode_scc_lab_value(series: pd.Series) -> pd.Series:
    """Inverse of _decode_scc_lab_value() (cervix SSF1). 3-character field."""
    value_re = re.compile(r'^SCC antigen (\d+(?:\.\d+)?) ng/mL$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v == 'Not applicable (treated at another hospital, no lab value)':
            return '988'
        if v == 'SCC antigen unknown / not tested':
            return '999'
        if v == 'SCC antigen <=0.1 ng/mL':
            return '001'
        if v == 'SCC antigen >=98.7 ng/mL':
            return '987'
        m = value_re.match(v)
        if m:
            code = round(float(m.group(1)) * 10)
            if 2 <= code <= 986:
                return str(code).zfill(3)
        m = re.match(r'^SCC antigen code (\d+)$', v)
        if m:
            return m.group(1)
        return _reverse_unlisted(v)

    return series.apply(_encode)


def encode_tumor_depth(series: pd.Series) -> pd.Series:
    """Inverse of _decode_tumor_depth() (stomach SSF4). 3-character field."""
    from tcr_decoder.ssf_registry import _decode_tumor_depth

    reverse = _reverse_from_decoder(
        _decode_tumor_depth, ['000', '980', '988', '998', '999'])
    value_re = re.compile(r'^Tumour depth (\d+(?:\.\d+)?) mm$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in reverse:
            return reverse[v]
        m = value_re.match(v)
        if m:
            code = round(float(m.group(1)) * 10)
            if 1 <= code <= 979:
                return str(code).zfill(3)
        m = re.match(r'^Code (\S+)$', v)
        if m:
            return m.group(1)
        return _reverse_unlisted(v)

    return series.apply(_encode)


def encode_crm(series: pd.Series) -> pd.Series:
    """Inverse of _decode_crm() (colorectum SSF4). 3-character field."""
    from tcr_decoder.ssf_registry import _decode_crm

    reverse = _reverse_from_decoder(
        _decode_crm, ['980', '988', '990', '991', '992', '993', '994', '995',
                      '996', '999'])
    value_re = re.compile(r'^CRM (\d+(?:\.\d+)?) mm$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in reverse:
            return reverse[v]
        m = value_re.match(v)
        if m:
            code = round(float(m.group(1)) * 10)
            if 0 <= code <= 979:
                return str(code).zfill(3)
        m = re.match(r'^Code (\S+)$', v)
        if m:
            return m.group(1)
        return _reverse_unlisted(v)

    return series.apply(_encode)


def encode_distance_to_anus(series: pd.Series) -> pd.Series:
    """Inverse of _decode_distance_to_anus() (rectum SSF9). 3 characters."""
    from tcr_decoder.ssf_registry import _decode_distance_to_anus

    reverse = _reverse_from_decoder(
        _decode_distance_to_anus, ['000', '988', '991', '992', '993', '999'])
    value_re = re.compile(r'^(\d+) mm from the anus$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in reverse:
            return reverse[v]
        m = value_re.match(v)
        if m and 1 <= int(m.group(1)) <= 150:
            return m.group(1).zfill(3)
        m = re.match(r'^Code (\S+)$', v)
        if m:
            return m.group(1)
        return _reverse_unlisted(v)

    return series.apply(_encode)


# ─────────────────────────────────────────────────────────────────────────────
# Prostate: Gleason patterns (SSF2/SSF4), Gleason score (SSF3/SSF5),
# biopsy cores (SSF6/SSF7)
# ─────────────────────────────────────────────────────────────────────────────

def encode_gleason_patterns(series: pd.Series, specimen: str) -> pd.Series:
    """Inverse of _decode_gleason_patterns(). 3-character field."""
    from tcr_decoder.ssf_registry import _decode_gleason_patterns

    # Build the reverse table from the decoder itself so the two can never
    # drift: every legal code decoded once, then indexed by its label.
    codes = ([f'{p}{s}' for p in range(1, 6) for s in list(range(1, 6)) + [9]]
             + ['099', '988', '999'])
    codes = [c.zfill(3) for c in codes]
    labels = _decode_gleason_patterns(pd.Series(codes, dtype=object), specimen)
    reverse = {}
    for code, label in zip(codes, labels):
        reverse.setdefault(label, code)

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in reverse:
            return reverse[v]
        m = re.match(r'^Code (\S+)$', v)
        if m:
            return m.group(1)
        return _reverse_unlisted(v)

    return series.apply(_encode)


def encode_gleason_score(series: pd.Series, specimen: str) -> pd.Series:
    """Inverse of _decode_gleason_score(). 3-character field."""
    from tcr_decoder.ssf_registry import _decode_gleason_score

    codes = [f'{i:03d}' for i in list(range(2, 11)) + [988, 999]]
    labels = _decode_gleason_score(pd.Series(codes, dtype=object), specimen)
    reverse = {}
    for code, label in zip(codes, labels):
        reverse.setdefault(label, code)

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in reverse:
            return reverse[v]
        m = re.match(r'^Code (\S+)$', v)
        if m:
            return m.group(1)
        return _reverse_unlisted(v)

    return series.apply(_encode)


def encode_biopsy_cores(series: pd.Series, kind: str) -> pd.Series:
    """Inverse of _decode_biopsy_cores(). 3-character field."""
    from tcr_decoder.ssf_registry import _decode_biopsy_cores

    sentinels = [988, 999] + ([998, 0] if kind == 'positive' else [])
    codes = [f'{i:03d}' for i in sentinels]
    labels = _decode_biopsy_cores(pd.Series(codes, dtype=object), kind)
    reverse = {}
    for code, label in zip(codes, labels):
        reverse.setdefault(label, code)
    count_re = re.compile(rf'^(\d+) core\(s\) {re.escape(kind)}$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in reverse:
            return reverse[v]
        m = count_re.match(v)
        if m:
            return m.group(1).zfill(3)
        m = re.match(r'^Code (\S+)$', v)
        if m:
            return m.group(1)
        return _reverse_unlisted(v)

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


# ─────────────────────────────────────────────────────────────────────────────
# Pancreas / ovary fields with a scaled or composite code
# ─────────────────────────────────────────────────────────────────────────────

def encode_ca19_9(series: pd.Series) -> pd.Series:
    """Inverse of _decode_ca19_9() (pancreas SSF3). 3-character field."""
    from tcr_decoder.ssf_registry import _decode_ca19_9

    tier_codes = ['000', '001', '988', '999'] + [
        str(i).zfill(3) for i in list(range(980, 998)) + [989, 990]]
    reverse = _reverse_from_decoder(_decode_ca19_9, sorted(set(tier_codes)))
    value_re = re.compile(r'^CA 19-9 (\d+\.\d) U/mL$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in reverse:
            return reverse[v]
        m = value_re.match(v)
        if m:
            code = round(float(m.group(1)) * 10)
            if 2 <= code <= 979:
                return str(code).zfill(3)
        m = re.match(r'^Code (\S+)$', v)
        if m:
            return m.group(1)
        return _reverse_unlisted(v)

    return series.apply(_encode)


def encode_mitotic_count(series: pd.Series) -> pd.Series:
    """Inverse of _decode_mitotic_count() (pancreas SSF5)."""
    from tcr_decoder.ssf_registry import _decode_mitotic_count

    reverse = _reverse_from_decoder(
        _decode_mitotic_count, ['110', '120', '130', '988', '999'])
    count_re = re.compile(r'^(\d+) mitoses per 10 HPF \(per 2 mm2\)$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in reverse:
            return reverse[v]
        m = count_re.match(v)
        if m and 0 <= int(m.group(1)) <= 21:
            return m.group(1).zfill(3)
        m = re.match(r'^Code (\S+)$', v)
        if m:
            return m.group(1)
        return _reverse_unlisted(v)

    return series.apply(_encode)


def encode_hba1c(series: pd.Series) -> pd.Series:
    """Inverse of _decode_hba1c() (pancreas SSF6)."""
    from tcr_decoder.ssf_registry import _decode_hba1c

    reverse = _reverse_from_decoder(
        _decode_hba1c,
        ['988', '999'] + [f'{h}{v:02d}' for h in '01'
                          for v in list(range(1, 95)) + [95, 96, 97, 98, 99]])

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in reverse:
            return reverse[v]
        m = re.match(r'^Code (\S+)$', v)
        if m:
            return m.group(1)
        return _reverse_unlisted(v)

    return series.apply(_encode)


def encode_ca125(series: pd.Series, timing: str) -> pd.Series:
    """Inverse of _decode_ca125() (ovary SSF1/SSF2)."""
    from tcr_decoder.ssf_registry import _decode_ca125

    tiers = [str(i) for i in list(range(901, 911)) + [920, 930, 931, 988, 999]]
    reverse = _reverse_from_decoder(
        lambda s: _decode_ca125(s, timing), tiers)
    when = 'before treatment' if timing == 'pre' else 'lowest after treatment'
    value_re = re.compile(rf'^CA-125 {re.escape(when)}: (\d+) U/mL$')

    def _encode(v):
        v = _clean(v)
        if not v:
            return ''
        if v in reverse:
            return reverse[v]
        m = value_re.match(v)
        if m and 1 <= int(m.group(1)) <= 900:
            return m.group(1).zfill(3)
        m = re.match(r'^Code (\S+)$', v)
        if m:
            return m.group(1)
        return _reverse_unlisted(v)

    return series.apply(_encode)
