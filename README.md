# TCR Decoder — Taiwan Cancer Registry Multi-Cancer SSF Decoder

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Decode Taiwan Cancer Registry (TCR) raw fields — including all Site-Specific Factors (SSF1–SSF10) — into clinically meaningful English labels. Supports **10 named cancer groups** plus a generic fallback profile.

---

## Coding Standard Versions

All field mappings and decoders in this package are written against the following **official TCR codebooks**:

| Document | Version | Effective Date |
|----------|---------|---------------|
| TCR Longform Coding Manual | `Longform-Manual_Official-version_20251224_W-1` | 2025-12-24 |
| TCR SSF Definitions Manual | `Cancer-SSF-Manual_Official-version_20251204_W` | 2025-12-04 |

Additional classification systems used:

| Standard | Version / Edition | Applied to |
|----------|------------------|-----------|
| ICD-O-3  | 3rd Edition      | Topography (TCODE1) and morphology (MCODE) |
| AJCC     | 7th & 8th Editions (auto-detected) | TNM staging |
| UICC TNM | 8th Edition      | Pathological staging fields (PT/PN/PM) |
| WHO Classification of Tumours | 2022 (Breast) | Nottingham grade, histology |
| St. Gallen Consensus | 2013 / 2015 | Molecular subtype (Luminal A/B, HER2-E, TNBC) |
| ASCO/CAP | ER/PR 2020 update; HER2 2023 update | ER/PR positivity thresholds and HER2 interpretation context |

> **Version traceability**: `from tcr_decoder import TCR_CODEBOOK_VERSION, TCR_SSF_MANUAL_VERSION`

---

## Supported Cancer Groups

| Group | ICD-O-3 | Key SSF Fields |
|-------|---------|---------------|
| **breast** | C50 | ER, PR, HER2, Ki67, Nottingham grade, sentinel LN |
| **lung** | C34 | Separate tumor nodules, visceral pleural invasion, ECOG/KPS, malignant pleural effusion, mediastinal LN sampling, EGFR, ALK |
| **colorectum** | C18–C21 | CEA, MSI (MSS/MSI-L/MSI-H), KRAS codon |
| **liver** | C22 | AFP, HBV/HCV, Child-Pugh class |
| **prostate** | C61 | PSA (×10 format), Gleason score, Grade Group |
| **stomach** | C16 | CEA, H. pylori, tumor depth, LVI |
| **thyroid** | C73 | Focality, vascular invasion, extrathyroidal extension, BRAF/RAS, thyroglobulin |
| **cervix** | C53 | SCC antigen value and SCC antigen vs normal |
| **nasopharynx** | C11 | EBV serology, plasma EBV DNA |
| **endometrium** | C54 | POLE, MMR, p53 (FIGO 2023 molecular) |
| **generic** | any | Numeric passthrough with sentinel decoding |

---

## Installation

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

---

## Quick Start

### Python API

```python
from tcr_decoder import TCRDecoder

# Auto-detect cancer group from TCODE1 (ICD-O-3 code)
dec = TCRDecoder("registry_data.xlsx")
dec.load().decode().validate().export("output.xlsx")

print(dec.cancer_group)   # e.g. 'breast'
print(dec.clean.columns)  # all decoded columns
print(dec.flags)          # clinical QA flags
```

### Force a specific cancer group

```python
dec = TCRDecoder("registry_data.xlsx", cancer_group="lung")
dec.load().decode().validate().export("lung_clean.xlsx")
```

### Programmatic SSF decoding

```python
from tcr_decoder import apply_ssf_profile, detect_cancer_group
import pandas as pd

group = detect_cancer_group("C50.1")   # → 'breast'

df = pd.DataFrame({
    "SSF1_raw": [70, 120, 888, 999],   # ER codes
    "SSF7_raw": [103, 510, 300, 999],  # HER2 codes
    # ... SSF2-10 also required
    **{f"SSF{i}_raw": [0]*4 for i in [2,3,4,5,6,8,9,10]}
})

result = apply_ssf_profile(df, "breast")
print(result[["ER_Status", "HER2_Status"]])
```

---

## Encoding (the reverse direction): clinical text → TCR raw codes

`TCRDecoder` turns a registry export into clinical text. `TCREncoder` is its
inverse: given a DataFrame shaped like `TCRDecoder(...).clean` (the
`Clinical_Clean` sheet), it reconstructs the raw `SSF1_raw … SSF10_raw` codes
plus a handful of structural fields (AJCC edition, surgery-type codes,
regional-LN-surgery codes, EBRT technique, LN_POSITI).

```python
from tcr_decoder import TCRDecoder, TCREncoder

dec = TCRDecoder("registry.xlsx").load().decode()

enc = TCREncoder(dec.clean)
raw = enc.encode()                 # DataFrame of {FIELD}_raw columns
print(enc.cancer_group)            # auto-detected the same way TCRDecoder does
print(enc.unencoded_columns)       # {column: reason} for anything it couldn't reconstruct
```

**Read this before assuming full coverage.** Most of `TCRDecoder`'s ~100
output columns are decoded by trusting a `{FIELD}_decoded` column that is
already present in the registry export (or, for a couple of fields, an
external `cancer_registry_mapping.py` that is not part of this repo) — there
is no code table in this package to invert for those, and `TCREncoder`
deliberately reports them in `unencoded_columns` instead of guessing. What it
*does* reconstruct with full fidelity is every SSF1-10 biomarker field across
all 16 cancer profiles, plus the structural fields with a real code table
(AJCC, per-site surgery codes, regional-node surgery codes, EBRT, LNEXAM,
LN_POSITI),
since those have real decode logic (and now, real inverse logic) in this
codebase.

### Comparing two files / validating round-trip fidelity

`compare_roundtrip` decodes a registry file, re-encodes it, and diffs the
result against the original raw codes — this is the "convert both ways and
compare" check:

```python
from tcr_decoder import compare_roundtrip, export_roundtrip_report

mismatches = compare_roundtrip("registry.xlsx")   # DataFrame: Patient_ID, Field, Original_Code, Roundtrip_Code
export_roundtrip_report("registry.xlsx", "roundtrip_report.xlsx")  # multi-sheet Excel report
```

**Verified groups must report ZERO mismatches.** Fourteen cancer groups
(breast, prostate, endometrium, thyroid, cervix, stomach, liver, lung,
colorectum, head & neck, esophagus, pancreas, ovary, bladder) have had their
ten SSF fields verified field-by-field against the printed code book: the official 編碼範圍 of each field is transcribed in
`tcr_decoder/code_ranges.py`, and `tests/test_breast_codebook_conformance.py`
enumerates every legal code to prove that decode understands all of them, that
no two codes share a decoded label, that `encode(decode(code)) == code` byte
for byte, and that encode only ever emits a submittable code at the field's
official width (a Nottingham score of 6 encodes to `060`, never `6`). A
mismatch on a breast field therefore means the *input file* holds a code
outside the official range, and `Roundtrip_Code` shows what the registry would
accept.

> **FHIR / 乳癌 IG**：把這些代碼表變成 FHIR CodeSystem/ValueSet、
> Questionnaire 與 Task 的產生器，以及乳癌摘錄原則與示範，已移到
> [TW-Breast-Cancer-FHIR-IG](https://github.com/erichuang777777/TW-Breast-Cancer-FHIR-IG)
> 的 `tcr_workbench/`。那個 repo 以本套件為相依，本 repo 只負責代碼表與雙向轉換。

### The breast validation data set

One command builds the full evidence file — every legal code with its code
book meaning, plus pairwise-complete synthetic cases run through the real
pipeline in both directions:

```bash
python -m tcr_decoder --build-validation breast_validation_dataset.xlsx
```

Sheets: `說明` (how to read it), `Summary`, `Field_Coverage` (all 1,187 legal
breast codes → 碼冊中文定義 → English clinical label → re-encoded code),
`Case_Combinations` (542 cases covering 100% of the 5,882 two-field value
pairs), `Case_Roundtrip` (5,420 field comparisons through the full
decode→encode pipeline), `Clinical_View` (each case's decoded clinical
picture plus its molecular subtype / NPI reading) and `Failures` (must be
empty). `tests/test_validation_dataset.py` runs the same builder and fails
if a single code or combination stops round-tripping.

> Code columns in that workbook are **text**: HER2 `000` and `100` are
> different codes. Read it back with `dtype=str` or pandas will infer `000`
> as the number 0 — exactly the defect this data set found in the decoder's
> own Excel reader (now fixed: all `*_raw` / `*_decoded` columns are read as
> text so fixed-width codes keep their leading zeros).

21,158 legal codes across all sixteen groups round-trip exactly, and every
one of them carries the code book's own Chinese definition in the validation
workbook.

Lymphoma and leukemia are keyed off the **morphology** code, not the primary
site (Cancer-SSF-Manual pp.194, 207), so `detect_cancer_group()` takes an
optional second argument:

```python
detect_cancer_group('C16.9')              # 'stomach'
detect_cancer_group('C16.9', '9699/3')    # 'lymphoma' — a gastric MALT lymphoma
detect_cancer_group('C42.2', '9835/3')    # 'lymphoma' — spleen
detect_cancer_group('C42.1', '9835/3')    # 'leukemia' — same morphology, marrow
```

M-9811-9837 is the one range the manual splits on the site, so neither code
can decide it alone. `TCRDecoder` passes `MCODE_raw` automatically when the
column is present; without a morphology column those cases fall back to their
site and a nodal lymphoma reads as `generic`.

Surgery of primary site (`PRESTYPE`, `STYPE95`) is defined **per primary
site** by Appendix B, so decoding it needs the topography code:

```python
decode_surgery(pd.Series(['660']), pd.Series(['C50.9']))
# 'Total (simple) mastectomy WITHOUT contralateral, implant reconstruction'
decode_surgery(pd.Series(['660']), pd.Series(['C34.1']))   # not a lung code
```

All 30 Appendix B tables (686 codes, 520 topography codes) are generated
straight from the PDF by `scripts/generate_surgery_codes.py`, so a new edition
of the manual is a re-run rather than a re-transcription.

Still outstanding: the other 61 Longform fields have no code table yet — see
[`docs/codebook_conformance_findings.md`](docs/codebook_conformance_findings.md).
The report's `Notes` sheet says the same.

---

## CLI

```bash
# Decode a registry file (cancer type auto-detected from TCODE1)
python -m tcr_decoder registry.xlsx

# Force cancer group
python -m tcr_decoder registry.xlsx output.xlsx --cancer lung

# List all supported cancer groups
python -m tcr_decoder --list-cancers

# Show SSF field definitions for a cancer group
python -m tcr_decoder --ssf-info colorectum

# Generate synthetic test data
python -m tcr_decoder --synth breast --n 200 --seed 42 --out test.xlsx --decode

# Round-trip check: decode a registry file, re-encode it, and report where
# the raw codes do not come back exactly (see "Encoding" section above)
python -m tcr_decoder registry.xlsx --roundtrip
```

---

## Synthetic Data Generator

No real patient data? Generate realistic synthetic data for testing:

```python
from tcr_decoder.synth import SyntheticTCRGenerator

gen = SyntheticTCRGenerator(cancer_group="breast", n=500, seed=42)
df = gen.generate()
gen.to_excel("synthetic_breast.xlsx")
print(gen.summary())
```

Supported cancer groups for synthesis: `breast`, `lung`, `colorectum`

---

## Key Decoding Features

### Sentinel Code Handling
All decoders handle TCR standard sentinel codes:

| Code | Meaning |
|------|---------|
| `888` | Not applicable (conversion after neoadjuvant) |
| `900–902` | No test done (clinical / radiographic / not documented) |
| `988` | Not applicable |
| `998` | Not applicable – not collected |
| `999` | Unknown / not stated |

### ER/PR (Breast SSF1/SSF2)
- Codes 0–100: percentage positivity (`ER Positive (70%)`)
- Code 0: negative (`ER Negative (0%)`)
- Code 120: `ER Negative (<1% or not specified)`
- W/I/S prefix: staining intensity codes

### HER2 (Breast SSF7)
- IHC-only era (codes 1xx): `IHC 0–3+`
- ISH-only era (codes 3xx): `ISH Negative/Positive/Equivocal`
- IHC+ISH combined era (codes 5xx–6xx): `IHC 2+ + ISH Positive — Positive`

### EGFR (Lung SSF6)
- Three-character alphabetic codes can encode concurrent mutations
- `A`: exon 19 deletion
- `B`: exon 21 L858R
- `E`: exon 20 insertion
- `D`: exon 18 G719X
- `X`: no mutation

---

## Project Structure

```
tcr_decoder/
├── __init__.py          # Public API (v2.0.0)
├── __main__.py          # CLI entry point
├── core.py              # TCRDecoder class (pipeline orchestrator)
├── encoder.py           # TCREncoder class (inverse pipeline orchestrator)
├── roundtrip.py         # compare_roundtrip / export_roundtrip_report
├── codemap.py           # CodeMap: bidirectional code<->label source of truth
├── code_ranges.py       # Official 編碼範圍 per field (breast, verified vs the manual)
├── validation.py        # Bidirectional validation data set builder
├── facts.py             # Observation/Resolution engine: keep every source,
│                        resolve per the code book's own rule
├── ssf_registry.py      # Multi-cancer SSF routing (11 profiles), decode + encode
├── decoders.py          # Breast-specific SSF decoders
├── encoders.py          # Inverse of decoders.py/ssf_registry.py's bespoke decoders
├── validators.py        # Clinical QA / consistency checks
├── derived.py           # Derived fields (staging, ratios)
├── synth.py             # Synthetic data generator
├── input_validator.py   # Pre-decode input validation
├── data_dictionary.py   # Data dictionary generator
├── utils.py             # Shared utilities
└── mappings.py          # Code mappings

tests/
├── conftest.py          # Session-scoped fixtures (synthetic data)
├── test_ssf_registry.py # SSF routing, cancer group detection
├── test_decoders.py     # Individual decoder tests (boundary, edge cases)
├── test_encoders.py     # Encode-direction + round-trip tests (all 11 profiles)
├── test_breast_codebook_conformance.py  # Exhaustive breast <-> code book conformance
├── test_validation_dataset.py           # Field + pairwise-case validation data set (16 groups)
├── test_pipeline.py     # End-to-end pipeline tests
├── test_synth.py        # Synthetic generator tests
└── test_adversarial.py  # Adversarial / stress tests
```

---

## Test Suite

```bash
# Run all tests
pytest tests/ -q

# With coverage
pytest tests/ --cov=tcr_decoder --cov-report=term-missing

# Run only adversarial tests
pytest tests/test_adversarial.py -v
```

**1800+ tests, all passing** | categories: sentinel chaos, boundary values, type injection, ICD-O-3 edge cases, profile contracts, roundtrip integrity, performance (10K rows), CLI smoke, contradictory data, rstrip regression, pipeline bug regression (Round 3), mathematical formula verification (Round 4), encode-direction round trips across every cancer profile (`test_encoders.py`), exhaustive code-book conformance over all 21,158 legal codes of all sixteen groups (`test_codebook_conformance.py`), and the bidirectional validation data set (`test_validation_dataset.py`)

---

## Requirements

- Python ≥ 3.9
- pandas ≥ 1.5
- numpy ≥ 1.23
- openpyxl ≥ 3.1

---

## License

MIT License — see [LICENSE](LICENSE) for details.

> **Privacy note**: This package processes cancer registry data. Never commit real patient data. Use `SyntheticTCRGenerator` for testing and development.
