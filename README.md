# TCR Decoder — Taiwan Cancer Registry Multi-Cancer SSF Decoder

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Decode Taiwan Cancer Registry (TCR) raw fields — including all Site-Specific Factors (SSF1–SSF10) — into clinically meaningful English labels. Supports **11 cancer groups** with cancer-specific biomarker decoding.

---

## Supported Cancer Groups

| Group | ICD-O-3 | Key SSF Fields |
|-------|---------|---------------|
| **breast** | C50 | ER, PR, HER2, Ki67, Nottingham grade, sentinel LN |
| **lung** | C34 | EGFR exon, ALK, ROS1, PD-L1% |
| **colorectum** | C18–C21 | CEA, MSI (MSS/MSI-L/MSI-H), KRAS codon |
| **liver** | C22 | AFP, HBV/HCV, Child-Pugh class |
| **prostate** | C61 | PSA (×10 format), Gleason score, Grade Group |
| **stomach** | C16 | Lauren type, H. pylori, HER2 |
| **thyroid** | C73 | Focality, BRAF, extrathyroidal extension |
| **cervix** | C53 | HPV, parametrial involvement |
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

### EGFR (Lung SSF3)
- Code 1: exon 19 deletion
- Code 2: exon 21 L858R
- Code 3: exon 20 insertion
- Code 4: exon 18 G719X
- Code 5: other mutation

---

## Project Structure

```
tcr_decoder/
├── __init__.py          # Public API (v2.0.0)
├── __main__.py          # CLI entry point
├── core.py              # TCRDecoder class (pipeline orchestrator)
├── ssf_registry.py      # Multi-cancer SSF routing (11 profiles)
├── decoders.py          # Breast-specific SSF decoders
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
├── test_pipeline.py     # End-to-end pipeline tests
├── test_synth.py        # Synthetic generator tests
└── test_adversarial.py  # Adversarial / stress tests (517 total)
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

**517 tests** | categories: sentinel chaos, boundary values, type injection, ICD-O-3 edge cases, profile contracts, roundtrip integrity, performance (10K rows), CLI smoke, contradictory data, rstrip regression

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
