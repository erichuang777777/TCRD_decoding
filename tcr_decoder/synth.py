"""
Synthetic TCR Data Generator — for testing and education purposes.

Generates realistic fake Taiwan Cancer Registry data in the expected
_raw / _decoded column format, without any patient privacy concerns.

Usage:
    from tcr_decoder.synth import SyntheticTCRGenerator

    # Generate 50 breast cancer patients → ready for TCRDecoder
    gen = SyntheticTCRGenerator(cancer_group='breast', n=50, seed=42)
    df  = gen.generate()
    gen.to_excel('synthetic_breast.xlsx')

    # Then decode it:
    from tcr_decoder import TCRDecoder
    TCRDecoder('synthetic_breast.xlsx').run('synthetic_breast_clean.xlsx')

    # Or from command line:
    python -m tcr_decoder.synth --cancer breast --n 100 --seed 42

Statistical distributions are modelled on published literature:
    Breast : PMID 36574899 (Taiwan breast cancer registry 2002-2019)
    Lung   : PMID 35088937 (Taiwan NSCLC EGFR prevalence)
    CRC    : PMID 34385350 (Taiwan colorectal cancer outcomes)
"""

from __future__ import annotations

import argparse
import random
import string
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _choice(rng: np.random.Generator, items, p=None):
    items = list(items)
    idx = rng.choice(len(items), p=p)
    return items[idx]


def _rand_date(rng: np.random.Generator, start: date, end: date) -> str:
    delta = (end - start).days
    d = start + timedelta(days=int(rng.integers(0, delta)))
    # Occasionally produce partial date (day=99) like real TCR
    if rng.random() < 0.05:
        return f'{d.year}/{d.month:02d}/99'
    return d.strftime('%Y/%m/%d')


def _pk(i: int, prefix: str = 'SYN') -> str:
    return f'{prefix}{i:05d}'


# ─────────────────────────────────────────────────────────────────────────────
# Cancer-specific field distributions
# ─────────────────────────────────────────────────────────────────────────────

class _BreastFields:
    """Realistic distributions for breast cancer SSF and staging fields."""

    TCODE1_CHOICES = ['C50.1', 'C50.2', 'C50.4', 'C50.9']
    TCODE1_P      = [0.20,    0.20,    0.25,    0.35]

    # Laterality: 96=bilateral(rare), 1=right, 2=left, 9=NA
    LAT_CHOICES = ['1', '2', '9', '96']
    LAT_P       = [0.48, 0.48, 0.03, 0.01]

    # Histology (MCODE): IDC=8500, ILC=8520, metaplastic=8575,8070
    MCODE_CHOICES = ['8500', '8520', '8575', '8070', '8522', '8480']
    MCODE_P       = [0.65,   0.10,   0.08,   0.07,   0.05,   0.05]

    # ER (SSF1): 0-100=%, 120=negative, 888=converted, 999=unknown
    @staticmethod
    def ssf1(rng):
        r = rng.random()
        if r < 0.65:   return int(rng.integers(50, 100))  # Positive (50-99%)
        if r < 0.73:   return int(rng.integers(1, 50))    # Low positive (1-49%)
        if r < 0.85:   return 120                          # Negative
        if r < 0.88:   return 888                          # Converted
        return 999

    # PR (SSF2)
    @staticmethod
    def ssf2(rng):
        r = rng.random()
        if r < 0.55:   return int(rng.integers(30, 100))
        if r < 0.65:   return int(rng.integers(1, 30))
        if r < 0.82:   return 120
        if r < 0.86:   return 888
        return 999

    # Neoadjuvant (SSF3): 0=no neoadj, 10=cCR, 11=pCR, 20-25=partial, 988=NA
    @staticmethod
    def ssf3(rng):
        r = rng.random()
        if r < 0.60:  return 0
        if r < 0.70:  return 988
        if r < 0.78:  return int(rng.choice([20, 21, 22, 23, 24, 25]))
        if r < 0.88:  return 10
        return 11

    # Sentinel LN examined (SSF4): 0=none, 1-10=count, 99=unknown
    @staticmethod
    def ssf4(rng):
        r = rng.random()
        if r < 0.10:  return 0
        if r < 0.85:  return int(rng.integers(1, 8))
        return 99

    # Sentinel LN positive (SSF5): 0=none, 1-N=count, 99=unknown
    @staticmethod
    def ssf5(ssf4_val, rng):
        if ssf4_val == 0:   return 0
        if ssf4_val == 99:  return 99
        r = rng.random()
        if r < 0.60:  return 0
        return int(rng.integers(1, max(2, ssf4_val + 1)))

    # Nottingham (SSF6): 30-90=score×10, 110/120/130=grade only, 999=unknown
    @staticmethod
    def ssf6(rng):
        r = rng.random()
        if r < 0.10:  return 999
        if r < 0.15:  return int(rng.choice([110, 120, 130]))
        scores = [30, 40, 50, 60, 70, 80, 90]
        p = [0.05, 0.10, 0.15, 0.25, 0.20, 0.15, 0.10]
        return int(rng.choice(scores, p=p))

    # HER2 IHC+ISH (SSF7): 100=IHC0, 101=IHC1+, 300=IHC3+(pos), 510=ISH+, 999=unknown
    @staticmethod
    def ssf7(rng):
        r = rng.random()
        if r < 0.35:  return 100   # IHC 0 Neg
        if r < 0.55:  return 101   # IHC 1+ Neg
        if r < 0.70:  return 200   # IHC 2+ Equivocal (no ISH)
        if r < 0.80:  return 510   # ISH Positive
        if r < 0.90:  return 300   # IHC 3+ Positive
        return 999

    # Paget's (SSF8): 0=no, 1=yes without mass, 2=yes with mass, 9=unknown
    @staticmethod
    def ssf8(rng):
        return int(rng.choice([0, 1, 2, 9], p=[0.88, 0.05, 0.05, 0.02]))

    # LVI_SSF (SSF9): 0=none, 1=present, 9=unknown
    @staticmethod
    def ssf9(rng):
        return int(rng.choice([0, 1, 9], p=[0.55, 0.30, 0.15]))

    # Ki67 (SSF10): 0-100=%, 999=unknown
    @staticmethod
    def ssf10(rng):
        r = rng.random()
        if r < 0.15:  return 999
        # Log-normal distribution centred ~20%
        val = int(np.clip(rng.lognormal(mean=3.0, sigma=0.7), 1, 99))
        return val

    # Path T stage
    PSTAGE_CHOICES = ['I', 'IA', 'IB', 'II', 'IIA', 'IIB', 'III', 'IIIA', 'IIIB', 'IIIC', 'IV']
    PSTAGE_P       = [0.00, 0.22, 0.04, 0.00, 0.28, 0.12, 0.00, 0.12,  0.04,  0.08,  0.10]


class _LungFields:
    """Distributions for lung cancer SSF fields per 2025 TCR codebook."""

    TCODE1_CHOICES = ['C34.1', 'C34.2', 'C34.3', 'C34.0', 'C34.9']
    TCODE1_P      = [0.35,    0.20,    0.20,    0.10,    0.15]

    LAT_CHOICES = ['1', '2', '9']
    LAT_P       = [0.45, 0.45, 0.10]

    MCODE_CHOICES = ['8140', '8070', '8041', '8255', '8260']
    MCODE_P       = [0.50,   0.25,   0.15,   0.05,   0.05]

    # SSF1 — Separate tumor nodules: 0=none, 10=ipsi same lobe, 20=ipsi diff lobe,
    #         30=both same+diff, 40=ipsi lobe unknown, 999=unknown
    @staticmethod
    def ssf1(rng):
        return int(rng.choice([0, 10, 20, 30, 40, 999], p=[0.65, 0.12, 0.08, 0.03, 0.02, 0.10]))

    # SSF2 — Visceral pleural invasion: 0=PL0, 10=PL1, 20=PL2, 30=PL3,
    #         40=beyond chest wall, 988=N/A, 999=unknown
    @staticmethod
    def ssf2(rng):
        return int(rng.choice([0, 10, 20, 30, 40, 988, 999], p=[0.50, 0.20, 0.12, 0.06, 0.02, 0.03, 0.07]))

    # SSF3 — Performance status ECOG: 0=PS0, 1=PS1, 2=PS2, 3=PS3, 4=PS4, 988=N/A, 999=unknown
    @staticmethod
    def ssf3(rng):
        return int(rng.choice([0, 1, 2, 3, 4, 988, 999], p=[0.25, 0.35, 0.20, 0.08, 0.02, 0.03, 0.07]))

    # SSF4 — Malignant pleural effusion: 0=no, 11=yes cytology neg, 12=pos borderline,
    #         13=pos malignant, 14=pericardial, 988=N/A, 999=unknown
    @staticmethod
    def ssf4(rng):
        return int(rng.choice([0, 11, 12, 13, 14, 988, 999], p=[0.60, 0.08, 0.05, 0.10, 0.02, 0.05, 0.10]))

    # SSF5 — Mediastinal LN sampling: 0=no dissection, 1=EBUS, 2=EUS,
    #         3=mediastinoscopy, 4=VATS, 5=thoracotomy, 988=N/A, 999=unknown
    @staticmethod
    def ssf5(rng):
        return int(rng.choice([0, 1, 2, 3, 4, 5, 988, 999], p=[0.30, 0.20, 0.10, 0.15, 0.08, 0.05, 0.05, 0.07]))

    # SSF6 — EGFR mutation (3-char alphabetic): per 2025 TCR codebook p.117
    #   XXX=no mutation, AXX=exon19del, BXX=L858R, EXX=exon20ins, GXX=T790M,
    #   VVV=mutated NOS, ZZZ=uninterpretable, 999=not tested
    @staticmethod
    def ssf6(rng):
        r = rng.random()
        if r < 0.15:  return '999'   # not tested
        if r < 0.40:  return 'XXX'   # WT (common in squamous/SCLC)
        if r < 0.60:  return 'AXX'   # exon 19 del
        if r < 0.76:  return 'BXX'   # L858R
        if r < 0.82:  return 'EXX'   # exon 20 ins
        if r < 0.86:  return 'GXX'   # T790M
        if r < 0.91:  return 'ABX'   # exon19 + L858R (compound)
        if r < 0.95:  return 'VVV'   # mutated NOS
        return 'ZZZ'                  # uninterpretable

    # SSF7 — ALK translocation: 10=positive, 20=negative, 30=uninterpretable, 999=not tested
    @staticmethod
    def ssf7(rng):
        return int(rng.choice([10, 20, 30, 999], p=[0.06, 0.78, 0.04, 0.12]))

    # SSF8 — Adenocarcinoma component (bitmask): 0=none, 1=micropapillary, 2=solid,
    #         3=micropapillary+solid, 4=cribriform, 988=N/A, 999=unknown
    @staticmethod
    def ssf8(rng):
        return int(rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 988, 999],
                               p=[0.35, 0.15, 0.15, 0.05, 0.05, 0.03, 0.03, 0.02, 0.10, 0.07]))

    # SSF9 — Tumor nodule count (for pathologic stage I/II only): 988=N/A, 999=unknown, or 2-10
    @staticmethod
    def ssf9(rng):
        return int(rng.choice([988, 999, 2, 3, 4, 5], p=[0.55, 0.20, 0.12, 0.07, 0.04, 0.02]))

    # SSF10 — Not defined in 2025 TCR codebook for lung; use 999 (unknown/not collected)
    @staticmethod
    def ssf10(rng):
        return 999

    PSTAGE_CHOICES = ['IA', 'IB', 'IIA', 'IIB', 'IIIA', 'IIIB', 'IV']
    PSTAGE_P       = [0.18,  0.10,  0.08,  0.08,  0.18,   0.10,   0.28]


class _ColorectumFields:
    """Distributions for colorectal cancer SSF fields per 2025 TCR codebook."""

    TCODE1_CHOICES = ['C18.0', 'C18.2', 'C18.6', 'C18.7', 'C19.9', 'C20.9']
    TCODE1_P      = [0.10,    0.20,    0.15,    0.20,    0.15,    0.20]

    LAT_CHOICES = ['9']
    LAT_P       = [1.0]

    MCODE_CHOICES = ['8140', '8480', '8144', '8010']
    MCODE_P       = [0.75,   0.10,   0.10,   0.05]

    # SSF1 — CEA lab value ×10 (code 20 = 2.0 ng/mL, 150 = 15.0 ng/mL): 988=N/A, 999=unknown
    @staticmethod
    def ssf1(rng):
        r = rng.random()
        if r < 0.12:  return 999
        if r < 0.15:  return 988
        # Pick a realistic CEA ×10 value: 5-9 normal range, >50 elevated
        choices = [5, 10, 15, 20, 30, 50, 80, 100, 150, 300, 500]
        p       = [0.12, 0.15, 0.12, 0.12, 0.10, 0.10, 0.08, 0.07, 0.06, 0.05, 0.03]
        return int(rng.choice(choices, p=p))

    # SSF2 — CEA vs. normal: 10=above normal, 20=normal, 30=borderline, 988=N/A, 999=unknown
    @staticmethod
    def ssf2(rng):
        return int(rng.choice([10, 20, 30, 988, 999], p=[0.50, 0.30, 0.08, 0.05, 0.07]))

    # SSF3 — Tumor regression grade (post-neoadjuvant): 0=complete, 1=TRG1, 2=TRG2,
    #         3=TRG3, 4=no regression, 988=N/A (no neoadjuvant), 999=unknown
    @staticmethod
    def ssf3(rng):
        return int(rng.choice([0, 1, 2, 3, 4, 988, 999], p=[0.05, 0.10, 0.15, 0.10, 0.10, 0.42, 0.08]))

    # SSF4 — Circumferential resection margin (CRM) in 0.1 mm: 988=N/A, 999=unknown
    @staticmethod
    def ssf4(rng):
        r = rng.random()
        if r < 0.10:  return 999
        if r < 0.25:  return 988   # colon (no mesorectal fascia)
        return int(rng.choice([0, 1, 5, 10, 20, 30, 50], p=[0.10, 0.15, 0.20, 0.20, 0.15, 0.12, 0.08]))

    # SSF5 — BRAF mutation: 10=V600E positive, 20=non-V600E positive,
    #         30=BRAF WT/negative, 988=N/A, 999=not tested
    @staticmethod
    def ssf5(rng):
        return int(rng.choice([10, 20, 30, 988, 999], p=[0.08, 0.02, 0.65, 0.05, 0.20]))

    # SSF6 — RAS mutation (3-char code): pos1=KRAS, pos2=NRAS, pos3='8' filler
    #   0=WT, 1=codon12, 2=codon13, 3=codon61, 4=multi, 5=other, 6=NOS, 7=uninterpret, 9=not tested
    #   Examples: '008'=both WT, '108'=KRAS codon12, '018'=NRAS codon12, '998'=both not tested
    @staticmethod
    def ssf6(rng):
        kras_choices = ['0', '1', '2', '3', '4', '6', '9']
        kras_p       = [0.52, 0.20, 0.10, 0.04, 0.02, 0.02, 0.10]
        kras = rng.choice(kras_choices, p=kras_p)
        # If KRAS tested and negative (0), NRAS also commonly tested
        if kras == '0':
            nras = rng.choice(['0', '1', '2', '9'], p=[0.85, 0.05, 0.03, 0.07])
        elif kras == '9':
            nras = '9'  # not tested either
        else:
            nras = '9'  # KRAS mutant → NRAS usually not tested
        return f'{kras}{nras}8'

    # SSF7 — Intestinal obstruction: 0=no, 10=yes, 988=N/A, 999=unknown
    @staticmethod
    def ssf7(rng):
        return int(rng.choice([0, 10, 988, 999], p=[0.75, 0.15, 0.05, 0.05]))

    # SSF8 — Intestinal perforation: 0=no, 10=yes, 988=N/A, 999=unknown
    @staticmethod
    def ssf8(rng):
        return int(rng.choice([0, 10, 988, 999], p=[0.85, 0.05, 0.05, 0.05]))

    # SSF9 — Distance to anus (mm): 0-150, 988=N/A, 991=not applicable (colon),
    #         992=not applicable (appendix), 993=no surgery, 999=unknown
    @staticmethod
    def ssf9(rng):
        site_r = rng.random()
        if site_r < 0.40:  return 991  # colon primary (not rectum)
        if site_r < 0.45:  return 993  # no surgery
        if site_r < 0.50:  return 999  # unknown
        # Rectum: 0-150 mm from anus
        return int(rng.choice([0, 20, 40, 60, 80, 100, 120, 150],
                               p=[0.05, 0.15, 0.20, 0.20, 0.15, 0.10, 0.10, 0.05]))

    # SSF10 — MSI/MMR status: 0=MSS, 10=MSI-L, 20=MSI-H/dMMR, 988=N/A, 999=not tested
    @staticmethod
    def ssf10(rng):
        return int(rng.choice([0, 10, 20, 988, 999], p=[0.65, 0.05, 0.15, 0.05, 0.10]))

    PSTAGE_CHOICES = ['I', 'II', 'IIA', 'IIB', 'III', 'IIIA', 'IIIB', 'IIIC', 'IV']
    PSTAGE_P       = [0.18, 0.05, 0.15, 0.05, 0.05, 0.12,  0.10,  0.10,  0.20]


# ─────────────────────────────────────────────────────────────────────────────
# Generic staging helpers (cancer-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

_STAGE_TO_TNM = {
    'IA':   ('T1', 'N0', 'M0'), 'IB':   ('T1', 'N0', 'M0'),
    'I':    ('T1', 'N0', 'M0'),
    'IIA':  ('T2', 'N0', 'M0'), 'IIB':  ('T2', 'N1', 'M0'),
    'II':   ('T2', 'N0', 'M0'),
    'IIIA': ('T3', 'N1', 'M0'), 'IIIB': ('T3', 'N2', 'M0'),
    'IIIC': ('T3', 'N3', 'M0'), 'III':  ('T3', 'N1', 'M0'),
    'IV':   ('T4', 'N2', 'M1'),
}

_HISTOLOGY_NAMES = {
    '8500': 'Infiltrating duct carcinoma, NOS',
    '8520': 'Lobular carcinoma, NOS',
    '8575': 'Metaplastic carcinoma, NOS',
    '8070': 'Squamous cell carcinoma, NOS',
    '8140': 'Adenocarcinoma, NOS',
    '8480': 'Mucinous adenocarcinoma',
    '8041': 'Small cell carcinoma, NOS',
    '8255': 'Adenocarcinoma with mixed subtypes',
    '8260': 'Papillary adenocarcinoma, NOS',
    '8522': 'Infiltrating duct and lobular carcinoma',
    '8144': 'Adenocarcinoma, intestinal type',
    '8010': 'Carcinoma, NOS',
}

_SEX_MAP   = {'1': 'Male', '2': 'Female'}
_BEHAV_MAP = {'3': 'Malignant, primary site', '2': 'In situ'}
_GRADE_MAP = {
    '1': 'Grade I: Well differentiated',
    '2': 'Grade II: Moderately differentiated',
    '3': 'Grade III: Poorly differentiated',
    '4': 'Grade IV: Undifferentiated',
    '9': 'Unknown grade',
}
_AJCC_MAP  = {
    '06': 'AJCC 6th edition (2002)',
    '07': 'AJCC 7th edition (2010)',
    '08': 'AJCC 8th edition (2018)',
}
_VSTA6_MAP = {'0': 'Alive', '1': 'Dead of cancer', '2': 'Dead of other cause'}
_CLASS_MAP = {
    '1': 'Class 1: Dx & first course Tx at this hospital',
    '2': 'Class 2: Dx at this hospital, Tx elsewhere',
    '3': 'Class 3: Dx & all Tx elsewhere (DC only)',
}

_CANCER_FIELDS_MAP = {
    'breast':      _BreastFields,
    'lung':        _LungFields,
    'colorectum':  _ColorectumFields,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticTCRGenerator:
    """Generate synthetic Taiwan Cancer Registry data for testing.

    Args:
        cancer_group: One of 'breast', 'lung', 'colorectum' (more to come)
        n: Number of synthetic patients to generate
        seed: Random seed for reproducibility
        dx_start: Earliest diagnosis year (default 2010)
        dx_end: Latest diagnosis year (default 2022)
    """

    SUPPORTED = list(_CANCER_FIELDS_MAP.keys())

    def __init__(
        self,
        cancer_group: str = 'breast',
        n: int = 100,
        seed: Optional[int] = 42,
        dx_start: int = 2010,
        dx_end: int = 2022,
    ):
        if cancer_group not in self.SUPPORTED:
            raise ValueError(
                f"cancer_group '{cancer_group}' not supported. "
                f"Choose from: {self.SUPPORTED}"
            )
        self.cancer_group = cancer_group
        self.n = n
        self.seed = seed
        self.dx_start = dx_start
        self.dx_end = dx_end
        self._rng = _rng(seed)
        self._fields_cls = _CANCER_FIELDS_MAP[cancer_group]
        self._df: Optional[pd.DataFrame] = None

    # ── helpers ──────────────────────────────────────────────────────────────

    def _sex(self) -> str:
        if self.cancer_group == 'breast':
            return _choice(self._rng, ['1', '2'], p=[0.01, 0.99])
        if self.cancer_group == 'prostate':
            return '1'
        return _choice(self._rng, ['1', '2'], p=[0.50, 0.50])

    def _age(self) -> int:
        # Age at diagnosis: normal(55, 12), clipped 20-90
        return int(np.clip(self._rng.normal(55, 12), 20, 90))

    def _dx_date(self) -> Tuple[str, int]:
        yr = int(self._rng.integers(self.dx_start, self.dx_end + 1))
        mo = int(self._rng.integers(1, 13))
        dy = int(self._rng.integers(1, 29))
        if self._rng.random() < 0.05:
            return f'{yr}/{mo:02d}/99', yr
        return f'{yr}/{mo:02d}/{dy:02d}', yr

    def _pstage(self) -> str:
        fc = self._fields_cls
        return _choice(self._rng, fc.PSTAGE_CHOICES, p=fc.PSTAGE_P)

    def _tnm_from_stage(self, stage: str) -> Tuple[str, str, str]:
        base = _STAGE_TO_TNM.get(stage, ('TX', 'NX', 'MX'))
        # Occasionally add substage suffixes
        t, n, m = base
        if self._rng.random() < 0.15 and t in ('T1', 'T2'):
            t += _choice(self._rng, ['a', 'b', 'c'])
        if m == 'M1' and self._rng.random() < 0.10:
            m = 'M0(i+) - ITC in bone marrow'
        return t, n, m

    def _lnexam(self) -> int:
        r = self._rng.random()
        if r < 0.05:  return 0
        if r < 0.10:  return 95   # sentinel only
        return int(self._rng.integers(1, 30))

    def _lnpositive(self, lnexam: int) -> str:
        if lnexam == 0:   return '0'
        if lnexam == 95:  return str(int(self._rng.integers(0, 4)))
        r = self._rng.random()
        if r < 0.55:  return '0'
        if r < 0.85:  return str(int(self._rng.integers(1, min(lnexam + 1, 20))))
        return '95'  # ≥95 (all positive)

    def _tx_flag(self, p_yes: float = 0.5) -> str:
        """Return code for treatment performed/not."""
        return '1' if self._rng.random() < p_yes else '0'

    def _surgery_type(self, cancer: str) -> str:
        if cancer == 'breast':
            # 2025 TCR 3-digit breast surgery codes (codebook Appendix B, p.370-372)
            # 200=lumpectomy, 310=skin-sparing, 410=nipple-sparing,
            # 510=areolar-sparing, 610=total, 700=radical, 800=mastectomy NOS
            return _choice(self._rng,
                           ['200', '310', '410', '510', '610', '700', '800', '000'],
                           p=[0.30, 0.10, 0.15, 0.15, 0.18, 0.05, 0.05, 0.02])
        # Generic 3-digit codes for other cancer sites
        return _choice(self._rng, ['100', '200', '300', '400', '000'],
                       p=[0.10, 0.30, 0.35, 0.20, 0.05])

    def _vital_status(self, age: int, surv_yr: float) -> str:
        # Simple hazard: older + longer followup = higher death probability
        p_dead = min(0.40, 0.05 + age / 1000 + surv_yr * 0.02)
        if self._rng.random() < p_dead:
            return '1'  # Dead of cancer
        if self._rng.random() < 0.05:
            return '2'  # Dead of other cause
        return '0'      # Alive

    # ── main generate ────────────────────────────────────────────────────────

    def generate(self) -> pd.DataFrame:
        """Generate synthetic data. Returns DataFrame in TCR _raw/_decoded format."""
        rows = []
        fc = self._fields_cls

        for i in range(1, self.n + 1):
            r = self._rng   # shorthand

            # Basic demographics
            pk        = _pk(i, prefix=self.cancer_group[:2].upper())
            sex_raw   = self._sex()
            age       = self._age()
            dx_date, dx_yr = self._dx_date()
            visit_date = dx_date  # simplification: same day

            # Diagnosis
            tcode1 = _choice(r, fc.TCODE1_CHOICES, p=fc.TCODE1_P)
            lat    = _choice(r, fc.LAT_CHOICES, p=fc.LAT_P)
            mcode  = _choice(r, fc.MCODE_CHOICES, p=fc.MCODE_P)
            behav  = '3'
            grade  = _choice(r, ['1', '2', '3', '4', '9'], p=[0.20, 0.35, 0.30, 0.05, 0.10])
            confer = _choice(r, ['1', '2', '3', '5', '7'], p=[0.05, 0.65, 0.10, 0.15, 0.05])

            # Tumour
            tsize = int(r.integers(5, 120)) if r.random() < 0.85 else 999
            pni   = _choice(r, ['0', '1', '9'], p=[0.60, 0.25, 0.15])
            lvi   = _choice(r, ['0', '1', '9'], p=[0.55, 0.30, 0.15])

            # Staging
            pstage      = self._pstage()
            pt, pn, pm  = self._tnm_from_stage(pstage)
            cstage_raw  = pstage   # simplification: clinical = pathologic
            ct, cn, cm  = pt, pn, pm

            # Determine AJCC edition by year
            if dx_yr < 2010:    ajcc = '06'
            elif dx_yr < 2018:  ajcc = '07'
            else:               ajcc = '08'

            # LN
            lnexam   = self._lnexam()
            lnpos    = self._lnpositive(lnexam)

            # SSF fields (cancer-specific)
            ssf1  = fc.ssf1(r)
            ssf2  = fc.ssf2(r)
            ssf3  = fc.ssf3(r)
            ssf4  = fc.ssf4(r)
            # ssf5: breast takes (ssf4, rng), others take only (rng)
            try:
                ssf5 = fc.ssf5(ssf4, r)   # breast
            except TypeError:
                ssf5 = fc.ssf5(r)          # lung, crc
            ssf6  = fc.ssf6(r)
            ssf7  = fc.ssf7(r)
            ssf8  = fc.ssf8(r)
            ssf9  = fc.ssf9(r)
            ssf10 = fc.ssf10(r)

            # Metastasis site
            meta1 = '70' if pm.startswith('M1') else '0'  # 70=bone

            # Surgery
            has_surg = r.random() < (0.85 if pm.startswith('M0') else 0.30)
            stype = self._surgery_type(self.cancer_group) if has_surg else '00'
            s_raw = '1' if has_surg else '0'
            marg  = _choice(r, ['0', '1', '2', '3', '9'], p=[0.55, 0.20, 0.10, 0.05, 0.10])
            # SLNSCO95: 2025 TCR codebook single-digit codes 0-7
            # 0=no LN, 1=diag biopsy, 2=SLNB, 3=dissection(unknown#), 4=1-3 LN,
            # 5=4+LN, 6=SLNB+dissection same surgery, 7=SLNB then dissection
            slnsco = _choice(r, ['0', '2', '3', '5', '6', '7', '9'],
                             p=[0.08, 0.28, 0.10, 0.15, 0.22, 0.12, 0.05])

            if has_surg:
                surg_days = int(r.integers(7, 60))
                dx_d = pd.to_datetime(dx_date.replace('/',''), format='%Y%m%d', errors='coerce')
                if pd.notna(dx_d):
                    sx_d = dx_d + pd.Timedelta(days=surg_days)
                    fsdate = sx_d.strftime('%Y/%m/%d')
                else:
                    fsdate = ''
            else:
                fsdate = ''

            # Radiation
            has_rt  = r.random() < 0.50
            r_raw   = '1' if has_rt else '0'
            rtar    = int(r.choice([0, 1, 2, 3, 7], p=[0.10, 0.30, 0.30, 0.20, 0.10])) if has_rt else 0
            rmod    = _choice(r, ['1', '2', '3', '9'], p=[0.55, 0.20, 0.20, 0.05]) if has_rt else '0'
            hdose   = int(r.integers(4000, 6600)) if has_rt else 0
            hno     = int(r.integers(15, 35)) if has_rt else 0

            # Systemic therapy
            has_chemo   = r.random() < 0.60
            has_hormone = r.random() < (0.65 if self.cancer_group == 'breast' else 0.05)
            has_target  = r.random() < 0.30
            has_immuno  = r.random() < 0.15
            c_raw   = '1' if has_chemo   else '0'
            h_raw   = '1' if has_hormone else '0'
            tar_raw = '1' if has_target  else '0'
            i_raw   = '1' if has_immuno  else '0'

            # Survival
            followup_months = int(r.integers(3, 120))
            surv_yr = round(followup_months / 12, 1)
            dx_dt = pd.to_datetime(dx_date.replace('/', '').replace('99', '15')[:8], errors='coerce')
            if pd.notna(dx_dt):
                lcd6_dt  = dx_dt + pd.Timedelta(days=followup_months * 30)
                lcd6_str = lcd6_dt.strftime('%Y/%m/%d')
            else:
                lcd6_str = ''

            vsta6 = self._vital_status(age, surv_yr)

            # Smoking triplet (XX,XX,XX): smoking,betelnut,alcohol
            smk  = _choice(r, ['00', '01', '02', '09'], p=[0.35, 0.45, 0.15, 0.05])
            btel = _choice(r, ['00', '01', '09'], p=[0.70, 0.25, 0.05])
            alc  = _choice(r, ['00', '01', '09'], p=[0.50, 0.45, 0.05])
            smoking_raw = f'{smk},{btel},{alc}'

            # Height / Weight
            height = int(r.normal(162, 8))
            weight = int(r.normal(62, 12))

            # Performance
            ecog = int(r.choice([0, 1, 2, 3, 9], p=[0.30, 0.35, 0.20, 0.05, 0.10]))

            # Class of case
            class95 = _choice(r, ['1', '2', '3'], p=[0.75, 0.15, 0.10])

            row = {
                'PK_raw':          pk,
                'SEX_raw':         sex_raw,
                'SEX_decoded':     _SEX_MAP.get(sex_raw, sex_raw),
                'AGE_raw':         age,
                'DX_YEAR_raw':     dx_yr,
                'DXDATE_raw':      dx_date,
                'VISTDATE_raw':    visit_date,
                'SMOKING_raw':     smoking_raw,
                'SEQ1_raw':        '1',
                'SEQ1_decoded':    'One primary only',
                'SEQ2_raw':        '1',
                'SEQ2_decoded':    '1st primary',
                'TCODE1_raw':      tcode1,
                'TCODE1_decoded':  tcode1,
                'LAT95_raw':       lat,
                'LAT95_decoded':   {'1': 'Right', '2': 'Left', '9': 'Not applicable', '96': 'Bilateral'}.get(lat, lat),
                'MCODE_raw':       mcode,
                'MCODE_decoded':   f'{mcode}: {_HISTOLOGY_NAMES.get(mcode, "NOS")}',
                'MCODE5_raw':      behav,
                'MCODE5_decoded':  _BEHAV_MAP.get(behav, behav),
                'MCODE6_raw':      grade,
                'MCODE6_decoded':  _GRADE_MAP.get(grade, grade),
                'MCODE6C_raw':     grade,
                'MCODE6C_decoded': _GRADE_MAP.get(grade, grade),
                'CONFER_raw':      confer,
                'CONFER_decoded':  {
                    '1': 'Positive histology', '2': 'Positive cytology',
                    '3': 'Radiology/imaging', '5': 'Clinical only', '7': 'Autopsy',
                }.get(confer, confer),
                'CSIZE95_raw':     tsize,
                'PNI_raw':         pni,
                'PNI_decoded':     {'0': 'None', '1': 'Present', '9': 'Unknown'}.get(pni, pni),
                'LVI_raw':         lvi,
                'LVI_decoded':     {'0': 'None', '1': 'Present', '9': 'Unknown'}.get(lvi, lvi),
                'LNEXAM_raw':      lnexam,
                'LN_POSITI_raw':   lnpos,
                'AJCC_raw':        ajcc,
                'AJCC_decoded':    _AJCC_MAP.get(ajcc, ajcc),
                'CT_raw':          ct,   'CT_decoded':  ct,
                'CN_raw':          cn,   'CN_decoded':  cn,
                'CM_raw':          cm,   'CM_decoded':  cm,
                'CSTG_raw':        f'Stage {cstage_raw}', 'CSTG_decoded': f'Stage {cstage_raw}',
                'PT_raw':          pt,   'PT_decoded':  pt,
                'PN_raw':          pn,   'PN_decoded':  pn,
                'PM_raw':          pm,   'PM_decoded':  pm,
                'PSTG_raw':        f'Stage {pstage}', 'PSTG_decoded': f'Stage {pstage}',
                'SUMSTG_raw':      f'Stage {pstage}', 'SUMSTG_decoded': f'Stage {pstage}',
                'OSTG_raw':        '0',  'OSTG_decoded':  'Not applicable',
                'OCSTG_raw':       '0',  'OCSTG_decoded': 'Not applicable',
                'OPSTG_raw':       '0',  'OPSTG_decoded': 'Not applicable',
                'META1_raw':       meta1, 'META1_decoded': 'Bone' if meta1 == '70' else 'None',
                'META2_raw':       '0',   'META2_decoded': 'None',
                'META3_raw':       '0',   'META3_decoded': 'None',
                'S_raw':           s_raw, 'S_decoded': 'Surgery performed' if has_surg else 'No surgery at this hospital',
                'FSDATE_raw':      fsdate,
                'PRESTYPE_raw':    '00',  'PRESTYPE_decoded': 'No outside surgery',
                'STYPE95_raw':     stype, 'STYPE95_decoded': stype,
                'MINS_raw':        _choice(r, ['0', '1', '9'], p=[0.70, 0.20, 0.10]),
                'MINS_decoded':    '',
                'MARG95_raw':      marg,
                'MARG95_decoded':  {'0': 'Margins uninvolved', '1': 'Margins involved', '2': 'Margins not evaluated', '3': 'No resection performed', '9': 'Unknown'}.get(marg, marg),
                'MARGDIS_raw':     int(r.integers(0, 30)) if marg == '0' else 999,
                'PRESLNSCO_raw':   '0',   'PRESLNSCO_decoded': 'No outside LN surgery',
                'SLNSCO95_raw':    slnsco, 'SLNSCO95_decoded': slnsco,
                'SSF1_raw':        ssf1,
                'SSF2_raw':        ssf2,
                'SSF3_raw':        ssf3,
                'SSF4_raw':        ssf4,
                'SSF5_raw':        ssf5,
                'SSF6_raw':        ssf6,
                'SSF7_raw':        ssf7,
                'SSF8_raw':        ssf8,
                'SSF9_raw':        ssf9,
                'SSF10_raw':       ssf10,
                'R_raw':           r_raw, 'R_decoded': 'Radiation performed' if has_rt else 'No radiation at this hospital',
                'RTAR_raw':        rtar,  'RTAR_decoded': str(rtar),
                'RMOD_raw':        rmod,  'RMOD_decoded': {'1': 'EBRT', '2': 'Brachytherapy', '3': 'Combination', '0': 'None', '9': 'Unknown'}.get(rmod, rmod),
                'EBRT_raw':        int(r.choice([0, 1, 4, 36, 68], p=[0.10, 0.15, 0.25, 0.30, 0.20])) if has_rt else 0,
                'HTAR_raw':        rtar if has_rt else 0,
                'HDOSE_raw':       hdose,
                'HNO_raw':         hno,
                'LTAR_raw':        0,
                'LDOSE_raw':       0,
                'LNO_raw':         0,
                'SEQRS_raw':       int(r.choice([0, 1, 2, 3], p=[0.30, 0.30, 0.25, 0.15])),
                'SEQRS_decoded':   '',
                'SEQLS_raw':       int(r.choice([0, 1, 2], p=[0.40, 0.35, 0.25])),
                'SEQLS_decoded':   '',
                'PREC_raw':        '0', 'PREC_decoded': 'No outside chemotherapy',
                'C_raw':           c_raw, 'C_decoded': 'Chemotherapy performed' if has_chemo else 'No chemotherapy at this hospital',
                'PREH_raw':        '0', 'PREH_decoded': 'No outside hormone therapy',
                'H_raw':           h_raw, 'H_decoded': 'Hormone therapy performed' if has_hormone else 'No hormone therapy at this hospital',
                'PREI_raw':        '0', 'PREI_decoded': 'No outside immunotherapy',
                'I_raw':           i_raw, 'I_decoded': 'Immunotherapy performed' if has_immuno else 'No immunotherapy at this hospital',
                'PREB_raw':        '0', 'PREB_decoded': 'No outside BMT/SCT',
                'B_raw':           '0', 'B_decoded': 'No BMT/SCT at this hospital',
                'PRETAR_raw':      '0', 'PRETAR_decoded': 'No outside targeted therapy',
                'TAR_raw':         tar_raw, 'TAR_decoded': 'Targeted therapy performed' if has_target else 'No targeted therapy at this hospital',
                'OTH_raw':         '0', 'OTH_decoded': 'None',
                'PREP_raw':        _choice(r, ['0', '1', '9'], p=[0.70, 0.25, 0.05]),
                'PREP_decoded':    '',
                'WATCHWAITING_raw': '0', 'WATCHWAITING_decoded': 'Not applicable',
                'VSTA_raw':        vsta6, 'VSTA_decoded':  _VSTA6_MAP.get(vsta6, vsta6),
                'CSTA_raw':        '0' if vsta6 == '0' else '1', 'CSTA_decoded': 'No evidence of disease' if vsta6 == '0' else 'With evidence of disease',
                'LCD_raw':         lcd6_str,
                'REDATE_raw':      '',
                'RETYPE95_raw':    '0', 'RETYPE95_decoded': 'No recurrence',
                'DIECAUSE_raw':    '0' if vsta6 == '0' else ('1' if vsta6 == '1' else '2'),
                'DIECAUSE_decoded': 'Not dead' if vsta6 == '0' else ('Cancer' if vsta6 == '1' else 'Non-cancer'),
                'VSTA6_raw':       vsta6, 'VSTA6_decoded': _VSTA6_MAP.get(vsta6, vsta6),
                'LCD6_raw':        lcd6_str,
                'SURVY6_raw':      surv_yr,
                'REDATE6_raw':     '',
                'RETYPE6_raw':     '0', 'RETYPE6_decoded': 'No recurrence',
                'DIECAUSE6_raw':   '0' if vsta6 == '0' else ('1' if vsta6 == '1' else '2'),
                'DIECAUSE6_decoded': 'Not dead' if vsta6 == '0' else ('Cancer' if vsta6 == '1' else 'Non-cancer'),
                'HEIGHT_raw':      height,
                'WEIGHT_raw':      weight,
                'KPSECOG_raw':     ecog, 'KPSECOG_decoded': f'ECOG {ecog}',
                'CLASS95_raw':     class95, 'CLASS95_decoded': _CLASS_MAP.get(class95, class95),
                'CLASSOFDIAG_raw': '1', 'CLASSOFDIAG_decoded': 'Diagnosed at this hospital',
                'CLASSOFTREAT_raw': class95, 'CLASSOFTREAT_decoded': _CLASS_MAP.get(class95, class95),
            }
            rows.append(row)

        self._df = pd.DataFrame(rows)
        return self._df

    def to_excel(self, path: str = 'synthetic_tcr.xlsx') -> Path:
        """Save to Excel in TCR format (sheet name: All_Fields_Decoded).

        The output is immediately loadable by TCRDecoder.
        """
        if self._df is None:
            self.generate()
        out = Path(path)
        with pd.ExcelWriter(str(out), engine='openpyxl') as writer:
            self._df.to_excel(writer, sheet_name='All_Fields_Decoded', index=False)
        print(f'Saved {len(self._df)} synthetic {self.cancer_group} patients → {out.name}')
        return out

    def summary(self) -> str:
        """Print a brief summary of the generated dataset."""
        if self._df is None:
            return 'Not generated yet. Call generate() first.'
        lines = [
            f'=== Synthetic TCR Dataset Summary ===',
            f'Cancer group : {self.cancer_group}',
            f'N patients   : {len(self._df)}',
            f'Seed         : {self.seed}',
            f'DX years     : {self.dx_start}–{self.dx_end}',
        ]
        # Stage distribution
        stage_col = 'PSTG_raw'
        if stage_col in self._df.columns:
            dist = self._df[stage_col].value_counts().head(6)
            lines.append(f'\nStage distribution:')
            for s, c in dist.items():
                lines.append(f'  {s:15s}: {c}')
        # SSF1 summary (cancer-specific)
        ssf1_col = 'SSF1_raw'
        if ssf1_col in self._df.columns:
            vals = pd.to_numeric(self._df[ssf1_col], errors='coerce')
            pos  = ((vals >= 1) & (vals <= 100)).sum()
            neg  = (vals == 120).sum()
            unk  = (vals == 999).sum()
            lines.append(f'\nSSF1 summary (n={len(vals.dropna())}):')
            if self.cancer_group == 'breast':
                lines.append(f'  ER Positive (1-100%): {pos}')
                lines.append(f'  ER Negative (120)   : {neg}')
                lines.append(f'  Unknown (999)       : {unk}')
            else:
                lines.append(f'  Range: {int(vals.min())}–{int(vals.max())}')
        return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(
        description='Generate synthetic TCR data for testing tcr_decoder.')
    parser.add_argument('--cancer', default='breast',
                        choices=SyntheticTCRGenerator.SUPPORTED,
                        help='Cancer group (default: breast)')
    parser.add_argument('--n', type=int, default=100,
                        help='Number of synthetic patients (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--out', default='',
                        help='Output Excel path (default: synthetic_{cancer}.xlsx)')
    parser.add_argument('--decode', action='store_true',
                        help='Also run TCRDecoder on the generated data')
    args = parser.parse_args()

    out_path = args.out or f'synthetic_{args.cancer}.xlsx'
    gen = SyntheticTCRGenerator(cancer_group=args.cancer, n=args.n, seed=args.seed)
    gen.generate()
    print(gen.summary())
    gen.to_excel(out_path)

    if args.decode:
        from tcr_decoder import TCRDecoder
        clean_path = out_path.replace('.xlsx', '_clean.xlsx')
        print(f'\nDecoding → {clean_path}')
        TCRDecoder(out_path).run(clean_path)
