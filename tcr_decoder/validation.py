# -*- coding: utf-8 -*-
"""Bidirectional validation dataset builder.

Produces the evidence that every TCR code -- and every combination of
codes across the ten SSF fields -- carries one unambiguous clinical meaning
and can be turned back into the exact original case record.

Two levels of evidence:

1. FIELD LEVEL (`build_field_coverage`) -- every single legal code of every
   breast SSF field, with the codebook's own Chinese definition, the English
   clinical label this package decodes it to, and the code that comes back
   out of the encoder. 100% of the official 編碼範圍, nothing sampled.

2. CASE LEVEL (`build_case_dataset` / `validate_cases`) -- whole synthetic
   patient records whose SSF values are laid out so that every PAIR of
   values across any two fields appears in at least one case (all-pairs /
   pairwise coverage). Each case is written as a real registry row, run
   through the full TCRDecoder pipeline, re-encoded with TCREncoder, and
   compared field by field against the original raw codes.

`export_validation_workbook()` writes the whole thing to a multi-sheet Excel
file that a human reviewer (or an auditor) can read without running Python.

Why pairwise and not every possible combination: the ten breast SSF fields
have >10^20 combinations, so exhaustive case coverage is impossible. Every
individual code is covered exhaustively at the field level, and the case
level covers every two-field interaction -- the standard combinatorial
testing result is that this catches the overwhelming majority of
interaction defects while staying a few hundred cases.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from tcr_decoder.code_ranges import CODE_RANGES, SUPPORTED_GROUPS
from tcr_decoder.ssf_registry import get_ssf_profile

SSF_KEYS: Tuple[str, ...] = tuple(f'SSF{i}' for i in range(1, 11))


# ─────────────────────────────────────────────────────────────────────────────
# Chinese definitions, transcribed from the Cancer-SSF-Manual (breast)
# ─────────────────────────────────────────────────────────────────────────────

_ZH_RECEPTOR = {'SSF1': ('ER', '動情激素接受體'), 'SSF2': ('PR', '黃體激素接受體')}

_ZH_ER_PR_SPECIAL = {
    '110': '陽性，但比例不明；或 Allred score 僅描述 3-8 分，未明示 intensity 及 positive cell',
    '111': '僅有前導性治療後的數值且為陽性',
    '120': '陰性：反應比例 <1%（不論染色強度）、比例未明示、僅表示為 (-)，'
           '或 Allred score 僅描述 1-2 分',
    '121': '僅有前導性治療後的數值且為陰性',
    '888': '前導性治療後的數值由陰性轉為陽性',
    '988': '不適用：以 Oncotype 進行檢測、Phyllodes tumor (M-9020)、'
           'Sarcoma (M-8800-8936, 8940-9136, 9141-9582)',
    '999': '不清楚檢測是否執行／病歷未記載／未檢驗',
}
_ZH_INTENSITY = {'S': '強染', 'I': '中染', 'W': '弱染'}

_ZH_SSF3 = {
    '010': 'Clinical complete response (cCR)，臨床完全緩解',
    '011': '病理資訊顯示 Pathological complete response (pCR)：乳房組織與淋巴結'
           '皆無殘餘侵襲性癌（N0(i+) 不視為 pCR）',
    '020': 'Partial response (PR)；Moderate response，部分緩解',
    '030': 'Stable disease (SD)；Minimal response，疾病穩定',
    '040': 'Progressive disease (PD)；Poor response；No response，疾病進展',
    '988': '不適用：沒有執行前導性療法，或未執行手術',
    '990': '治療後有縮小反應，但反應程度未進一步說明',
    '999': '有進行前導性治療但療效不明／病歷未記載或不詳',
}

_ZH_SSF4_SPECIAL = {
    '000': '沒有進行哨兵淋巴結手術',
    '988': '不適用（例如前導性治療後手術才做哨兵淋巴結檢查）',
    '996': '病歷記載有哨兵淋巴結檢查但數目不詳；或確實取樣但病理報告未發現淋巴結組織',
    '999': '不清楚檢測是否執行／病歷未記載',
}
_ZH_SSF5_SPECIAL = {
    '000': '沒有哨兵淋巴結侵犯；或淋巴結僅有 isolated tumor cell (ITCs) 侵犯；'
           '或取樣未發現淋巴結組織且該組織未被侵犯',
    '988': '不適用',
    '996': '病歷記載哨兵淋巴結有被侵犯但數目不詳；或取樣未發現淋巴結組織但該組織有被侵犯',
    '999': '不清楚哨兵淋巴結是否有被侵犯／病歷未記載',
}

_ZH_SSF6 = {
    '110': 'Low Grade，BR grade 1',
    '120': 'Medium Grade，BR grade 2',
    '130': 'High Grade，BR grade 3',
    '988': '不適用：Sarcoma (M-8800-8936, 8940-9136, 9141-9582)、'
           'Phyllodes tumor (M-9020)、原位癌',
    '999': '既無 BR grade 亦無 BR score／不詳／病歷未記載',
}

_ZH_HER2_IHC = {
    '0': 'IHC 0 (negative or score 0)',
    '1': 'IHC 1+ (score 1+)',
    '2': 'IHC 2+ (equivocal or score 2+)',
    '3': 'IHC 3+ (positive or score 3+)',
    '9': 'IHC 不詳',
}
_ZH_HER2_ISH = {
    '0': 'ISH negative / not amplified',
    '1': 'ISH positive / amplified',
    '2': 'ISH equivocal',
}
_ZH_HER2_SPECIAL = {
    '000': 'IHC 0 (negative or score 0) 且有描述染色比例為 0%',
    '004': 'IHC 0 (negative or score 0) 且有描述染色比例 >0%、≦10%（Ultralow）',
    '100': 'IHC 0 (negative or score 0) 且未描述染色比例',
    '101': 'IHC 1+ (score 1+)',
    '102': 'IHC 2+ (equivocal or score 2+)',
    '103': 'IHC 3+ (positive or score 3+)',
    '200': '僅 100-107 診斷年適用：CISH negative / not amplified',
    '201': '僅 100-107 診斷年適用：CISH positive / amplified',
    '202': '僅 100-107 診斷年適用：CISH equivocal',
    '300': '僅 100-107 診斷年適用：FISH negative / not amplified',
    '301': '僅 100-107 診斷年適用：FISH positive / amplified',
    '302': '僅 100-107 診斷年適用：FISH equivocal',
    '400': '僅 100-107 診斷年適用：其它檢驗，HER2 陰性',
    '401': '僅 100-107 診斷年適用：其它檢驗，HER2 陽性',
    '402': '僅 100-107 診斷年適用：其它檢驗，HER2 equivocal',
    '888': '前導性治療後 HER2 檢驗值由陰性轉為陽性；或無治療前數值但治療後為陽性',
    '900': 'HER2 陰性，其它檢驗方式或外院檢驗方式不詳',
    '901': 'HER2 陽性，其它檢驗方式或外院檢驗方式不詳',
    '902': 'HER2 equivocal，其它檢驗方式或外院檢驗方式不詳',
    '988': '不適用：Phyllodes tumor (M-9020)、Sarcoma',
    '999': '不清楚檢測是否執行／病歷並未記載／沒有檢測',
}
_ZH_HER2_600 = {
    '600': 'IHC 0（染色比例 0%）且 ISH negative / not amplified（114 診斷年起）',
    '601': 'IHC 0（染色比例 0%）且 ISH positive / amplified（114 診斷年起）',
    '602': 'IHC 0（染色比例 0%）且 ISH equivocal（114 診斷年起）',
    '640': 'IHC 0（0%<染色比例≦10%，Ultralow）且 ISH negative / not amplified（114 診斷年起）',
    '641': 'IHC 0（0%<染色比例≦10%，Ultralow）且 ISH positive / amplified（114 診斷年起）',
    '642': 'IHC 0（0%<染色比例≦10%，Ultralow）且 ISH equivocal（114 診斷年起）',
}

_ZH_SSF8 = {
    '000': '沒有 Paget disease',
    '010': '有 Paget disease',
    '988': '不適用：病理標本不含乳頭、乳暈之檢查',
    '999': '病歷未記載或不詳',
}

_ZH_SSF9 = {
    '000': '腫瘤內並無淋巴管或血管的侵犯',
    '010': '腫瘤內有淋巴管或血管的侵犯',
    '988': '不適用',
    '990': '局部切除病理未描述 LVI，後續切除標本無殘餘腫瘤；或前導性治療後'
           '切除標本無殘餘腫瘤或敘述無淋巴管、血管侵犯',
    '999': '病歷未記載或不詳',
}

_ZH_SSF10 = {
    '988': '不適用：Phyllodes tumor (M-9020)、Sarcoma、'
           '外院已切除大部分腫瘤且無外院資訊',
    '998': '有檢驗，但百分比不明',
    '999': '不清楚檢測是否執行／病歷未記載／沒有檢驗',
}


def _zh_breast(ssf_key: str, code: str) -> str:
    """The code book's own Chinese definition of one breast SSF code."""
    code = str(code).strip()

    if ssf_key in _ZH_RECEPTOR:
        abbr, full = _ZH_RECEPTOR[ssf_key]
        if code in _ZH_ER_PR_SPECIAL:
            return _ZH_ER_PR_SPECIAL[code]
        if code[:1].upper() in _ZH_INTENSITY and code[1:].isdigit():
            pct = int(code[1:]) or 100          # '00' 代表 100%
            return f'{_ZH_INTENSITY[code[0].upper()]}，{abbr} 反應比例 {pct}%'
        if code.isdigit():
            n = int(code)
            if n == 0:
                return f'{abbr} 0%'
            if 1 <= n <= 100:
                return f'{abbr} 反應比例 {n}%，但表現(染色)強度未明示'
        return f'{full}：未定義代碼 {code}'

    if ssf_key == 'SSF3':
        return _ZH_SSF3.get(code, f'未定義代碼 {code}')

    if ssf_key in ('SSF4', 'SSF5'):
        special = _ZH_SSF4_SPECIAL if ssf_key == 'SSF4' else _ZH_SSF5_SPECIAL
        if code in special:
            return special[code]
        if code.isdigit() and 1 <= int(code) <= 89:
            what = '檢查' if ssf_key == 'SSF4' else '侵犯'
            return f'依實際{what}之哨兵淋巴結數目：{int(code)} 顆'
        return f'未定義代碼 {code}'

    if ssf_key == 'SSF6':
        if code in _ZH_SSF6:
            return _ZH_SSF6[code]
        if code.isdigit() and code.endswith('0') and 3 <= int(code) // 10 <= 9:
            return f'BR score {int(code) // 10} 分'
        return f'未定義代碼 {code}'

    if ssf_key == 'SSF7':
        if code in _ZH_HER2_SPECIAL:
            return _ZH_HER2_SPECIAL[code]
        if code in _ZH_HER2_600:
            return _ZH_HER2_600[code]
        if len(code) == 3 and code[0] == '5':
            ihc = _ZH_HER2_IHC.get(code[1], f'IHC 碼 {code[1]}')
            ish = _ZH_HER2_ISH.get(code[2], f'ISH 碼 {code[2]}')
            return f'{ihc} 且 {ish}'
        return f'未定義代碼 {code}'

    if ssf_key == 'SSF8':
        return _ZH_SSF8.get(code, f'未定義代碼 {code}')

    if ssf_key == 'SSF9':
        return _ZH_SSF9.get(code, f'未定義代碼 {code}')

    if ssf_key == 'SSF10':
        if code in _ZH_SSF10:
            return _ZH_SSF10[code]
        if code[:1].upper() == 'A' and code[1:].isdigit():
            return f'Ki-67 檢驗值 0.{int(code[1:])}%（小於 1%、大於 0%，無條件捨去至小數第一位）'
        if code.isdigit():
            n = int(code)
            if n == 0:
                return 'Ki-67 檢驗值 0%'
            if 1 <= n <= 100:
                return f'Ki-67 檢驗值 {n}%'
        return f'未定義代碼 {code}'

    return f'未定義欄位 {ssf_key}'


# ─────────────────────────────────────────────────────────────────────────────
# Chinese definitions, transcribed from the Cancer-SSF-Manual (prostate)
# ─────────────────────────────────────────────────────────────────────────────

_ZH_PSA_TIERS = {
    '980': 'PSA 98.0 ng/ml（僅 100-104 診斷年個案適用）',
    '981': 'PSA 98.0-199.9 ng/ml', '982': 'PSA 200.0-299.9 ng/ml',
    '983': 'PSA 300.0-399.9 ng/ml', '984': 'PSA 400.0-499.9 ng/ml',
    '985': 'PSA 500.0-599.9 ng/ml', '986': 'PSA 600.0-699.9 ng/ml',
    '987': 'PSA 700.0-799.9 ng/ml', '989': 'PSA 800.0-899.9 ng/ml',
    '990': 'PSA 900.0-999.9 ng/ml', '991': 'PSA 1000-1999 ng/ml',
    '992': 'PSA 2000-2999 ng/ml', '993': 'PSA 3000-3999 ng/ml',
    '994': 'PSA 4000-4999 ng/ml', '995': 'PSA 5000-5999 ng/ml',
    '996': 'PSA 6000-6999 ng/ml', '997': 'PSA 7000-7999 ng/ml',
    '998': 'PSA ≧8000 ng/ml',
    '988': '不適用，已於外院開始首次療程且無外院檢驗值',
    '999': '病歷未記載或不詳／沒有檢驗',
}

_ZH_GLEASON_NA = {
    'biopsy': ('不適用：未執行細針切片或經尿道攝護腺刮除術；或 PIN III',
               '執行細針切片或經尿道攝護腺刮除術，但病理報告未記載級數／不詳'),
    'prostatectomy': ('不適用：未執行攝護腺全切除或大體解剖；PIN III 或 '
                      'No residual tumor；手術前已執行前導性治療',
                      '執行攝護腺全切除或大體解剖，但病理報告未記載級數／不詳'),
}

_ZH_PROSTATE_CORES_EXAMINED = {
    '988': '不適用；未執行切片（例如僅執行 TURP）',
    '999': '執行切片，但切片條數檢查數未知／不詳',
}
_ZH_PROSTATE_CORES_POSITIVE = {
    '000': '切片取得所有切片條數均為陰性',
    '988': '不適用：未執行切片；或 PIN III 個案',
    '998': '切片後確定為陽性，但不清楚陽性數目是否正確',
    '999': '切片呈現陽性，但切片條數陽性數未知／不詳',
}

_ZH_PROSTATE_T_STAGING = {
    '000': '未執行肛門指檢與影像檢查',
    '010': '僅執行肛門指檢，未進行影像檢查',
    '020': '未進行肛門指檢，有執行 TRUS 或 erMRI 影像檢查',
    '030': '有執行肛門指檢與 TRUS 或 erMRI 影像檢查',
    '040': '未進行肛門指檢與 TRUS 或 erMRI，但有執行 MRI 影像檢查',
    '050': '有執行肛門指檢與 MRI 影像檢查，但未進行 TRUS 或 erMRI',
    '988': '不適用',
    '999': '病歷未記載或不詳',
}


def _zh_prostate(ssf_key: str, code: str) -> str:
    """The code book's own Chinese definition of one prostate SSF code."""
    code = str(code).strip()

    if ssf_key == 'SSF1':
        if code in _ZH_PSA_TIERS:
            return _ZH_PSA_TIERS[code]
        if code.isdigit():
            n = int(code)
            if n == 1:
                return 'PSA ≦0.1 ng/ml'
            if 2 <= n <= 979:
                return f'PSA {n / 10:.1f} ng/ml'
        return f'未定義代碼 {code}'

    if ssf_key in ('SSF2', 'SSF4'):
        specimen = 'biopsy' if ssf_key == 'SSF2' else 'prostatectomy'
        na_text, unknown_text = _ZH_GLEASON_NA[specimen]
        if code == '988':
            return na_text
        if code == '999':
            return unknown_text
        if code == '099':
            return '主要級數不明，次要級數不明'
        if code.isdigit():
            primary, secondary = int(code) // 10, int(code) % 10
            if 1 <= primary <= 5 and 1 <= secondary <= 5:
                return f'主要級數{primary}，次要級數{secondary}'
            if 1 <= primary <= 5 and secondary == 9:
                return f'主要級數{primary}，次要級數不明'
        return f'未定義代碼 {code}'

    if ssf_key in ('SSF3', 'SSF5'):
        specimen = 'biopsy' if ssf_key == 'SSF3' else 'prostatectomy'
        na_text, unknown_text = _ZH_GLEASON_NA[specimen]
        if code == '988':
            return na_text
        if code == '999':
            return unknown_text
        if code.isdigit() and 2 <= int(code) <= 10:
            return f"Gleason's Score {int(code)} 分"
        return f'未定義代碼 {code}'

    if ssf_key == 'SSF6':
        if code in _ZH_PROSTATE_CORES_EXAMINED:
            return _ZH_PROSTATE_CORES_EXAMINED[code]
        if code.isdigit() and 1 <= int(code) <= 100:
            return f'檢查 {int(code)} 切片條數'
        return f'未定義代碼 {code}'

    if ssf_key == 'SSF7':
        if code in _ZH_PROSTATE_CORES_POSITIVE:
            return _ZH_PROSTATE_CORES_POSITIVE[code]
        if code.isdigit() and 1 <= int(code) <= 100:
            return f'{int(code)} 切片條數呈現陽性'
        return f'未定義代碼 {code}'

    if ssf_key == 'SSF8':
        return _ZH_PROSTATE_T_STAGING.get(code, f'未定義代碼 {code}')

    if ssf_key in ('SSF9', 'SSF10'):
        if code == '988':
            return '攝護腺癌未收錄此欄位，一律編碼 988（手冊 p.1 收案條件）'
        return f'未定義代碼 {code}'

    return f'未定義欄位 {ssf_key}'


# ─────────────────────────────────────────────────────────────────────────────
# Chinese definitions, transcribed from the Cancer-SSF-Manual (子宮體癌)
# ─────────────────────────────────────────────────────────────────────────────

_ZH_UTERUS_FIGO = {
    '001': 'Non-squamous or non-morular solid growth pattern ≦5%（Grade 1）',
    '002': 'Non-squamous or non-morular solid growth pattern = 6-50%（Grade 2）',
    '003': 'Non-squamous or non-morular solid growth pattern > 50%（Grade 3）',
    '987': '碼冊 p.157 編碼範圍列出 987，但 p.163 碼表未定義此碼（需向登記中心確認）',
    '988': '不適用',
    '999': '不詳／病歷未記載',
}
_ZH_POLE = {
    '010': 'POLE 基因檢驗結果，有突變',
    '020': 'POLE 基因檢驗結果，無突變',
    '030': '有進行 POLE 基因檢驗，但結果無法判讀',
    '999': '病歷未記載或不詳／沒有檢測',
}
_ZH_MSI_UTERUS = {
    '000': 'MSI stable（MSS）；且／或 MMR intact（pMMR，無 MMR 蛋白核表現缺失）',
    '010': 'MSI unstable low（MSI-L）',
    '020': 'MSI unstable high（MSI-H）；且／或 一個以上 MMR 蛋白核表現缺失（dMMR）',
    '988': '不適用：GIST、NETs 及 High grade dysplasia；或於外院檢查且無外院結果',
    '999': '病歷未記載或不詳／沒有檢測／MSI-I（indeterminate）／MSI-equivocal',
}
_ZH_P53 = {
    '010': 'p53 蛋白表現異常',
    '020': 'p53 蛋白表現正常',
    '030': '有進行 p53 蛋白檢測，但結果無法判讀',
    '999': '僅接受 TP53 基因檢測但無 p53 蛋白檢測／病歷未記載或不詳／沒有檢測',
}


def _zh_endometrium(ssf_key: str, code: str) -> str:
    """The code book's own Chinese definition of one uterine-corpus SSF code."""
    code = str(code).strip()
    if ssf_key in ('SSF1', 'SSF2'):
        # Identical code scheme to breast ER/PR (p.153/155).
        return _zh_breast(ssf_key, code)
    if ssf_key == 'SSF3':
        return _ZH_UTERUS_FIGO.get(code, f'未定義代碼 {code}')
    if ssf_key == 'SSF4':
        return _ZH_POLE.get(code, f'未定義代碼 {code}')
    if ssf_key == 'SSF5':
        return _ZH_MSI_UTERUS.get(code, f'未定義代碼 {code}')
    if ssf_key == 'SSF6':
        return _ZH_P53.get(code, f'未定義代碼 {code}')
    if ssf_key in ('SSF7', 'SSF8', 'SSF9', 'SSF10'):
        if code == '988':
            return '子宮體癌未收錄此欄位，一律編碼 988（手冊 p.1 收案條件）'
        return f'未定義代碼 {code}'
    return f'未定義欄位 {ssf_key}'


def _zh_thyroid(ssf_key: str, code: str) -> str:
    """Thyroid collects no SSFs (manual p.1): 988 is the only legal code."""
    if str(code).strip() == '988':
        return '甲狀腺非 SSF 收錄部位，SSF1-SSF20 一律編碼 988（手冊 p.1 收案條件）'
    return f'未定義代碼 {code}'


# ─────────────────────────────────────────────────────────────────────────────
# Chinese definitions: 子宮頸 / 胃 / 肝 / 肺 / 結直腸 / 頭頸部
# ─────────────────────────────────────────────────────────────────────────────

def _zh_lab_x10(analyte: str, code: str, na_text: str, unknown_text: str) -> str:
    """001 = ≦0.1, 002-986 = 值x10, 987 = ≧98.7, 988 = 不適用, 999 = 不詳."""
    if code == '988':
        return na_text
    if code == '999':
        return unknown_text
    if code.isdigit():
        n = int(code)
        if n == 1:
            return f'{analyte} ≦0.1 ng/ml'
        if 2 <= n <= 986:
            return f'{analyte} {n / 10:.1f} ng/ml'
        if n == 987:
            return f'{analyte} ≧98.7 ng/ml'
    return f'未定義代碼 {code}'


def _zh_vs_normal(analyte: str, code: str, na_text: str) -> str:
    table = {
        '010': f'個案的{analyte}檢驗值陽性／大於正常值',
        '020': f'個案的{analyte}檢驗值陰性／正常值；正常值以內',
        '030': f'個案的{analyte}檢驗值為臨界值；不確定陽性或陰性',
        '988': na_text,
        '999': f'病歷未記載或不詳／未進行{analyte}檢驗',
    }
    return table.get(code, f'未定義代碼 {code}')


_ZH_CEA_NA = '不適用：GIST 及 NETs；或已於外院開始首次療程且無外院檢驗值'
_ZH_SCC_NA = ('不適用：子宮頸原位癌；治療前切片非鱗狀癌或鱗狀腺侵襲癌；'
              '或已於外院開始首次療程且無外院檢驗值')


def _zh_cervix(ssf_key: str, code: str) -> str:
    code = str(code).strip()
    if ssf_key == 'SSF1':
        return _zh_lab_x10('血清 SCC-Ag', code, _ZH_SCC_NA,
                           '病歷未記載或不詳／未進行血清 SCC-Ag 檢驗')
    if ssf_key == 'SSF2':
        return _zh_vs_normal('血清 SCC-Ag', code, _ZH_SCC_NA)
    if code == '988':
        return '子宮頸癌僅收錄 SSF1-SSF2，其餘欄位一律編碼 988（手冊 p.1）'
    return f'未定義代碼 {code}'


_ZH_H_PYLORI = {
    '000': '檢驗結果為陰性',
    '001': '病理組織 (histology) 檢驗結果為陽性',
    '002': '細菌培養檢驗結果為陽性',
    '003': '快速尿素酶試驗 (rapid urease test) 結果為陽性',
    '004': '血清抗體檢驗結果為陽性',
    '005': '尿素呼氣試驗 (urea breath test) 檢驗結果為陽性',
    '006': '幽門桿菌糞便抗原檢查法 (HpSA Test) 檢驗結果為陽性',
    '007': '聚合酶連鎖反應 (PCR) 結果為陽性',
    '008': '檢驗結果為陽性，但方法不詳',
    '010': '任二項或超過二項之檢驗結果為陽性',
    '988': '不適用：GIST 及 NETs',
    '999': '病歷未記載或不詳／沒有檢驗',
}

_ZH_LVI = {
    '000': '腫瘤內並無淋巴管或血管的侵犯',
    '010': '腫瘤內有淋巴管或血管的侵犯',
    '988': '不適用',
    '990': '無殘餘腫瘤，無法評估淋巴管或血管侵犯',
    '999': '病歷未記載或不詳',
}


def _zh_depth(code: str, na_text: str) -> str:
    specials = {
        '980': '腫瘤之深度 ≧98mm',
        '988': na_text,
        '998': '沒有手術標本（未接受手術個案）；或手術前已接受放射治療或全身性治療',
        '999': '病歷的病理報告中缺乏深度的數據',
    }
    if code in specials:
        return specials[code]
    if code.isdigit():
        n = int(code)
        if n == 0:
            return '實際腫瘤深度為 0 mm（<0.1mm）'
        if 1 <= n <= 979:
            return f'實際腫瘤深度 {n / 10:.1f} mm（以 0.1mm 為單位）'
    return f'未定義代碼 {code}'


def _zh_stomach(ssf_key: str, code: str) -> str:
    code = str(code).strip()
    if ssf_key == 'SSF1':
        return _zh_lab_x10('癌胚抗原 CEA', code, _ZH_CEA_NA,
                           '病歷未記載或不詳／未進行癌胚抗原 CEA 檢驗')
    if ssf_key == 'SSF2':
        return _zh_vs_normal('癌胚抗原 CEA', code, _ZH_CEA_NA)
    if ssf_key == 'SSF3':
        return _ZH_H_PYLORI.get(code, f'未定義代碼 {code}')
    if ssf_key == 'SSF4':
        return _zh_depth(code, '不適用：有接受手術且非病理 T1 個案；GIST 及 NETs')
    if ssf_key == 'SSF5':
        return _ZH_LVI.get(code, f'未定義代碼 {code}')
    if code == '988':
        return '胃癌僅收錄 SSF1-SSF5，其餘欄位一律編碼 988（手冊 p.1）'
    return f'未定義代碼 {code}'


_ZH_ISHAK = {
    '000': 'Ishak F0：無纖維化 (No fibrosis)',
    '001': 'Ishak F1：部分匯管區纖維化擴張，有或無短纖維隔',
    '002': 'Ishak F2：多數匯管區纖維化擴張，有或無短纖維隔',
    '003': 'Ishak F3：多數匯管區纖維化擴張，偶見 P-P 橋接',
    '004': 'Ishak F4：匯管區纖維化擴張並有明顯 P-P 及 P-C 橋接',
    '005': 'Ishak F5：明顯橋接並偶見結節（不完全肝硬化）',
    '006': 'Ishak F6：肝硬化 (probable or definite)',
    '007': '無病理報告或病理未描述纖維化程度，但超音波／電腦斷層／核磁共振報告'
           '有提及肝硬化',
    '008': '無病理報告或病理未描述纖維化程度，且超音波／電腦斷層／核磁共振報告'
           '無提及肝硬化',
    '988': '不適用',
    '999': '病歷未記載或不詳',
}

_ZH_CHILD_PUGH_CLASS = {'1': 'A', '2': 'B', '3': 'C'}

_ZH_HEPATITIS = {
    '000': '沒有檢驗，亦無帶原／感染史',
    '001': '沒有檢驗，但病歷記載曾有帶原／感染史',
    '010': '檢驗結果為陰性，且無帶原／感染史',
    '011': '檢驗結果為陰性，但病歷記載曾有帶原／感染史',
    '020': '檢驗結果為陽性',
    '999': '病歷未記載或不詳',
}


def _zh_liver(ssf_key: str, code: str) -> str:
    code = str(code).strip()
    if ssf_key == 'SSF1':
        specials = {
            '988': '不適用，於外院已接受首次療程且無外院檢驗值',
            '991': 'AFP 超過儀器最大稀釋值，且儀器最大稀釋值介於 400-6000 ng/ml',
            '992': 'AFP 超過儀器最大稀釋值，且儀器最大稀釋值介於 6001-9879 ng/ml',
            '993': 'AFP 實際數值 ≧9880 ng/ml；或超過儀器最大稀釋值且該值 ≧9880 ng/ml',
            '999': '不詳／首次治療前無檢驗 AFP',
        }
        if code in specials:
            return specials[code]
        if code[:1].upper() == 'A' and code[1:].isdigit():
            return f'AFP 實際數值 {int(code[1:])} ng/ml（<1 或 1-99，2021 診斷年以後適用）'
        if code.isdigit():
            n = int(code)
            if 0 <= n <= 9:
                return f'AFP 實際數值 {n * 10}-{n * 10 + 9} ng/ml（1-99 除以 10，2020 年以前適用）'
            if 10 <= n <= 99:
                return f'AFP 實際數值 {n * 10}-{n * 10 + 9} ng/ml（100-999 去除個位數）'
            if 100 <= n <= 987:
                return f'AFP 實際數值 {n * 10}-{n * 10 + 9} ng/ml（1000-9879 除以 10）'
        return f'未定義代碼 {code}'
    if ssf_key == 'SSF2':
        return _ZH_ISHAK.get(code, f'未定義代碼 {code}')
    if ssf_key == 'SSF3':
        if code == '999':
            return 'Child-Pugh 分級與分數皆不詳'
        if len(code) == 3 and code[0] in _ZH_CHILD_PUGH_CLASS and code[1:].isdigit():
            grade = _ZH_CHILD_PUGH_CLASS[code[0]]
            score = code[1:]
            if score == '99':
                return f'Child-Pugh 分級 {grade}，病歷僅有分級未提及分數'
            return f'Child-Pugh 分級 {grade}，Child-Pugh Score {int(score)} 分'
        return f'未定義代碼 {code}'
    if ssf_key in ('SSF4', 'SSF5'):
        analyte = '肌酸(酐)' if ssf_key == 'SSF4' else '總膽紅素'
        if code == '988':
            return '不適用，於外院已接受首次療程且無外院檢驗值'
        if code == '999':
            return f'{analyte}檢驗值不詳／未檢驗'
        if code.isdigit():
            n = int(code)
            if n == 1:
                return f'{analyte} ≦0.1 mg/dL'
            if 2 <= n <= 986:
                return f'{analyte} {n / 10:.1f} mg/dL'
            if n == 987:
                return f'{analyte} ≧98.7 mg/dL'
        return f'未定義代碼 {code}'
    if ssf_key == 'SSF6':
        if code == '988':
            return '不適用，於外院已接受首次療程且無外院檢驗值'
        if code == '997':
            return 'INR >6.0 或超過儀器最大值'
        if code == '999':
            return 'INR 不詳／未檢驗'
        if code.isdigit() and 1 <= int(code) <= 60:
            return f'凝血酶原時間國際正常比值 (INR) {int(code) / 10:.1f}'
        return f'未定義代碼 {code}'
    if ssf_key in ('SSF7', 'SSF8'):
        virus = 'B 型肝炎表面抗原 (HBsAg)' if ssf_key == 'SSF7' else 'C 型肝炎抗體 (Anti-HCV)'
        text = _ZH_HEPATITIS.get(code)
        return f'{virus}：{text}' if text else f'未定義代碼 {code}'
    if code == '988':
        return '肝癌僅收錄 SSF1-SSF8，其餘欄位一律編碼 988（手冊 p.1）'
    return f'未定義代碼 {code}'


_ZH_LUNG = {
    'SSF1': {
        '000': '沒有出現另外肺腫瘤結節（病灶）；或原位癌',
        '010': '非主病灶的另外肺腫瘤結節，同側同葉',
        '020': '非主病灶的另外肺腫瘤結節，同側不同葉',
        '030': '非主病灶的另外肺腫瘤結節，同側同葉和同側不同葉 (010+020)',
        '040': '非主病灶的另外肺腫瘤結節，同側，但同葉／不同葉不明',
        '999': '病歷未記載或不詳',
    },
    'SSF2': {
        '000': '沒有證據顯示臟層肋膜被波及；完全沒有侵犯彈性層 (PL0)',
        '010': '腫瘤侵犯到臟層肋膜的彈性層但未超過臟層肋膜 (PL1)',
        '020': '腫瘤侵犯到臟層肋膜的表面 (PL2)',
        '030': '腫瘤侵犯到壁層肋膜 (PL3)',
        '040': '腫瘤侵犯到肋膜，但沒有詳細敘述',
        '988': '不適用，未針對原發部位進行手術',
        '999': '病歷未記載或不詳',
    },
    'SSF3': {
        '000': 'ECOG=0；KPS=100',
        '001': 'ECOG=1；KPS=80/90',
        '002': 'ECOG=2；KPS=60/70',
        '003': 'ECOG=3；KPS=40/50',
        '004': 'ECOG=4；KPS=10/20/30',
        '005': 'ECOG=5；KPS=0',
        '988': '不適用',
        '998': '沒有評估',
        '999': '病歷未記載或不詳',
    },
    'SSF4': {
        '000': '影像學及／或細胞學檢查沒有發現；或醫師判定非惡性肋膜積水',
        '011': '影像有肋膜積水、未執行細胞學檢查，醫師判定為惡性',
        '012': '影像有肋膜積水、細胞學陰性或 atypical，醫師判定為惡性',
        '013': '細胞學檢查證實為惡性肋膜積水',
        '014': '影像有肋膜積水、未執行細胞學檢查，醫師未判定為惡性',
        '015': '影像有肋膜積水、細胞學陰性或 atypical，醫師未判定為惡性',
        '988': '不適用，M0 的個案',
        '999': '病歷未記載或不詳',
    },
    'SSF7': {
        '010': 'ALK 基因檢驗結果，有突變',
        '020': 'ALK 基因檢驗結果，無突變',
        '030': '有進行 ALK 基因檢驗，但結果無法判讀',
        '999': '不知道是否有檢驗／病歷未呈現報告／沒有做檢測',
    },
    'SSF8': {
        '000': '腫瘤型態未包含微乳突型、實體型和篩狀型',
        '001': '腫瘤型態僅包含微乳突型 (micropapillary)',
        '002': '腫瘤型態僅包含實體型 (solid)',
        '003': '腫瘤型態包含微乳突型和實體型',
        '004': '腫瘤型態僅包含篩狀型 (cribriform)／複合腺型',
        '005': '腫瘤型態包含微乳突型和篩狀型／複合腺型',
        '006': '腫瘤型態包含實體型和篩狀型／複合腺型',
        '007': '腫瘤型態包含微乳突型、實體型和篩狀型／複合腺型',
        '988': '不適用，非腺癌個案',
        '999': '病歷未記載或不詳',
    },
}

_ZH_EGFR_LETTERS = {
    'A': 'Exon 19 缺失', 'B': 'Exon 21 L858R', 'C': 'Exon 18 E709',
    'D': 'Exon 18 G719X', 'E': 'Exon 20 插入', 'F': 'Exon 20 S768I',
    'G': 'Exon 20 T790M', 'H': 'Exon 21 L861', 'U': '其他突變',
}


def _zh_lung(ssf_key: str, code: str) -> str:
    code = str(code).strip()
    if ssf_key in _ZH_LUNG:
        text = _ZH_LUNG[ssf_key].get(code)
        if text:
            return text
        return f'未定義代碼 {code}'
    if ssf_key == 'SSF5':
        if code == '988':
            return '不適用（小細胞肺癌或未執行手術）'
        if code == '999':
            return '不詳／有廓清但位置不明'
        if code.isdigit() and 0 <= int(code) <= 8:
            n = int(code)
            return ('無執行縱膈腔淋巴結取樣或廓清' if n == 0
                    else f'取樣或廓清縱膈腔淋巴結 {n} 個位置')
        return f'未定義代碼 {code}'
    if ssf_key == 'SSF6':
        if code == '999':
            return 'EGFR 不詳／未檢驗'
        if code == 'XXX':
            return 'EGFR 無突變'
        if code == 'VVV':
            return 'EGFR 有突變，型別不明'
        if code == 'ZZZ':
            return 'EGFR 檢驗結果無法判讀'
        if len(code) == 3:
            names = [_ZH_EGFR_LETTERS[c] for c in code if c in _ZH_EGFR_LETTERS]
            if names and all(c in _ZH_EGFR_LETTERS or c == 'X' for c in code):
                return 'EGFR 突變：' + '＋'.join(names)
        return f'未定義代碼 {code}'
    if ssf_key == 'SSF9':
        if code == '988':
            return ('不適用：T0、非病理期別 0-2 期、T3N0、未執行手術、僅單顆腫瘤，'
                    '或非於申報醫院診療且無外院資料')
        if code == '999':
            return '病歷未記載或不詳'
        if code == '021':
            return '腫瘤顆數大於 20 顆以上'
        if code.isdigit() and 2 <= int(code) <= 20:
            return f'腫瘤顆數為 {int(code)} 顆'
        return f'未定義代碼 {code}'
    if ssf_key == 'SSF10' and code == '988':
        return '肺癌僅收錄 SSF1-SSF9，SSF10 一律編碼 988（手冊 p.1）'
    return f'未定義代碼 {code}'


_ZH_CRC = {
    'SSF3': {
        '000': '腫瘤縮小等級 0 級 (Complete response)：完全反應，無殘餘腫瘤',
        '010': '腫瘤縮小等級 1 級 (Moderate response)：單個或少量殘餘癌細胞',
        '020': '腫瘤縮小等級 2 級 (Minimal response)：殘餘癌細胞組織纖維化',
        '030': '腫瘤縮小等級 3 級 (Poor response)：廣泛殘餘癌細胞',
        '988': '不適用：GIST、NETs 及 High grade dysplasia；沒有術前治療；'
               '沒有手術治療；無病理組織學確認',
        '990': '有反應，但反應程度未進一步說明',
        '999': '治療反應不詳／病歷未記載',
    },
    'SSF5': {
        '000': 'BRAF 基因正常／無突變／wild type',
        '001': 'BRAF 有突變：含 V600E (c.1799T>A) 變異',
        '002': 'BRAF 有突變：非 V600E 變異',
        '006': '僅描述 BRAF 有突變，但不確定何種密碼子變異',
        '007': '有進行 BRAF 檢驗，但報告描述為無法判讀',
        '988': '不適用：GIST、NETs 及 High grade dysplasia；或於外院檢查且無結果',
        '990': '碼冊 p.69 編碼範圍列出 990，但碼表未定義此碼（需向登記中心確認）',
        '991': '碼冊 p.69 編碼範圍列出 991，但碼表未定義此碼（需向登記中心確認）',
        '999': '病歷未記載或不詳／沒有檢驗',
    },
    'SSF7': {
        '000': '任一影像檢查未發現腸阻塞，或手術中未發現腸阻塞',
        '010': '任一影像檢查發現腸阻塞，或手術中發現腸阻塞',
        '988': '不適用：無進行任一影像檢查亦無接受手術；GIST、NETs 及 High grade dysplasia',
        '999': '病歷未記載或不詳',
    },
    'SSF8': {
        '000': '任一影像檢查未發現腸穿孔，或手術中未發現腸穿孔',
        '010': '任一影像檢查發現腸穿孔，或手術中發現腸穿孔',
        '980': '碼冊 p.58 編碼範圍列出 980，但碼表未定義此碼（需向登記中心確認）',
        '988': '不適用：無進行任一影像檢查亦無接受手術；GIST、NETs 及 High grade dysplasia',
        '999': '病歷未記載或不詳',
    },
    'SSF10': {
        '000': 'MSS／微星體穩定；且／或 MMR 功能正常 (pMMR)',
        '010': 'MSI-L：低度不穩定',
        '020': 'MSI-H：高度不穩定；且／或 MMR 功能缺失 (dMMR)',
        '988': '不適用：GIST、NETs 及 High grade dysplasia；或於外院檢查且無結果',
        '999': '病歷未記載或不詳／沒有檢測／MSI 判定不確定',
    },
}

_ZH_RAS_DIGIT = {
    '0': 'wild type（無突變）', '1': 'Codon 12 突變', '2': 'Codon 13 突變',
    '3': 'Codon 61 突變', '4': '多密碼子突變（≧2 個，含 12/13/61 之一）',
    '5': '非 12/13/61 之突變', '6': '有突變但密碼子不明', '7': '無法判讀',
    '9': '未檢驗',
}

_ZH_CRM = {
    '980': '腫瘤環切緣 ≧98mm',
    '988': '不適用：GIST、NETs 及 High grade dysplasia；沒有接受手術；'
           '於外院手術且無環切緣資料；僅局部切除且無描述 CRM；病理報告明示不適用',
    '990': '病理標本無殘餘腫瘤 (no residual tumor)',
    '991': '腫瘤環切緣陰性、距離沒有說明',
    '992': '腫瘤環切緣介於 1mm 至 2mm 之間',
    '993': '腫瘤環切緣介於 2mm 至 3mm 之間',
    '994': '腫瘤環切緣介於 3mm 至 4mm 之間',
    '995': '腫瘤環切緣介於 4mm 至 5mm 之間',
    '996': '腫瘤環切緣 >5mm',
    '999': '腫瘤環切緣不詳／病歷未記載／僅描述 distal and proximal margins',
}

_ZH_ANUS = {
    '000': '小於 1mm',
    '988': '不適用：GIST、NETs 及 High grade dysplasia；於外院已接受首次治療'
           '且無外院報告；直腸乙狀結腸 (C19.9)',
    '991': '直腸腫瘤位於直腸上 1/3',
    '992': '直腸腫瘤位於直腸中 1/3',
    '993': '直腸腫瘤位於直腸下 1/3',
    '999': '病歷未記載或不詳／療前未評估',
}


def _zh_colorectum(ssf_key: str, code: str) -> str:
    code = str(code).strip()
    if ssf_key == 'SSF1':
        return _zh_lab_x10('癌胚抗原 CEA', code, _ZH_CEA_NA,
                           '病歷未記載或不詳／未進行癌胚抗原 CEA 檢驗')
    if ssf_key == 'SSF2':
        return _zh_vs_normal('癌胚抗原 CEA', code, _ZH_CEA_NA)
    if ssf_key in _ZH_CRC:
        return _ZH_CRC[ssf_key].get(code, f'未定義代碼 {code}')
    if ssf_key == 'SSF4':
        if code in _ZH_CRM:
            return _ZH_CRM[code]
        if code.isdigit() and 0 <= int(code) <= 979:
            return f'腫瘤環切緣 {int(code) / 10:.1f} mm（以 0.1mm 為單位）'
        return f'未定義代碼 {code}'
    if ssf_key == 'SSF6':
        if code == '988':
            return '不適用：GIST、NETs 及 High grade dysplasia；或於外院檢查且無結果'
        if code == '998':
            return 'RAS 檢驗結果未記載／沒有檢驗'
        if (len(code) == 3 and code[2] == '8'
                and code[0] in _ZH_RAS_DIGIT and code[1] in _ZH_RAS_DIGIT):
            return (f'KRAS：{_ZH_RAS_DIGIT[code[0]]}；'
                    f'NRAS：{_ZH_RAS_DIGIT[code[1]]}（第三碼為空白處，固定為 8）')
        return f'未定義代碼 {code}'
    if ssf_key == 'SSF9':
        if code in _ZH_ANUS:
            return _ZH_ANUS[code]
        if code.isdigit() and 1 <= int(code) <= 150:
            return f'直腸腫瘤下緣與肛門口的距離 {int(code)} mm'
        return f'未定義代碼 {code}'
    return f'未定義代碼 {code}'


_ZH_HN_NODE_SIZE = {
    '000': '頸部淋巴結沒有被侵犯（N0）',
    '987': '被侵犯的淋巴結大小 ≧987 亳米',
    '988': '不適用，已於外院診療且無外院資料',
    '990': '僅顯微鏡下見到微小病灶且未記載大小，臨床觸診摸不到頸部淋巴結',
    '991': '只能確定淋巴結小於 10 亳米',
    '992': '只能確定淋巴結介於 10-20 亳米',
    '993': '只能確定淋巴結介於 20-30 亳米',
    '994': '只能確定淋巴結介於 30-40 亳米',
    '995': '只能確定淋巴結介於 40-50 亳米',
    '996': '只能確定淋巴結介於 50-60 亳米',
    '997': '只能確定淋巴結大於 60 亳米',
    '999': '頸部淋巴結情況不明／病歷中未記載／無法被評估',
}

_ZH_HN_ECE = {
    '000': '沒有頸部淋巴結莢膜外侵犯情形',
    '001': '臨床評估可能有莢膜外侵犯，或臨床上有固定不動的頸部淋巴結，'
           '但病理報告未陳述',
    '002': '臨床評估有莢膜外侵犯，手術前先接受前導性治療，術後病理報告陳述'
           '無淋巴結侵犯或無莢膜外侵犯',
    '005': '病理報告敘述有頸部淋巴結莢膜外侵犯存在',
    '988': '不適用：頸部淋巴結都沒有被侵犯；或於外院執行檢查或治療且無外院資料',
    '999': '頸部淋巴結情況不明／病歷中未記載／無法被評估',
}

_ZH_HN_REGIONS = {
    'SSF3': ('第 I 區', '第 II 區', '第 III 區'),
    'SSF4': ('第 IV 區', '第 V 區', '後咽區'),
    'SSF5': ('第 VI 區', '第 VII 區', '顏面淋巴結'),
    'SSF6': ('側咽區', '腮腺區', '後枕/耳後區'),
}
_ZH_HN_LEVEL_STATUS = {'0': '未侵犯', '1': '有侵犯',
                       '8': '跨區侵犯，無法區分正確侵犯區域'}

_ZH_HN_ENE_FINAL = {
    '0': '無 ENE 侵犯 ENE(-)', '1': '有 ENE 侵犯 ENE(+)',
    '8': '不使用 overt ENE（AJCC 第八版第 9/10/14 章）',
    '9': '未描述 ENE 侵犯狀態',
}
_ZH_HN_ENE_DETAIL = {
    '0': '無 ENE 侵犯 ENE(-)',
    '1': '有 ENE 侵犯且有描述特徵 ENE(+)',
    '2': '有 ENE 侵犯但無描述特徵 ENE(+)',
    '9': '未描述或未執行相關檢查',
}
_ZH_HN_ENE_PATH = {
    '000': '有病理證實淋巴結轉移，但無病理淋巴結外侵犯',
    '199': '有病理 ENE，侵犯距離 ≦2mm，但無確實距離數據',
    '210': '有病理 ENE，侵犯距離 ≧9.9mm',
    '299': '有病理 ENE，侵犯距離 >2mm 但無確實距離數據；或肉眼／巨觀已發現 ENE',
    '399': '有病理 ENE，但無確實距離數據',
    '988': '不適用：淋巴結切片/excision/dissection 後為 pN0；無淋巴結手術；'
           '或 pNx（未見惡性侵犯且無淋巴組織）',
    '998': '於外院接受淋巴結切片或手術，外院病理淋巴結外侵犯狀態不詳',
    '999': '病理報告描述無法評估或未評估／區域淋巴結侵犯但不知 ENE 情形／不詳',
}


def _zh_head_neck(ssf_key: str, code: str) -> str:
    code = str(code).strip()
    if ssf_key == 'SSF1':
        if code in _ZH_HN_NODE_SIZE:
            return _ZH_HN_NODE_SIZE[code]
        if code.isdigit() and 1 <= int(code) <= 986:
            return f'頸部淋巴結被侵犯的最大徑 {int(code)} 亳米'
        return f'未定義代碼 {code}'
    if ssf_key == 'SSF2':
        return _ZH_HN_ECE.get(code, f'未定義代碼 {code}')
    if ssf_key in _ZH_HN_REGIONS:
        regions = _ZH_HN_REGIONS[ssf_key]
        if code == '988':
            return '不適用，於外院執行檢查或治療且無外院資料'
        if code == '999':
            return '頸部該區域淋巴結情況不明／病歷中未記載／無法被評估'
        if len(code) == 3 and all(c in _ZH_HN_LEVEL_STATUS for c in code):
            if code == '000':
                return f'{regions[0]}、{regions[1]}、{regions[2]} 都未侵犯（N stage 為 0）'
            return '；'.join(f'{r}：{_ZH_HN_LEVEL_STATUS[c]}'
                             for r, c in zip(regions, code))
        return f'未定義代碼 {code}'
    if ssf_key == 'SSF7':
        specials = {
            '980': '腫瘤之深度 ≧98mm',
            '987': '腫瘤病理報告屬於原位癌',
            '988': '不適用，非口腔癌個案；於外院執行檢查或治療且無外院資料',
            '990': '腫瘤只有微小侵犯或小區域轉移，且未描述腫瘤深度',
            '997': '再次切除後病理標本無殘餘腫瘤，沒有深度可測量，且前次報告亦未描述',
            '998': '沒有手術標本（未接受手術）；或手術前已接受放射治療或全身性治療',
            '999': '病歷的病理報告中缺乏深度的數據',
        }
        if code in specials:
            return specials[code]
        if code.isdigit():
            n = int(code)
            if n == 0:
                return '實際腫瘤深度為 0 mm'
            if 1 <= n <= 979:
                return f'實際腫瘤深度 {n / 10:.1f} mm（以 0.1mm 為單位）'
        return f'未定義代碼 {code}'
    if ssf_key == 'SSF8':
        specials = {
            '000': '小於 1mm 且未明示手術邊緣狀態；或手術切緣陽性',
            '980': '手術切緣距離 ≧98mm',
            '987': '腫瘤病理報告屬於原位癌',
            '988': '不適用，非口腔癌個案；於外院執行檢查或治療且無外院資訊',
            '990': '再次切除後病理標本無殘餘腫瘤，切緣距離不清楚',
            '998': '沒有手術標本（未接受手術）；或手術前已接受放射治療或全身性治療',
            '999': '病理報告中缺乏切緣距離的數據',
        }
        if code in specials:
            return specials[code]
        if code.isdigit() and 1 <= int(code) <= 979:
            return f'手術邊緣狀態為陰性，切緣距離 {int(code) / 10:.1f} mm'
        return f'未定義代碼 {code}'
    if ssf_key == 'SSF9':
        if code == '988':
            return '不適用，臨床 N category 判定為 cN0'
        if code == '998':
            return '於外院診療且無外院資料；或術後才確診之個案'
        if (len(code) == 3 and code[0] in _ZH_HN_ENE_FINAL
                and code[1] in _ZH_HN_ENE_DETAIL and code[2] in _ZH_HN_ENE_DETAIL):
            return (f'綜合判定：{_ZH_HN_ENE_FINAL[code[0]]}；'
                    f'影像檢查：{_ZH_HN_ENE_DETAIL[code[1]]}；'
                    f'理學檢查：{_ZH_HN_ENE_DETAIL[code[2]]}')
        return f'未定義代碼 {code}'
    if ssf_key == 'SSF10':
        if code in _ZH_HN_ENE_PATH:
            return _ZH_HN_ENE_PATH[code]
        if len(code) == 3 and code.isdigit():
            head, dist = code[0], int(code[1:])
            if head == '1' and 1 <= dist <= 20:
                return f'有病理 ENE，侵犯距離 ≦2mm，實測 {dist / 10:.1f} mm'
            if head == '2' and 21 <= dist <= 98:
                return f'有病理 ENE，侵犯距離 >2mm，實測 {dist / 10:.1f} mm'
        return f'未定義代碼 {code}'
    return f'未定義欄位 {ssf_key}'


# ─────────────────────────────────────────────────────────────────────────────
# Chinese definitions: 食道 / 胰臟 / 卵巢 / 膀胱
# ─────────────────────────────────────────────────────────────────────────────

_ZH_ESOPHAGUS = {
    'SSF1': {
        '000': '無使用正子掃描電腦斷層檢查',
        '020': '僅首次治療前有使用正子掃描電腦斷層檢查',
        '030': '僅首次治療後有使用正子掃描電腦斷層檢查',
        '040': '首次治療前後均有使用正子掃描電腦斷層檢查',
        '988': '不適用',
        '999': '病歷未記載或不詳',
    },
    'SSF2': {
        '000': '無使用微創手術切除食道癌',
        '010': '有使用微創手術切除食道癌',
        '988': '不適用',
        '999': '病歷未記載或不詳',
    },
    'SSF3': {
        '000': 'No viable cancer cells（完全反應，score 0）',
        '001': 'Single cells or rare small groups of cancer cells（近乎完全反應，score 1）',
        '002': '殘餘癌細胞但有明顯腫瘤退縮，多於單顆或少量細胞（部分反應，score 2）',
        '003': '廣泛殘餘癌細胞且無明顯退縮（差或無反應，score 3）',
        '010': '病理資訊顯示腫瘤全消 (Complete response)',
        '020': '病理資訊顯示腫瘤縮小 ≧50%',
        '030': '病理資訊顯示腫瘤縮小 <50%',
        '040': '病理資訊顯示腫瘤未縮小',
        '988': '不適用：未進行前導性治療；未進行手術治療；GIST 及 Sarcoma',
        '990': '前導性療法後腫瘤有縮小反應，但未說明反應程度',
        '999': '有進行前導性治療但療效不明／病歷未記載或不詳',
    },
    'SSF4': {
        '010': '臨床資訊顯示完全反應（CR，100%）',
        '020': '臨床資訊顯示部份反應（PR，50%）',
        '030': '臨床資訊顯示病情穩定（SD）',
        '040': '臨床資訊顯示漸進性疾病（PD）',
        '988': '不適用：沒有執行放射治療；僅執行化學治療；有執行手術切除；'
               'GIST 及 Sarcoma；僅執行遠端部位放射治療',
        '990': '治療後有縮小反應，但反應程度未進一步說明',
        '999': '有進行前導性治療但療效不明／病歷未記載或不詳',
    },
}


def _zh_esophagus(ssf_key: str, code: str) -> str:
    code = str(code).strip()
    if ssf_key in _ZH_ESOPHAGUS:
        return _ZH_ESOPHAGUS[ssf_key].get(code, f'未定義代碼 {code}')
    if code == '988':
        return '食道癌僅收錄 SSF1-SSF4，其餘欄位一律編碼 988（手冊 p.1）'
    return f'未定義代碼 {code}'


_ZH_CA19_9_TIERS = {
    '980': '98.0-99.9 U/ml', '981': '100.0-199.9 U/ml', '982': '200.0-299.9 U/ml',
    '983': '300.0-399.9 U/ml', '984': '400.0-499.9 U/ml', '985': '500.0-599.9 U/ml',
    '986': '600.0-699.9 U/ml', '987': '700.0-799.9 U/ml', '989': '800.0-899.9 U/ml',
    '990': '900.0-999.9 U/ml', '991': '1000.0-1999.9 U/ml',
    '992': '2000.0-2999.9 U/ml', '993': '3000.0-3999.9 U/ml',
    '994': '4000.0-4999.9 U/ml', '995': '5000.0-5999.9 U/ml',
    '996': '6000.0-6999.9 U/ml', '997': '≧7000.0 U/ml',
}

_ZH_MITOTIC = {
    '110': '<2 mitoses per 10 HPF / 2 mm2（Grade 1）',
    '120': '2-20 mitoses per 10 HPF / 2 mm2（Grade 2）',
    '130': '>20 mitoses per 10 HPF / 2 mm2（Grade 3）',
    '988': '不適用，組織型態非神經內分泌瘤 (NET)',
    '999': '病歷未記載或不詳',
}

_ZH_HBA1C_VALUE = {
    95: '9.5-9.9%', 96: '10.0-10.9%', 97: '11.0-11.9%', 98: '≧12.0%',
    99: '不詳或沒有檢驗',
}


def _zh_pancreas(ssf_key: str, code: str) -> str:
    code = str(code).strip()
    if ssf_key == 'SSF1':
        return _zh_lab_x10('癌胚抗原 CEA', code, _ZH_CEA_NA,
                           '病歷未記載或不詳／未進行癌胚抗原 CEA 檢驗')
    if ssf_key == 'SSF2':
        return _zh_vs_normal('癌胚抗原 CEA', code, _ZH_CEA_NA)
    if ssf_key == 'SSF3':
        if code in _ZH_CA19_9_TIERS:
            return f'血清 CA 19-9 {_ZH_CA19_9_TIERS[code]}'
        if code == '988':
            return '不適用，於外院已接受首次療程且無外院檢驗值'
        if code == '999':
            return '不詳／首次治療前無檢驗 CA 19-9'
        if code.isdigit():
            n = int(code)
            if n == 0:
                return '血清 CA 19-9 0.0 U/ml'
            if n == 1:
                return '血清 CA 19-9 ≦0.1 U/ml'
            if 2 <= n <= 979:
                return f'血清 CA 19-9 {n / 10:.1f} U/ml'
        return f'未定義代碼 {code}'
    if ssf_key == 'SSF4':
        # Identical Ki-67 scheme to breast SSF10 (p.97).
        return _zh_breast('SSF10', code)
    if ssf_key == 'SSF5':
        if code in _ZH_MITOTIC:
            return _ZH_MITOTIC[code]
        if code.isdigit() and 0 <= int(code) <= 21:
            return f'{int(code)} of mitoses per 10 HPF (per 2 mm2)'
        return f'未定義代碼 {code}'
    if ssf_key == 'SSF6':
        if code == '988':
            return '不適用，於外院執行檢查且無外院檢驗結果及糖尿病病史紀錄'
        if code == '999':
            return '不詳或沒有檢驗'
        if len(code) == 3 and code[0] in '01' and code[1:].isdigit():
            history = '有糖尿病病史' if code[0] == '1' else '無糖尿病病史'
            value = int(code[1:])
            if value in _ZH_HBA1C_VALUE:
                return f'{history}；糖化血色素 {_ZH_HBA1C_VALUE[value]}'
            if 1 <= value <= 94:
                return f'{history}；糖化血色素 {value / 10:.1f}%'
        return f'未定義代碼 {code}'
    if code == '988':
        return '胰臟癌僅收錄 SSF1-SSF6，其餘欄位一律編碼 988（手冊 p.1）'
    return f'未定義代碼 {code}'


_ZH_CA125_TIERS = {
    '901': '901-1000 U/ml', '902': '1001-2000 U/ml', '903': '2001-3000 U/ml',
    '904': '3001-4000 U/ml', '905': '4001-5000 U/ml', '906': '5001-6000 U/ml',
    '907': '6001-7000 U/ml', '908': '7001-8000 U/ml', '909': '8001-9000 U/ml',
    '910': '9001-10000 U/ml', '920': '10001-20000 U/ml',
    '930': '20001-30000 U/ml', '931': '≧30001 U/ml',
}

_ZH_OVARY_RESIDUAL = {
    '000': '沒有殘存腫瘤',
    '010': '殘存腫瘤 ≦1 公分，未接受術前化學治療',
    '020': '殘存腫瘤 ≦1 公分，有接受術前化學治療',
    '030': '殘存腫瘤 >1 公分，未接受術前化學治療',
    '040': '殘存腫瘤 >1 公分，有接受術前化學治療',
    '988': '不適用；未手術',
    '990': '以肉眼觀看有殘存腫瘤，但未記錄大小且未接受術前化學治療',
    '991': '以肉眼觀看有殘存腫瘤，但未記錄大小且有接受術前化學治療',
    '999': '病歷未記載或不詳',
}


def _zh_ovary(ssf_key: str, code: str) -> str:
    code = str(code).strip()
    if ssf_key in ('SSF1', 'SSF2'):
        when = '療前' if ssf_key == 'SSF1' else '療後最低'
        if code in _ZH_CA125_TIERS:
            return f'{when}血清 CA125 檢驗值 {_ZH_CA125_TIERS[code]}'
        if code == '988':
            return '不適用'
        if code == '999':
            return '病歷未記載或不詳'
        if code.isdigit() and 1 <= int(code) <= 900:
            return f'{when}血清 CA125 檢驗值 {int(code)} U/ml'
        return f'未定義代碼 {code}'
    if ssf_key == 'SSF3':
        return _ZH_OVARY_RESIDUAL.get(code, f'未定義代碼 {code}')
    if code == '988':
        return '卵巢癌僅收錄 SSF1-SSF3，其餘欄位一律編碼 988（手冊 p.1）'
    return f'未定義代碼 {code}'


_ZH_BLADDER = {
    'SSF1': {
        '010': 'Low grade',
        '020': 'High grade',
        '988': '不適用：原發部位無病理檢查，或病理組織非泌尿上皮相關癌症',
        '999': '不知 WHO/ISUP 分級／病歷未記載',
    },
    'SSF2': {
        '000': '無區域淋巴結侵犯',
        '010': '區域淋巴結侵犯，但無淋巴結夾膜外侵犯',
        '020': '區域淋巴結侵犯，且有淋巴結夾膜外侵犯',
        '030': '區域淋巴結侵犯，但不知夾膜外是否有侵犯情形',
        '988': '不適用',
        '999': '不知區域淋巴結是否受侵犯／無法評估／病歷未記載',
    },
    'SSF3': {
        '000': '病理標本未包含固有肌肉層',
        '010': '病理標本有包含固有肌肉層',
        '988': '不適用：未執行 TURBT；於外院接受 TURBT 且無病理報告；'
               'TURBT 僅為診斷性處置',
        '999': '病理標本未描述是否含有固有肌肉層／不詳',
    },
}


def _zh_bladder(ssf_key: str, code: str) -> str:
    code = str(code).strip()
    if ssf_key in _ZH_BLADDER:
        return _ZH_BLADDER[ssf_key].get(code, f'未定義代碼 {code}')
    if code == '988':
        return '膀胱癌僅收錄 SSF1-SSF3，其餘欄位一律編碼 988（手冊 p.1）'
    return f'未定義代碼 {code}'


# ─────────────────────────────────────────────────────────────────────────────
# 淋巴瘤（碼冊 pp.194-206）
# ─────────────────────────────────────────────────────────────────────────────

_ZH_LYM_NA_HAEM = '不適用：ICD-O-3 M-9811-9837 且 C42.0、C42.1、C42.4'

_ZH_LYM_HIV = {
    '001': '陰性。',
    '002': '陽性。',
    '988': _ZH_LYM_NA_HAEM,
    '999': '病歷未記載或不詳；沒有檢驗 HIV。',
}

_ZH_LYM_B = {
    '000': '無 B 症狀。',
    '010': '有以下任一 B 症狀：發燒、夜間盜汗、體重減輕。',
    '988': '不適用：ICD-O-3 M-9811-9837 且 C42.0/C42.1/C42.4；M-9731-9732、'
           '9734；非 M-9650-9663。',
    '999': '不詳。',
}

_ZH_LYM_IPI_BAND = {
    '990': '僅記載低風險 (Low risk)。',
    '991': '僅記載中低風險 (Low intermediate risk)。',
    '992': '僅記載中高風險 (High intermediate risk)。',
    '993': '僅記載高風險 (High risk)。',
    '994': '僅記載中風險 (Intermediate risk)。',
    '988': '不適用：M-9650-9663；M-9700-9701、9731-9732、9734、9749、9751-9759、'
           '9761-9762、9766；M-9811-9837 且 C42.0/C42.1/C42.4；濾泡性淋巴瘤僅'
           '評估 FLIPI 未評估 IPI。',
    '999': '病歷未記載、不詳或沒有評估。',
}

_ZH_LYM_FLIPI_BAND = {
    '990': '僅記載低風險 (Low risk)。',
    '991': '僅記載中風險 (Intermediate risk)。',
    '992': '僅記載高風險 (High risk)。',
    '988': '不適用：非 ICD-O-3 M-95973、96903-96983 者。',
    '999': '病歷未記載、不詳或沒有評估。',
}

_ZH_LYM_HTLV1 = {
    '000': '沒有檢驗 HTLV-1。',
    '001': '陰性。',
    '002': '陽性。',
    '988': _ZH_LYM_NA_HAEM,
    '999': '病歷未記載或不詳。',
}

_ZH_LYM_CMV = {
    '000': '沒有檢驗 CMV。',
    '001': '沒有發生 CMV 感染。',
    '002': '發生 CMV 感染但未造成疾病。',
    '003': '發生 CMV 感染且造成疾病。',
    '988': '不適用。',
    '999': '病歷未記載或不詳。',
}

_ZH_LEU_CMV = {
    '001': '沒有發生 CMV 感染。',
    '002': '發生 CMV 感染但未造成疾病。',
    '003': '發生 CMV 感染且造成疾病。',
    '988': '不適用：ICD-O-3 M-9811-9837（C42.0、C42.1、C42.4 除外）。',
    '999': '病歷未記載或不詳；沒有檢驗 CMV。',
}


def _zh_hepatitis(marker: str, history: str, na_text: str) -> dict:
    """HBsAg / Anti-HCV：左數第 2 碼為檢驗結果，第 3 碼為病史。"""
    return {
        '000': f'沒有檢驗，亦無{history}。',
        '001': f'沒有檢驗，但病歷記載曾有{history}。',
        '010': f'檢驗結果為陰性，且無{history}。',
        '011': f'檢驗結果為陰性，但病歷記載曾有{history}。',
        '020': '檢驗結果為陽性。',
        '988': na_text,
        '999': '不詳。',
    }


_ZH_ACUTE_HEP = {
    '001': '無急性肝炎發作（GOT/GPT 未超過正常值上限 5 倍）。',
    '002': '有急性肝炎發作（GOT 或 GPT 超過正常值上限 5 倍）。',
    '999': '病歷未記載或不詳；沒有 GOT 或 GPT 任一項檢驗。',
}

_ZH_LYM_ESR_HEAD = {
    **{f'{i:02d}': f'ESR {i} mm/h。' for i in range(1, 51)},
    '51': 'ESR 數值 >50 mm/h。',
    '98': '不適用：非 ICD-O-3 M-9650-9663 者。',
    '99': '病歷未記載、不詳或沒有檢驗。',
}
_ZH_LYM_IPS_TAIL = {
    **{str(i): f'IPS {i} 分。' for i in range(8)},
    '8': '不適用：非 ICD-O-3 M-9650-9663 者。',
    '9': '病歷未記載、不詳或沒有評估。',
}


def _zh_lymphoma(ssf_key: str, code: str) -> str:
    c = str(code).strip()
    if ssf_key == 'SSF1':
        return _ZH_LYM_HIV.get(c, '')
    if ssf_key == 'SSF2':
        return _ZH_LYM_B.get(c, '')
    if ssf_key in ('SSF3', 'SSF4'):
        label = 'IPI' if ssf_key == 'SSF3' else 'FLIPI'
        bands = _ZH_LYM_IPI_BAND if ssf_key == 'SSF3' else _ZH_LYM_FLIPI_BAND
        if c.isdigit() and 0 <= int(c) <= 5:
            return f'{label} {int(c)} 分。'
        return bands.get(c, '')
    if ssf_key == 'SSF5':
        return _ZH_LYM_HTLV1.get(c, '')
    if ssf_key == 'SSF6':
        return _ZH_LYM_CMV.get(c, '')
    if ssf_key == 'SSF7':
        return _zh_hepatitis('HBsAg', 'B 肝帶原史', _ZH_LYM_NA_HAEM).get(c, '')
    if ssf_key == 'SSF8':
        return _zh_hepatitis('Anti-HCV', 'C 型肝炎感染史', _ZH_LYM_NA_HAEM).get(c, '')
    if ssf_key == 'SSF9':
        return {**_ZH_ACUTE_HEP, '988': _ZH_LYM_NA_HAEM}.get(c, '')
    if ssf_key == 'SSF10' and len(c) == 3:
        head = _ZH_LYM_ESR_HEAD.get(c[:2])
        tail = _ZH_LYM_IPS_TAIL.get(c[2])
        if head and tail:
            return f'第1-2碼 {head} 第3碼 {tail}'
    return ''


# ─────────────────────────────────────────────────────────────────────────────
# 白血病（碼冊 pp.207-222）
# ─────────────────────────────────────────────────────────────────────────────

_ZH_LEU_NA_MARROW = '不適用：ICD-O-3 M-9811-9837（C42.0、C42.1、C42.4 除外）。'

# 碼冊 pp.209-212 的染色體／分子生物學定義本身即以英文基因名稱書寫，
# 中文僅出現在 000／090-092 與 8XX 說明，因此此處沿用碼冊原文。
_ZH_LEU_BUCKETS = {
    '090': '一種異常，其他未列之變化；或二種異常，其中一種（或二種皆）非表列。',
    '091': '同時有兩種（含）以上表列之變化。',
    '092': '三種（含）以上變化。',
}


def _zh_leukemia_study(code: str, mapping, kind: str) -> str:
    c = str(code).strip()
    if c == '988':
        return _ZH_LEU_NA_MARROW
    if c == '998':
        return f'有執行{kind}，但結果無法判斷。'
    if c == '999':
        return f'病歷未記載或不詳，或未執行{kind}。'
    post = c.startswith('8') and len(c) == 3
    base = f'0{c[1:]}' if post else c
    if base in _ZH_LEU_BUCKETS:
        text = _ZH_LEU_BUCKETS[base]
    else:
        try:
            text = mapping[int(base)]
        except (KeyError, ValueError):
            return ''
        text = '正常。' if base == '000' else f'{text}。'
    if post:
        return f'{text}（化學治療／免疫治療／標靶治療後之{kind}）'
    return text


_ZH_LEU_INDUCTION = {
    '001': '完全緩解 (complete remission)。',
    '002': '部份緩解 (partial remission)。',
    '988': '不適用。',
    '990': '首次前導治療後之反應評估非部分 (002) 或完全緩解 (001)，'
           '例如 no remission、incomplete remission 等。',
    '999': '病歷未記載或不詳。',
}

_ZH_LEU_AGVHD = {
    '000': '沒有發生過 aGVHD。',
    '010': '有發生過 aGVHD 但嚴重度不明。',
    '011': '有發生過第一級 aGVHD。',
    '012': '有發生過第二級 aGVHD。',
    '013': '有發生過第三級 aGVHD。',
    '014': '有發生過第四級 aGVHD。',
    '988': '不適用：未接受異體幹細胞移植，或接受自體幹細胞移植。',
    '999': '病歷未記載或不詳。',
}

_ZH_LEU_CGVHD = {
    '000': '沒有發生 cGVHD。',
    '001': '發生 cGVHD 但嚴重度不明。',
    '002': '發生侷限期 (limited stage) cGVHD。',
    '003': '發生廣泛期 (extensive stage) cGVHD。',
    '988': '不適用：未接受異體幹細胞移植。',
    '999': '病歷未記載或不詳。',
}

_ZH_LEU_MRD_HEAD = {
    **{f'{i:02d}': f'自診斷日起至申報前最後一次分子檢驗之實足月數 {i} 個月。'
       for i in range(25)},
    '98': '不適用：非 ICD-O-3 M-9875/3；未使用藥物治療。',
    '99': '病歷未記載或不詳；間隔超過 24 個月；未執行分子檢驗。',
}
_ZH_LEU_MRD_TAIL = {
    '0': 'No log reduction (=0) 或上升。',
    '1': '0 < log reduction < 1。',
    '2': '1 ≦ log reduction < 2。',
    '3': '2 ≦ log reduction < 3。',
    '4': '3 ≦ log reduction < 4。',
    '5': 'log reduction ≧ 4。',
    '6': 'log reduction 僅描述為 undetectable，且未描述數值。',
    '8': '不適用：非 ICD-O-3 M-9875/3；未使用藥物治療。',
    '9': '病歷未記載或不詳；間隔超過 24 個月；未執行分子檢驗。',
}


def _zh_leukemia(ssf_key: str, code: str) -> str:
    from tcr_decoder.ssf_registry import _LEU_KARYOTYPE, _LEU_MOLECULAR
    c = str(code).strip()
    if ssf_key == 'SSF1':
        return _zh_leukemia_study(c, _LEU_KARYOTYPE, '染色體檢查')
    if ssf_key == 'SSF2':
        return _zh_leukemia_study(c, _LEU_MOLECULAR, '分子生物學檢查')
    if ssf_key == 'SSF3':
        return _ZH_LEU_INDUCTION.get(c, '')
    if ssf_key == 'SSF4':
        return _ZH_LEU_AGVHD.get(c, '')
    if ssf_key == 'SSF5':
        return _ZH_LEU_CGVHD.get(c, '')
    if ssf_key == 'SSF6':
        return _ZH_LEU_CMV.get(c, '')
    if ssf_key == 'SSF7':
        return _zh_hepatitis('HBsAg', 'B 肝帶原史', _ZH_LEU_NA_MARROW).get(c, '')
    if ssf_key == 'SSF8':
        return _zh_hepatitis('Anti-HCV', 'C 型肝炎感染史', _ZH_LEU_NA_MARROW).get(c, '')
    if ssf_key == 'SSF9':
        return {**_ZH_ACUTE_HEP, '988': _ZH_LEU_NA_MARROW}.get(c, '')
    if ssf_key == 'SSF10' and len(c) == 3:
        head = _ZH_LEU_MRD_HEAD.get(c[:2])
        tail = _ZH_LEU_MRD_TAIL.get(c[2])
        if head and tail:
            return f'第1-2碼 {head} 第3碼 {tail}'
    return ''


_ZH_BY_GROUP = {
    'breast': _zh_breast,
    'prostate': _zh_prostate,
    'endometrium': _zh_endometrium,
    'thyroid': _zh_thyroid,
    'cervix': _zh_cervix,
    'stomach': _zh_stomach,
    'liver': _zh_liver,
    'lung': _zh_lung,
    'colorectum': _zh_colorectum,
    'head_neck': _zh_head_neck,
    'esophagus': _zh_esophagus,
    'pancreas': _zh_pancreas,
    'ovary': _zh_ovary,
    'bladder': _zh_bladder,
    'lymphoma': _zh_lymphoma,
    'leukemia': _zh_leukemia,
}


# Groups whose Chinese code-table wording has been transcribed from the
# manual. For any other group the workbook shows the decoder's English
# clinical label instead, clearly marked, rather than pretending the Chinese
# definition exists.
ZH_TRANSCRIBED = tuple(sorted(_ZH_BY_GROUP))


def zh_definition(cancer_group: str, ssf_key: str, code: str) -> str:
    """The code book's own Chinese definition of one SSF code."""
    fn = _ZH_BY_GROUP.get(cancer_group)
    if fn is None:
        return ''
    return fn(ssf_key, code)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Field-level coverage: every legal code of every SSF field
# ─────────────────────────────────────────────────────────────────────────────

def _sorted_codes(codes) -> List[str]:
    """Numeric codes first in numeric order, then letter codes alphabetically."""
    numeric = sorted((c for c in codes if c.isdigit()), key=int)
    letters = sorted(c for c in codes if not c.isdigit())
    return numeric + letters


def _decode_encode_field(profile, cancer_group, ssf_key, raw):
    """Decode then encode one field, falling back to the generic SSF pair for
    fields a cancer group does not define (decoder=None)."""
    from tcr_decoder.encoders import encode_generic_ssf
    from tcr_decoder.ssf_registry import _generic_ssf

    field = profile.fields[ssf_key]
    decoded = (field.decoder(raw) if field.decoder is not None
               else _generic_ssf(raw, unit=field.unit))
    encoded = (field.encoder(decoded) if field.encoder is not None
               else encode_generic_ssf(decoded, unit=field.unit))
    return decoded, encoded.astype(str)


def build_field_coverage(cancer_group: str = 'breast') -> pd.DataFrame:
    """Every legal code of one cancer group: code -> meaning -> code."""
    _check_group(cancer_group)
    profile = get_ssf_profile(cancer_group)
    rows = []
    for ssf_key in SSF_KEYS:
        width, codes, ref = CODE_RANGES[cancer_group][ssf_key]
        field = profile.fields[ssf_key]
        raw = pd.Series(_sorted_codes(codes), dtype=object)
        decoded, reencoded = _decode_encode_field(profile, cancer_group, ssf_key, raw)
        for code, label, back in zip(raw, decoded, reencoded):
            rows.append({
                'SSF': ssf_key,
                '欄位名稱': field.column_name,
                '原始代碼': code,
                '碼冊中文定義': (zh_definition(cancer_group, ssf_key, code)
                             or f'（中文定義尚未轉錄）{label}'),
                '解碼臨床意義 (EN)': label,
                '反編碼結果': back,
                '往返一致': back == code,
                '欄位長度': width,
                '寬度正確': len(back) == width,
                '碼冊出處': ref,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Case-level coverage: pairwise combinations across all ten SSF fields
# ─────────────────────────────────────────────────────────────────────────────

# One representative per semantic class of each field. Every individual code
# is already covered exhaustively by build_field_coverage(); these drive the
# COMBINATION coverage, where the interesting thing is the interaction
# between fields (e.g. "pCR + Ki-67 unknown + LVI not assessable").
_BREAST_CLASSES: Dict[str, Tuple[str, ...]] = {
    'SSF1':  ('000', '001', '050', '100', 'W05', 'I50', 'S00',
              '110', '111', '120', '121', '888', '988', '999'),
    'SSF2':  ('000', '001', '050', '100', 'W05', 'I50', 'S00',
              '110', '111', '120', '121', '888', '988', '999'),
    'SSF3':  ('010', '011', '020', '030', '040', '988', '990', '999'),
    'SSF4':  ('000', '001', '010', '089', '988', '996', '999'),
    'SSF5':  ('000', '001', '010', '089', '988', '996', '999'),
    'SSF6':  ('030', '050', '060', '070', '090', '110', '120', '130', '988', '999'),
    'SSF7':  ('000', '004', '100', '101', '102', '103', '200', '201', '300', '301',
              '400', '401', '500', '501', '502', '510', '511', '512', '520', '521',
              '522', '530', '531', '532', '590', '591', '600', '601', '640', '641',
              '888', '900', '901', '902', '988', '999'),
    'SSF8':  ('000', '010', '988', '999'),
    'SSF9':  ('000', '010', '988', '990', '999'),
    'SSF10': ('000', '001', '013', '014', '030', '031', '100',
              'A00', 'A05', 'A09', '988', '998', '999'),
}

_PROSTATE_CLASSES: Dict[str, Tuple[str, ...]] = {
    'SSF1':  ('001', '045', '500', '979', '980', '981', '990', '991',
              '997', '998', '988', '999'),
    'SSF2':  ('011', '033', '034', '043', '055', '039', '099', '988', '999'),
    'SSF3':  ('002', '006', '007', '009', '010', '988', '999'),
    'SSF4':  ('011', '033', '034', '043', '055', '049', '099', '988', '999'),
    'SSF5':  ('002', '006', '007', '009', '010', '988', '999'),
    'SSF6':  ('001', '012', '024', '100', '988', '999'),
    'SSF7':  ('000', '001', '005', '100', '988', '998', '999'),
    'SSF8':  ('000', '010', '020', '030', '040', '050', '988', '999'),
    'SSF9':  ('988',),
    'SSF10': ('988',),
}

_ENDOMETRIUM_CLASSES: Dict[str, Tuple[str, ...]] = {
    'SSF1':  ('000', '050', '100', 'W05', 'S00', '110', '120', '888', '988', '999'),
    'SSF2':  ('000', '050', '100', 'I50', 'S00', '110', '120', '888', '988', '999'),
    'SSF3':  ('001', '002', '003', '987', '988', '999'),
    'SSF4':  ('010', '020', '030', '999'),
    'SSF5':  ('000', '010', '020', '988', '999'),
    'SSF6':  ('010', '020', '030', '999'),
    'SSF7':  ('988',),
    'SSF8':  ('988',),
    'SSF9':  ('988',),
    'SSF10': ('988',),
}

_THYROID_CLASSES: Dict[str, Tuple[str, ...]] = {f'SSF{i}': ('988',) for i in range(1, 11)}

_NOT_COLLECTED_CLASS = ('988',)

_CERVIX_CLASSES: Dict[str, Tuple[str, ...]] = {
    'SSF1':  ('001', '045', '500', '986', '987', '988', '999'),
    'SSF2':  ('010', '020', '030', '988', '999'),
    **{f'SSF{i}': _NOT_COLLECTED_CLASS for i in range(3, 11)},
}

_STOMACH_CLASSES: Dict[str, Tuple[str, ...]] = {
    'SSF1':  ('001', '045', '500', '987', '988', '999'),
    'SSF2':  ('010', '020', '030', '988', '999'),
    'SSF3':  ('000', '001', '005', '008', '010', '988', '999'),
    'SSF4':  ('000', '015', '500', '980', '988', '998', '999'),
    'SSF5':  ('000', '010', '988', '990', '999'),
    **{f'SSF{i}': _NOT_COLLECTED_CLASS for i in range(6, 11)},
}

_LIVER_CLASSES: Dict[str, Tuple[str, ...]] = {
    'SSF1':  ('A00', 'A05', 'A99', '000', '050', '500', '987', '988',
              '991', '992', '993', '999'),
    'SSF2':  ('000', '003', '006', '008', '988', '999'),
    'SSF3':  ('105', '106', '199', '207', '299', '310', '315', '399', '999'),
    'SSF4':  ('001', '045', '500', '987', '988', '999'),
    'SSF5':  ('001', '045', '500', '987', '988', '999'),
    'SSF6':  ('001', '012', '060', '997', '988', '999'),
    'SSF7':  ('000', '001', '010', '011', '020', '999'),
    'SSF8':  ('000', '001', '010', '011', '020', '999'),
    'SSF9':  _NOT_COLLECTED_CLASS,
    'SSF10': _NOT_COLLECTED_CLASS,
}

_LUNG_CLASSES: Dict[str, Tuple[str, ...]] = {
    'SSF1':  ('000', '010', '020', '030', '040', '999'),
    'SSF2':  ('000', '010', '020', '030', '040', '988', '999'),
    'SSF3':  ('000', '002', '005', '988', '998', '999'),
    'SSF4':  ('000', '011', '013', '015', '988', '999'),
    'SSF5':  ('000', '003', '008', '988', '999'),
    'SSF6':  ('AXX', 'ABX', 'ABC', 'VVV', 'XXX', 'ZZZ', '999'),
    'SSF7':  ('010', '020', '030', '999'),
    'SSF8':  ('000', '003', '007', '988', '999'),
    'SSF9':  ('002', '010', '021', '988', '999'),
    'SSF10': _NOT_COLLECTED_CLASS,
}

_COLORECTUM_CLASSES: Dict[str, Tuple[str, ...]] = {
    'SSF1':  ('001', '045', '500', '987', '988', '999'),
    'SSF2':  ('010', '020', '030', '988', '999'),
    'SSF3':  ('000', '010', '020', '030', '988', '990', '999'),
    'SSF4':  ('000', '050', '980', '988', '990', '992', '996', '999'),
    'SSF5':  ('000', '001', '002', '006', '007', '988', '990', '999'),
    'SSF6':  ('008', '118', '438', '798', '988', '998'),
    'SSF7':  ('000', '010', '988', '999'),
    'SSF8':  ('000', '010', '980', '988', '999'),
    'SSF9':  ('000', '050', '150', '988', '991', '993', '999'),
    'SSF10': ('000', '010', '020', '988', '999'),
}

_HEAD_NECK_CLASSES: Dict[str, Tuple[str, ...]] = {
    'SSF1':  ('000', '015', '045', '986', '987', '988', '990', '993', '997', '999'),
    'SSF2':  ('000', '001', '002', '005', '988', '999'),
    'SSF3':  ('000', '100', '110', '111', '018', '888', '988', '999'),
    'SSF4':  ('000', '010', '011', '808', '988', '999'),
    'SSF5':  ('000', '001', '111', '088', '988', '999'),
    'SSF6':  ('000', '100', '011', '880', '988', '999'),
    'SSF7':  ('000', '030', '979', '980', '987', '988', '990', '997', '998', '999'),
    'SSF8':  ('000', '025', '979', '980', '987', '988', '990', '998', '999'),
    'SSF9':  ('000', '110', '122', '199', '899', '999', '988', '998'),
    'SSF10': ('000', '101', '120', '199', '210', '221', '298', '299', '399',
              '988', '998', '999'),
}

_ESOPHAGUS_CLASSES: Dict[str, Tuple[str, ...]] = {
    'SSF1':  ('000', '020', '030', '040', '988', '999'),
    'SSF2':  ('000', '010', '988', '999'),
    'SSF3':  ('000', '001', '002', '003', '010', '020', '030', '040',
              '988', '990', '999'),
    'SSF4':  ('010', '020', '030', '040', '988', '990', '999'),
    **{f'SSF{i}': ('988',) for i in range(5, 11)},
}

_PANCREAS_CLASSES: Dict[str, Tuple[str, ...]] = {
    'SSF1':  ('001', '045', '500', '987', '988', '999'),
    'SSF2':  ('010', '020', '030', '988', '999'),
    'SSF3':  ('000', '001', '350', '979', '980', '991', '997', '988', '999'),
    'SSF4':  ('000', '014', '100', 'A05', '988', '998', '999'),
    'SSF5':  ('000', '005', '021', '110', '120', '130', '988', '999'),
    'SSF6':  ('001', '058', '094', '095', '098', '099', '158', '199',
              '988', '999'),
    **{f'SSF{i}': ('988',) for i in range(7, 11)},
}

_OVARY_CLASSES: Dict[str, Tuple[str, ...]] = {
    'SSF1':  ('001', '350', '900', '901', '910', '920', '931', '988', '999'),
    'SSF2':  ('001', '350', '900', '905', '930', '931', '988', '999'),
    'SSF3':  ('000', '010', '020', '030', '040', '988', '990', '991', '999'),
    **{f'SSF{i}': ('988',) for i in range(4, 11)},
}

_BLADDER_CLASSES: Dict[str, Tuple[str, ...]] = {
    'SSF1':  ('010', '020', '988', '999'),
    'SSF2':  ('000', '010', '020', '030', '988', '999'),
    'SSF3':  ('000', '010', '988', '999'),
    **{f'SSF{i}': ('988',) for i in range(4, 11)},
}

_LYMPHOMA_CLASSES: Dict[str, Tuple[str, ...]] = {
    'SSF1':  ('001', '002', '988', '999'),
    'SSF2':  ('000', '010', '988', '999'),
    'SSF3':  ('000', '003', '005', '988', '990', '994', '999'),
    'SSF4':  ('000', '002', '005', '988', '990', '992', '999'),
    'SSF5':  ('000', '001', '002', '988', '999'),
    'SSF6':  ('000', '001', '002', '003', '988', '999'),
    'SSF7':  ('000', '001', '010', '011', '020', '988', '999'),
    'SSF8':  ('000', '001', '010', '011', '020', '988', '999'),
    'SSF9':  ('001', '002', '988', '999'),
    # One representative per branch of the composite: a real ESR, the >50
    # ceiling, and each of the two 2-character sentinels, crossed with a
    # score, the not-applicable digit and the unknown digit.
    'SSF10': ('253', '510', '517', '988', '999', '983', '992'),
}

_LEUKEMIA_CLASSES: Dict[str, Tuple[str, ...]] = {
    'SSF1':  ('000', '001', '013', '051', '090', '092', '801', '892',
              '988', '998', '999'),
    'SSF2':  ('000', '010', '013', '051', '055', '090', '092', '803', '892',
              '988', '998', '999'),
    'SSF3':  ('001', '002', '988', '990', '999'),
    'SSF4':  ('000', '010', '011', '014', '988', '999'),
    'SSF5':  ('000', '001', '002', '003', '988', '999'),
    'SSF6':  ('001', '002', '003', '988', '999'),
    'SSF7':  ('000', '001', '010', '011', '020', '988', '999'),
    'SSF8':  ('000', '001', '010', '011', '020', '988', '999'),
    'SSF9':  ('001', '002', '988', '999'),
    'SSF10': ('000', '115', '246', '988', '999', '983', '992'),
}

EQUIVALENCE_CLASSES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    'breast': _BREAST_CLASSES,
    'esophagus': _ESOPHAGUS_CLASSES,
    'pancreas': _PANCREAS_CLASSES,
    'ovary': _OVARY_CLASSES,
    'bladder': _BLADDER_CLASSES,
    'head_neck': _HEAD_NECK_CLASSES,
    'cervix': _CERVIX_CLASSES,
    'stomach': _STOMACH_CLASSES,
    'liver': _LIVER_CLASSES,
    'lung': _LUNG_CLASSES,
    'colorectum': _COLORECTUM_CLASSES,
    'prostate': _PROSTATE_CLASSES,
    'endometrium': _ENDOMETRIUM_CLASSES,
    'thyroid': _THYROID_CLASSES,
    'lymphoma': _LYMPHOMA_CLASSES,
    'leukemia': _LEUKEMIA_CLASSES,
}


def _check_group(cancer_group: str):
    if cancer_group not in SUPPORTED_GROUPS:
        raise ValueError(
            f'{cancer_group!r} has no transcribed code ranges yet. '
            f'Verified groups: {", ".join(SUPPORTED_GROUPS)}. '
            f'See docs/codebook_conformance_findings.md.')
    if cancer_group not in EQUIVALENCE_CLASSES:
        raise ValueError(f'{cancer_group!r} has no equivalence classes defined')


def _all_pairs(domains: Dict[str, Sequence[str]]) -> List[Dict[str, str]]:
    """Greedy all-pairs (pairwise) case generation.

    Returns a list of cases (field -> value) such that for every pair of
    fields, every combination of one value from each appears in at least one
    case. Greedy IPO-style: repeatedly build the case that covers the most
    still-uncovered pairs.
    """
    keys = list(domains)
    uncovered = set()
    for a, b in itertools.combinations(keys, 2):
        for va in domains[a]:
            for vb in domains[b]:
                uncovered.add((a, va, b, vb))

    cases: List[Dict[str, str]] = []
    while uncovered:
        # Seed the case with one uncovered pair, then greedily fill the rest.
        seed = next(iter(sorted(uncovered)))
        case = {seed[0]: seed[1], seed[2]: seed[3]}
        for key in keys:
            if key in case:
                continue
            best_value, best_gain = domains[key][0], -1
            for value in domains[key]:
                gain = sum(
                    1 for other, chosen in case.items()
                    if ((other, chosen, key, value) in uncovered
                        or (key, value, other, chosen) in uncovered)
                )
                if gain > best_gain:
                    best_value, best_gain = value, gain
            case[key] = best_value
        for a, b in itertools.combinations(keys, 2):
            uncovered.discard((a, case[a], b, case[b]))
        cases.append({k: case[k] for k in keys})
    return cases


def build_case_combinations(cancer_group: str = 'breast') -> pd.DataFrame:
    """Pairwise-complete SSF combinations, one row per synthetic case."""
    _check_group(cancer_group)
    classes = EQUIVALENCE_CLASSES[cancer_group]
    cases = _all_pairs({k: classes[k] for k in SSF_KEYS})
    df = pd.DataFrame(cases, columns=list(SSF_KEYS))
    df.insert(0, 'Case_ID', [f'VAL{i + 1:04d}' for i in range(len(df))])
    return df


def pairwise_coverage(cases: pd.DataFrame,
                      cancer_group: str = 'breast') -> Tuple[int, int]:
    """(covered pairs, total pairs) across the ten SSF fields."""
    classes = EQUIVALENCE_CLASSES[cancer_group]
    total = 0
    covered = set()
    for a, b in itertools.combinations(SSF_KEYS, 2):
        total += len(classes[a]) * len(classes[b])
        for va, vb in zip(cases[a], cases[b]):
            covered.add((a, va, b, vb))
    return len(covered), total


# ─────────────────────────────────────────────────────────────────────────────
# 3. Whole-case round trip through the real pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_case_dataset(cases: Optional[pd.DataFrame] = None,
                       seed: int = 20260814,
                       cancer_group: str = 'breast') -> pd.DataFrame:
    """Full raw registry rows carrying the designed SSF combinations.

    The non-SSF columns come from the synthetic generator so the rows are
    complete, realistic registry records that TCRDecoder can actually load.
    """
    from tcr_decoder.synth import SyntheticTCRGenerator

    if cases is None:
        cases = build_case_combinations(cancer_group)
    gen = SyntheticTCRGenerator(cancer_group=cancer_group, n=len(cases), seed=seed)
    raw = gen.generate().reset_index(drop=True)
    cases = cases.reset_index(drop=True)
    for ssf_key in SSF_KEYS:
        raw[f'{ssf_key}_raw'] = cases[ssf_key].values
        decoded_col = f'{ssf_key}_decoded'
        if decoded_col in raw.columns:
            # This file's own "already decoded" column would be stale for the
            # codes we just substituted; blank it so nothing downstream can
            # silently trust it instead of the code table.
            raw[decoded_col] = ''
    raw.insert(0, 'Case_ID', cases['Case_ID'].values)
    return raw


def validate_cases(raw: Optional[pd.DataFrame] = None,
                   tmp_dir: Optional[Union[str, Path]] = None,
                   cancer_group: str = 'breast'):
    """Run the designed cases through decode -> encode and diff per field.

    Returns (per_field_results, clinical_view, decoder), where
    per_field_results has one row per (case, SSF field).
    """
    import tempfile
    from tcr_decoder.core import TCRDecoder
    from tcr_decoder.encoder import TCREncoder

    if raw is None:
        raw = build_case_dataset(cancer_group=cancer_group)
    tmp_dir = Path(tmp_dir or tempfile.mkdtemp())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    xlsx = tmp_dir / '_validation_cases.xlsx'
    with pd.ExcelWriter(str(xlsx), engine='openpyxl') as writer:
        raw.drop(columns=['Case_ID']).to_excel(
            writer, sheet_name='All_Fields_Decoded', index=False)

    dec = TCRDecoder(str(xlsx))
    dec.load(skip_input_check=True).decode()
    try:
        # Module 2 turns the decoded SSF combination into the clinical
        # reading a reviewer actually cares about (molecular subtype, NPI),
        # which is the point of the Clinical_View sheet. Failing to compute a
        # score must not invalidate the round-trip evidence, so it is
        # best-effort.
        dec.decode_with_scores()
    except Exception:  # pragma: no cover - scores are a bonus, not the proof
        pass
    clean = dec.clean
    enc = TCREncoder(clean, cancer_group=cancer_group)
    reencoded = enc.encode(on_error='empty')

    profile = get_ssf_profile(cancer_group)
    rows = []
    for ssf_key in SSF_KEYS:
        col = profile.fields[ssf_key].column_name
        original = raw[f'{ssf_key}_raw'].astype(str).reset_index(drop=True)
        labels = clean[col].astype(str).reset_index(drop=True)
        back = (reencoded[f'{ssf_key}_raw'].astype(str).reset_index(drop=True)
                if f'{ssf_key}_raw' in reencoded.columns
                else pd.Series([''] * len(original)))
        for i in range(len(original)):
            rows.append({
                'Case_ID': raw['Case_ID'].iloc[i],
                'SSF': ssf_key,
                '欄位名稱': col,
                '原始代碼': original[i],
                '碼冊中文定義': (zh_definition(cancer_group, ssf_key, original[i])
                             or f'（中文定義尚未轉錄）{labels[i]}'),
                '解碼臨床意義 (EN)': labels[i],
                '反編碼結果': back[i],
                '往返一致': back[i] == original[i],
            })
    per_field = pd.DataFrame(rows)

    clinical_cols = ['Case_ID'] + [profile.fields[k].column_name for k in SSF_KEYS]
    clinical_view = clean.copy()
    clinical_view.insert(0, 'Case_ID', raw['Case_ID'].values)
    extra = [c for c in ('Molecular_Subtype', 'NPI_Score', 'NPI_Group',
                         'Triple_Negative', 'HR_Status')
             if c in clinical_view.columns]
    clinical_view = clinical_view[[c for c in clinical_cols if c in clinical_view.columns] + extra]
    return per_field, clinical_view, dec


# ─────────────────────────────────────────────────────────────────────────────
# 4. Workbook export
# ─────────────────────────────────────────────────────────────────────────────

_README_ROWS = [
    ('目的', '證明乳癌每一個 TCR 代碼、以及欄位之間的每一組兩兩組合，'
             '都能明確對應到唯一的臨床意義，並且能完整還原回原始病例代碼。'),
    ('依據', '《癌症部位特定因子編碼手冊》民國114年12月修訂，乳癌 pp.119-151；'
             '各欄位官方編碼範圍見 tcr_decoder/code_ranges.py'),
    ('欄位碼表 (Field_Coverage)',
     '涵蓋 10 個 SSF 欄位「全部」合法代碼，逐碼列出碼冊中文定義、'
     '本工具解碼出的臨床意義、以及反編碼回去的代碼。「往返一致」全部應為 TRUE。'),
    ('組合設計 (Case_Combinations)',
     '以 pairwise（all-pairs）方式設計的病例：任兩個欄位的任一組取值組合，'
     '至少出現在一個病例中。10 個欄位的全組合超過 10^20 種，無法窮舉；'
     '單碼已在 Field_Coverage 窮舉，此處保證的是「欄位之間的交互作用」全覆蓋。'),
    ('病例往返 (Case_Roundtrip)',
     '每個病例寫成真實的登記檔案列，走完整 TCRDecoder 解碼流程後，'
     '再用 TCREncoder 反編碼，逐欄位比對是否回到原始代碼。'),
    ('臨床檢視 (Clinical_View)',
     '每個病例解碼後的臨床欄位（ER/PR/HER2/Ki-67…），以及由該組合推導出的'
     '分子亞型與 NPI 預後分組，供人工核讀臨床意義是否合理。'),
    ('判讀方式', 'Summary 工作表的「失敗筆數」必須為 0；'
                 '任何一筆 False 都代表該代碼或組合無法無損雙向轉換。'),
    ('重新讀取注意事項',
     '代碼欄位在 Excel 中是「文字」，保留前置 0（例如 HER2 的 000 與 100 是不同的碼）。'
     '若用 pandas 讀回本檔，請加 dtype=str，否則 000 會被推斷成數字 0 而失去意義。'),
    ('重建方式', 'python -m tcr_decoder --build-validation <輸出檔.xlsx>'),
]


def export_validation_workbook(output_path: Union[str, Path],
                              cancer_group: str = 'breast') -> Dict[str, object]:
    """Build every sheet and write the validation workbook.

    Returns a summary dict (also written as the Summary sheet).
    """
    _check_group(cancer_group)
    field_cov = build_field_coverage(cancer_group)
    cases = build_case_combinations(cancer_group)
    raw = build_case_dataset(cases, cancer_group=cancer_group)
    per_field, clinical_view, _dec = validate_cases(raw, cancer_group=cancer_group)
    covered_pairs, total_pairs = pairwise_coverage(cases, cancer_group)

    summary = {
        '癌別': f'{cancer_group} ({get_ssf_profile(cancer_group).site_label})',
        '欄位數': len(SSF_KEYS),
        '涵蓋合法代碼數': int(len(field_cov)),
        '欄位往返失敗筆數': int((~field_cov['往返一致']).sum()),
        '欄位寬度錯誤筆數': int((~field_cov['寬度正確']).sum()),
        '病例數': int(len(cases)),
        '病例欄位比對筆數': int(len(per_field)),
        '病例往返失敗筆數': int((~per_field['往返一致']).sum()),
        '兩兩組合覆蓋': f'{covered_pairs}/{total_pairs}',
        '兩兩組合覆蓋率': f'{100.0 * covered_pairs / total_pairs:.1f}%',
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(str(out), engine='openpyxl') as writer:
        pd.DataFrame(_README_ROWS, columns=['項目', '說明']).to_excel(
            writer, sheet_name='說明', index=False)
        pd.DataFrame([{'項目': k, '數值': v} for k, v in summary.items()]).to_excel(
            writer, sheet_name='Summary', index=False)
        field_cov.to_excel(writer, sheet_name='Field_Coverage', index=False)
        cases.to_excel(writer, sheet_name='Case_Combinations', index=False)
        per_field.to_excel(writer, sheet_name='Case_Roundtrip', index=False)
        clinical_view.to_excel(writer, sheet_name='Clinical_View', index=False)
        failures = pd.concat([
            field_cov[~field_cov['往返一致']].assign(層級='欄位碼表'),
            per_field[~per_field['往返一致']].assign(層級='病例'),
        ], ignore_index=True)
        (failures if len(failures) else
         pd.DataFrame([{'結果': '無任何往返失敗'}])).to_excel(
            writer, sheet_name='Failures', index=False)

    summary['輸出檔案'] = str(out)
    return summary
