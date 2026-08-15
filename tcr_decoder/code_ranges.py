"""Official TCR code ranges (編碼範圍), transcribed from the printed manuals.

Every SSF field in the Cancer-SSF-Manual states a field width (欄位長度) and
an explicit list of legal codes (編碼範圍). This module holds that list for
the cancer groups whose profile has been verified field-by-field against the
manual, so two things can be checked mechanically instead of by eye:

  * decode() accepts every legal code and never silently drops one, and
  * encode() only ever emits a code the registry would actually accept
    (right characters AND right width -- '6' is not a legal Nottingham
    score code, '060' is).

`SUPPORTED_GROUPS` is the honest list of what is covered so far. See
docs/codebook_conformance_findings.md for the audit of the remaining cancer
groups, several of which do not yet match the manual and so are deliberately
absent here rather than half-transcribed.
"""

from typing import Dict, FrozenSet, Iterable, Tuple


def _num_range(lo: int, hi: int, width: int) -> Iterable[str]:
    return (str(i).zfill(width) for i in range(lo, hi + 1))


def _letter_range(letter: str, lo: int, hi: int) -> Iterable[str]:
    return (f'{letter}{i:02d}' for i in range(lo, hi + 1))


def _codes(*groups: Iterable[str]) -> FrozenSet[str]:
    out = set()
    for g in groups:
        out.update(g)
    return frozenset(out)


# Any SSF a site does not collect is coded 988 (manual p.1 收案條件).
_NOT_COLLECTED = _codes(('988',))


# ─────────────────────────────────────────────────────────────────────────────
# Breast (Cancer-SSF-Manual, 民國114年12月修訂, pp.119-151)
# ─────────────────────────────────────────────────────────────────────────────

_ER_PR_CODES = _codes(
    _num_range(0, 100, 3),                       # 000-100: % positive cells
    ('110', '111', '120', '121'),                # shared special codes
    ('888', '988', '999'),
    _letter_range('I', 0, 99),                   # intensity + proportion
    _letter_range('S', 0, 99),
    _letter_range('W', 0, 99),
)

_HER2_CODES = _codes(
    ('000', '004'),
    _num_range(100, 103, 3), _num_range(200, 202, 3), _num_range(300, 302, 3),
    _num_range(400, 402, 3), _num_range(500, 502, 3), _num_range(510, 512, 3),
    _num_range(520, 522, 3), _num_range(530, 532, 3), _num_range(590, 592, 3),
    _num_range(600, 602, 3), _num_range(640, 642, 3),
    ('888',), _num_range(900, 902, 3), ('988', '999'),
)

# ssf_key -> (field width, legal codes, manual page reference)
BREAST: Dict[str, Tuple[int, FrozenSet[str], str]] = {
    'SSF1':  (3, _ER_PR_CODES, 'p.121-124 動情激素接受體檢測 (ER)'),
    'SSF2':  (3, _ER_PR_CODES, 'p.125-128 黃體激素接受體檢測 (PR)'),
    'SSF3':  (3, _codes(('010', '011', '020', '030', '040', '988', '990', '999')),
              'p.129-130 前導性療法之療效'),
    'SSF4':  (3, _codes(_num_range(0, 89, 3), ('988', '996', '999')),
              'p.131 哨兵淋巴結檢查數目'),
    'SSF5':  (3, _codes(_num_range(0, 89, 3), ('988', '996', '999')),
              'p.132 哨兵淋巴結侵犯數目'),
    # The 編碼範圍 line on p.134 omits 988, but the code table on p.141 lists
    # it (不適用: sarcoma / phyllodes / 原位癌), so it is legal.
    'SSF6':  (3, _codes(('030', '040', '050', '060', '070', '080', '090',
                         '110', '120', '130', '988', '999')),
              'p.134/141 Nottingham 或 BR 的分數/級數'),
    'SSF7':  (3, _HER2_CODES, 'p.136-145 HER2 IHC/ISH'),
    'SSF8':  (3, _codes(('000', '010', '988', '999')), 'p.142/148 Paget 氏症'),
    'SSF9':  (3, _codes(('000', '010', '988', '990', '999')),
              'p.143 淋巴管或血管侵犯 (LVI)'),
    'SSF10': (3, _codes(_letter_range('A', 0, 9), _num_range(0, 100, 3),
                        ('988', '998', '999')),
              'p.144/151 Ki-67'),
}

# ─────────────────────────────────────────────────────────────────────────────
# Prostate (Cancer-SSF-Manual, 民國114年12月修訂, pp.175-191)
# ─────────────────────────────────────────────────────────────────────────────

# Gleason primary+secondary pattern: tens digit = primary 1-5, units digit =
# secondary 1-5 or 9 (unknown); 099 = both unknown.
_GLEASON_PATTERN_CODES = _codes(
    (f'{p}{s}'.zfill(3) for p in range(1, 6) for s in list(range(1, 6)) + [9]),
    ('099', '988', '999'),
)
_GLEASON_SCORE_CODES = _codes(_num_range(2, 10, 3), ('988', '999'))

PROSTATE: Dict[str, Tuple[int, FrozenSet[str], str]] = {
    # 001-999 with NO 000: 001 = <=0.1 ng/mL.
    'SSF1':  (3, _codes(_num_range(1, 999, 3)), 'p.183-184 攝護腺特異抗原指數 (PSA)'),
    'SSF2':  (3, _GLEASON_PATTERN_CODES,
              'p.179-180 細針切片/TURP 檢體 Gleason 主要與次要型態級數'),
    'SSF3':  (3, _GLEASON_SCORE_CODES, 'p.181-182 細針切片/TURP 檢體 Gleason 氏分數'),
    'SSF4':  (3, _GLEASON_PATTERN_CODES,
              'p.183-184 攝護腺全切除/大體解剖檢體 Gleason 主要與次要型態級數'),
    'SSF5':  (3, _GLEASON_SCORE_CODES, 'p.185-186 攝護腺全切除/解剖檢體 Gleason 氏分數'),
    'SSF6':  (3, _codes(_num_range(1, 100, 3), ('988', '999')),
              'p.187 切片條數檢查數目'),
    'SSF7':  (3, _codes(_num_range(0, 100, 3), ('988', '998', '999')),
              'p.188 切片條數陽性數目'),
    'SSF8':  (3, _codes(('000', '010', '020', '030', '040', '050', '988', '999')),
              'p.191 臨床腫瘤期別 (T-stage) 診斷方式'),
    # SSF9/SSF10 are not collected for prostate: the manual's rule is that any
    # SSF a site does not collect is coded 988.
    'SSF9':  (3, _codes(('988',)), 'p.1 收案條件：未收錄之欄位一律編碼 988'),
    'SSF10': (3, _codes(('988',)), 'p.1 收案條件：未收錄之欄位一律編碼 988'),
}

# ─────────────────────────────────────────────────────────────────────────────
# Uterine corpus / endometrium (Cancer-SSF-Manual, pp.151-161)
# ─────────────────────────────────────────────────────────────────────────────

ENDOMETRIUM: Dict[str, Tuple[int, FrozenSet[str], str]] = {
    'SSF1':  (3, _ER_PR_CODES, 'p.153 動情激素接受體檢測 (ER)'),
    'SSF2':  (3, _ER_PR_CODES, 'p.155 黃體激素接受體檢測 (PR)'),
    # 987 is listed in the 編碼範圍 line on p.157 but has no row in the code
    # table on p.163; kept legal and decoded as explicitly undefined.
    'SSF3':  (3, _codes(_num_range(1, 3, 3), ('987', '988', '999')),
              'p.157 非內膜細胞組織型態混合百分比 / FIGO grade'),
    'SSF4':  (3, _codes(('010', '020', '030', '999')), 'p.158 POLE 基因突變'),
    'SSF5':  (3, _codes(('000', '010', '020', '988', '999')), 'p.159 微星體不穩定檢測'),
    'SSF6':  (3, _codes(('010', '020', '030', '999')), 'p.161 p53 腫瘤抑制蛋白檢測'),
    'SSF7':  (3, _NOT_COLLECTED, 'p.1 收案條件：未收錄之欄位一律編碼 988'),
    'SSF8':  (3, _NOT_COLLECTED, 'p.1 收案條件：未收錄之欄位一律編碼 988'),
    'SSF9':  (3, _NOT_COLLECTED, 'p.1 收案條件：未收錄之欄位一律編碼 988'),
    'SSF10': (3, _NOT_COLLECTED, 'p.1 收案條件：未收錄之欄位一律編碼 988'),
}

# ─────────────────────────────────────────────────────────────────────────────
# Cervix (pp.147-150), Stomach (pp.37-43), Liver (pp.79-90),
# Lung (pp.103-117), Colorectum (pp.45-76)
# ─────────────────────────────────────────────────────────────────────────────

# CEA / SCC lab value share one scheme: 001 = <=0.1, 002-986 = value x10,
# 987 = >=98.7, 988 = not applicable, 999 = unknown.
_LAB_VALUE_X10_CODES = _codes(_num_range(1, 988, 3), ('999',))
# "vs normal range" companion field.
_LAB_VS_NORMAL_CODES = _codes(('010', '020', '030', '988', '999'))

CERVIX: Dict[str, Tuple[int, FrozenSet[str], str]] = {
    'SSF1':  (3, _LAB_VALUE_X10_CODES, 'p.149 SCC 抗原檢驗值'),
    'SSF2':  (3, _LAB_VS_NORMAL_CODES, 'p.150 SCC 抗原檢驗正常值'),
    **{f'SSF{i}': (3, _NOT_COLLECTED,
                   'p.1 收案條件：子宮頸癌僅收錄 SSF1-2，其餘一律編碼 988')
       for i in range(3, 11)},
}

STOMACH: Dict[str, Tuple[int, FrozenSet[str], str]] = {
    'SSF1':  (3, _LAB_VALUE_X10_CODES, 'p.39 癌胚抗原 CEA 檢驗值'),
    'SSF2':  (3, _LAB_VS_NORMAL_CODES, 'p.40 癌胚抗原 CEA 檢驗正常值'),
    'SSF3':  (3, _codes(_num_range(0, 8, 3), ('010', '988', '999')),
              'p.41 幽門螺旋桿菌'),
    'SSF4':  (3, _codes(_num_range(0, 980, 3), ('988', '998', '999')),
              'p.42 病理報告中的腫瘤深度'),
    'SSF5':  (3, _codes(('000', '010', '988', '990', '999')),
              'p.43 淋巴管或血管侵犯'),
    **{f'SSF{i}': (3, _NOT_COLLECTED,
                   'p.1 收案條件：胃癌僅收錄 SSF1-5，其餘一律編碼 988')
       for i in range(6, 11)},
}

LIVER: Dict[str, Tuple[int, FrozenSet[str], str]] = {
    'SSF1':  (3, _codes(_letter_range('A', 0, 99), _num_range(0, 987, 3),
                        ('988',), _num_range(991, 993, 3), ('999',)),
              'p.81 AFP 甲型胎兒蛋白檢驗值'),
    'SSF2':  (3, _codes(_num_range(0, 8, 3), ('988', '999')), 'p.83 肝纖維化的程度'),
    'SSF3':  (3, _codes(('105', '106', '199', '207', '208', '209', '299',
                         '310', '311', '312', '313', '314', '315', '399', '999')),
              'p.84 Child-Pugh 分數'),
    'SSF4':  (3, _codes(_num_range(1, 988, 3), ('999',)), 'p.86 肌酸(酐) 檢驗值'),
    'SSF5':  (3, _codes(_num_range(1, 988, 3), ('999',)), 'p.87 總膽紅素檢驗值'),
    'SSF6':  (3, _codes(_num_range(1, 60, 3), ('997', '988', '999')),
              'p.88 凝血酶原時間國際正常比值 (INR)'),
    'SSF7':  (3, _codes(('000', '001', '010', '011', '020', '999')),
              'p.89 B 型肝炎表面抗原 (HBsAg)'),
    'SSF8':  (3, _codes(('000', '001', '010', '011', '020', '999')),
              'p.90 C 型肝炎抗體 (Anti-HCV)'),
    **{f'SSF{i}': (3, _NOT_COLLECTED,
                   'p.1 收案條件：肝癌僅收錄 SSF1-8，其餘一律編碼 988')
       for i in range(9, 11)},
}

# Lung SSF6 (EGFR) is a 3-letter combinatorial code: each position is one of
# the mutation letters A-U (or V/X/Z for the special whole-code values), so
# the legal set is every 3-letter combination the decoder recognises plus
# VVV/XXX/ZZZ/999. Built from the decoder's own letter table.
def _lung_egfr_codes() -> FrozenSet[str]:
    """AAA-UXX, VVV, XXX, ZZZ, 999.

    The field records a SET of up to three concurrent mutations, written as
    mutation letters first and X-padded on the right ("AAA-UXX" in the
    manual's range shorthand). Codes with an X in the middle ('AXA') are not
    canonical -- they decode to the same two mutations as 'AAX', so they are
    not part of the legal set; encode always produces the padded-right form.
    """
    from tcr_decoder.ssf_registry import _LUNG_EGFR_LETTER_MAP

    mutations = sorted(k for k in _LUNG_EGFR_LETTER_MAP
                       if k not in ('V', 'X', 'Z'))
    out = {'VVV', 'XXX', 'ZZZ', '999'}
    for a in mutations:
        out.add(f'{a}XX')
        for b in mutations:
            out.add(f'{a}{b}X')
            for c in mutations:
                out.add(f'{a}{b}{c}')
    return frozenset(out)


LUNG: Dict[str, Tuple[int, FrozenSet[str], str]] = {
    'SSF1':  (3, _codes(('000', '010', '020', '030', '040', '999')),
              'p.105 同側肺但非主病灶的另外肺腫瘤'),
    'SSF2':  (3, _codes(('000', '010', '020', '030', '040', '988', '999')),
              'p.106 波及臟層膜 / 彈性層'),
    'SSF3':  (3, _codes(_num_range(0, 5, 3), ('988', '998', '999')),
              'p.107 首次治療前生活功能狀態的評估'),
    'SSF4':  (3, _codes(('000',), _num_range(11, 15, 3), ('988', '999')),
              'p.108 惡性肋膜積水'),
    'SSF5':  (3, _codes(_num_range(0, 8, 3), ('988', '999')),
              'p.109 縱膈腔淋巴結取樣或廓清'),
    'SSF6':  (3, None, 'p.111 EGFR 基因突變'),          # filled in below
    'SSF7':  (3, _codes(('010', '020', '030', '999')), 'p.113 ALK 基因轉位突變'),
    'SSF8':  (3, _codes(_num_range(0, 7, 3), ('988', '999')),
              'p.114 特定腺癌腫瘤組成'),
    'SSF9':  (3, _codes(_num_range(2, 21, 3), ('988', '999')), 'p.117 腫瘤顆數'),
    'SSF10': (3, _NOT_COLLECTED,
              'p.1 收案條件：肺癌僅收錄 SSF1-9，SSF10 一律編碼 988'),
}

COLORECTUM: Dict[str, Tuple[int, FrozenSet[str], str]] = {
    # C18/C21 (colon) and C19/C20 (rectosigmoid/rectum) differ slightly:
    # rectum adds 990/991 to SSF5 and has SSF9; colon adds 980 to SSF8. One
    # profile serves both, so the legal set is the union of the two schemas.
    'SSF1':  (3, _LAB_VALUE_X10_CODES, 'p.47/63 癌胚抗原 CEA 檢驗值'),
    'SSF2':  (3, _LAB_VS_NORMAL_CODES, 'p.48/64 癌胚抗原 CEA 檢驗正常值'),
    'SSF3':  (3, _codes(('000', '010', '020', '030', '988', '990', '999')),
              'p.49/65 腫瘤縮小等級或分數'),
    'SSF4':  (3, _codes(_num_range(0, 980, 3), _num_range(990, 996, 3),
                        ('988', '999')),
              'p.51/67 病理環切緣 (CRM)'),
    'SSF5':  (3, _codes(_num_range(0, 2, 3), ('006', '007', '988', '990', '991', '999')),
              'p.53/69 BRAF 基因突變（990/991 僅直腸適用）'),
    # "008-988" is range shorthand: the real codes are
    # {KRAS digit}{NRAS digit}8 with each digit in 0-7 or 9 (p.54), plus the
    # two whole-field sentinels.
    'SSF6':  (3, _codes((f'{k}{n}8' for k in '012345679' for n in '012345679'),
                        ('988', '998')),
              'p.54/70 RAS 基因突變'),
    'SSF7':  (3, _codes(('000', '010', '988', '999')), 'p.57/73 腸阻塞'),
    'SSF8':  (3, _codes(('000', '010', '980', '988', '999')),
              'p.58/74 腸穿孔（980 僅結腸適用）'),
    'SSF9':  (3, _codes(_num_range(0, 150, 3), ('988', '991', '992', '993', '999')),
              'p.75 直腸腫瘤下緣與肛門口的距離（僅 C19/C20）'),
    'SSF10': (3, _codes(('000', '010', '020', '988', '999')),
              'p.59/76 微星體不穩定檢測 (MSI)'),
}

LUNG['SSF6'] = (3, _lung_egfr_codes(), 'p.111 EGFR 基因突變')


# ─────────────────────────────────────────────────────────────────────────────
# Esophagus (pp.31-36), Pancreas (pp.91-101), Ovary (pp.163-167),
# Bladder (pp.169-173)
# ─────────────────────────────────────────────────────────────────────────────

ESOPHAGUS: Dict[str, Tuple[int, FrozenSet[str], str]] = {
    'SSF1': (3, _codes(('000', '020', '030', '040', '988', '999')),
             'p.33 正子掃描電腦斷層檢查'),
    'SSF2': (3, _codes(('000', '010', '988', '999')), 'p.34 使用微創手術切除'),
    'SSF3': (3, _codes(_num_range(0, 3, 3),
                       ('010', '020', '030', '040', '988', '990', '999')),
             'p.35 前導性療法後的病理反應'),
    'SSF4': (3, _codes(('010', '020', '030', '040', '988', '990', '999')),
             'p.36 放射治療後的臨床反應'),
    **{f'SSF{i}': (3, _NOT_COLLECTED,
                   'p.1 收案條件：食道癌僅收錄 SSF1-4，其餘一律編碼 988')
       for i in range(5, 11)},
}

PANCREAS: Dict[str, Tuple[int, FrozenSet[str], str]] = {
    'SSF1': (3, _LAB_VALUE_X10_CODES, 'p.93 癌胚抗原 CEA 檢驗值'),
    'SSF2': (3, _LAB_VS_NORMAL_CODES, 'p.94 癌胚抗原 CEA 檢驗正常值'),
    'SSF3': (3, _codes(_num_range(0, 997, 3), ('999',)),
             'p.95 Carbohydrate Antigen 19-9'),
    'SSF4': (3, _codes(_letter_range('A', 0, 9), _num_range(0, 100, 3),
                       ('988', '998', '999')),
             'p.97 Ki-67（與乳癌 SSF10 同碼制）'),
    'SSF5': (3, _codes(_num_range(0, 21, 3), ('110', '120', '130', '988', '999')),
             'p.98 Mitotic Count'),
    'SSF6': (3, _codes(_num_range(1, 99, 3), _num_range(101, 199, 3),
                       ('988', '999')),
             'p.100 糖化血色素 HbA1c'),
    **{f'SSF{i}': (3, _NOT_COLLECTED,
                   'p.1 收案條件：胰臟癌僅收錄 SSF1-6，其餘一律編碼 988')
       for i in range(7, 11)},
}

_CA125_CODES = _codes(_num_range(1, 910, 3), ('920', '930', '931', '988', '999'))

OVARY: Dict[str, Tuple[int, FrozenSet[str], str]] = {
    'SSF1': (3, _CA125_CODES, 'p.165 療前 CA 125 檢驗值'),
    'SSF2': (3, _CA125_CODES, 'p.166 療後最低 CA 125 檢驗值'),
    'SSF3': (3, _codes(('000', '010', '020', '030', '040', '988', '990',
                        '991', '999')),
             'p.167 腫瘤手術後之殘存腫瘤狀態及大小'),
    **{f'SSF{i}': (3, _NOT_COLLECTED,
                   'p.1 收案條件：卵巢癌僅收錄 SSF1-3，其餘一律編碼 988')
       for i in range(4, 11)},
}

BLADDER: Dict[str, Tuple[int, FrozenSet[str], str]] = {
    'SSF1': (3, _codes(('010', '020', '988', '999')), 'p.171 WHO/ISUP 分級'),
    'SSF2': (3, _codes(('000', '010', '020', '030', '988', '999')),
             'p.172 區域淋巴結夾膜外侵犯情形'),
    'SSF3': (3, _codes(('000', '010', '988', '999')), 'p.173 固有肌肉層病理標本'),
    **{f'SSF{i}': (3, _NOT_COLLECTED,
                   'p.1 收案條件：膀胱癌僅收錄 SSF1-3，其餘一律編碼 988')
       for i in range(4, 11)},
}


# ─────────────────────────────────────────────────────────────────────────────
# Head & neck (Cancer-SSF-Manual pp.3-29)
# ─────────────────────────────────────────────────────────────────────────────

# SSF3-SSF6 use one positional scheme: each character is one nodal region,
# 0 = not involved, 1 = involved, 8 = cross-region, cannot localise.
_HN_LEVEL_CODES = _codes(
    (f'{a}{b}{c}' for a in '018' for b in '018' for c in '018'),
    ('988', '999'),
)

# SSF9 is a composite: 1st char = overall clinical ENE call (0/1/8/9),
# 2nd = imaging finding (0/1/2/9), 3rd = physical exam finding (0/1/2/9).
_HN_ENE_CLINICAL_CODES = _codes(
    (f'{a}{b}{c}' for a in '0189' for b in '0129' for c in '0129'),
    ('988', '998'),
)

# SSF10: 000, 101-120, 199, 210, 221-298, 299, 399, 988, 998, 999.
_HN_ENE_PATH_CODES = _codes(
    ('000',), _num_range(101, 120, 3), ('199', '210'),
    _num_range(221, 298, 3), ('299', '399', '988', '998', '999'),
)

HEAD_NECK: Dict[str, Tuple[int, FrozenSet[str], str]] = {
    'SSF1':  (3, _codes(_num_range(0, 988, 3), _num_range(990, 997, 3), ('999',)),
              'p.7-8 被侵犯的頸部淋巴結的大小'),
    'SSF2':  (3, _codes(('000', '001', '002', '005', '988', '999')),
              'p.10-11 頸部淋巴結莢膜外侵犯情形'),
    'SSF3':  (3, _HN_LEVEL_CODES, 'p.12-13 頸部第 I-III 區淋巴結侵犯範圍'),
    'SSF4':  (3, _HN_LEVEL_CODES, 'p.14-15 頸部第 IV-V 區及後咽區淋巴結侵犯範圍'),
    'SSF5':  (3, _HN_LEVEL_CODES, 'p.16-17 頸部第 VI-VII 區及顏面淋巴結侵犯範圍'),
    'SSF6':  (3, _HN_LEVEL_CODES, 'p.18-19 側咽/腮腺/後枕耳後區淋巴結侵犯範圍'),
    'SSF7':  (3, _codes(_num_range(0, 980, 3),
                        ('987', '988', '990', '997', '998', '999')),
              'p.20-21 病理報告中的腫瘤深度'),
    'SSF8':  (3, _codes(_num_range(0, 980, 3),
                        ('987', '988', '990', '998', '999')),
              'p.22-23 腫瘤細胞與手術切緣的最近距離'),
    # SSF9 has no 編碼範圍 line in the manual; the legal set is the product of
    # its three documented character alphabets plus the two whole-field codes.
    'SSF9':  (3, _HN_ENE_CLINICAL_CODES, 'p.24-25/30-31 臨床淋巴結外侵犯狀態'),
    'SSF10': (3, _HN_ENE_PATH_CODES, 'p.29-30 病理淋巴結外侵犯狀態'),
}


# ─────────────────────────────────────────────────────────────────────────────
# Thyroid: NOT an SSF-collecting site (manual p.1), so every SSF is 988.
# ─────────────────────────────────────────────────────────────────────────────

THYROID: Dict[str, Tuple[int, FrozenSet[str], str]] = {
    f'SSF{i}': (3, _NOT_COLLECTED,
                'p.1 收案條件：非收錄部位，SSF1-SSF20 應編碼 988')
    for i in range(1, 11)
}

# ─────────────────────────────────────────────────────────────────────────────
# Lymphoma (Cancer-SSF-Manual, 民國114年12月修訂, pp.194-206)
#
# Selected by MORPHOLOGY, not by primary site. Which SSFs a case must report
# depends on its M-code bucket (p.194); every field still has to accept its
# whole 編碼範圍, so the ranges below are per field, not per bucket.
# ─────────────────────────────────────────────────────────────────────────────

# 000/001 = not tested (with / without a history), 010/011 = tested negative
# (with / without a history), 020 = tested positive. Same shape for HBsAg and
# anti-HCV, in both the lymphoma and the leukemia chapter.
_HEPATITIS_SEROLOGY = _codes(
    ('000', '001', '010', '011', '020', '988', '999'))

_LYM_ESR_HEADS = tuple(f'{i:02d}' for i in range(1, 52)) + ('98', '99')
_LYM_IPS_TAILS = tuple(str(i) for i in range(10))
_LYM_ESR_IPS = _codes(h + t for h in _LYM_ESR_HEADS for t in _LYM_IPS_TAILS)

LYMPHOMA: Dict[str, Tuple[int, FrozenSet[str], str]] = {
    'SSF1':  (3, _codes(('001', '002', '988', '999')),
              'p.195 後天人類免疫不全病毒感染狀況 (HIV)'),
    'SSF2':  (3, _codes(('000', '010', '988', '999')),
              'p.196 診斷時全身性之症狀 (B symptoms)'),
    'SSF3':  (3, _codes(_num_range(0, 5, 3), ('988',),
                        _num_range(990, 994, 3), ('999',)),
              'p.197-198 IPI score'),
    # 990-992 mean different bands here than in SSF3: FLIPI has three bands
    # (low / intermediate / high), IPI has five.
    'SSF4':  (3, _codes(_num_range(0, 5, 3), ('988',),
                        _num_range(990, 992, 3), ('999',)),
              'p.199 FLIPI score'),
    # The 編碼範圍 line on p.200 reads "988" alone, but the code table on the
    # same page defines 000/001/002/999 too. Both are accepted: a real HTLV-1
    # result must still decode and round-trip.
    'SSF5':  (3, _codes(_num_range(0, 2, 3), ('988', '999')),
              'p.200 HTLV-1 感染狀況（編碼範圍與碼表不一致，取聯集）'),
    'SSF6':  (3, _codes(_num_range(0, 3, 3), ('988', '999')),
              'p.201 巨細胞病毒感染狀況 (CMV)'),
    'SSF7':  (3, _HEPATITIS_SEROLOGY, 'p.202 B 型肝炎表面抗原 (HBsAg)'),
    'SSF8':  (3, _HEPATITIS_SEROLOGY, 'p.203 C 型肝炎抗體 (Anti-HCV)'),
    'SSF9':  (3, _codes(('001', '002', '988', '999')),
              'p.204 急性肝炎發作'),
    'SSF10': (3, _LYM_ESR_IPS,
              'p.205-206 何杰金氏淋巴瘤預後因子 ESR (第1-2碼) 與 IPS (第3碼)'),
}

# ─────────────────────────────────────────────────────────────────────────────
# Leukemia (Cancer-SSF-Manual, 民國114年12月修訂, pp.207-222)
# ─────────────────────────────────────────────────────────────────────────────

def _post_treatment(base: Iterable[str]) -> FrozenSet[str]:
    """8XX: '8' plus the last two digits of a pre-treatment finding code.

    p.209/211 -- a chromosome or molecular study done after chemotherapy,
    immunotherapy or targeted therapy reports the same finding shifted into
    the 800 block, so t(8;21) is 001 before treatment and 801 after it.
    """
    return frozenset('8' + c[-2:] for c in base)


_LEU_KARYOTYPE_CODES = _codes(
    _num_range(0, 7, 3), ('013',), _num_range(21, 27, 3),
    ('041', '042', '051', '061'), _num_range(90, 92, 3),
)

_LEU_MOLECULAR_CODES = _codes(
    _num_range(0, 13, 3), _num_range(21, 25, 3), ('041', '042'),
    _num_range(51, 55, 3), _num_range(90, 92, 3),
)

_LEU_MRD_HEADS = tuple(f'{i:02d}' for i in range(25)) + ('98', '99')
_LEU_MRD_TAILS = ('0', '1', '2', '3', '4', '5', '6', '8', '9')
_LEU_MRD = _codes(h + t for h in _LEU_MRD_HEADS for t in _LEU_MRD_TAILS)

LEUKEMIA: Dict[str, Tuple[int, FrozenSet[str], str]] = {
    'SSF1':  (3, _codes(_LEU_KARYOTYPE_CODES,
                        _post_treatment(_LEU_KARYOTYPE_CODES),
                        ('988', '998', '999')),
              'p.209-210 白血病染色體檢查'),
    # The 編碼範圍 line on p.211 stops at 090-091, but the code table on p.212
    # defines 092 (三種以上異常). Included, or a legally-coded 092 would be
    # rejected as illegal.
    'SSF2':  (3, _codes(_LEU_MOLECULAR_CODES,
                        _post_treatment(_LEU_MOLECULAR_CODES),
                        ('988', '998', '999')),
              'p.211-212 白血病分子生物學檢查（碼表另有 092，已納入）'),
    'SSF3':  (3, _codes(('001', '002', '988', '990', '999')),
              'p.213 首次前導化學治療後反應'),
    'SSF4':  (3, _codes(('000',), _num_range(10, 14, 3), ('988', '999')),
              'p.214-215 急性移植體對抗宿主疾病 (aGVHD)'),
    'SSF5':  (3, _codes(_num_range(0, 3, 3), ('988', '999')),
              'p.216 慢性移植體對抗宿主疾病 (cGVHD)'),
    # Leukemia's CMV table has no 000: "not tested" is folded into 999,
    # unlike the lymphoma chapter's version of the same field.
    'SSF6':  (3, _codes(_num_range(1, 3, 3), ('988', '999')),
              'p.217 巨細胞病毒感染狀況 (CMV)'),
    'SSF7':  (3, _HEPATITIS_SEROLOGY, 'p.218 B 型肝炎表面抗原 (HBsAg)'),
    'SSF8':  (3, _HEPATITIS_SEROLOGY, 'p.219 C 型肝炎抗體 (Anti-HCV)'),
    'SSF9':  (3, _codes(('001', '002', '988', '999')),
              'p.220 急性肝炎發作'),
    'SSF10': (3, _LEU_MRD,
              'p.221-222 最近一次治療反應的微量殘餘疾病（第1-2碼月數、'
              '第3碼 log reduction）'),
}

# Longform-Manual structural fields. Only fields with a real bidirectional
# decoder/encoder pair are listed; the rest of the 99 Longform fields are
# still passthrough and have nothing to check yet.
LONGFORM: Dict[str, Tuple[int, FrozenSet[str], str]] = {
    'LNEXAM':    (2, _codes(_num_range(0, 90, 2), _num_range(95, 99, 2)),
                  'Longform p.127-129 區域淋巴結檢查數目'),
    'LN_POSITI': (2, _codes(_num_range(0, 90, 2), ('95', '97', '98', '99')),
                  'Longform p.130-131 區域淋巴結侵犯數目'),
    'LAT95':     (1, _codes(_num_range(0, 5, 1), ('9',)),
                  'Longform p.87-88 側性 (#2.7)'),
    'MCODE5':    (1, _codes(('2', '3')),
                  'Longform p.91-92 性態碼 (#2.9)'),
    # Code 3 is legal only for the haematolymphoid morphologies (M9590-9993);
    # the solid-tumour table on p.102 has no 3. The 編碼範圍 line states 1-9
    # for the field as a whole, so 3 is legal here and decode picks the table
    # from MCODE.
    'CONFER':    (1, _codes(_num_range(1, 9, 1)),
                  'Longform p.102-104 癌症確診方式 (#2.11)'),
    'PNI':       (1, _codes(('0', '1', '7', '8', '9')),
                  'Longform p.114-115 神經侵襲 (#2.13.1)'),
    'LVI':       (1, _codes(('0', '1', '7', '8', '9')),
                  'Longform p.117-118 淋巴管或血管侵犯 (#2.13.2)'),
    # 首次療程的全身性治療 (#4.3.x). Both halves of each pair share the
    # modality codes; only the reporting hospital carries the 8x block for
    # "planned but not given, and here is why".
    'PREC':      (2, _codes(_num_range(0, 13, 2), _num_range(20, 21, 2),
                            _num_range(30, 31, 2), ('99',)),
                  'Longform p.268-269 外院化學治療 (#4.3.2)'),
    'C':         (2, _codes(_num_range(0, 13, 2), _num_range(20, 21, 2),
                            _num_range(30, 31, 2), _num_range(81, 83, 2),
                            _num_range(85, 88, 2), ('99',)),
                  'Longform p.271-273 申報醫院化學治療 (#4.3.3)'),
    'PREH':      (2, _codes(_num_range(0, 3, 2), _num_range(20, 21, 2),
                            _num_range(30, 31, 2), ('99',)),
                  'Longform p.277-278 外院荷爾蒙/類固醇治療 (#4.3.5)'),
    'H':         (2, _codes(_num_range(0, 3, 2), _num_range(20, 21, 2),
                            _num_range(30, 31, 2), _num_range(82, 83, 2),
                            _num_range(85, 88, 2), ('99',)),
                  'Longform p.279-281 申報醫院荷爾蒙/類固醇治療 (#4.3.6)'),
    'PREI':      (2, _codes(_num_range(0, 7, 2), _num_range(20, 23, 2),
                            _num_range(30, 33, 2), _num_range(40, 41, 2),
                            ('99',)),
                  'Longform p.282-284 外院免疫治療 (#4.3.8)'),
    'I':         (2, _codes(_num_range(0, 7, 2), _num_range(20, 23, 2),
                            _num_range(30, 33, 2), _num_range(40, 41, 2),
                            _num_range(82, 83, 2), _num_range(85, 88, 2),
                            ('99',)),
                  'Longform p.285-287 申報醫院免疫治療 (#4.3.9)'),
    'PRETAR':    (2, _codes(_num_range(0, 1, 2), _num_range(20, 21, 2),
                            _num_range(30, 31, 2), ('99',)),
                  'Longform p.292-293 外院標靶治療 (#4.3.13)'),
    'TAR':       (2, _codes(_num_range(0, 1, 2), _num_range(20, 21, 2),
                            _num_range(30, 31, 2), _num_range(82, 83, 2),
                            _num_range(85, 88, 2), ('99',)),
                  'Longform p.294-296 申報醫院標靶治療 (#4.3.14)'),
    'OTH':       (2, _codes(_num_range(0, 3, 2), ('99',)),
                  'Longform p.301-302 其他治療 (#4.5.1)'),
    'PREP':      (1, _codes(_num_range(0, 7, 1), ('9',)),
                  'Longform p.298-300 申報醫院緩和照護 (#4.4)'),
    'PRESLNSCO': (1, _codes(_num_range(0, 7, 1), ('9',)),
                  'Longform p.203 外院區域淋巴結手術範圍'),
    'SLNSCO95':  (1, _codes(_num_range(0, 7, 1), ('9',)),
                  'Longform p.207 申報醫院區域淋巴結手術範圍'),
}

# Appendix B surgery codes are SITE-SPECIFIC. The 編碼範圍 line is the same
# for every site (000, 100-800, 900, 980, 990 -- Longform-Manual p.186/188),
# but which of those codes exist, and what they mean, comes from the site's
# own Appendix B table: all 30 of them live in tcr_decoder/surgery_codes.py,
# generated from the manual by scripts/generate_surgery_codes.py.
SURGERY_FIELD_WIDTH = 3


def legal_surgery_codes(tcode1) -> FrozenSet[str]:
    """Legal PRESTYPE / STYPE95 codes for one primary site."""
    from tcr_decoder.surgery_codes import legal_surgery_codes as _lookup
    return _lookup(tcode1)


CODE_RANGES: Dict[str, Dict[str, Tuple[int, FrozenSet[str], str]]] = {
    'breast': BREAST,
    'prostate': PROSTATE,
    'endometrium': ENDOMETRIUM,
    'thyroid': THYROID,
    'cervix': CERVIX,
    'stomach': STOMACH,
    'liver': LIVER,
    'lung': LUNG,
    'colorectum': COLORECTUM,
    'head_neck': HEAD_NECK,
    'esophagus': ESOPHAGUS,
    'pancreas': PANCREAS,
    'ovary': OVARY,
    'bladder': BLADDER,
    'lymphoma': LYMPHOMA,
    'leukemia': LEUKEMIA,
    '_longform': LONGFORM,
}

SUPPORTED_GROUPS = ('breast', 'prostate', 'endometrium', 'thyroid',
                    'cervix', 'stomach', 'liver', 'lung', 'colorectum',
                    'head_neck', 'esophagus', 'pancreas', 'ovary', 'bladder',
                    'lymphoma', 'leukemia')


def legal_codes(cancer_group: str, ssf_key: str) -> FrozenSet[str]:
    """Legal codes for one field, or an empty set if not transcribed yet."""
    entry = CODE_RANGES.get(cancer_group, {}).get(ssf_key)
    return entry[1] if entry else frozenset()


def field_width(cancer_group: str, ssf_key: str) -> int:
    """Official field width in characters, or 0 if not transcribed yet."""
    entry = CODE_RANGES.get(cancer_group, {}).get(ssf_key)
    return entry[0] if entry else 0


def is_legal_code(cancer_group: str, ssf_key: str, code: str) -> bool:
    """True if `code` is listed in this field's 編碼範圍.

    Returns True for a field with no transcribed range (nothing to check
    against) and for a blank value (an empty cell is not an illegal code).
    """
    codes = legal_codes(cancer_group, ssf_key)
    if not codes:
        return True
    s = str(code).strip()
    return not s or s in codes
