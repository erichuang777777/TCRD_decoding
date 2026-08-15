
# -*- coding: utf-8 -*-
"""Official Chinese wording for the Longform structural fields.

Mirrors ssf_registry's zh_definition() for the site-specific factors: the
FHIR CodeSystem `display` must be the code book's own Chinese text, with the
English label as an `en` designation -- not the other way around, which is
what tcr_workbench.ig_export._structural_concepts() did until this file
existed (there was nothing for it to call).

Keyed by raw field tag (LNEXAM, SEX, RTAR, ...) then by the SAME key the
field's CodeMap/decoder uses, so a caller can always zip one against the
other. The additive fields (RTAR/RMOD/HTAR/LTAR/SEQRS/SEQLS) store Chinese
COMPONENT names only, not one entry per combination -- combinations are
built the same way the English label is, by joining the components that sum
to the code, so this file only needs the handful of Chinese phrases the
manual actually prints rather than transcribing every subset sum by hand.
"""

from typing import Dict

# ─────────────────────────────────────────────────────────────────────────────
# 側性／性態碼／確診方式／神經侵襲／LVI (p.87-118)
# ─────────────────────────────────────────────────────────────────────────────

ZH_LAT95: Dict[int, str] = {
    0: '不是成對器官。',
    1: '原發起源位在右側。',
    2: '原發起源位在左側。',
    3: '成對器官，只有單側侵犯，但起源於左側或右側則不清楚。',
    4: '成對器官，雙側侵犯但起源之側位不清楚，且病歷描述是單一原發。',
    5: '原發部位為身體中線的腫瘤（成對器官）。',
    9: '成對器官，但其側位不清楚。',
}

ZH_MCODE5: Dict[int, str] = {
    2: '原位（in situ），不具侵襲性。',
    3: '侵襲性（invasive）或微侵襲。',
}

_ZH_CONFER_SHARED: Dict[int, str] = {
    1: '組織病理學確診(Positive histology)。',
    2: '細胞學確診(Positive cytology)。',
    4: '經顯微鏡檢證實，但未描述其確診方式。',
    6: '直接察看診斷為癌症，但未經顯微鏡檢證實。',
    7: '放射線或其他影像學檢查診斷為癌症，但未經顯微鏡檢證實。',
    8: '僅臨床診斷(不包含5、6、7)。',
    9: '不清楚是否經顯微鏡檢證實。',
}
ZH_CONFER_SOLID: Dict[int, str] = {
    **_ZH_CONFER_SHARED,
    5: '實驗室檢驗/腫瘤標記檢查為陽性，但未經顯微鏡檢證實。',
}
ZH_CONFER_HAEM: Dict[int, str] = {
    **_ZH_CONFER_SHARED,
    3: '組織病理學確診加上免疫表現分型陽性同時/或基因檢測為陽性。',
    5: '實驗室檢驗/腫瘤標記檢查為陽性，或免疫表現分型陽性同時/或基因檢測為陽性。',
}


def _zh_invasion(subject: str) -> Dict[int, str]:
    return {
        0: f'無{subject}。',
        1: f'有{subject}。',
        7: '無法判斷是否侵犯（NA、樣本太小、樣本不足、未描述、無惡性細胞）。',
        8: '不適用（GIST、NET、高度分化不良、原位癌、淋巴瘤及白血病、'
           '漿細胞骨髓瘤、中樞神經系統惡性腫瘤、原發部位不詳或未執行病理檢查）。',
        9: '病歷未記載或不詳。',
    }


ZH_PNI = _zh_invasion('神經侵襲')
ZH_LVI = _zh_invasion('淋巴管或血管侵犯')


# ─────────────────────────────────────────────────────────────────────────────
# 性別／個案分類／診斷狀態分類／治療狀態分類／生存狀態／首次復發型式／
# 首次治療前生活功能狀態評估 (pp.65, 28-29, 82-88, 234, 317-319, 335-337)
# ─────────────────────────────────────────────────────────────────────────────

ZH_SEX: Dict[int, str] = {
    1: '男性。',
    2: '女性。',
    3: '其他，例如：雙性人(hermaphrodite)。',
    4: '變性人。',
    9: '不詳或在病歷上未記載。',
}

ZH_CLASS_OF_CASE: Dict[int, str] = {
    0: '申報醫院診斷，但未於申報醫院接受首次療程。',
    1: '申報醫院診斷，並於申報醫院接受全部或部份的首次療程。',
    2: '他院診斷，於申報醫院接受全部或部份的首次療程。',
    3: '他院診斷，未於申報醫院接受任何首次療程；因復發或持續癌症問題至申報醫院就診。',
    5: '屍體解剖時才診斷為癌症。',
    7: '僅有病理檢查報告，個案未因診斷或治療癌症而到申報醫院（不需申報）。',
    8: '僅由死亡診斷證明書診斷為癌症（不需申報）。',
    9: '不詳，病歷上未記載足以決定個案分類的資訊。',
}

ZH_CLASS_OF_DIAGNOSIS: Dict[int, str] = {
    1: '於申報醫院診斷。',
    2: '於他院診斷，因該癌症於首次療程內/未復發/復發狀況不明下，至申報醫院就診。',
    3: '於他院診斷，因該癌症復發或惡化後，才至申報醫院就診。',
    5: '屍體解剖時才診斷為癌症。',
    7: '僅有病理檢查報告，個案未因診斷或治療癌症而到申報醫院。',
    8: '僅由死亡診斷證明書診斷為癌症。',
}

ZH_CLASS_OF_TREATMENT: Dict[int, str] = {
    0: '未於申報醫院接受任何治療即死亡。',
    1: '僅於申報醫院接受首次療程，未於他院接受任何首次療程。',
    2: '未於申報醫院接受任何首次療程（僅於他院接受，或主要治療計畫由他院規劃）。',
    3: '於申報醫院接受部分首次療程，也於他院接受部分首次療程。',
    4: '首次療程為不予治療或再密切觀察，或僅給予非腫瘤切除之緩和性手術/疼痛控制/支持療法。',
    5: '首次療程僅有另類治療。',
    6: '首次療程為個案拒絕治療。',
    7: '他院診斷及治療，因癌症或其治療的併發症至申報醫院求診。',
    8: '他院診斷個案，為了其他疾病至申報醫院求診（不需申報）。',
    9: '首次療程不詳，也無法確認個案拒絕治療。',
}

ZH_VSTA: Dict[int, str] = {
    0: '死亡。',
    1: '存活。',
}

ZH_RETYPE95: Dict[int, str] = {
    0:  '個案治療後 disease-free 且沒有復發。',
    4:  '侵襲癌以原位癌的形式復發。',
    6:  '原位癌以原位癌的形式復發。',
    10: '局部復發，且無足夠的資訊可供編碼為 13-17。',
    13: '侵襲癌於局部復發。',
    14: '侵襲癌於套管處(trocar)復發。',
    15: '侵襲癌同時於局部及套管處復發。',
    16: '原位癌於局部復發，其他未詳細描述。',
    17: '原位癌同時於局部及套管處復發。',
    20: '區域性復發，且無足夠資訊可供編碼為 21-27。',
    21: '侵襲癌僅於鄰近組織或器官復發。',
    22: '侵襲癌僅於區域淋巴結復發。',
    25: '侵襲癌同時於鄰近組織或器官及區域淋巴結復發。',
    26: '原位癌於區域性復發，其他未詳細描述。',
    27: '原位癌同時於鄰近組織或器官以及區域淋巴結復發。',
    30: '侵襲癌之復發為鄰近組織/器官或區域淋巴結復發，合併局部或套管處/手術切開處復發。',
    36: '原位癌之復發為區域性復發，合併局部或套管處復發。',
    40: '遠端復發，且無足夠資訊可供編碼為 46-62。',
    46: '原位癌於遠端復發。',
    51: '侵襲癌僅於腹膜發生遠端復發，或腹水呈現惡性細胞。',
    52: '侵襲癌僅於肺臟(包括 visceral pleura)發生遠端復發。',
    53: '侵襲癌僅於 pleura 發生遠端復發，或肋膜滲液呈現惡性細胞。',
    54: '侵襲癌僅於肝發生遠端復發。',
    55: '侵襲癌僅於骨骼發生遠端復發，但原發部位的骨骼除外。',
    56: '侵襲癌僅於中樞神經系統(CNS)發生遠端復發，但 external eye 除外。',
    57: '侵襲癌僅於皮膚發生遠端復發，但原發部位的皮膚除外。',
    58: '侵襲癌僅於遠端淋巴結復發。',
    59: '侵襲癌僅發生全身性(systemic)遠端復發（如 lymphoma、leukemia、骨髓轉移等）。',
    60: '侵襲癌之復發為單一或多處遠端部位復發，合併局部或區域復發。',
    62: '侵襲癌於多處遠端部位復發。',
    70: '個案癌症確診後從未 disease-free 過。',
    88: '癌症有復發，但復發型式不詳。',
    99: '不確定癌症是否曾復發或曾 disease-free 過。',
}

# 首次治療前生活功能狀態評估 (#7.6, p.335-337)：第 1-2 碼 KPS 十分位，
# 第 3 碼 ECOG。碼冊自己給的組合定義如下（p.337「若病歷同時記載KPS與ECOG
# PS評估值...」表），000-004 為僅記載 ECOG（KPS 未評估，第 1-2 碼固定 00）。
ZH_KPSECOG: Dict[int, str] = {
    0:   '僅記載 ECOG PS 0（KPS 未評估）：活動性與生病之前無異，不受疾病影響。',
    1:   '僅記載 ECOG PS 1（KPS 未評估）：無法做劇烈活動，但可以走動與從事輕鬆或坐著的工作。',
    2:   '僅記載 ECOG PS 2（KPS 未評估）：可以走動，可以完全自我照顧，但無法工作。',
    3:   '僅記載 ECOG PS 3（KPS 未評估）：自我照顧能力有限，一半以上的清醒時刻需臥床或坐輪椅。',
    4:   '僅記載 ECOG PS 4（KPS 未評估）：處於完全失能狀態，生活完全無法自理。',
    5:   '未治療即死亡（KPS=0 及/或 ECOG PS=5）。',
    100: 'KPS=100，ECOG PS=0：正常，沒有任何抱怨，確定沒有疾病。',
    104: 'KPS=10，ECOG PS=4：病況緊急，很快有死亡的危險。',
    204: 'KPS=20，ECOG PS=4：病情嚴重，尚未有死亡的危險。',
    209: 'KPS=20（ECOG 未評估）：病情嚴重，尚未有死亡的危險。',
    303: 'KPS=30，ECOG PS=3：嚴重傷殘，尚未有死亡的危險。',
    304: 'KPS=30，ECOG PS=4：嚴重傷殘，尚未有死亡的危險。',
    309: 'KPS=30（ECOG 未評估）：嚴重傷殘，尚未有死亡的危險。',
    403: 'KPS=40，ECOG PS=3：傷殘，需要特別照顧及幫助。',
    409: 'KPS=40（ECOG 未評估）：傷殘，需要特別照顧及幫助。',
    502: 'KPS=50，ECOG PS=2：需要考慮別人幫助，經常給予醫療照顧。',
    503: 'KPS=50，ECOG PS=3：需要考慮別人幫助，經常給予醫療照顧。',
    509: 'KPS=50（ECOG 未評估）：需要考慮別人幫助，經常給予醫療照顧。',
    602: 'KPS=60，ECOG PS=2：有時需要別人幫助，但能照顧自己大部分需要。',
    609: 'KPS=60（ECOG 未評估）：有時需要別人幫助，但能照顧自己大部分需要。',
    701: 'KPS=70，ECOG PS=1：可以自我照顧，但無法從事正常活動。',
    702: 'KPS=70，ECOG PS=2：可以自我照顧，但無法從事正常活動。',
    709: 'KPS=70（ECOG 未評估）：可以自我照顧，但無法從事正常活動。',
    801: 'KPS=80，ECOG PS=1：可以稍微正常活動，已經有一些疾病的症狀。',
    809: 'KPS=80（ECOG 未評估）：可以稍微正常活動，已經有一些疾病的症狀。',
    900: 'KPS=90，ECOG PS=0：可以正常活動，有一些疾病症狀。',
    901: 'KPS=90，ECOG PS=1：可以正常活動，有一些疾病症狀。',
    909: 'KPS=90（ECOG 未評估）：可以正常活動，有一些疾病症狀。',
    988: '不適用：首次療程已在他院執行，無法取得治療前之生活功能狀態評估值。',
    999: '病歷未記載或不詳。',
}

# ─────────────────────────────────────────────────────────────────────────────
# 首次療程的全身性治療 (#4.3.x, #4.4, #4.5.1, pp.268-303)
# ─────────────────────────────────────────────────────────────────────────────

ZH_CHEMO_MODALITY: Dict[int, str] = {
    1:  '接受全身性化學治療。',
    2:  '接受全身性化學治療，且只有一種化學藥物(僅適用於2017(含)診斷年之前的個案)。',
    3:  '接受全身性化學治療，且超過一種以上的化學藥物(僅適用於2017(含)診斷年之前的個案)。',
    4:  '個案原發部位僅接受局部動脈栓塞化學治療(TACE)。',
    5:  '個案原發部位接受局部動脈栓塞化學(TACE)及全身性化學治療。',
    6:  '個案原發部位接受局部動脈栓塞化學(TACE)及其他局部性化學治療。',
    7:  '個案原發部位接受局部動脈栓塞化學(TACE)、其他局部性及全身性化學治療。',
    8:  '個案僅接受局部性化學治療(不包含TACE)。',
    9:  '個案同時接受全身性及局部性化學治療(不包含TACE)。',
    10: '肝轉移個案接受局部動脈栓塞化學治療(TACE)。',
    11: '肝轉移個案接受局部動脈栓塞化學(TACE)及全身性化學治療。',
    12: '肝轉移個案接受局部動脈栓塞化學治療(TACE)及其他局部性化學治療。',
    13: '肝轉移個案接受局部動脈栓塞化學治療(TACE)、其他局部性及全身性化學治療。',
}
ZH_HORMONE_MODALITY: Dict[int, str] = {
    1: '在首次療程中有接受全身性荷爾蒙/類固醇治療。',
    2: '接受局部性荷爾蒙/類固醇治療。',
    3: '接受全身性與局部性荷爾蒙/類固醇治療。',
}
ZH_IMMUNO_MODALITY: Dict[int, str] = {
    1: '接受全身性免疫藥物治療。',
    2: '接受局部性免疫藥物治療。',
    3: '接受全身性與局部性免疫藥物治療。',
    4: '僅接受免疫細胞治療。',
    5: '同時接受免疫細胞治療與全身性免疫藥物治療。',
    6: '同時接受免疫細胞治療與局部性免疫藥物治療。',
    7: '同時接受免疫細胞治療、全身性與局部性免疫藥物治療。',
}
ZH_TARGETED_MODALITY: Dict[int, str] = {
    1: '在首次療程中有接受標靶治療。',
}


def _zh_trial(therapy: str, cellular: bool = False) -> Dict[int, str]:
    codes = {
        20: f'僅接受臨床試驗{therapy}。',
        21: f'同時接受{therapy}及臨床試驗{therapy}。',
        30: f'僅接受雙盲試驗{therapy}。',
        31: f'同時接受{therapy}及雙盲試驗{therapy}。',
    }
    if cellular:
        codes.update({
            22: f'同時接受免疫細胞治療及臨床試驗{therapy}。',
            23: f'同時接受免疫細胞治療與全身性或/與局部性藥物治療及臨床試驗{therapy}。',
            32: f'同時接受免疫細胞治療及雙盲試驗{therapy}。',
            33: f'同時接受免疫細胞治療與全身性或/與局部性藥物治療及雙盲試驗{therapy}。',
            40: '僅接受臨床試驗免疫細胞治療。',
            41: '同時接受全身性或/與局部性藥物治療及臨床試驗免疫細胞治療。',
        })
    return codes


def _zh_therapy_map(therapy: str, modality: Dict[int, str],
                    this_hospital: bool, extra: Dict[int, str] = None,
                    cellular: bool = False) -> Dict[int, str]:
    codes = {0: f'未接受{therapy}，{therapy}非首次療程的一部份。',
             **modality, **_zh_trial(therapy, cellular)}
    if this_hospital:
        codes.update(extra or {})
        codes.update({
            82: f'{therapy}因禁忌症或個案其他危險因素(併發症、年邁)而未建議或給予。',
            83: f'{therapy}因疾病進展而未建議或給予。',
            85: f'{therapy}是既定之首次療程計畫中的一部分，但因個案未接受治療前即死亡或病危出院。',
            86: f'{therapy}雖然是既定之首次療程計畫中的一部分，未執行且病歷也未記載未執行的原因；'
                f'或於他院執行{therapy}。',
            87: f'{therapy}雖然是既定之首次療程計畫中的一部分，但未執行，且病歷記載個案或其家屬拒絕此項治療。',
            88: f'{therapy}雖然是既定之首次療程計畫中的一部分，但摘錄時尚未執行。',
        })
    codes[99] = f'由於病歷未記載，所以不知道是否{therapy}有被建議或是已經執行。'
    return codes


ZH_CHEMO_OTHER = _zh_therapy_map('化學治療', ZH_CHEMO_MODALITY, False)
ZH_CHEMO_THIS = _zh_therapy_map(
    '化學治療', ZH_CHEMO_MODALITY, True,
    {81: '化學治療是既定之首次療程，但因基因檢測結果而未建議或給予。'})
ZH_HORMONE_OTHER = _zh_therapy_map('荷爾蒙/類固醇治療', ZH_HORMONE_MODALITY, False)
ZH_HORMONE_THIS = _zh_therapy_map('荷爾蒙/類固醇治療', ZH_HORMONE_MODALITY, True)
ZH_IMMUNO_OTHER = _zh_therapy_map('免疫治療', ZH_IMMUNO_MODALITY, False, cellular=True)
ZH_IMMUNO_THIS = _zh_therapy_map('免疫治療', ZH_IMMUNO_MODALITY, True, cellular=True)
ZH_TARGETED_OTHER = _zh_therapy_map('標靶治療', ZH_TARGETED_MODALITY, False)
ZH_TARGETED_THIS = _zh_therapy_map('標靶治療', ZH_TARGETED_MODALITY, True)

ZH_OTHER_TREATMENT: Dict[int, str] = {
    0: '未接受其他治療，其他治療非首次療程的一部份。',
    1: '在申報醫院的首次療程中接受其他治療。',
    2: '在外院的首次療程中接受其他治療。',
    3: '個案在申報醫院及外院的首次療程中接受其他治療。',
    99: '由於病歷未記載，所以不知道其他治療是否有被建議或是已經執行。',
}

ZH_PALLIATIVE_CARE: Dict[int, str] = {
    0: '個案未接受緩和照護。',
    1: '接受以減輕症狀為目的，而非為了診斷、分期或治療所進行之手術(可包括bypass 繞道手術)。',
    2: '僅接受以減輕症狀為目的，而非為了診斷、分期或治療所進行之放射治療。',
    3: '僅接受以減輕症狀為目的，而非為了診斷、分期或治療所進行之局部藥物或全身性藥物治療。',
    4: '僅接受或轉介疼痛治療，並未接受其他緩和照護。',
    5: '編碼 1、2 及 3 任二項或二項以上但不包括 4。',
    6: '編碼 1、2 及 3 任一項或一項以上且包括 4。',
    7: '有接受或轉介緩和照護，但在病歷中未提及緩和照護的型式。',
    9: '不確定是否曾接受或轉介緩和照護；病歷未記載。',
}


# ─────────────────────────────────────────────────────────────────────────────
# 放射治療 (#4.2.1.x, #4.2.2.2.1, #4.2.2.3.1, #4.1.4.1) -- additive fields
# store Chinese COMPONENT names; combinations are assembled the same way the
# English label is.
# ─────────────────────────────────────────────────────────────────────────────

ZH_TARGET_VOLUME_COMPONENTS: Dict[int, str] = {
    1:  '原發腫瘤(T)',
    2:  '區域淋巴結(N)',
    4:  '遠端轉移(M)',
    8:  '廣泛淋巴區域(mini-mantle/mantle/inverted-Y 或全身淋巴照射；限何杰金氏及非何杰金氏淋巴癌)',
    16: '全身/全骨髓',
    32: '全皮膚(限卡波西氏肉瘤、原發表皮淋巴癌或其他需全身皮膚電子射線放射治療者)',
}
ZH_RTAR_ZERO = '個案未接受放射治療。'
ZH_RTAR_UNKNOWN = '不知道個案是否有接受放射治療。'
ZH_RTAR_NOS = '個案有接受放射治療，但放射範圍未明示；或放射治療是單純內分泌處置。'

ZH_HTAR_LTAR_ZERO = '個案未接受體外放射治療，或是有體外放射治療但沒有 CTV。'
ZH_HTAR_LTAR_UNKNOWN = '不知道個案是否有接受體外放射治療。'
ZH_HTAR_LTAR_NOS = '個案有接受放射治療，但放射範圍未明示；或放射治療是單純內分泌處置。'

ZH_RT_MODALITY_COMPONENTS: Dict[int, str] = {
    1:  '一般體外放射治療(包括鈷六十機、光子/電子射線直線加速器、螺旋式斷層治療機)',
    2:  '放射手術(加馬刀、Linac-based、電腦刀、Zap-X)',
    4:  '近距放射治療(組織插種、molds、seeds、needles 或腔內放射性物質)',
    8:  '放射線同位素治療(注射放射性同位素，如碘-131、鍶-89)',
    16: '質子治療',
    32: '其他帶電荷粒子或中子治療',
    64: '硼捉中子治療(BNCT)',
}
ZH_RMOD_ZERO = '個案未接受放射治療。'
ZH_RMOD_UNKNOWN = '不知道個案是否有接受放射治療。'
ZH_RMOD_NOS = '個案有接受放射治療，但未明示使用的治療儀器或方法。'

ZH_SEQRS_COMPONENTS: Dict[int, str] = {
    1: '手術前放射治療(縮小腫瘤)',
    2: '手術中放射治療(IORT)',
    4: '手術後放射治療(降低局部復發率)',
}
ZH_SEQRS_SPECIAL: Dict[int, str] = {
    -9: '不清楚是否有手術或放射治療。',
    -8: '首次療程中個案有原發部位及區域淋巴結放射治療，但未接受原發部位及區域淋巴結手術；'
        '或無原發部位或區域淋巴結手術，接受遠端部位預防性放射治療。',
    -7: '結內淋巴癌、血液腫瘤疾病或遠端轉移個案；或首次療程包含手術及放射治療，但治療範圍不同。',
    -6: '有兩次以上原發部位或區域淋巴結手術，無法決定與放射治療的順序。',
    -1: '首次癌症療程中有手術也有放射治療，但順序未明。',
    0:  '首次癌症療程中個案未接受放射治療；或首次療程不含放射治療亦不含手術。',
}

ZH_SEQLS_COMPONENTS: Dict[int, str] = {
    1: '導引/前導性輔助療法(手術前進行全身性治療)',
    2: '同步/併行療法(同步全身性藥物放射治療 CSRT，或手術旁全身性藥物治療)',
    4: '輔助療法(手術後進行全身性治療)',
}
ZH_SEQLS_SPECIAL: Dict[int, str] = {
    -9: '不清楚是否有區域療法與全身性治療。',
    -8: '首次療程有全身性藥物治療而無區域治療(不論是否有局部性藥物治療)。',
    -7: '首次療程僅有局部性藥物治療而未使用全身性藥物治療(不論是否有區域治療)。',
    -1: '首次療程中有區域治療與全身性治療，但順序未明；或結內淋巴癌、血液腫瘤疾病或遠端轉移個案。',
    0:  '首次療程中不含化學藥物、荷爾蒙、免疫及標靶治療，不論是否有接受區域治療。',
}

ZH_RT_STATUS: Dict[int, str] = {
    0:  '個案僅於申報醫院接受首次療程的放射治療。',
    1:  '放射治療非既定之首次療程計畫中的一部分。',
    2:  '放射治療因禁忌症或個案其他危險因素(併發症、年邁)而未建議或給予。',
    3:  '放射治療因疾病進展而未建議或給予。',
    4:  '個案於申報醫院接受首次療程的放射治療，但病人因個人因素未完成既定放射治療之療程。',
    5:  '放射治療是既定之首次療程計畫中的一部分，但因個案未接受前即死亡或病危出院。',
    6:  '放射治療是既定之首次療程計畫中的一部分，未執行且病歷也未記載未執行的原因。',
    7:  '放射治療是既定之首次療程計畫中的一部分，但病歷記載個案或其家屬拒絕放射治療。',
    8:  '放射治療雖是既定之首次療程計畫中的一部分，但摘錄時尚未執行。',
    9:  '個案僅於外院接受首次療程的放射治療。',
    10: '個案於外院及申報醫院皆接受首次療程的放射治療。',
    99: '由於病歷未記載，所以不知道放射治療是否有被建議或是已經執行。',
}

ZH_MINIMALLY_INVASIVE: Dict[int, str] = {
    0: '僅接受開放性手術(open surgery)，未接受微創或機械臂輔助手術。',
    1: '有接受內視鏡手術。',
    2: '有接受胸腔鏡或腹腔鏡手術，或類似手術。',
    3: '有接受機械臂輔助手術(Robotic Surgery)。',
    4: '有接受微創手術或機械臂輔助手術，合併或轉換開放性手術。',
    8: '不適用(原發部位未接受手術；手術方式編碼100-190；攝護腺癌手術方式編碼210-270；'
       '於外院接受原發部位手術；2017年(含)之前診斷；或造血/網狀內皮/免疫增生/骨髓增生疾病)。',
    9: '不詳。',
}


# ─────────────────────────────────────────────────────────────────────────────
# Combine the additive-field component tables into per-code Chinese text the
# same way longform_codes._additive_map()/_therapy_map() build the English
# label -- so this file only needs the manual's own component phrases, not
# one entry per subset sum.
# ─────────────────────────────────────────────────────────────────────────────

def _zh_additive(components: Dict[int, str], zero: str, unknown: str,
                 nos: str) -> Dict[int, str]:
    import itertools
    codes = {-9: unknown, -1: nos, 0: zero}
    for r in range(1, len(components) + 1):
        for combo in itertools.combinations(components, r):
            codes[sum(combo)] = '＋'.join(components[c] for c in sorted(combo))
    return codes


ZH_RTAR = _zh_additive(ZH_TARGET_VOLUME_COMPONENTS, ZH_RTAR_ZERO,
                      ZH_RTAR_UNKNOWN, ZH_RTAR_NOS)
ZH_HTAR = _zh_additive(ZH_TARGET_VOLUME_COMPONENTS, ZH_HTAR_LTAR_ZERO,
                      ZH_HTAR_LTAR_UNKNOWN, ZH_HTAR_LTAR_NOS)
ZH_LTAR = ZH_HTAR
ZH_RMOD = _zh_additive(ZH_RT_MODALITY_COMPONENTS, ZH_RMOD_ZERO,
                       ZH_RMOD_UNKNOWN, ZH_RMOD_NOS)


def _zh_seq_additive(components: Dict[int, str],
                     special: Dict[int, str]) -> Dict[int, str]:
    import itertools
    codes = dict(special)
    for r in range(1, len(components) + 1):
        for combo in itertools.combinations(components, r):
            codes[sum(combo)] = '＋'.join(components[c] for c in sorted(combo))
    return codes


ZH_SEQRS = _zh_seq_additive(ZH_SEQRS_COMPONENTS, ZH_SEQRS_SPECIAL)
ZH_SEQLS = _zh_seq_additive(ZH_SEQLS_COMPONENTS, ZH_SEQLS_SPECIAL)


# ─────────────────────────────────────────────────────────────────────────────
# Public registry: raw field tag -> {code: Chinese text}, mirroring
# longform_codes.LONGFORM_CODE_MAPS's keys exactly.
# ─────────────────────────────────────────────────────────────────────────────

LONGFORM_ZH: Dict[str, Dict] = {
    'LAT95':       ZH_LAT95,
    'MCODE5':      ZH_MCODE5,
    'PNI':         ZH_PNI,
    'LVI':         ZH_LVI,
    'PREC':        ZH_CHEMO_OTHER,
    'C':           ZH_CHEMO_THIS,
    'PREH':        ZH_HORMONE_OTHER,
    'H':           ZH_HORMONE_THIS,
    'PREI':        ZH_IMMUNO_OTHER,
    'I':           ZH_IMMUNO_THIS,
    'PRETAR':      ZH_TARGETED_OTHER,
    'TAR':         ZH_TARGETED_THIS,
    'OTH':         ZH_OTHER_TREATMENT,
    'PREP':        ZH_PALLIATIVE_CARE,
    'RTAR':        ZH_RTAR,
    'RMOD':        ZH_RMOD,
    'HTAR':        ZH_HTAR,
    'LTAR':        ZH_LTAR,
    'SEQRS':       ZH_SEQRS,
    'SEQLS':       ZH_SEQLS,
    'R':           ZH_RT_STATUS,
    'MINS':        ZH_MINIMALLY_INVASIVE,
    'SEX':         ZH_SEX,
    'CLASS95':     ZH_CLASS_OF_CASE,
    'CLASSOFDIAG': ZH_CLASS_OF_DIAGNOSIS,
    'CLASSOFTREAT': ZH_CLASS_OF_TREATMENT,
    'VSTA':        ZH_VSTA,
    'RETYPE95':    ZH_RETYPE95,
    'KPSECOG':     ZH_KPSECOG,
}

# CONFER needs the morphology-selected table, handled separately since it is
# not a plain CodeMap in longform_codes.py either.
LONGFORM_ZH_CONFER_SOLID = ZH_CONFER_SOLID
LONGFORM_ZH_CONFER_HAEM = ZH_CONFER_HAEM


def zh_longform(tag: str, code) -> str:
    """The code book's own Chinese text for one Longform structural field.

    Mirrors tcr_decoder.validation.zh_definition() for the SSF fields.
    Returns '' if this field or code has no transcribed Chinese text yet
    (Appendix B surgery codes, AJCC, EBRT and the two node-surgery fields
    predate this module and are a documented follow-up, not silently
    guessed at -- see docs/codebook_conformance_findings.md).
    """
    table = LONGFORM_ZH.get(tag)
    if table is None:
        return ''
    try:
        key = int(str(code).strip())
    except (TypeError, ValueError):
        return ''
    return table.get(key, '')
