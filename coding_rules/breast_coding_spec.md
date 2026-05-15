# 乳癌 TCR 編碼規格 (Breast Cancer TCR Coding Specification)

> **用途**：本文件定義從臨床自由文字（病理報告、治療計畫書）抽取結構化事實，
> 並對應到台灣癌症登記（TCR）代碼的完整規則與邏輯。
>
> **對應模組**：`tcr_decoder/decoders.py`、`tcr_decoder/ssf_registry.py`（乳癌 SSF profile）
>
> **最後更新**：2026-05-15

---

## 總覽：乳癌需要抽取的臨床事實

| # | 臨床事實 | 對應 TCR 欄位 | 資料類型 |
|---|---|---|---|
| 1 | ER 染色百分比 / 陰陽性 | SSF1 | 數字 0–100 或定性 |
| 2 | PR 染色百分比 / 陰陽性 | SSF2 | 數字 0–100 或定性 |
| 3 | 新輔助治療反應 | SSF3 | 類別 |
| 4 | Sentinel LN 切除數目 | SSF4 | 整數 |
| 5 | Sentinel LN 陽性數目 | SSF5 | 整數 |
| 6 | Nottingham 評分 / 分級 | SSF6 | 整數 3–9 或 Grade 1–3 |
| 7 | HER2 IHC 分數 + ISH 結果 + 診斷年份 | SSF7 | 複合條件 |
| 8 | Paget's Disease 是否存在 | SSF8 | 布林 |
| 9 | LVI（淋巴血管侵犯）是否存在 | SSF9 | 布林 |
| 10 | Ki-67 百分比 | SSF10 | 數字 0–100 |
| 11 | 手術方式 | STYPE95 | 文字對應 |
| 12 | 腫瘤大小（mm） | PT / CT | 數字 → T stage |
| 13 | 淋巴結切除 / 陽性數目 | PN | 整數 → N stage |

---

## SSF1：ER（雌激素受體）

### 需要抽取的事實
- `er_percent`：數字 0–100（若病理報告有寫百分比）
- `er_staining_intensity`：`weak` / `intermediate` / `strong`（若有標注）
- `er_qualitative`：`positive` / `negative`（若只有定性描述）
- `context`：`pre_treatment` / `post_neoadjuvant`

### 編碼規則
```
若 context == post_neoadjuvant:
    若 er_qualitative == positive:    → SSF1 = 111
    若 er_qualitative == negative:    → SSF1 = 121

若 er_percent 有值:
    若 er_percent == 0:               → SSF1 = 0   (陰性 0%)
    若 er_percent >= 1:
        若 intensity == weak:         → SSF1 = "W{er_percent}"  (e.g. W15)
        若 intensity == intermediate: → SSF1 = "I{er_percent}"
        若 intensity == strong:       → SSF1 = "S{er_percent}"
        若 intensity 未知:            → SSF1 = er_percent  (純數字)

若只有定性描述（無百分比）:
    若 er_qualitative == positive:    → SSF1 = 110  (陽性，比例不明)
    若 er_qualitative == negative:    → SSF1 = 120  (陰性，<1% 或未說明)

若 converted neg→pos after neoadjuvant: → SSF1 = 888
若 不適用（Oncotype/Phyllodes/肉瘤）:    → SSF1 = 988
若 未記錄 / 不明:                        → SSF1 = 999
```

### 人工審查觸發條件
- 報告只寫「ER (+)」但無百分比 → 輸出 110，標記提醒補充百分比
- 百分比介於 0–1% 之間 → 臨床意義模糊，flag

---

## SSF2：PR（黃體素受體）

> 規則與 SSF1 完全相同，將所有 `er_` 替換為 `pr_`，代碼對應不變。

---

## SSF3：新輔助治療反應

### 需要抽取的事實
- `neoadjuvant_given`：布林（是否曾接受新輔助治療）
- `response_type`：`cCR` / `pCR` / `partial` / `stable` / `progressive`

### 編碼規則
```
若 neoadjuvant_given == False:    → SSF3 = 988  (不適用)

若 response_type == cCR:          → SSF3 = 010  (臨床完全緩解)
若 response_type == pCR:          → SSF3 = 011  (病理完全緩解)
若 response_type == partial:      → SSF3 = 020  (部分緩解)
若 response_type == stable:       → SSF3 = 030  (穩定 / 輕微緩解)
若 response_type == progressive:  → SSF3 = 040  (疾病進展)

若 縮小但未分類:                  → SSF3 = 990
若 未評估:                        → SSF3 = 999
```

---

## SSF4 / SSF5：Sentinel 淋巴結切除數 / 陽性數

### 需要抽取的事實
- `slnb_performed`：布林
- `sln_examined`：整數（切除數）
- `sln_positive`：整數（陽性數）

### 編碼規則
```
若 slnb_performed == False:       → SSF4 = 988, SSF5 = 988

若 slnb_performed == True:
    SSF4 = sln_examined  (0–89)
    SSF5 = sln_positive  (0–89)

    若 切除但找不到組織 / 數目不明:  → SSF4 = 996

交叉驗證：sln_positive ≤ sln_examined，否則 flag ERROR
```

---

## SSF6：Nottingham 分級（Bloom-Richardson）

### 需要抽取的事實
- `nottingham_score`：整數 3–9（管狀形成 + 核分裂 + 細胞核多形性之總和）
- OR `nottingham_grade`：1 / 2 / 3（若只有分級沒有分數）

### 編碼規則
```
若有 nottingham_score:
    score 3–5:  → SSF6 = score  (Grade 1，分化良好)
    score 6–7:  → SSF6 = score  (Grade 2，中度分化)
    score 8–9:  → SSF6 = score  (Grade 3，分化不良)

若只有 nottingham_grade:
    Grade 1:    → SSF6 = 110
    Grade 2:    → SSF6 = 120
    Grade 3:    → SSF6 = 130

若不適用（Phyllodes / 肉瘤）: → SSF6 = 988
若未記錄:                      → SSF6 = 999
```

---

## SSF7：HER2 綜合狀態 ⚠️ 最複雜 — 依診斷年份不同

### 需要抽取的事實
- `diagnosis_year`：整數（決定使用哪個代碼系統）
- `her2_ihc`：`0` / `1+` / `2+` / `3+` / `ultralow`
- `her2_ish_result`：`amplified` / `not_amplified` / `equivocal` / `not_done`

### 年份分界
| 診斷年份 | TCR 內部年碼 | 使用代碼系統 |
|---|---|---|
| 2000–2007 | yr 100–107 | 舊版單碼或三碼（0/1/2/3 或 100–402） |
| 2008–2013 | yr 108–113 | IHC+ISH 組合碼 500 系列 |
| 2014+ | yr 114+ | 同上，另加 Ultralow 600 系列 |

### 編碼規則（2008+ 主流案例）
```
her2_ihc == "3+":
    ISH negative  → SSF7 = 530
    ISH positive  → SSF7 = 531
    ISH equivocal → SSF7 = 532
    ISH not_done  → SSF7 = 103  (IHC 3+ Positive，無 ISH)

her2_ihc == "2+":
    ISH amplified     → SSF7 = 521  (Positive)
    ISH not_amplified → SSF7 = 520  (Negative)
    ISH equivocal     → SSF7 = 522  (Equivocal)
    ISH not_done      → SSF7 = 102  (Equivocal，未做 ISH)

her2_ihc == "1+":
    ISH not_done      → SSF7 = 101  (Negative, Low HER2)
    ISH amplified     → SSF7 = 511  (Positive)
    ISH not_amplified → SSF7 = 510  (Negative)

her2_ihc == "0":
    ISH not_done      → SSF7 = 100  (Negative)
    ISH amplified     → SSF7 = 501  (Positive by ISH)
    ISH not_amplified → SSF7 = 500  (Negative)

不適用（Phyllodes / 肉瘤）: → SSF7 = 988
未做任何 HER2 檢測:         → SSF7 = 999
```

### 2014+ Ultralow（IHC 0 但 0% < 染色 ≤ 10%）
```
her2_ihc == "ultralow":
    ISH not_amplified → SSF7 = 640
    ISH amplified     → SSF7 = 641
    ISH equivocal     → SSF7 = 642
```

### 人工審查觸發條件
- IHC 2+ 但未做 ISH → flag「建議補 FISH/CISH 結果」
- HER2 結論與 IHC + ISH 邏輯矛盾 → flag WARN

---

## SSF8：Paget's Disease of Nipple

### 編碼規則
```
有 Paget's disease:                        → SSF8 = 010
無 Paget's disease:                        → SSF8 = 0
不適用（檢體未含乳頭 / 乳暈）:              → SSF8 = 988
未記錄:                                    → SSF8 = 999
```

---

## SSF9：淋巴血管侵犯（LVI）

### 編碼規則
```
LVI 存在（lymphovascular invasion present）: → SSF9 = 010
LVI 不存在:                                  → SSF9 = 0
新輔助治療後無殘留腫瘤（LVI 無法評估）:       → SSF9 = 990
未記錄:                                      → SSF9 = 999
```

---

## SSF10：Ki-67 增生指數

### 需要抽取的事實
- `ki67_percent`：數字（0–100，或 <1% 的小數）

### 編碼規則
```
若 ki67_percent >= 1:
    SSF10 = round(ki67_percent)  (整數 1–100)

若 ki67_percent < 1:
    SSF10 = f"A{round(ki67_percent * 10):02d}"
    # 例：0.5% → A05，0.1% → A01

分類（僅供參考，不影響代碼）：
    0–13%  → Low
    14–30% → Intermediate
    >30%   → High

若檢測但無具體數字:  → SSF10 = 998
若不適用:            → SSF10 = 988
若未記錄:            → SSF10 = 999
```

---

## 手術代碼（STYPE95）

### 主要對應邏輯（2025 新版 3 碼）

| 手術描述關鍵字 | 代碼 |
|---|---|
| Lumpectomy / BCS / partial mastectomy / 象限切除 / segmentectomy | 200 |
| Re-excision of margins（切緣再切） | 240 |
| Central lumpectomy（含乳頭乳暈切除） | 290 |
| Skin-sparing mastectomy（保留皮膚乳房切除） | 300 系列 |
| Nipple-sparing mastectomy / NSM（保留乳頭） | 400 系列 |
| Areola-sparing mastectomy | 500 系列 |
| Total / simple mastectomy（全乳切除） | 600 系列 |
| Radical / modified radical mastectomy | 700 系列 |

### 次碼規則（300 / 400 / 500 / 600 / 700 系列內）
```
基礎碼 + 對側處置：
    x00：單側，無對側切除資訊
    x10：不含對側切除
    x20：含對側切除

重建加碼（x10→x14, x20→x24）：
    x11：重建 NOS
    x12：自體組織重建
    x13：植入物重建
    x14：複合重建
```

---

## TNM 分期（AJCC 第 8 版）

### T stage（腫瘤大小）
```
大小 == 0 且確認 pCR:   → T0
大小 ≤ 5 mm:            → T1a
5 < 大小 ≤ 10 mm:       → T1b
10 < 大小 ≤ 20 mm:      → T1c
20 < 大小 ≤ 50 mm:      → T2
大小 > 50 mm:            → T3
侵犯皮膚 / 胸壁:         → T4（需進一步判斷 a/b/c/d）
炎性乳癌:                → T4d
```

### N stage（淋巴結）
```
0 個陽性:                         → pN0
僅 ITC（≤ 0.2mm）:               → pN0(i+)
1–3 個 Level I/II 腋下 LN 陽性:  → pN1
4–9 個 LN 陽性:                  → pN2
≥ 10 個 LN 陽性:                 → pN3
```

### 新輔助治療後 TNM
```
新輔助治療後的 TNM 前綴加 yp：
    ypT0 ypN0 → 代表 pCR
    T/N/M 代碼填 888 → 「新輔助治療後，分期不適用於術前分期」
```

---

## 重要交叉驗證規則

| 驗證項目 | 條件 | 動作 |
|---|---|---|
| Sentinel LN 邏輯 | sln_positive > sln_examined | Flag ERROR |
| HER2 一致性 | IHC 3+ 且 ISH not amplified | Flag WARN（少見，需確認）|
| TNBC 自動標記 | ER- PR- HER2- | 標記為 Triple Negative |
| ER 轉換 | 治療前 ER- 治療後 ER+ | SSF1 = 888，需記錄 |
| Nottingham 一致性 | score 3–5 但 grade 填 3 | Flag WARN |
| Ki67 與分子亞型 | ER+ HER2- Ki67 <14% | 符合 Luminal A，可自動標記 |

---

## 尚未定義 / 需人工確認的情況

1. **皮膚侵犯程度**：T4a/T4b/T4c 分類需要影像 + 病理聯合判斷
2. **內乳淋巴結**：N stage 分類複雜，建議人工確認
3. **多灶性腫瘤**：T stage 以最大病灶為準，需旗標提醒
4. **各院縮寫差異**：例如「SLNB」vs「sentinel node biopsy」vs「前哨淋巴結切片」
5. **HER2 ultralow**：2023 年新定義，舊版報告不會有此描述

---

## 待辦事項

- [ ] 加入 Oncotype DX / MammaPrint 對 SSF1 的影響說明
- [ ] 完成 AJCC 8th 完整 stage group 表（T × N × M 組合）
- [ ] 收集各合作醫院常見手術縮寫對照表
- [ ] 定義 LLM extraction prompt template（基於本規格）
- [ ] 建立 Python 規則引擎（基於本規格）
