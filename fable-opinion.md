# TCR Decoder 後續工作執行規劃

> 撰寫者：Fable 5（本次 session 的深入研究者）
> 撰寫日期：2026-07-04
> 對象分支：`claude/taiwan-cancer-registry-converter-qvv2r0`（PR #1，draft，目標 main）

---

## 給接手模型的話（請先讀這段）

這份文件是為了「把本次 session 累積的判斷與脈絡，交給後續沒有這次研究背景的人／模型」而寫的。本次 session 幫 `tcr_decoder` 這個台灣癌症登記（TCR）解碼工具新增了「編碼」方向（臨床文字 → 原始代碼），並順手修掉了 4 個既有解碼錯誤。這些工作**已經做完、已驗證、有測試**，你不需要重做（詳見下方「已完成、不要重做」章節）。

這份文件真正的價值在**後續工作的判斷**，不只是任務清單。使用方式：

1. **動手前先讀對應章節的「判斷與理由」**。每一項工作我都寫了「該不該做／優先順序／為什麼」，有些我認為不值得大改（例如 Allred score，見工作 5），照著我的判斷做，不要自己重新發明方向。
2. **每項工作都附了「具體修改步驟」與「完成後驗證指令」**。改完一定要跑對應的測試指令，確認沒有破壞既有功能。
3. **正確性的唯一依據是兩份官方 PDF 手冊**（在 `docs/` 底下），不是憑感覺猜。任何欄位對照表都要以 PDF 為準。用 `pypdf` 讀取；注意 pypdf 對「表格」的文字擷取常常跳行、欄位錯位，遇到表格要特別小心，不要照抄可能有誤的行。
4. **環境注意事項**：這台機器上 `pip install -e .` 目前是壞的（見工作 2）。本 session 全程用 `pip install pandas numpy openpyxl pytest` 直接裝依賴、用 `python -m pytest` 跑測試來繞過。若 `import cryptography` 出現 cffi 錯誤，跑 `pip install --force-reinstall cffi`。
5. **誠實原則**：文件裡凡是我標「不確定」的地方，就是真的無法從程式碼或 PDF 直接驗證，請你也不要假裝有把握——寧可標記待人工確認，也不要做出「看起來合理但其實是錯的」代碼。這在醫療資料上尤其重要。

### 目前的環境版本（會影響某些測試）
- Python 3.11
- pandas **3.0.3**（很新，是造成兩個既有測試失敗的原因之一）
- pytest **9.1.1**（很新，`pytest.warns(None)` 舊寫法已被移除）
- pypdf 6.14.2

### 全套測試現況
在本分支上跑 `python -m pytest tests/ -q`：**941 passed, 2 failed**。那 2 個 failed（`test_series_all_same_group`、`test_leading_zero_pk_preserved_as_string`）是**跟本次工作無關的既有環境問題**，我已用 `git stash` 驗證過在本分支改動之前就會失敗。它們的修法見工作 3。

---

## 已完成、已驗證的事（不要重做）

以下都已經 commit 進本分支，並有測試覆蓋。列出來是讓你知道「這些不用碰」：

1. **`CodeMap`（`tcr_decoder/codemap.py`）**：單一雙向 code↔label 對照表，`decode()`/`encode()` 共用同一份 dict，兩個方向不會漂移。`utils.py` 的 `_map_decode()` 已改為委派給它，解碼行為完全不變。

2. **`tcr_decoder/encoders.py`**：針對「不只是查表」的複雜欄位手寫的反向（編碼）函式：ER/PR 染色強度+比例、HER2 三碼組合、Ki-67、Nottingham、AFP、PSA、肺癌 EGFR/大腸直腸 RAS 多字元組合碼、EBRT 位元遮罩加總等。每個都刻意模仿對應解碼函式「連無法辨識的代碼也原樣通過、不報錯」的容錯行為。

3. **`TCREncoder`（`tcr_decoder/encoder.py`）**：吃一個像 `TCRDecoder(...).clean` 的 DataFrame，重建 11 種癌別的 SSF1-10 原始代碼，加上少數本專案有完整對照表的結構性欄位（AJCC 版本、STYPE95/PRESTYPE、PRESLNSCO/SLNSCO95、EBRT、LN_POSITI）。**刻意不去猜**其餘約 90 個解碼欄位——那些欄位的解碼是「信任輸入檔已存在的 `_decoded` 欄位」或依賴外部檔 `cancer_registry_mapping.py`（不在本 repo），本專案沒有對照表可反查，硬猜會做出「看似合理但實際錯」的代碼。這些欄位列在 `TCREncoder.unencoded_columns` 並附原因。**這個「不猜」的決定是刻意的、正確的，請維持，不要在後續工作裡去「補齊」它們。**

4. **`compare_roundtrip()` / `export_roundtrip_report()`（`tcr_decoder/roundtrip.py`）**：解碼 → 再編碼 → 跟原始代碼比對差異，並有 CLI：`python -m tcr_decoder registry.xlsx --roundtrip`。

5. **4 個既有解碼錯誤的修正**（都直接對照 PDF 原文驗證過，都有 `tests/test_encoders.py` 測試）：
   - 乳癌 SSF1/SSF2 ER/PR：`S00`/`W00`/`I00` 是「100%」不是「0%」（手冊內部頁碼 p.121，PDF index 126）。
   - 攝護腺 SSF1 PSA：`980-998`（≥98 ng/mL）原本完全沒處理，已補完整分級表（p.177-178）。
   - 肝癌 SSF1 AFP：`A00` 是「<1 ng/mL」不是「0 ng/mL」（p.81）。
   - HER2 `300-302`：補上「(legacy...)」註記、改標為 FISH（與 200-202/400-402 一致）。

---

## 後續工作

以下每項都標了優先順序（高／中／低）與理由。**建議的執行順序：工作 2（最小最安全）→ 工作 3 → 工作 1 → 工作 4 → 工作 5（可不做）**。

---

### 工作 1 — 修正 lung SSF3（`Performance_Status`）欄位撞名 bug　【優先：高】

#### 這是什麼問題
`core.py` 裡，肺癌 SSF3（ECOG/KPS 一項癌別特定因子）的解碼結果，跟另一個通用欄位 `KPSECOG` 的解碼結果，**都寫進同一個輸出欄位名 `Performance_Status`**。因為 `KPSECOG` 那一行（`core.py:595`）在 pipeline 中比 SSF profile 晚執行，所以對肺癌病人來說，SSF3 的解碼結果被 `KPSECOG` 的值**靜默覆蓋掉**，SSF3 的資訊實際上遺失了。

這個 bug 目前已被 `core.py:586-594` 的 `# KNOWN ISSUE` 註解標記，`TCREncoder` 也已在 `encoder.py:92-101` 的 `_SSF_PIPELINE_OVERRIDDEN` 針對性處理（讓它不 crash、標為 unencoded），**但根本原因（欄位撞名）還沒修**。

#### 為什麼值得做（理由 + 優先高）
這是一個真正的**資料正確性 bug**：肺癌病人的 ECOG/KPS SSF3 值目前是被丟掉的，使用者拿到的 `Performance_Status` 其實是 KPSECOG 而不是 SSF3。因為涉及臨床資料遺失，優先度定為「高」。但它牽涉多個檔案的引用，所以要小心做完整。

#### 具體修改步驟
建議把**肺癌 SSF3 的輸出欄位改名**（例如 `Performance_Status_SSF3`），讓兩個欄位並存、不再互相覆蓋。通用的 `Performance_Status`（= KPSECOG，適用所有癌別）維持原名不動。

1. **`tcr_decoder/ssf_registry.py:969`**：把 lung SSF3 的 `SSFFieldDef` 第二個參數（`column_name`）從 `'Performance_Status'` 改成 `'Performance_Status_SSF3'`：
   ```python
   'SSF3':  SSFFieldDef('SSF3', 'Performance_Status_SSF3',
                        'Performance status (ECOG/KPS) before treatment',
                        decoder=_decode_lung_ssf3_ecog),
   ```
   注意：`_ENCODER_WIRING`（`ssf_registry.py:1373`）用的 key 是 `('lung', 'SSF3')`（用 SSF key，不是 column_name），所以編碼那邊**不需要改**——這是個好消息，改動被侷限在 column_name。

2. **`tcr_decoder/core.py:586-595`**：拿掉 `# KNOWN ISSUE` 註解區塊（不再是 issue）。`out['Performance_Status'] = en(self._dec('KPSECOG'))` 這行**保留不動**（通用欄位）。改完後，肺癌病人會同時有 `Performance_Status`（KPSECOG）與 `Performance_Status_SSF3`（SSF3），互不覆蓋。

3. **`tcr_decoder/encoder.py:81-102`**：把 `_SSF_PIPELINE_OVERRIDDEN` 裡 `('lung', 'Performance_Status')` 這一整個 entry **刪掉**。因為改名後 SSF3 有自己的欄位 `Performance_Status_SSF3`，而它在 `_ENCODER_WIRING` 有 encoder（`_LUNG_SSF3_MAP.encode`），會被正常編碼成 `SSF3_raw`，不再是「無法反查」的欄位。

4. **`tcr_decoder/data_dictionary.py:115` 與 `:274`**：目前 `'Performance_Status': ('KPSECOG', 'mapping', 'ECOG/KPS performance status')` 與 `'Performance_Status': '7.1'`。這兩處指的是**通用 KPSECOG 欄位**（癌登欄位序號 7.1），語意正確，**保留**。但要**新增**一筆給肺癌 SSF3 的欄位（例如 `'Performance_Status_SSF3': ('SSF3', ...)`，癌登欄位序號請去 PDF 查肺癌 SSF3 的正確序號，不要瞎填）。若 data_dictionary 的產生邏輯只針對實際出現的欄位、缺欄位不會報錯，這步可視為選配，但建議補齊以免資料字典不完整。

5. **測試同步更新**（這是這項工作最容易漏的地方）：
   - `tests/test_pipeline.py:82`：`REQUIRED_LUNG_SSF` 清單裡的 `'Performance_Status'` 要改成 `'Performance_Status_SSF3'`（因為這個清單本意是檢查「肺癌 SSF 欄位都在」，指的就是 SSF3）。
   - `tests/test_ssf_registry.py:319`：`assert 'ECOG 0' in result['Performance_Status'].iloc[0]` 要改成 `result['Performance_Status_SSF3']`（這裡測的是 `apply_ssf_profile` 直接輸出，欄位名已改）。
   - `tests/test_encoders.py:406-410`：`test_lung_performance_status_collision_is_flagged` 這個測試的語意要改。改名後 `Performance_Status`（= KPSECOG）**仍然**會在 `unencoded_columns` 裡（因為 KPSECOG 沒有本地對照表），所以 `assert 'Performance_Status' in enc.unencoded_columns` 這行**技術上還會通過**，但測試名稱與註解的「collision」語意已不成立。建議：把測試改名為 `test_lung_generic_performance_status_has_no_codetable`，並**新增**一個 assertion 驗證 `Performance_Status_SSF3` 有被成功編碼成 `SSF3_raw`（`assert 'SSF3_raw' in raw.columns` 之類，並確認它不在 unencoded_columns）。
   - `tests/test_synth.py:103`：這行 `assert len(invalid) == 0, f'Invalid Performance_Status codes: {invalid}'` 只是錯誤訊息字串，**先實際打開看它驗證的是哪個欄位**（很可能是 `KPSECOG_raw`，與這次改名無關）。若真的無關就不用動，只是確認一下。

#### 風險與注意
- 主要風險是「漏改某個引用點」導致測試紅或欄位對不上。上面第 5 點已列出**所有** grep 到 `Performance_Status` 的位置，動手前請自己再 `grep -rn "Performance_Status" tcr_decoder/ tests/` 一次確認沒有新增的引用點。
- `report.py`、`html_report.py` 我已 grep 過，**沒有**硬編 `Performance_Status`（它們是動態列欄位），所以不用改。
- 不要把通用 `Performance_Status`（KPSECOG）也一起改名——那會影響所有癌別、破壞更多測試，且它本來就沒問題。

#### 完成後驗證
```bash
python -m pytest tests/test_pipeline.py tests/test_ssf_registry.py tests/test_encoders.py tests/test_synth.py -q
# 再跑全套確認沒有連帶破壞
python -m pytest tests/ -q
```
預期：除了工作 3 那兩個既有失敗外，其餘全綠。並手動驗證一次肺癌流程，確認 `Performance_Status_SSF3` 有值、`Performance_Status`（KPSECOG）也有值、兩者不同：
```bash
python -m tcr_decoder --synth lung --n 20 --seed 1 --out /tmp/lung.xlsx --decode
# 或用 python 直接 TCRDecoder 檢查 clean DataFrame 有這兩欄
```

---

### 工作 2 — 修好 `pyproject.toml` 的 build-backend　【優先：高（但工作量最小）】

#### 這是什麼問題
`pyproject.toml` 目前是：
```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"
```
`build-backend = "setuptools.backends.legacy:build"` 是**錯的字串**，這個 module path 不存在，導致 `pip install -e .` 完全跑不起來。本 session 全程靠直接 `pip install pandas numpy openpyxl pytest` 繞過。

#### 為什麼值得做
獨立、極小、風險極低，但影響「別人能不能照 README 安裝這個套件」。README 第 61 行就寫著 `pip install -e .`，目前那行是壞的，任何照做的人第一步就會失敗。優先度高只是因為 CP 值極高，不是因為緊急。

#### 具體修改步驟
把 `pyproject.toml` 的 build-backend 改成正確值：
```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"
```

#### 風險與注意
幾乎沒有風險。唯一要確認的是改完後真的裝得起來。

#### 完成後驗證
```bash
cd /home/user/TCRD_decoding
pip install -e . 2>&1 | tail -20      # 應該成功，不再報 legacy backend 錯誤
python -c "import tcr_decoder; print(tcr_decoder.__version__)"
python -m pytest tests/ -q            # 確認裝成套件後測試仍可跑
```
若這台機器 `pip install -e .` 因為網路/沙箱因素仍失敗，至少確認錯誤訊息**不再是** build-backend 相關（而是別的環境問題），就算修好了。

---

### 工作 3 — 兩個既有測試失敗　【優先：中】

這兩個都跟本次工作無關，是新版 pandas/pytest 造成的。**都值得修，且都很小、風險低。** 修它們能讓「全套綠燈」，對之後加 CI（工作 4）很有幫助——CI 不能容忍已知紅燈。

#### 3a. `tests/test_adversarial.py::TestICDO3EdgeCases::test_series_all_same_group`
- **失敗原因（已實際驗證）**：`test_adversarial.py:357` 用了 `with pytest.warns(None) as rec:`。`pytest.warns(None)` 這種「斷言不會有 warning」的舊寫法在 pytest 7 起 deprecated、pytest 8+ 直接 raise `TypeError: exceptions must be derived from Warning, not <class 'NoneType'>`。這跟被測程式碼無關，純粹是測試寫法過時。
- **修法**：這個測試的本意（看註解 `# Ideally no UserWarning for a uniform series`）是「同質序列不該噴 warning」。用新版寫法改寫：
  ```python
  import warnings
  def test_series_all_same_group(self):
      homog = pd.Series(['C50.1', 'C50.2', 'C50.9'])
      with warnings.catch_warnings():
          warnings.simplefilter('error')   # 任何 warning 都會變成 error
          group = detect_cancer_group_from_series(homog)
      assert group == 'breast'
  ```
  或者更寬鬆、只斷言結果（若你不確定內部是否真的完全不噴 warning）：先用 `warnings.catch_warnings(record=True)` 收集，斷言沒有 `UserWarning`。**建議先跑一次看 `detect_cancer_group_from_series` 對同質輸入到底會不會噴 warning**，再決定用 `simplefilter('error')`（嚴格）還是只斷言結果（寬鬆）。若它其實會噴無害 warning，用嚴格版會讓測試繼續紅，那就改成寬鬆版。
- **風險**：低。只動測試檔，不動產品程式碼。

#### 3b. `tests/test_round3_bugs.py::TestK1_LeadingZeroPatientID::test_leading_zero_pk_preserved_as_string`
- **失敗原因（已實際驗證）**：`test_round3_bugs.py:66` 斷言 `assert pk.dtype == object`。在 pandas 3.0，從 Excel 讀進來的字串欄位 dtype 變成 `StringDtype(storage='python', na_value=nan)`（新的預設字串型別），不再是 `object`。**但這個測試真正要保護的行為——leading zero 沒被吃掉、`'0001234'` 沒變成 `1234`——其實是完全正常的**（我驗證過 `pk.tolist()` 仍是 `['0001234', '0005678', '0099999']`，值完全正確）。壞的只是那句過度嚴格的 dtype 斷言。
- **修法**：把 dtype 斷言放寬，改成斷言「值正確且不是數值型別」。建議：
  ```python
  pk = dec._raw_df['PK_raw']
  # 重點是 leading zero 有保留、不是被當數字讀進來（object 或 pandas 3.0 的 string 都可接受）
  assert not pd.api.types.is_numeric_dtype(pk)
  assert pk.tolist() == ['0001234', '0005678', '0099999']
  ```
  這樣在 pandas 2.x（object）與 3.0（StringDtype）都會過，且仍能抓到「被當成整數讀進來」這個真正的 regression。
- **注意**：同檔的 `test_mixed_numeric_and_zero_padded_pk`（:69）也做類似檢查，但它只斷言 `.tolist()` 的值，沒有斷言 dtype，所以**沒有壞**。改 3b 時不用動它，但可以順手確認它仍綠。
- **風險**：低。只動測試檔的斷言方式，保留原本要保護的行為。

#### 完成後驗證
```bash
python -m pytest tests/test_adversarial.py::TestICDO3EdgeCases::test_series_all_same_group \
                 tests/test_round3_bugs.py::TestK1_LeadingZeroPatientID -q
python -m pytest tests/ -q     # 目標：943 passed, 0 failed
```

---

### 工作 4 — 加上 GitHub Actions CI　【優先：中】

#### 這是什麼問題
本 repo 完全沒有 CI（開 PR 後 `get_check_runs` 回傳 0 個 check）。沒有 `.github/` 目錄。

#### 為什麼值得做（我的判斷）
這個專案有 900+ 測試、且是醫療正確性導向的工具，正是最該有 CI 的那種專案——每次 PR 自動跑測試能防止回歸。**我建議加**，但有一個前提：**先做完工作 3**，否則 CI 一上線就是紅的（那兩個既有失敗會讓 CI fail），反而讓 CI 失去意義。所以順序上工作 4 應排在工作 3 之後。優先度定「中」而非「高」，是因為它不影響工具本身的正確性，只是流程改善。

#### 具體修改步驟
新增檔案 `.github/workflows/test.yml`：
```yaml
name: tests

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ['3.9', '3.11', '3.12']
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ -q
```

#### 風險與注意
- **matrix 的 Python 版本要小心**：本地是 3.11 + pandas 3.0。pandas 3.0 需要 Python ≥3.10，所以在 CI 的 3.9 job 上會裝到較舊的 pandas（<3.0），那兩個「pandas 3.0 才會壞」的測試在 3.9 上本來就是綠的——但反過來，**若你只做寬鬆修法（工作 3），要確認修完的測試在新舊 pandas 都過**（工作 3 我給的修法已考慮這點：用 `is_numeric_dtype` 而非寫死 dtype，兩版都過）。若懶得處理跨版本差異，可先只保留 `'3.11'` 單一版本，之後再擴充 matrix。
- `requirements.txt` 已含 pytest，所以 `pip install -r requirements.txt` 就夠跑測試，不需要額外裝。
- 不要在 CI 裡跑 `pip install -e .`，除非工作 2 已修好——否則 CI 會卡在安裝步驟。用 `pip install -r requirements.txt` 最保險。
- **這個 workflow 我沒有實際在 GitHub Actions runner 上跑過驗證**（本 session 不 push），YAML 語法我有把握，但 runner 上實際的 pandas 解析、跨版本相容性請以第一次 CI 執行結果為準，紅了就依 log 調整。

#### 完成後驗證
本地無法直接驗證 GitHub Actions。可做的替代驗證：
```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/test.yml'))"  # YAML 語法檢查
```
真正的驗證是 push 後看 PR 的 checks 是否綠。

---

### 工作 5 — ER/PR 的 Allred score 替代編碼制　【優先：低（我傾向「先不要大改」，理由見下）】

這一項我花最多時間查證 PDF。**結論先講：現有 `decode_er_pr`/`encode_er_pr` 其實已經「意外地」把 Allred 編碼制處理得相當好，所以原本擔心的「真實 Allred 資料會被誤判」風險比想像中小很多。我建議只做「補測試 + 補一個文件註解」的輕量處理，不要為它做大規模重寫。** 下面把查證結果完整交代，讓你能自己判斷。

#### 我查證到什麼程度（PDF 原文）
我讀了 `Cancer-SSF-Manual_Official-version_20251204_W.pdf` 的以下頁面（PDF index，非內部頁碼）：
- **SSF1（ER）**：index **126**（內部 p.121，欄位定義 + 兩種編碼制說明 + 範例）、**127**（內部 p.122，Allred 僅有分數時的原則 + 「ER 反應比例」對照表）、**128**（內部 p.123，Allred score 對照表 + 共用編碼 110/111/120/121/888/988/999）。
- **SSF2（PR）**：index **131**（內部 p.126）、**132**（內部 p.127）。SSF2 的 Allred 表與 SSF1 **完全相同**，只是把「ER」換成「PR」。

手冊明確寫這個欄位有**兩種編碼制**，依病理報告描述方式擇一摘錄：

**制度 A：「ER/PR 反應比例」（現有程式支援的）**
| 代碼 | 意義 |
|------|------|
| `000` | ER 0%（陰性） |
| `001-100` | 反應比例%，但染色強度未明示 |
| `S00-S99` | 強染 + 反應比例（`00`=100%） |
| `I00-I99` | 中染 + 反應比例 |
| `W00-W99` | 弱染 + 反應比例 |

**制度 B：「Allred score」（現有程式沒有明確支援的）**
- 第一碼 = 訊號強度（intensity）：`0`=None(score 0)、`W`=Weak(1)、`I`=Intermediate(2)、`S`=Strong(3)
- 第二、三碼 = 細胞比例（proportion），依陽性細胞比例上下限**取平均值**登錄：

| 第二三碼 | proportion score | positive cell % |
|----------|------------------|-----------------|
| `00` | 0 | 0% |
| `120`（**存疑，見下**） | 1 | <1% |
| `06` | 2 | 1-10%（平均約 6） |
| `22` | 3 | 11-33%（平均 22） |
| `49` | 4 | 34-66%（平均約 49） |
| `84` | 5 | ≧67%（平均約 84） |

- 手冊範例：Allred Score = 8 (3+5) → 第一碼 S、第二三碼 84 → **`S84`**。
- 手冊範例：全 0（intensity 0 + proportion 0）→ **`000`**。
- **僅知道 Allred 總分、不知細分**時的 fallback（共用編碼）：Score 0 → `000`；Score 1-2 → `120`（陰性，比例不明）；Score 3-8 → `110`（陽性，比例不明）。

#### 關鍵洞察：為什麼現有程式其實已經處理得不錯
Allred 制度 B 的「第二三碼」被**刻意設計成等於平均陽性細胞百分比**（00, 06, 22, 49, 84 就是各區間的中位/平均%）。而現有 `decode_er_pr` 的邏輯是「第一碼是強度字母、後兩碼當成百分比」。這兩者**天然吻合**。我實際跑了現有 decoder 驗證（下表是實測輸出）：

| 輸入代碼 | 現有 decode 輸出 | 是否合理 |
|----------|------------------|----------|
| `000` | ER Negative (0%) | ✅ 正確（Allred 全 0 = 陰性） |
| `S84` | ER Positive (Strong staining, 84%) | ✅ 忠實反映 Allred 3+5 |
| `S06` | ER Positive (Strong staining, 6%) | ✅ 忠實反映 Allred 3+2 |
| `I22` | ER Positive (Intermediate staining, 22%) | ✅ |
| `S49` | ER Positive (Strong staining, 49%) | ✅ |
| `S00` | ER Positive (Strong staining, 100%) | ✅（制度 A；Allred 不會產生 S00，見下） |
| `120` | ER Negative (<1% or not specified) | ✅（共用碼已處理） |
| `110` | ER Positive (proportion unclear) | ✅（共用碼已處理） |

**為什麼 `S00`=100% 的修正（本 session 做的）跟 Allred 不衝突**：在 Allred 制度 B，若強度是 Strong，代碼絕不會是 `S00`（proportion `00` 代表 0% 細胞，而強度 Strong 又代表有訊號，兩者矛盾；Allred 的「陰性」一律編為 `000`，第一碼是數字 0 不是字母 S）。所以 `S00`/`W00`/`I00` 只會由制度 A 產生，一律代表 100%。本 session 的修正正確、與 Allred 無衝突。

#### 唯一真正不確定的地方（請人工確認）
Allred proportion score **1（<1%）**那一列，pypdf 擷取出來是「`120` / 1 / <1%」。但 `120` 是**三碼**，不符合「第二三碼」應為兩碼的格式，而且 `120` 剛好又是下方「共用編碼」裡的陰性碼。**我的最佳猜測**是：當 proportion score = 1（<1%，無法取兩碼平均）時，整個欄位直接編為共用碼 `120`（陰性、比例<1%），而非「字母 + 兩碼」。**但這是 pypdf 表格擷取跳行造成的判讀，我無法百分百確定。**

> ⚠️ **給接手者**：這一格請**用真正的 PDF 檢視器打開 `docs/Cancer-SSF-Manual` 內部頁碼 p.123（SSF1）或 p.127（SSF2）親眼確認**那個表格，不要照抄 pypdf 的輸出。若你也是模型、無法開圖形 PDF，就**標記為待人工確認、不要寫死任何 <1% 的特殊處理**。

#### 我的判斷與建議（為什麼定「低」優先）
1. **實務上台灣醫院哪種較常見？我不確定。** 我沒有台灣病理報告的實務分佈資料，無法從程式碼或 PDF 得知。手冊只說「依病理報告描述方式擇一」。**請勿假裝知道**——若要做決策，需人工向臨床端確認。
2. 因為現有 decoder 對「有完整分數的 Allred 碼」已能忠實解讀（如上表），真正的功能缺口很小，主要只有 proportion score 1（<1%）那一格，而那格又存疑。為這麼小的缺口做「判斷輸入是哪種制度、避免衝突」的大改，CP 值低、且引入 bug 的風險高於效益。
3. **建議的輕量處理（若要動的話）**：
   - 在 `tests/test_encoders.py` 的 `TestEncodeErPr` 補一組**Allred 範例代碼的解碼測試**（`000`, `S84`, `S06`, `I22`, `S49`, `120`, `110`），斷言它們解碼成「臨床上合理」的文字（就是上表那些），把「Allred 已被涵蓋」這件事**用測試釘住**，避免未來有人「好心」去改 `decode_er_pr` 反而弄壞 Allred。
   - 在 `decode_er_pr` 的 docstring（`decoders.py:16-21`）補一句：說明本欄位有 A/B 兩種編碼制，制度 B（Allred）的 00/06/22/49/84 因為等於平均陽性%，被現有「字母+百分比」邏輯自然涵蓋；proportion score 1（<1%）尚未特別處理、且其代碼待人工對 PDF 確認。
   - **不要**去加「先判斷是 A 還是 B 制度」的分支邏輯——兩制度的可辨識代碼幾乎不重疊（唯一可能重疊的 `S00` 已論證只屬制度 A），加判斷反而製造 bug。
4. 若後續拿到真實資料、發現 Allred 的 <1% 碼真的以某特定形式出現且被誤判，再回頭針對那一格做最小修正即可。

#### 完成後驗證（若採輕量處理）
```bash
python -m pytest tests/test_encoders.py::TestEncodeErPr -q
python -m pytest tests/ -q
```

---

### 工作 6 — 我在讀程式碼時的其他觀察與建議

以下都標明是**我自己的判斷**，不是既定事實，供接手者參考，優先度普遍偏低。

1. **`compare_roundtrip` 的 leading-zero 比對是靠 `lstrip('0')`（`roundtrip.py:104`）**【判斷：目前 OK，但值得留意】
   round-trip 會把 `000`→`0`、`022`→`22`、`06`→`6` 視為「等價、非真正 mismatch」（因為 `o.lstrip('0') == g.lstrip('0')`）。我實測 Allred 的 `000`/`022`/`06` 正是走這條路被判為等價。這在臨床意義上是對的（前導零不改變數值），但要注意一個邊界：若某欄位真的區分 `0` 與 `00`（例如某些 sentinel），`lstrip('0')` 會把兩者都變成空字串再比較，第 104 行有 `o.lstrip('0') != ''` 的保護避免「全 0」誤判等價。我看過覺得目前是安全的，但如果未來新增「以前導零數量表達意義」的欄位，這條規則要重新檢視。

2. **`encode_*` 系列的「原樣通過」容錯是雙面刃**【判斷：設計上正確，但要有測試守住】
   `encoders.py` 裡大量 `return v`（無法辨識就原樣回傳）刻意模仿 decoder 的容錯。這讓 round-trip 不會 crash，但也意味著「一個 encoder 少寫了一個 label pattern」時，它不會報錯、而是靜默把 label 文字當成 raw code 傳回去，可能產生看似成功實則錯誤的 `_raw` 值。目前是靠 `test_encoders.py` 的 round-trip 測試（decode→encode 應還原）來守住這件事。**建議**：未來任何時候擴充 decoder 的輸出格式（新增一種 label），一定要同步在對應 encoder 加 pattern **並加 round-trip 測試**，否則會靜默漏編碼。這是這套雙向設計最脆弱的地方。

3. **測試涵蓋率的落差**【判斷：可加強但非緊急】
   `test_encoders.py` 對 SSF 生醫欄位的 round-trip 覆蓋很紮實，但**結構性欄位**（AJCC/PRESTYPE/STYPE95/PRESLNSCO/SLNSCO95/EBRT/LN_POSITI，走 `STRUCTURAL_FIELD_ENCODERS`）的 round-trip 覆蓋我看得比較少。建議補一組「用 `core.py` 的 `AJCC_MAP`/`PRESTYPE_MAP` 等每個 key 做 decode→encode 還原」的參數化測試，確保結構欄位的雙向一致性也被釘住。

4. **`README.md` 的測試數字與實際不符**【判斷：小事，可順手修】
   README 寫「**940+ tests**」，實際是 943（941 pass + 2 既有 fail）。做完工作 3 後會是 943 全綠。順手把數字對齊即可，非必要。

5. **PDF 內部頁碼 vs PDF index 的落差固定為 +5**【給後續查 PDF 的人的提示】
   `Cancer-SSF-Manual` 的「內部頁碼」比 `PdfReader` 的 0-based index 小 5（例如內部 p.121 = index 126）。程式碼註解與 commit message 引用的頁碼都是**內部頁碼**。你要用 pypdf 查證某個註解引用的頁碼時，記得 `index = 內部頁碼 + 5`。`Longform-Manual` 我這次沒逐頁核對其 offset，若要查它請自己先抓一頁對一下 offset 再換算。

---

## 附錄：常用指令速查

```bash
# 跑全套測試
python -m pytest tests/ -q

# 跑單一測試檔 / 單一測試
python -m pytest tests/test_encoders.py -q
python -m pytest "tests/test_round3_bugs.py::TestK1_LeadingZeroPatientID" -q

# 產生合成資料並解碼（手動驗證用；支援 breast/lung/colorectum）
python -m tcr_decoder --synth lung --n 20 --seed 1 --out /tmp/lung.xlsx --decode

# round-trip 檢查（解碼→編碼→比對）
python -m tcr_decoder registry.xlsx --roundtrip

# 讀 PDF 某一頁（記得 index = 內部頁碼 + 5，for Cancer-SSF-Manual）
python3 -c "from pypdf import PdfReader; print(PdfReader('docs/Cancer-SSF-Manual_Official-version_20251204_W.pdf').pages[126].extract_text())"

# 環境問題急救
pip install pandas numpy openpyxl pytest        # 繞過壞掉的 pip install -e .
pip install --force-reinstall cffi              # 若 import cryptography 報 cffi 錯
```

## 附錄：本文件涉及的關鍵檔案與行號

| 工作 | 檔案:行 | 內容 |
|------|---------|------|
| 1 | `tcr_decoder/ssf_registry.py:969` | lung SSF3 的 column_name（要改名） |
| 1 | `tcr_decoder/ssf_registry.py:1373` | `('lung','SSF3')` encoder wiring（key 不用改） |
| 1 | `tcr_decoder/core.py:586-595` | KNOWN ISSUE 註解 + KPSECOG 那行 |
| 1 | `tcr_decoder/encoder.py:81-102` | `_SSF_PIPELINE_OVERRIDDEN`（刪 lung entry） |
| 1 | `tcr_decoder/data_dictionary.py:115, 274` | Performance_Status 資料字典 |
| 1 | `tests/test_pipeline.py:82` | `REQUIRED_LUNG_SSF` |
| 1 | `tests/test_ssf_registry.py:319` | ECOG 斷言 |
| 1 | `tests/test_encoders.py:406-410` | collision 測試 |
| 2 | `pyproject.toml:3` | build-backend |
| 3a | `tests/test_adversarial.py:357` | `pytest.warns(None)` |
| 3b | `tests/test_round3_bugs.py:66` | `pk.dtype == object` |
| 4 | `.github/workflows/test.yml` | 新增檔 |
| 5 | `tcr_decoder/decoders.py:16-58` | `decode_er_pr` |
| 5 | `tcr_decoder/encoders.py:107-149` | `encode_er_pr` |
