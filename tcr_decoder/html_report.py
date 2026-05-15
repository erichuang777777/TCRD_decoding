"""Interactive HTML data browser for decoded TCR clinical data.

Generates a self-contained HTML file (Bootstrap + DataTables via CDN) with:
- Summary statistics header cards
- Filterable, sortable, paginated table of all decoded fields

Requires internet access for CDN assets (Bootstrap 5, DataTables 1.13).

Usage
-----
    from tcr_decoder.html_report import generate_html_report
    generate_html_report(decoder.clean, "browser.html", cancer_group="breast")
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

_SKIP_SUFFIXES = ('_raw',)
_SKIP_PREFIXES = ('QA_', 'SSF_Raw')
_ALWAYS_INCLUDE = {'Patient_ID'}
_SKIP_EXACT = {'TCODE1', 'TCODE2'}
_MAX_CARDINALITY = 60


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for col in df.columns:
        if col in _SKIP_EXACT:
            continue
        if any(col.startswith(p) for p in _SKIP_PREFIXES):
            continue
        if any(col.endswith(s) for s in _SKIP_SUFFIXES):
            continue
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if col not in _ALWAYS_INCLUDE and s.dtype == object and s.nunique() > _MAX_CARDINALITY:
            continue
        keep.append(col)

    if 'Patient_ID' in keep:
        keep = ['Patient_ID'] + [c for c in keep if c != 'Patient_ID']
    return df[keep]


def _summary_stats(df: pd.DataFrame) -> list[tuple[str, str]]:
    cards: list[tuple[str, str]] = [('Total Patients', f'{len(df):,}')]

    if 'Age_at_Diagnosis' in df.columns:
        age = pd.to_numeric(df['Age_at_Diagnosis'], errors='coerce').dropna()
        if len(age):
            q1, med, q3 = age.quantile([0.25, 0.50, 0.75])
            cards.append(('Age (median)', f'{med:.0f} yrs (IQR {q1:.0f}–{q3:.0f})'))

    for col in ('Pathologic_Stage_Combined', 'Clinical_Stage', 'Combined_Stage'):
        if col in df.columns:
            vc = df[col].value_counts()
            if len(vc):
                cards.append(('Top Stage', f'{vc.index[0]} ({100 * vc.iloc[0] / len(df):.0f}%)'))
            break

    for col, pos_val, label in [
        ('ER_Status', 'Positive', 'ER+'),
        ('HER2_Status', 'Positive', 'HER2+'),
        ('EGFR_Mutation', 'Mutated', 'EGFR mut'),
        ('MSI_Status', 'MSI-H', 'MSI-H'),
    ]:
        if col in df.columns:
            total = df[col].notna().sum()
            if total:
                cards.append((label, f'{100 * (df[col] == pos_val).sum() / total:.0f}%'))

    if 'OS_Days' in df.columns:
        os_mo = pd.to_numeric(df['OS_Days'], errors='coerce').dropna() / 30.44
        if len(os_mo):
            cards.append(('Median OS', f'{os_mo.median():.1f} mo'))

    return cards


_TEMPLATE = """\
<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="https://cdn.datatables.net/1.13.7/css/dataTables.bootstrap5.min.css" rel="stylesheet">
<style>
body{font-size:.85rem;background:#f5f6fa}
.stat-card{background:#fff;border:1px solid #e3e6f0;border-radius:10px;padding:12px 20px;min-width:120px}
.stat-card .label{font-size:.72rem;color:#6c757d;text-transform:uppercase;letter-spacing:.5px}
.stat-card .value{font-size:1.2rem;font-weight:700;color:#2c3e50}
table.dataTable thead th{background:#2c3e50;color:#fff;border:none}
table.dataTable tbody td{max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
table.dataTable tbody td:hover{white-space:normal;max-width:none;background:#fffde7;cursor:default}
.dataTables_wrapper .dataTables_filter input{border-radius:20px;border:1px solid #ced4da;padding:4px 14px}
</style>
</head>
<body>
<nav class="navbar navbar-dark" style="background:#2c3e50">
  <div class="container-fluid">
    <span class="navbar-brand fw-bold">__TITLE__</span>
    <small class="text-white-50">Generated __DATE__</small>
  </div>
</nav>
<div class="container-fluid py-3">
  <div class="d-flex flex-wrap gap-2 mb-3">
__STAT_CARDS__
  </div>
  <div class="card shadow-sm">
    <div class="card-body p-2">
      <table id="t" class="table table-sm table-hover" style="width:100%">
        <thead><tr>__HEADERS__</tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>
<script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
<script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/1.13.7/js/dataTables.bootstrap5.min.js"></script>
<script>
$(function(){
  $('#t').DataTable({
    data:__JSON_DATA__,
    columns:__JSON_COLS__.map(c=>({title:c,data:c,defaultContent:''})),
    pageLength:25,
    lengthMenu:[[10,25,50,100,-1],[10,25,50,100,'All']],
    order:[],scrollX:true,
    dom:'<"d-flex justify-content-between align-items-center mb-2"lf>rt<"d-flex justify-content-between mt-2"ip>',
    language:{search:'',searchPlaceholder:'Search all columns...',
      lengthMenu:'Show _MENU_ rows',info:'Patients _START_–_END_ of _TOTAL_',
      infoEmpty:'No patients',zeroRecords:'No match'}
  });
});
</script>
</body>
</html>
"""


def generate_html_report(
    df: pd.DataFrame,
    output_path: str | Path,
    cancer_group: str = 'generic',
) -> Path:
    """Generate an interactive HTML data browser for decoded clinical data.

    Parameters
    ----------
    df : pd.DataFrame
        Decoded clinical data (``TCRDecoder.clean``).
    output_path : str | Path
        Destination file path (``.html`` added if absent).
    cancer_group : str
        Used for the report title and cancer-specific stat cards.

    Returns
    -------
    Path
        Resolved path of the saved HTML file.

    Notes
    -----
    Requires internet access (Bootstrap 5, DataTables loaded from CDN).
    """
    out = Path(output_path).with_suffix('.html')
    browser_df = _select_columns(df)
    stats = _summary_stats(df)

    stat_cards = '\n'.join(
        f'    <div class="stat-card">'
        f'<div class="label">{lbl}</div><div class="value">{val}</div></div>'
        for lbl, val in stats
    )
    headers = ''.join(f'<th>{c}</th>' for c in browser_df.columns)

    clean = browser_df.where(pd.notna(browser_df), None)
    records = clean.to_dict(orient='records')

    title = f"TCR Clinical Browser — {cancer_group.replace('_', ' ').title()} (n={len(df):,})"

    html = (
        _TEMPLATE
        .replace('__TITLE__', title)
        .replace('__DATE__', date.today().isoformat())
        .replace('__STAT_CARDS__', stat_cards)
        .replace('__HEADERS__', headers)
        .replace('__JSON_DATA__', json.dumps(records, ensure_ascii=False))
        .replace('__JSON_COLS__', json.dumps(list(browser_df.columns), ensure_ascii=False))
    )

    out.write_text(html, encoding='utf-8')
    return out.resolve()
