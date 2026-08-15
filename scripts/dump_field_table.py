# -*- coding: utf-8 -*-
"""Print one Longform field's code table straight from the manual PDF.

    python scripts/dump_field_table.py 2.7 2.11 2.13.1

Used when transcribing a field into code_ranges.py: the markdown conversion
interleaves the example table with the code table, so this reads the PDF page
the field index points at and prints it in source order.
"""
import re
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    sys.exit('PyMuPDF is required: pip install pymupdf')

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / 'docs' / 'Longform-Manual_Official-version_20251224_W-1.pdf'


def page_lines(page):
    """[(x, text)] in visual reading order.

    Sorted by y then x, not by PyMuPDF block order: a two-column code table
    is emitted as two separate blocks, so block order prints every definition
    with none of the codes.
    """
    out = []
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines', []):
            t = ' '.join(''.join(s['text'] for s in line['spans']).split())
            if t:
                x0, y0 = line['spans'][0]['bbox'][0], line['spans'][0]['bbox'][1]
                out.append((round(y0, 1), round(x0, 1), t))
    out.sort()
    return [(x, t) for _y, x, t in out]


def markdown_tables(printed_page: int):
    """The converted markdown's pipe-tables for a printed page.

    These are the safer source for a plain code table: they state the
    code/definition pairing explicitly. The PDF's own two-column layout puts
    a code and its definition on slightly different baselines, so reading it
    in y order can shift every definition by one row -- which looks entirely
    plausible and is entirely wrong.

    Markdown page numbering is the PDF page index + 1, i.e. printed + 9.
    """
    md = ROOT / 'codebook_md'
    out = []
    for path in sorted(md.glob('longform_chunk_*.md')):
        text = path.read_text(encoding='utf-8')
        for n in (printed_page + 8, printed_page + 9, printed_page + 10):
            start = text.find(f'## Page {n}\n')
            if start < 0:
                continue
            end = text.find(f'## Page {n + 1}\n', start)
            seg = text[start:end if end > 0 else len(text)]
            for block in re.split(r'\n(?=### Table)', seg):
                if block.lstrip().startswith('### Table') and '| 編碼' in block:
                    out.append(block.rstrip())
    return out


def main():
    from tcr_decoder.longform_fields import LONGFORM_FIELDS

    if len(sys.argv) < 2:
        sys.exit(__doc__)
    doc = fitz.open(PDF)
    for seq in sys.argv[1:]:
        f = LONGFORM_FIELDS.get(seq)
        if f is None:
            print(f'#{seq}: not in the field index')
            continue
        print('=' * 78)
        print(f'#{seq}  {f.name_zh} / {f.name_en}')
        print(f'欄位長度：{f.width}   編碼範圍：{f.code_range}   '
              f'（碼冊 p.{f.page}）')
        print('=' * 78)

        tables = markdown_tables(f.page)
        if tables:
            print('--- 碼表（markdown，代碼↔定義配對可信） ---')
            for block in tables:
                print(block)
            print()

        print('--- PDF 原文（版面與縮排，配對不可信） ---')
        # The code table can spill onto the next page.
        for idx in (f.page + 8, f.page + 9):
            if idx >= doc.page_count:
                continue
            for x, t in page_lines(doc[idx]):
                # Do NOT drop bare digits here: '0'-'9' are codes, and the
                # printed page number looks exactly the same.
                if t.startswith('附錄'):
                    continue
                print(f'  {x:6.1f}  {t}')
            print('  ' + '-' * 40 + f' (end of p.{idx - 8})')
        print()


if __name__ == '__main__':
    main()
