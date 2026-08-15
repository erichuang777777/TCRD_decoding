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
