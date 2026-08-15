# -*- coding: utf-8 -*-
"""Generate tcr_decoder/surgery_codes.py from Appendix B of the Longform manual.

    python scripts/generate_surgery_codes.py [--review]

Appendix B gives one surgery-code table per primary site. The tables are
hierarchical, and the hierarchy is carried ONLY by indentation:

    300 Skin-sparing mastectomy
      310 WITHOUT removal of uninvolved contralateral breast
        311 Reconstruction, NOS
    400 Nipple-sparing mastectomy
      410 WITHOUT removal of uninvolved contralateral breast

"WITHOUT removal of uninvolved contralateral breast" is the printed text of
both 310 and 410. Taking it verbatim gives two codes the same decoded label,
which makes encode() unable to tell them apart -- `encode(decode('410'))`
would hand back '310' and silently rewrite the operation. So each label is
qualified with its ancestors.

The indentation is lost by PDF-to-markdown conversion, so this reads the PDF
directly and clusters the x-coordinate of each code line into levels. A few
pages have noisy coordinates (a child printed further left than its parent);
those are detected, listed, and taken from OVERRIDES instead of guessed.

Nothing here is trusted on its own: the generated tables must still pass
test_codebook_conformance.py, which enumerates every code and requires decode
to be injective and encode(decode(c)) == c.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    sys.exit('PyMuPDF is required: pip install pymupdf')

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / 'docs' / 'Longform-Manual_Official-version_20251224_W-1.pdf'
OUT = ROOT / 'tcr_decoder' / 'surgery_codes.py'

# Appendix B runs from the Oral Cavity table to the last site table. Page
# numbers are 0-based PDF indices; the printed page number is 8 lower.
FIRST_PAGE, LAST_PAGE = 348, 399

CODE_RE = re.compile(r'^([0-9][0-9A-Z][0-9])\s+(\S.*)$')
BARE_CODE_RE = re.compile(r'^([0-9][0-9A-Z][0-9])\s*$')
# A prose line that introduces the codes indented beneath it.
HEADING_RE = re.compile(r'^(Any combination|Combination) .*WITH$', re.I)
LEVEL_TOLERANCE = 12.0     # points; two x within this are the same level

PROSE_PREFIX = (
    '[', 'No specimen', 'Specimen sent', 'Any combination', 'Code ', 'Codes',
    'Note:', '附錄B', '(Except', 'Do not code', 'T-Code', 'For invasive',
    'Surgical procedures', 'Debulking is', 'Unknown whether', 'The ', 'A ',
    'This ', 'If ', 'Use ', 'Assign ', 'Includes', 'Excludes',
)

# Sites whose printed coordinates are too noisy to derive the nesting from.
# Transcribed by hand from the printed tables.
OVERRIDES = {
    'Breast': {
        '310': 'Skin-sparing mastectomy WITHOUT removal of the contralateral breast',
        '311': 'Skin-sparing mastectomy WITHOUT contralateral, reconstruction NOS',
        '312': 'Skin-sparing mastectomy WITHOUT contralateral, tissue reconstruction',
        '313': 'Skin-sparing mastectomy WITHOUT contralateral, implant reconstruction',
        '314': 'Skin-sparing mastectomy WITHOUT contralateral, combined reconstruction',
        '320': 'Skin-sparing mastectomy WITH removal of the contralateral breast',
        '321': 'Skin-sparing mastectomy WITH contralateral, reconstruction NOS',
        '322': 'Skin-sparing mastectomy WITH contralateral, tissue reconstruction',
        '323': 'Skin-sparing mastectomy WITH contralateral, implant reconstruction',
        '324': 'Skin-sparing mastectomy WITH contralateral, combined reconstruction',
        '410': 'Nipple-sparing mastectomy WITHOUT removal of the contralateral breast',
        '411': 'Nipple-sparing mastectomy WITHOUT contralateral, reconstruction NOS',
        '412': 'Nipple-sparing mastectomy WITHOUT contralateral, tissue reconstruction',
        '413': 'Nipple-sparing mastectomy WITHOUT contralateral, implant reconstruction',
        '414': 'Nipple-sparing mastectomy WITHOUT contralateral, combined reconstruction',
        '420': 'Nipple-sparing mastectomy WITH removal of the contralateral breast',
        '421': 'Nipple-sparing mastectomy WITH contralateral, reconstruction NOS',
        '422': 'Nipple-sparing mastectomy WITH contralateral, tissue reconstruction',
        '423': 'Nipple-sparing mastectomy WITH contralateral, implant reconstruction',
        '424': 'Nipple-sparing mastectomy WITH contralateral, combined reconstruction',
        '510': 'Areolar-sparing mastectomy WITHOUT removal of the contralateral breast',
        '530': 'Areolar-sparing mastectomy WITHOUT contralateral, reconstruction NOS',
        '540': 'Areolar-sparing mastectomy WITHOUT contralateral, tissue reconstruction',
        '550': 'Areolar-sparing mastectomy WITHOUT contralateral, implant reconstruction',
        '560': 'Areolar-sparing mastectomy WITHOUT contralateral, combined reconstruction',
        '520': 'Areolar-sparing mastectomy WITH removal of the contralateral breast',
        '570': 'Areolar-sparing mastectomy WITH contralateral, reconstruction NOS',
        '580': 'Areolar-sparing mastectomy WITH contralateral, tissue reconstruction',
        '590': 'Areolar-sparing mastectomy WITH contralateral, implant reconstruction',
        '630': 'Areolar-sparing mastectomy WITH contralateral, combined reconstruction',
        '610': 'Total (simple) mastectomy WITHOUT removal of the contralateral breast',
        '640': 'Total (simple) mastectomy WITHOUT contralateral, reconstruction NOS',
        '650': 'Total (simple) mastectomy WITHOUT contralateral, tissue reconstruction',
        '660': 'Total (simple) mastectomy WITHOUT contralateral, implant reconstruction',
        '670': 'Total (simple) mastectomy WITHOUT contralateral, combined reconstruction',
        '620': 'Total (simple) mastectomy WITH removal of the contralateral breast',
        '680': 'Total (simple) mastectomy WITH contralateral, reconstruction NOS',
        '690': 'Total (simple) mastectomy WITH contralateral, tissue reconstruction',
        '730': 'Total (simple) mastectomy WITH contralateral, implant reconstruction',
        '740': 'Total (simple) mastectomy WITH contralateral, combined reconstruction',
        '710': 'Radical mastectomy WITHOUT removal of the contralateral breast',
        '720': 'Radical mastectomy WITH removal of the contralateral breast',
        '760': 'Bilateral mastectomy for a single tumour involving both breasts',
    },
}

# Two sites whose entire table is printed as prose, not as a code list.
EXTRA = [
    ('Hematopoietic / Reticuloendothelial / Immunoproliferative / '
     'Myeloproliferative Disease',
     ('C420', 'C421', 'C423', 'C424'), '366',
     [('980', 'All haematopoietic / reticuloendothelial / immunoproliferative '
              '/ myeloproliferative sites, WITH or WITHOUT surgical treatment'),
      ('990', 'Death certificate only')]),
    ('Unknown and Ill-Defined Primary Sites',
     ('C760', 'C761', 'C762', 'C763', 'C764', 'C765', 'C766', 'C767', 'C768',
      'C809'), '391',
     [('980', 'All unknown and ill-defined primary sites, WITH or WITHOUT '
              'surgical treatment'),
      ('990', 'Death certificate only')]),
]


def page_lines(page):
    """[(x, text)] for one PDF page, in reading order.

    The printed page number is dropped here, at the top of the page where it
    lives. It cannot be filtered later by shape: '348' is a page number and
    '110' is a surgery code, and both are bare three-digit lines.
    """
    out = []
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines', []):
            text = ''.join(s['text'] for s in line['spans'])
            if text.strip():
                out.append((round(line['spans'][0]['bbox'][0], 1),
                            ' '.join(text.split())))
    while out and (out[0][1].startswith('附錄B') or out[0][1].isdigit()):
        out.pop(0)
    return out


def expand_sites(header: str):
    """'C030–C039, C079' -> ('C030', ..., 'C039', 'C079')."""
    norm = header.replace('.', '').replace('–', '-').replace('—', '-')
    out = []
    for m in re.finditer(r'C(\d{2,3})\s*-\s*C?(\d{2,3})|C(\d{2,3})', norm):
        if m.group(3):
            out.append(f'C{m.group(3)}'.ljust(4, '0')[:4])
        else:
            lo, hi = m.group(1), m.group(2)
            if len(lo) == len(hi) and lo <= hi:
                out += [f'C{str(v).zfill(len(lo))}'.ljust(4, '0')[:4]
                        for v in range(int(lo), int(hi) + 1)]
    return tuple(sorted({s for s in out if len(s) == 4}))


def collect_blocks(doc):
    """Split Appendix B into one block per site table."""
    blocks, cur = [], None
    for idx in range(FIRST_PAGE, LAST_PAGE + 1):
        lines = page_lines(doc[idx])
        printed = str(idx - 8)
        # A new table starts where a 'Codes' line follows the site heading.
        marker = next((i for i, (_x, t) in enumerate(lines) if t == 'Codes'), None)
        if marker is not None:
            head = [t for _x, t in lines[:marker] if not t.startswith('附錄B')]
            sites = expand_sites(' '.join(head))
            if sites:
                name = re.sub(r'C\s?\d{2}[\d.–\-]*', '', head[0]).strip(' ,–-')
                cur = {'name': name, 'sites': sites, 'page': printed, 'rows': []}
                blocks.append(cur)
                lines = lines[marker + 1:]
        if cur is not None:
            cur['rows'] += [(x, t, printed) for x, t in lines]
    return blocks


def strip_note(text: str) -> str:
    """Drop a SEER note printed on the same line as the code definition."""
    return re.sub(r'\s*\[(SEER )?Note:.*$', '', text.strip()).strip()


def parse_rows(rows):
    """[(x, code, text, page)] -- code lines only, wrapped text rejoined."""
    out, pending, prev_code = [], None, False
    for x, text, page in rows:
        if text.startswith('附錄B'):
            continue
        m = CODE_RE.match(text)
        if m:
            out.append([x, m.group(1), strip_note(m.group(2)), page])
            pending, prev_code = None, True
            continue
        if BARE_CODE_RE.match(text):
            pending, prev_code = (x, BARE_CODE_RE.match(text).group(1), page), False
            continue
        if HEADING_RE.match(text):
            # "Any combination of 200 or 260-270 WITH" is a heading, not
            # commentary: the codes indented under it are combinations, and
            # dropping it makes them inherit the wrong parent.
            out.append([x, None, text, page])
            pending, prev_code = None, False
            continue
        if text.startswith(PROSE_PREFIX) or re.match(r'^[說註例（(]', text):
            pending, prev_code = None, False
            continue
        if pending:
            out.append([pending[0], pending[1], text, pending[2]])
            pending, prev_code = None, True
        elif prev_code and out and out[-1][1] is not None:
            out[-1][2] += ' ' + text          # definition wrapped onto next line
            prev_code = False
    return out


def levels_for(rows):
    """Assign a nesting level to each code line from its x-coordinate."""
    xs = sorted({x for x, *_ in rows})
    buckets = []
    for x in xs:
        if buckets and x - buckets[-1][-1] <= LEVEL_TOLERANCE:
            buckets[-1].append(x)
        else:
            buckets.append([x])
    level_of = {x: i for i, b in enumerate(buckets) for x in b}
    return [level_of[x] for x, *_ in rows]


def noisy(rows, levels):
    """True if a code is printed further left than the table's own margin.

    That only happens when the PDF's text layer misplaces a line, and it makes
    a child look like a section head. Such tables are taken from OVERRIDES.
    """
    if not rows:
        return False
    base = Counter(levels).most_common(1)[0][0]
    first = levels[0]
    return any(l < min(base, first) for l in levels)


def qualify(rows, levels, overrides):
    """Make every label unique, staying as close to the printed text as possible.

    A label is left verbatim when it is already unique in this table. Only the
    ones the manual repeats under different parents ("WITHOUT hysterectomy",
    "Facial nerve spared", "Tissue") get their ancestor chain prefixed -- so
    the common case reads exactly as printed, and the ambiguous case reads as
    "Bilateral (salpingo-) oophorectomy — WITHOUT hysterectomy".
    """
    chains, stack = [], {}
    for (_x, code, text, _page), lvl in zip(rows, levels):
        stack = {k: v for k, v in stack.items() if k < lvl}
        chains.append([stack[k] for k in sorted(stack)] + [text])
        stack[lvl] = text

    coded = [(r, c) for r, c in zip(rows, chains) if r[1] is not None]
    repeated = {t for t, n in Counter(c[-1] for _r, c in coded).items() if n > 1}
    out = []
    for (_x, code, text, _page), chain in coded:
        if code in overrides:
            out.append((code, overrides[code]))
        elif text in repeated:
            out.append((code, ' — '.join(chain)))
        else:
            out.append((code, text))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--review', action='store_true',
                    help='print every code and its resolved label, then exit')
    args = ap.parse_args()

    doc = fitz.open(PDF)
    tables, flagged = [], []
    for b in collect_blocks(doc):
        rows = parse_rows(b['rows'])
        if sum(1 for r in rows if r[1] is not None) < 2:
            continue
        levels = levels_for(rows)
        ov = OVERRIDES.get(b['name'], {})
        if noisy(rows, levels):
            flagged.append(b['name'])
            if not ov:
                # No hand transcription for a table we cannot read reliably.
                # Fall back to the verbatim labels and let the uniqueness
                # check below decide whether that is good enough.
                levels = [0] * len(rows)
        tables.append({'name': b['name'], 'sites': b['sites'], 'page': b['page'],
                       'codes': qualify(rows, levels, ov)})

    for name, sites, page, codes in EXTRA:
        tables.append({'name': name, 'sites': sites, 'page': page, 'codes': codes})

    if args.review:
        for t in tables:
            print(f"\n=== {t['name']}  (p.{t['page']}, {len(t['sites'])} sites)")
            for c, d in t['codes']:
                print(f'  {c}  {d}')
        return

    problems = []
    for t in tables:
        labels = Counter(d for _, d in t['codes'])
        codes = Counter(c for c, _ in t['codes'])
        dup_l = [k for k, v in labels.items() if v > 1]
        dup_c = [k for k, v in codes.items() if v > 1]
        if dup_l or dup_c:
            problems.append((t['name'], dup_l[:3], dup_c[:3]))

    owner, clashes = {}, []
    for t in tables:
        for s in t['sites']:
            if s in owner and owner[s] != t['name']:
                clashes.append((s, owner[s], t['name']))
            owner[s] = t['name']

    print(f"{len(tables)} site tables, {sum(len(t['codes']) for t in tables)} "
          f"codes, {len(owner)} topography codes routed")
    if flagged:
        print('  noisy coordinates (taken from OVERRIDES): '
              + ', '.join(flagged))
    for name, dup_l, dup_c in problems:
        print(f'  AMBIGUOUS {name}: labels={dup_l} codes={dup_c}')
    for s, a, c in clashes:
        print(f'  SITE CLASH {s}: {a} vs {c}')
    if problems or clashes:
        sys.exit('refusing to generate an ambiguous table')

    render(tables)


def render(tables):
    L = ['# -*- coding: utf-8 -*-',
         '"""Appendix B: surgery of primary site, one code table per site.',
         '',
         'GENERATED by scripts/generate_surgery_codes.py -- edit that, not this.',
         '',
         'Both surgery fields (外院 PRESTYPE #4.1.3, 申報醫院 STYPE95 #4.1.4)',
         'read these tables. The manual states one 編碼範圍 for both',
         '(000, 100-800, 900, 980, 990) and then defines what those codes MEAN',
         'per primary site, so the same code is a different operation in a',
         'different organ: decoding needs the topography code, not just the',
         'cancer group.',
         '',
         "Labels are qualified with their ancestors ('Skin-sparing mastectomy",
         "WITHOUT removal of the contralateral breast' rather than the printed",
         "'WITHOUT removal of uninvolved contralateral breast'), because the",
         'manual reuses the same wording under several parents and encode()',
         'has to be able to tell those codes apart.',
         '"""',
         '',
         'import re',
         'from typing import Dict, FrozenSet, Optional, Tuple',
         '',
         '',
         '# site name -> (topography codes, {code: definition}, manual page)',
         'SURGERY_TABLES: Dict[str, Tuple[Tuple[str, ...], Dict[str, str], str]] = {']
    for t in sorted(tables, key=lambda x: int(x['page'])):
        L.append(f'    {t["name"]!r}: (')
        sites = ', '.join(repr(s) for s in t['sites'])
        if len(t['sites']) == 1:
            sites += ','          # ('C199') is a string, not a 1-tuple
        L.append(f'        ({sites}),')
        L.append('        {')
        for c, d in t['codes']:
            L.append(f'            {c!r}: {d!r},')
        L.append('        },')
        L.append(f"        'Longform 附錄B p.{t['page']}',")
        L.append('    ),')
    L += ['}',
          '',
          '',
          '_BY_SITE: Dict[str, str] = {',
          '    site: name',
          '    for name, (sites, _codes, _ref) in SURGERY_TABLES.items()',
          '    for site in sites',
          '}',
          '',
          '',
          'def surgery_table_name(tcode1) -> Optional[str]:',
          '    """Which Appendix B table applies to this topography code."""',
          '    if tcode1 is None:',
          '        return None',
          "    digits = re.sub(r'[^0-9]', '', str(tcode1))",
          '    if len(digits) < 3:',
          '        return None',
          "    return _BY_SITE.get(f'C{digits[:3]}')",
          '',
          '',
          'def surgery_codes(tcode1) -> Dict[str, str]:',
          '    """{code: definition} for this site, or {} if the site is unlisted."""',
          '    name = surgery_table_name(tcode1)',
          '    return SURGERY_TABLES[name][1] if name else {}',
          '',
          '',
          'def legal_surgery_codes(tcode1) -> FrozenSet[str]:',
          '    """Official 編碼範圍 for this site\'s surgery fields."""',
          '    return frozenset(surgery_codes(tcode1))',
          '']
    OUT.write_text('\n'.join(L), encoding='utf-8')
    print(f'wrote {OUT.relative_to(ROOT)} ({len(L)} lines)')


if __name__ == '__main__':
    main()
