"""
CLI entry point for TCR Decoder v2.0.

Usage:
    # Decode a real registry file (auto-detects cancer type)
    python -m tcr_decoder input.xlsx
    python -m tcr_decoder input.xlsx output.xlsx
    python -m tcr_decoder input.xlsx --cancer lung

    # Validate only (no export)
    python -m tcr_decoder input.xlsx --validate-only

    # Generate synthetic data + decode (for testing)
    python -m tcr_decoder --synth breast --n 100
    python -m tcr_decoder --synth lung   --n 200 --seed 2024

    # List supported cancer groups
    python -m tcr_decoder --list-cancers

    # Show SSF field definitions for a cancer group
    python -m tcr_decoder --ssf-info lung
"""

import sys
import argparse
from pathlib import Path


def _print_banner():
    print('┌─────────────────────────────────────────────────────┐')
    print('│  Taiwan Cancer Registry Decoder  v2.0               │')
    print('│  Multi-cancer SSF decoding │ Auto cancer detection  │')
    print('└─────────────────────────────────────────────────────┘')


def cmd_list_cancers():
    """List all supported cancer groups."""
    from tcr_decoder import list_supported_cancers
    df = list_supported_cancers()
    print('\nSupported cancer groups:\n')
    print(f"  {'Group':15s} {'Site':35s} {'ICD-O-3':22s} {'Custom SSF'}")
    print('  ' + '-' * 80)
    for _, row in df.iterrows():
        grp = row['Cancer_Group']
        label = row['Site_Label']
        codes = row['ICD_O_3_Codes']
        custom = f"{row['Custom_Decoders']}/{row['Total_SSF_Fields']}"
        marker = '●' if row['Custom_Decoders'] > 0 else '○'
        print(f"  {marker} {grp:14s} {label:35s} {codes:22s} {custom}")
    print()
    print('  ● = has custom clinical decoders   ○ = generic numeric passthrough')
    print('  Run --ssf-info <group> to see SSF field details\n')


def cmd_ssf_info(cancer_group: str):
    """Show SSF field definitions for a cancer group."""
    from tcr_decoder import get_ssf_profile, detect_cancer_group
    profile = get_ssf_profile(cancer_group)
    print(f'\nSSF Field Definitions — {profile.site_label}')
    print(f'ICD-O-3: {", ".join(profile.site_codes) or "(any)"}')
    print(f'{profile.notes}\n')
    print(f"  {'SSF':6s} {'Output Column':35s} {'Decoder':10s} {'Description'}")
    print('  ' + '-' * 90)
    for ssf_key, fdef in profile.fields.items():
        tag = '✓ custom' if fdef.decoder else '○ generic'
        unit = f' [{fdef.unit}]' if fdef.unit else ''
        print(f"  {ssf_key:6s} {fdef.column_name:35s} {tag:10s} {fdef.description[:45]}{unit}")
    print()


def cmd_synth(cancer: str, n: int, seed: int, out: str, decode: bool):
    """Generate synthetic data, optionally decode it."""
    from tcr_decoder.synth import SyntheticTCRGenerator
    try:
        gen = SyntheticTCRGenerator(cancer_group=cancer, n=n, seed=seed)
    except ValueError as e:
        print(f'ERROR: {e}')
        print(f'Supported: {SyntheticTCRGenerator.SUPPORTED}')
        sys.exit(1)

    gen.generate()
    print(gen.summary())
    out_path = out or f'synthetic_{cancer}.xlsx'
    gen.to_excel(out_path)

    if decode:
        from tcr_decoder import TCRDecoder
        clean_path = out_path.replace('.xlsx', '_clean.xlsx')
        print(f'\nDecoding → {clean_path}')
        dec = TCRDecoder(out_path)
        dec.load(skip_input_check=True).decode().validate().export(clean_path)
        _print_flags(dec.flags)
        print(f'\n✓ Clean data: {clean_path}')


def cmd_decode(input_path: str, output_path: str, sheet: str,
               cancer_group: str, validate_only: bool):
    """Decode a real registry file."""
    from tcr_decoder import TCRDecoder
    inp = Path(input_path)
    if not inp.exists():
        print(f'ERROR: Input file not found: {input_path}')
        sys.exit(1)

    out = output_path or inp.stem + '_clean.xlsx'

    decoder = TCRDecoder(inp, sheet_name=sheet,
                         cancer_group=cancer_group if cancer_group else None)
    decoder.load()
    decoder.decode()
    decoder.validate()

    print(f'\n{"="*60}')
    print(f'  CLINICAL FLAGS SUMMARY  ({decoder.cancer_group or "generic"} profile)')
    print(f'{"="*60}')
    _print_flags(decoder.flags)

    if not validate_only:
        decoder.export(out)
        print(f'\n✓ Output saved to: {out}')

    print(f'\n  Cancer group : {decoder.cancer_group}')
    print(f'  Patients     : {len(decoder.clean)}')
    print(f'  Columns      : {len(decoder.clean.columns)}')
    print(f'  Flags total  : {len(decoder.flags)}')


def _print_flags(flags):
    """Pretty-print clinical flags grouped by severity."""
    import pandas as pd
    if len(flags) == 0:
        print('\n  ✓ No flags raised.')
        return
    for sev in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
        subset = flags[flags['Severity'] == sev]
        if len(subset) == 0:
            continue
        icons = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🔵', 'INFO': '⚪'}
        print(f'\n  {icons.get(sev,"•")} [{sev}] ({len(subset)} flags)')
        for flag, group in subset.groupby('Flag'):
            n = len(group)
            detail = group.iloc[0]['Detail'][:75]
            print(f'    • {flag} ({n}×): {detail}')


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(
        prog='python -m tcr_decoder',
        description='Taiwan Cancer Registry Decoder v2.0 — Multi-cancer SSF decoding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python -m tcr_decoder registry.xlsx                     # auto-detect cancer
  python -m tcr_decoder registry.xlsx --cancer lung        # force lung profile
  python -m tcr_decoder --synth breast --n 100 --decode   # test with synthetic data
  python -m tcr_decoder --list-cancers                     # show all cancer groups
  python -m tcr_decoder --ssf-info colorectum              # SSF fields for CRC
        """)

    # Mode: synth vs decode vs info
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--synth', metavar='CANCER',
                      help='Generate synthetic data for CANCER group (breast/lung/colorectum/...)')
    mode.add_argument('--list-cancers', action='store_true',
                      help='List all supported cancer groups and exit')
    mode.add_argument('--ssf-info', metavar='CANCER',
                      help='Show SSF field definitions for a cancer group')

    # Decode mode args
    parser.add_argument('input', nargs='?', default=None,
                        help='Input Excel file (required unless --synth / --list-cancers)')
    parser.add_argument('output', nargs='?', default=None,
                        help='Output Excel file (default: <input>_clean.xlsx)')
    parser.add_argument('--sheet', default='All_Fields_Decoded',
                        help='Sheet name in input Excel (default: All_Fields_Decoded)')
    parser.add_argument('--cancer', default=None,
                        metavar='GROUP',
                        help='Force cancer group instead of auto-detecting from TCODE1')
    parser.add_argument('--validate-only', action='store_true',
                        help='Run validation only, do not export output file')

    # Synth mode args
    parser.add_argument('--n', type=int, default=100,
                        help='Number of synthetic patients (default: 100, used with --synth)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42, used with --synth)')
    parser.add_argument('--out', default='',
                        help='Output path (used with --synth)')
    parser.add_argument('--decode', action='store_true',
                        help='Also decode the synthetic data (used with --synth)')

    args = parser.parse_args()

    _print_banner()

    if args.list_cancers:
        cmd_list_cancers()
    elif args.ssf_info:
        cmd_ssf_info(args.ssf_info)
    elif args.synth:
        cmd_synth(cancer=args.synth, n=args.n, seed=args.seed,
                  out=args.out, decode=args.decode)
    elif args.input:
        cmd_decode(input_path=args.input, output_path=args.output,
                   sheet=args.sheet, cancer_group=args.cancer,
                   validate_only=args.validate_only)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
