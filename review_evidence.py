"""Print the recorded E20 endpoints or verify the distributed evidence, using the standard library."""
import argparse
import csv
import hashlib
import os
import re
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'SHA256SUMS.txt'
GROUPS = ('O2M', 'O2O', 'Dual')


def digest(path):
    with path.open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def expected_files():
    expected = {}
    for line in MANIFEST.read_text(encoding='utf-8-sig').splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
        if not match:
            raise ValueError('Malformed checksum line')
        checksum, name = match.groups()
        path = PurePosixPath(name)
        if (path.is_absolute() or '..' in path.parts or ':' in name or '\\' in name
                or path.as_posix() != name or name in expected or name == MANIFEST.name):
            raise ValueError(f'Invalid or repeated manifest path: {name}')
        expected[name] = checksum
    return expected


def verify():
    expected = expected_files()
    actual = set()
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__'}]
        for file in files:
            path = Path(folder) / file
            if path != MANIFEST and file != '.git':
                actual.add(path.relative_to(ROOT).as_posix())
    missing = expected.keys() - actual
    extra = actual - expected.keys()
    changed = [p for p in sorted(actual & expected.keys()) if digest(ROOT / p) != expected[p]]
    for label, names in [('missing', missing), ('extra', extra), ('changed', changed)]:
        for name in sorted(names):
            print(f'{label}: {name}')
    if missing or extra or changed:
        return 1
    print(f'{len(actual)} files verified, SHA-256 {digest(MANIFEST)} (manifest).')
    return 0


def rows(name, expected):
    path = ROOT / name
    if digest(path) != expected[name]:
        raise ValueError(f'Evidence hash mismatch: {name}')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def results(weights_dir):
    expected = expected_files()
    subset = rows('evidence/validation_all.csv', expected)
    full = rows('full_val/results.csv', expected)
    weights = {r['filename']: r for r in rows('evidence/weight-fingerprints.csv', expected)}
    print('E20 last | AP (%) | subset 250 overlaps with the 2,510-image official validation set')
    print('group  subset AP50-95  subset AP50  full AP50-95  full AP50')
    for group in GROUPS:
        a, = [r for r in subset if (r['scope'], r['epoch'], r['group']) == ('subset250', '20', group)]
        b, = [r for r in full if (r['epoch'], r['group']) == ('20', group)]
        name = f'{group}_E20_last.pt'
        info = weights[name]
        if b['checkpoint_sha256'] != info['sha256']:
            raise ValueError(f'Checkpoint fingerprints disagree for {group}')
        if weights_dir is not None:
            checkpoint = weights_dir / name
            if checkpoint.stat().st_size != int(info['bytes']) or digest(checkpoint) != info['sha256']:
                raise ValueError(f'Checkpoint bytes differ for {group}')
        values = [100 * float(r[key]) for r in (a, b) for key in ('ap5095', 'ap50')]
        print(f'{group:5s}  {values[0]:14.3f}  {values[1]:11.3f}  {values[2]:12.3f}  {values[3]:9.3f}')
    print('Checked three local checkpoint files.' if weights_dir else 'Checked recorded checkpoint fingerprints. Weight binaries are external.')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest='command', required=True)
    cmds.add_parser('verify', help='check repository file membership and SHA-256')
    p = cmds.add_parser('results', help='read saved endpoint tables')
    p.add_argument('--weights-dir', type=Path, help='optional external directory containing the three named E20 last files')
    args = parser.parse_args()
    try:
        return verify() if args.command == 'verify' else results(args.weights_dir)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    sys.exit(main())
