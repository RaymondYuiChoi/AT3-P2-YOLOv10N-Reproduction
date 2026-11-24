"""Plan or assemble a fresh historical AT3 replay from authorized local archives. No model is started."""
import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from review_evidence import ROOT, digest, expected_files

COMMIT = '799ff3be47d21173bcf29b351820d4b8e955e0fe'
FEAS = 'AT3-P2-FEASIBILITY-01'
EXP = 'AT3-P2-YOLOv10N-EXP20-01'
PILOT = 'AT3-P2-YOLOv10N-PILOT-01'
NUM = 'AT3-P2-YOLOv10N-NUMERIC-01'
FROZEN = ('dataset.json', 'freeze.json', 'initialization.json', 'paired_plans.json',
          'paired_image_pixels.json', 'initial_graph_costs.csv', 'weight_transfer.json')
OVERLAY = ('at3_variant.py', 'ultralytics/nn/modules/block.py')


def git(folder, *args):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(folder), *args], text=True).strip()


def replay_inputs(history):
    research = history / FEAS / 'sources/yolov10'
    if git(research, 'rev-parse', 'HEAD') != COMMIT or git(research, 'status', '--porcelain'):
        raise ValueError('A clean upstream checkout at the pinned revision is required.')
    paused = history / EXP
    decision = 'AT3_P2_YOLOv10N_数值稳定性验收与实验恢复建议_V1.0.md'
    pairs = [(ROOT / d, Path(d)) for d in ('configs', 'tools', 'patches')]
    pairs += [(paused / 'datasets', Path('datasets')), (paused / 'runs/O2M', Path('runs/O2M')),
              (history / NUM / 'source_records' / decision, Path('source_records') / decision)]
    for name in FROZEN:
        pairs.append((ROOT / 'evidence' / name, Path('evidence') / name))
    for group in ('O2M', 'O2O', 'Dual'):
        name = 'init_' + group + '.pt'
        pairs.append((history / PILOT / 'weights' / name, Path('weights') / name))
    fixed = json.loads((ROOT / 'evidence/replay-inputs.json').read_text(encoding='utf-8'))
    for rel, checksum in fixed.items():
        if digest(history / rel) != checksum:
            raise ValueError(f'Historical input mismatch: {rel}')
    for src, dest in pairs:
        if not src.exists():
            raise ValueError(f'Missing input for {dest}')
    if (paused / 'datasets/VOC2007/JPEGImages/full_val').exists():
        raise ValueError('The paused dataset already contains the derived full validation images.')
    runtime = history / FEAS / 'envs/v9/Scripts/python.exe'
    if not runtime.is_file():
        raise ValueError('The recorded Windows Python 3.10.11 environment is missing.')
    return research, pairs


def restore_model_sources(worktree):
    archive = json.loads((ROOT / 'evidence/resume_gate.json').read_text(encoding='utf-8'))
    hashes = archive['locked_files']
    for name, patch in zip(OVERLAY, ('01_historical_variants.patch', '02_psa_eval_fp32.patch')):
        git(worktree, 'apply', '--check', str(ROOT / 'patches' / patch))
        git(worktree, 'apply', str(ROOT / 'patches' / patch))
        path = worktree / name
        canonical = path.read_bytes().replace(b'\r\n', b'\n')
        view = b'\n'.join(line.rstrip(b' \t') for line in canonical.split(b'\n'))
        if view != (ROOT / 'research/yolov10' / name).read_bytes():
            raise ValueError(f'Research source differs from its review copy: {name}')
        # The historical Windows block.py used CRLF. The standalone single-head module used LF.
        path.write_bytes(canonical.replace(b'\n', b'\r\n') if name == OVERLAY[1] else canonical)
        if digest(path) != hashes['research/yolov10/' + name]:
            raise ValueError(f'Historical model source checksum mismatch: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'prepare'))
    parser.add_argument('--history-root', type=Path, required=True, help='parent of the four historical archives')
    parser.add_argument('--name', default='AT3-P2-REPLAY', help='name of a new, unused sibling directory')
    args = parser.parse_args()
    history = args.history_root.resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.name) or history.is_relative_to(ROOT):
        parser.error('Use a simple new directory name and a history root outside this repository.')
    dest = history / args.name
    if dest.exists():
        parser.error('The requested replay directory already exists.')
    try:
        for name, checksum in expected_files().items():
            if digest(ROOT / name) != checksum:
                raise ValueError(f'Changed repository file: {name}')
        research, pairs = replay_inputs(history)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'{exc}\n')
    print(f'Pinned upstream {COMMIT}. Checked archived inputs for {len(pairs)} destinations.')
    if args.mode == 'plan':
        print('Plan complete. No files copied and no model executed.')
        return
    partial = Path(tempfile.mkdtemp(prefix=args.name + '-partial-', dir=history))
    try:
        for source, path in pairs:
            target = partial / path
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copyfile(source, target)
        worktree = partial / 'research/yolov10'
        worktree.parent.mkdir(parents=True)
        subprocess.run(['git', 'clone', '--no-hardlinks', '--no-checkout', str(research), str(worktree)], check=True)
        git(worktree, '-c', 'core.autocrlf=false', 'checkout', '--detach', COMMIT)
        restore_model_sources(worktree)
        git(worktree, 'config', '--local', 'core.autocrlf', 'true')
        if git(worktree, 'diff') != git(history / NUM / 'research/yolov10', 'diff'):
            raise ValueError('The reconstructed diff differs from the NUMERIC archive.')
        if git(worktree, 'diff', '--name-only') != OVERLAY[1] or git(worktree, 'ls-files', '--others', '--exclude-standard') != OVERLAY[0]:
            raise ValueError('Reconstructed research tree differs from the two recorded source changes.')
        for name in ('logs', 'figures', 'full_val'):
            (partial / name).mkdir(exist_ok=True)
        if dest.exists():
            raise FileExistsError(dest)
        partial.rename(dest)
    except BaseException:
        print(f'Incomplete work preserved in {partial}')
        raise
    print(f'Replay inputs assembled at {dest}.')


if __name__ == '__main__':
    main()
