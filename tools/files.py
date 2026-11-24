"""Common paths and portable, explicit evidence formats."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets' / 'VOC2007'
EVIDENCE = ROOT / 'evidence'
REPO = ROOT / 'research' / 'yolov10'
PRISTINE = ROOT.parent / 'AT3-P2-FEASIBILITY-01' / 'sources' / 'yolov10'
COMMIT = '799ff3be47d21173bcf29b351820d4b8e955e0fe'
SEED = 2007
GROUPS = ('O2M', 'O2O', 'Dual')
NAMES = 'aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor'.split()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open('w', encoding='utf-8-sig', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
