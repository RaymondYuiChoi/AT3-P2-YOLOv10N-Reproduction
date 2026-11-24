"""Seal the final deliverable including the Git copy, and independently stream-check every listed byte."""
import argparse
import difflib
import hashlib
import subprocess
from pathlib import Path
from files import COMMIT,EVIDENCE,PRISTINE,REPO,ROOT,digest,read_json,write_json


def verify_history():
    original=read_json(EVIDENCE/'historical_provenance.json')['inventories']
    for name,item in original.items():
        base=ROOT.parent/name; manifest=base/'evidence/SHA256SUMS.txt'
        assert digest(manifest)==item['manifest_sha256']
        rows=manifest.read_text(encoding='utf-8-sig').splitlines()
        assert len(rows)==item['files_verified']
        for row in rows:
            checksum,relative=row.split('  ',1);assert digest(base/relative)==checksum,relative
    for relative,expected in read_json(EVIDENCE/'resume_gate.json')['locked_files'].items():
        assert digest(ROOT/relative)==expected,relative
    for directory in (PRISTINE,REPO):
        assert subprocess.check_output(['git','--no-optional-locks','-C',str(directory),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','--no-optional-locks','-C',str(PRISTINE),'status','--porcelain'])
    assert subprocess.check_output(['git','--no-optional-locks','-C',str(REPO),'diff','--name-only'],text=True).strip()=='ultralytics/nn/modules/block.py'
    write_json(EVIDENCE/'provenance_after.json',dict(historical_inventories_unchanged=original,official_clean=True,gate_locked_files_unchanged=True,commit=COMMIT))


def patches():
    exp=ROOT.parent/'AT3-P2-YOLOv10N-EXP20-01/tools';num=ROOT.parent/'AT3-P2-YOLOv10N-NUMERIC-01/tools'
    diff=[]
    for target in sorted((ROOT/'tools').iterdir()):
        if target.suffix not in ('.py','.mjs','.ps1'):continue
        old=exp/target.name
        if not old.is_file():old=num/target.name
        source=old.read_text(encoding='utf-8-sig').splitlines(keepends=True) if old.is_file() else []
        after=target.read_text(encoding='utf-8-sig').splitlines(keepends=True)
        diff.extend(difflib.unified_diff(source,after,fromfile='a/tools/'+target.name if source else '/dev/null',tofile='b/tools/'+target.name))
    (ROOT/'patches/03_实验与审计脚本.patch').write_text(''.join(diff),encoding='utf-8')


DEST=EVIDENCE/'SHA256SUMS.txt'
def inventory():
    return {p.relative_to(ROOT).as_posix():p for p in ROOT.rglob('*') if p.is_file() and p!=DEST}


def verify():
    report=read_json(EVIDENCE/'delivery_validation.json');assert report['passed']
    assert all((ROOT/relative).is_file() for relative in report['reports']['verified_by_final_seal'])
    paths=inventory();rows=DEST.read_text('utf-8').splitlines();assert len(paths)==len(rows)
    for row in rows:
        checksum,relative=row.split('  ',1);h=hashlib.sha256()
        with paths.pop(relative).open('rb') as source:
            for block in iter(lambda:source.read(4*1024*1024),b''):h.update(block)
        assert checksum==h.hexdigest(),relative
    assert not paths
    print(f'PASS {len(rows)} files. Manifest SHA256 {digest(DEST)}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','seal','verify'));mode=parser.parse_args().mode
    if mode=='prepare':verify_history();patches()
    if mode=='seal':
        files=inventory();DEST.write_text(''.join(f'{digest(files[key])}  {key}\n' for key in sorted(files)),encoding='utf-8')
    if mode=='verify':verify()
