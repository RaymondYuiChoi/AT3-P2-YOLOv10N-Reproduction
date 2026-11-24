"""The full official Val is constructed/evaluated only after three completed training runs."""
import hashlib
import tarfile
import time
from datetime import datetime

import runtime
import torch
from evaluation import validate
from files import DATA,EVIDENCE,GROUPS,NAMES,ROOT,digest,read_json,write_csv,write_json
from prepare_voc import ARCHIVE,BASE,parse_annotation
from recheck import restore
from runtime import seed_all
from voc_loader import FrozenVOC


def extract_full():
    output=ROOT/'full_val'
    before=read_json(EVIDENCE/'freeze.json')
    assert digest(ARCHIVE)==before['archive_sha256']
    manifests=DATA/'manifests'
    directory=DATA/'JPEGImages/full_val'
    directory.mkdir(parents=True,exist_ok=False)
    lookup={record['id']:record for record in read_json(manifests/'val.json')}
    train=(manifests/'train.txt').read_text().split()
    records,hashes=[],[]
    with tarfile.open(ARCHIVE,'r:') as archive:
        def get(name):
            return archive.extractfile(BASE+name).read()
        ids_content=get('ImageSets/Main/val.txt')
        assert ids_content==(manifests/'official_val.txt').read_bytes()
        (manifests/'full_val.txt').write_bytes(ids_content)
        ids=ids_content.decode().split()
        assert len(set(ids))==len(ids)==2510 and not set(train)&set(ids) and set(lookup)<=set(ids)
        for image_id in ids:
            image=get(f'JPEGImages/{image_id}.jpg')
            xml=get(f'Annotations/{image_id}.xml')
            record=parse_annotation(xml,image_id)
            image_hash=hashlib.sha256(image).hexdigest()
            if image_id in lookup:
                assert lookup[image_id]==record and image_hash==digest(DATA/'JPEGImages/val'/(image_id+'.jpg'))
            (directory/(image_id+'.jpg')).write_bytes(image)
            records.append(record)
            hashes.append(dict(image_id=image_id,jpeg_sha256=image_hash,xml_sha256=hashlib.sha256(xml).hexdigest()))
    with (manifests/'file_hashes.csv').open(encoding='utf-8-sig') as source:
        import csv
        train_jpegs={item['jpeg_sha256'] for item in csv.DictReader(source) if item['split']=='train'}
    assert not {item['jpeg_sha256'] for item in hashes}&train_jpegs
    write_json(manifests/'full_val.json',records)
    write_csv(output/'source_hashes.csv',hashes)
    counts=[]
    for index,name in enumerate(NAMES):
        regular=[sum(obj['cls']==index and not obj['difficult'] for obj in record['objects']) for record in records]
        difficult=sum(obj['cls']==index and obj['difficult'] for record in records for obj in record['objects'])
        counts.append(dict(class_id=index,name=name,regular_gt=sum(regular),regular_images=sum(value>0 for value in regular),difficult_gt=difficult))
    write_csv(output/'class_support.csv',counts)
    write_json(output/'split.json',dict(created=datetime.now().astimezone().isoformat(),images=2510,subset_identical=250,train_overlap=0,
               train_jpeg_hash_overlap=0,archive_sha256=before['archive_sha256'],official_list_sha256=digest(manifests/'full_val.txt'),
               annotations_sha256=digest(manifests/'full_val.json'),regular_gt=sum(row['regular_gt'] for row in counts),
               difficult_gt=sum(row['difficult_gt'] for row in counts),endpoint='fixed epoch 20, one inference pass per group'))


def main():
    for group in GROUPS:
        status=read_json(ROOT/'runs'/group/'status.json')
        assert status['status']=='complete' and status['epochs']==20
    seed_all()
    extract_full()
    dataset=FrozenVOC('full_val')
    output=ROOT/'full_val'
    split=read_json(output/'split.json')
    summary=[]
    for group in GROUPS:
        model,epoch,sha=restore(group,'last')
        assert epoch==20 and sha==digest(ROOT/'runs'/group/'weights/epoch_20.pt')
        folder=output/group
        folder.mkdir()
        torch.cuda.synchronize()
        start=time.perf_counter()
        val=validate(model,group,dataset,folder/'epoch_20.npz',extra_raw=group=='O2M')
        torch.cuda.synchronize()
        assert val['evaluated_images']==2510 and val['regular_gt']==split['regular_gt']
        assert bool(val['nms']['calls'])==(group=='O2M')
        summary.append(dict(group=group,epoch=20,checkpoint_sha256=sha,images=2510,regular_gt=val['regular_gt'],
                       ap5095=val['ap5095'],ap50=val['ap50'],predictions=val['detections'],wall_s=time.perf_counter()-start))
        print(group,summary[-1],flush=True)
        del model
    write_csv(output/'results.csv',summary)


if __name__=='__main__':
    main()
