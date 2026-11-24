"""Independent, CPU-only reconciliation of labels, step logs and AP from saved detections."""
import csv
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from files import COMMIT, DATA, EVIDENCE, GROUPS, NAMES, PRISTINE, REPO, ROOT, digest, read_json, write_csv, write_json


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def check_voc():
    frozen={}
    files={x['image_id']:x for x in read_csv(DATA/'manifests/file_hashes.csv')}
    assert len(files)==1250
    for split in ('train','val'):
        records=read_json(DATA/'manifests'/f'{split}.json')
        classes=read_csv(DATA/'manifests/class_counts.csv')
        pool=set((DATA/'manifests'/f'official_{split}.txt').read_text().split())
        frozen[split]={r['id'] for r in records}
        assert len(frozen[split])==len(records) and frozen[split]<=pool
        boxes=np.zeros(20,int)
        images=np.zeros(20,int)
        diffs=np.zeros(20,int)
        for r in records:
            i=r['id']
            xml=DATA/'Annotations'/split/(i+'.xml')
            label=DATA/'labels'/split/(i+'.txt')
            assert digest(xml)==files[i]['xml_sha256'] and digest(label)==files[i]['yolo_label_sha256']
            assert digest(DATA/'JPEGImages'/split/(i+'.jpg'))==files[i]['jpeg_sha256']
            root=ET.parse(xml).getroot()
            assert [int(root.findtext('size/'+k)) for k in ('width','height')]==[r['width'],r['height']]
            truth=[]
            seen=set()
            objects=root.findall('object')
            assert len(objects)==len(r['objects'])
            for obj,stored in zip(objects,r['objects']):
                c=NAMES.index(obj.findtext('name'))
                d=int(obj.findtext('difficult'))
                corners=[int(obj.findtext('bndbox/'+k)) for k in ('xmin','ymin','xmax','ymax')]
                corners[:2]=[v-1 for v in corners[:2]]
                assert stored['cls']==c and stored['xyxy']==corners and stored['difficult']==d
                if d:
                    diffs[c]+=1
                    continue
                seen.add(c)
                boxes[c]+=1
                b=np.array(corners)/[r['width'],r['height'],r['width'],r['height']]
                truth.append([c,*((b[:2]+b[2:])*.5),*(b[2:]-b[:2])])
            for c in seen:
                images[c]+=1
            assert np.allclose(np.loadtxt(label,ndmin=2),truth,rtol=0,atol=5.1e-10)
        expected=[r for r in classes if r['split']==split]
        assert (boxes==[int(r['regular_boxes']) for r in expected]).all()
        assert (images==[int(r['regular_images']) for r in expected]).all()
        assert (diffs==[int(r['difficult_boxes']) for r in expected]).all()
    assert not frozen['train']&frozen['val']
    return len(files)


def overlaps(b,rows):
    if not len(b):
        return np.empty((0,len(rows)),np.float32)
    a=np.array([o['xyxy'] for o in b],np.float32)[:,None,:]
    p=rows[None,:,:4]
    inter=np.maximum(0,np.minimum(a[:,:,2:],p[:,:,2:])-np.maximum(a[:,:,:2],p[:,:,:2])).prod(-1)
    union=(a[:,:,2:]-a[:,:,:2]).prod(-1)+(p[:,:,2:]-p[:,:,:2]).prod(-1)-inter
    cls=np.array([o['cls'] for o in b])[:,None]
    return inter/(union+np.float32(1e-7))*(cls==rows[:,5])


def independent_ap(file,records):
    with np.load(file) as z:
        ids,offsets,det=z['ids'],z['offsets'],z['detections']
    assert ids.tolist()==[r['id'] for r in records] and len(offsets)==251 and offsets[-1]==len(det)
    assert np.isfinite(det).all() and np.all((det[:,5]>=0)&(det[:,5]<20)&(det[:,5]==np.floor(det[:,5])))
    thresholds=np.array([.5,.55,.6,.65,.7,.75,.8,.85,.9,.95],np.float32)
    tp=np.zeros((len(det),10),bool)
    ign=np.zeros_like(tp)
    gt=np.zeros(20,int)
    for n,r in enumerate(records):
        start,stop=offsets[n:n+2]
        p=det[start:stop]
        regular=[o for o in r['objects'] if not o['difficult']]
        hard=[o for o in r['objects'] if o['difficult']]
        for o in regular:
            gt[o['cls']]+=1
        reg=overlaps(regular,p)
        diff=overlaps(hard,p)
        if len(diff):
            ign[start:stop]=(diff.max(0)[:,None]>=thresholds)&(reg.max(0)[:,None]<thresholds)
        for j,t in enumerate(thresholds):
            a,b=np.where(reg>=t)
            if not len(a):
                continue
            priority=np.argsort(reg[a,b])[::-1]
            a,b=a[priority],b[priority]
            # Official v1.1 unmatched tie order: first best GT per detection, then first retained detection per GT.
            _,pick=np.unique(b,return_index=True)
            a,b=a[pick],b[pick]
            _,pick=np.unique(a,return_index=True)
            tp[start+b[pick],j]=True
    aps=np.zeros((20,10))
    grid=np.linspace(0,1,101)
    for j in range(10):
        keep=np.flatnonzero(~ign[:,j])
        order=keep[np.argsort(-det[keep,4])]
        for c in range(20):
            flags=tp[order[det[order,5]==c],j]
            if not len(flags):
                continue
            hits=flags.cumsum()
            rec=np.r_[0.,hits/gt[c],1.]
            prec=np.r_[1.,hits/np.arange(1,len(flags)+1),0.]
            envelope=np.maximum.accumulate(prec[::-1])[::-1]
            aps[c,j]=np.trapz(np.interp(grid,rec,envelope),grid)
    return aps, int((det[:,4]>.1).sum())


def main():
    inputs=check_voc()
    val=read_json(DATA/'manifests/val.json')
    checks,combined=[],[]
    order_by_epoch={i:set() for i in range(1,6)}
    for g in GROUPS:
        run=ROOT/'runs'/g
        epochs=read_csv(run/'epochs.csv')
        steps=read_csv(run/'steps.csv')
        assert len(epochs)==5 and len(steps)==1250 and read_json(run/'status.json')['status']=='complete'
        for i,e in enumerate(epochs,1):
            batch=steps[(i-1)*250:i*250]
            assert [int(s['batch']) for s in batch]==list(range(1,251))
            assert all(int(s['epoch'])==i for s in batch)
            assert int(e['updates'])==sum(int(s['optimizer_update']) for s in batch)
            assert int(e['updates'])+int(e['amp_skips'])==250
            assert all(int(s['finite_gradients'])==1 for s in batch)
            assert np.isclose(np.mean([float(s['objective']) for s in batch]),float(e['train_loss']))
            assert np.isclose(4*sum(float(e[k]) for k in ('box_sum','cls_sum','dfl_sum')),float(e['train_loss']),rtol=1e-6)
            for arm in ('many','one'):
                if arm+'_positives' in e:
                    assert int(e[arm+'_gt'])==2508
                    for k in ('positives','positive_weight','unmatched'):
                        assert int(e[arm+'_'+k])==sum(int(s[arm+'_'+k]) for s in batch)
            order_by_epoch[i].add((e['order_flip_sha256'],e['training_pixels_sha256']))
            a,n=independent_ap(run/'validation'/f'epoch_{i}.npz',val)
            v=read_json(run/'validation'/f'epoch_{i}.json')
            delta=float(np.abs(a-np.array(v['ap_by_class_iou'])).max())
            assert delta<1e-7,(g,i,delta)
            assert abs(a.mean()-float(e['ap5095']))<1e-8 and abs(a[:,0].mean()-float(e['ap50']))<1e-8
            with np.load(run/'validation'/f'epoch_{i}.npz') as saved:
                nonempty_images=int((np.diff(saved['offsets'])>0).sum())
            assert v['nms']['calls']==(nonempty_images if g=='O2M' else 0)
            assert int(e['gpu_samples'])>0 and float(e['peak_alloc_mib'])<=float(e['peak_reserved_mib'])
            checks.append(dict(group=g,epoch=i,map_abs_diff=abs(a.mean()-float(e['ap5095'])),matrix_max_abs_diff=delta,
                               high_conf_predictions=n,classes_with_ap50_above_zero=int((a[:,0]>0).sum())))
            combined.append(e)
    assert all(len(v)==1 for v in order_by_epoch.values())
    columns=list(dict.fromkeys(k for row in combined for k in row))
    write_csv(EVIDENCE/'epochs_all.csv',[{k:e.get(k,'') for k in columns} for e in combined])
    write_csv(EVIDENCE/'ap_reconciliation.csv',checks)
    for folder in (PRISTINE,REPO):
        assert subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()==COMMIT
    status=subprocess.check_output(['git','-C',str(PRISTINE),'status','--porcelain'],text=True)
    assert not status.strip()
    p=subprocess.run(['git','diff','--no-index','--','/dev/null','at3_variant.py'],cwd=REPO,capture_output=True)
    assert p.returncode==1 and b'new file mode' in p.stdout
    patch=EVIDENCE/'yolov10n-variants.patch'
    patch.write_bytes(p.stdout.replace(b'\r\n',b'\n'))
    subprocess.run(['git','-C',str(PRISTINE),'apply','--check',str(patch)],check=True)
    write_json(EVIDENCE/'audit.json',dict(passed=True,xml_jpeg_labels_verified=inputs,epochs=15,steps=3750,
               metrics_recalculated_from_saved_detections=15,per_class_iou_cells=3000,
               max_per_class_iou_abs_diff=max(r['matrix_max_abs_diff'] for r in checks),
               matched_epoch_order_flips_and_pixel_hashes=5,pristine_clean=True,patch_applies_to_pristine=True,
               checkpoint_reloads=len(read_json(EVIDENCE/'independent_validation.json'))))
    print('PASS 1,250 originals, 3,750 step records, 15 validation splits, protected repository and patch.')


if __name__=='__main__':
    main()
