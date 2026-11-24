"""Recompute the 20×10 AP matrix and effective prediction counts directly from serialized boxes."""
import csv
import numpy as np
from audit import overlaps
from files import DATA,EVIDENCE,GROUPS,NAMES,ROOT,read_json,write_csv,write_json

THRESHOLDS=np.arange(50,100,5,dtype=np.float32)/100


def read_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as source:
        return list(csv.DictReader(source))


def recalc(ids,offsets,det,records):
    assert ids.tolist()==[item['id'] for item in records] and len(offsets)==len(ids)+1
    assert offsets[0]==0 and offsets[-1]==len(det) and (np.diff(offsets)<=300).all() and (np.diff(offsets)>=0).all()
    assert det.shape==(len(det),6) and np.isfinite(det).all() and (det[:,4]>.001).all() and (det[:,4]<=1).all()
    assert np.all((det[:,5]>=0)&(det[:,5]<20)&(det[:,5]==np.floor(det[:,5])))
    positives=np.zeros((len(det),10),bool)
    ignored=np.zeros_like(positives)
    gt_total=np.zeros(20,int)
    for index,record in enumerate(records):
        start,stop=offsets[index:index+2]
        rows=det[start:stop]
        assert (rows[:,0]>=0).all() and (rows[:,2]<=record['width']).all() and (rows[:,1]>=0).all() and (rows[:,3]<=record['height']).all()
        regular=[obj for obj in record['objects'] if not obj['difficult']]
        hard=[obj for obj in record['objects'] if obj['difficult']]
        for obj in regular:
            gt_total[obj['cls']]+=1
        iou=overlaps(regular,rows)
        if hard:
            high=overlaps(hard,rows).max(0)
            low=iou.max(0) if regular else np.zeros(len(rows),np.float32)
            ignored[start:stop]=(high[:,None]>=THRESHOLDS)&(low[:,None]<THRESHOLDS)
        for column,threshold in enumerate(THRESHOLDS):
            targets,outputs=np.where(iou>=threshold)
            if not len(targets):
                continue
            best=np.argsort(iou[targets,outputs])[::-1]
            targets,outputs=targets[best],outputs[best]
            unique=np.unique(outputs,return_index=True)[1]
            targets,outputs=targets[unique],outputs[unique]
            unique=np.unique(targets,return_index=True)[1]
            positives[start+outputs[unique],column]=True
    aps=np.zeros((20,10))
    effective=np.zeros((20,10),int)
    hits=np.zeros_like(effective)
    grid=np.linspace(0,1,101)
    assert (gt_total>0).all()
    for column in range(10):
        active=np.flatnonzero(~ignored[:,column])
        order=active[np.argsort(-det[active,4])]
        for index in range(20):
            flags=positives[order[det[order,5]==index],column]
            if not len(flags):
                continue
            count=flags.cumsum()
            effective[index,column]=len(flags)
            hits[index,column]=count[-1]
            recall=np.r_[0.,count/gt_total[index],1.]
            precision=np.r_[1.,count/np.arange(1,len(flags)+1),0.]
            aps[index,column]=np.trapz(np.interp(grid,recall,np.maximum.accumulate(precision[::-1])[::-1]),grid)
    return aps,effective,hits,gt_total


def compare(file,group,epoch,scope,records):
    saved=read_json(file.with_suffix('.json'))
    with np.load(file,allow_pickle=False) as archive:
        ids,offsets,det=archive['ids'],archive['offsets'],archive['detections']
    aps,effective,hits,gt_total=recalc(ids,offsets,det,records)
    error=float(np.max(np.abs(aps-np.array(saved['ap_by_class_iou']))))
    assert error<1e-7,(group,epoch,scope,error)
    assert abs(aps.mean()-saved['ap5095'])<1e-8 and abs(aps[:,0].mean()-saved['ap50'])<1e-8
    assert saved['detections']==len(det) and saved['regular_gt']==int(gt_total.sum())
    assert saved['nms']['calls']==(int((np.diff(offsets)>0).sum()) if group=='O2M' else 0)
    rows=[]
    for index,name in enumerate(NAMES):
        actual=saved['per_class'][index]
        sel=det[:,5]==index
        assert actual['name']==name and actual['gt']==gt_total[index] and effective[index].tolist()==actual['effective_by_iou']
        assert hits[index].tolist()==actual['tp_by_iou'] and int(sel.sum())==actual['predictions']
        assert actual['above_010']==int((sel&(det[:,4]>.10)).sum()) and actual['above_001']==int((sel&(det[:,4]>.01)).sum())
        rows.append(dict(scope=scope,group=group,epoch=epoch,class_id=index,name=name,gt=int(gt_total[index]),
                         fp50=int(effective[index,0]-hits[index,0]),fn50=int(gt_total[index]-hits[index,0]),
                         eligible_box_class=saved['before_filter_eligible_by_class'][index],
                         ap5095=aps[index].mean(),ap50=aps[index,0],predictions=actual['predictions'],
                         effective50=effective[index,0],tp50=hits[index,0],p_gt_010=actual['above_010'],p_gt_001=actual['above_001'],
                         **{f'AP{round(float(threshold)*100)}':aps[index,column] for column,threshold in enumerate(THRESHOLDS)},
                         **{f'effective_IoU{round(float(threshold)*100)}':effective[index,column] for column,threshold in enumerate(THRESHOLDS)}))
    audit=dict(scope=scope,group=group,epoch=epoch,images=len(ids),regular_gt=gt_total.sum(),detections=len(det),p_gt_010=int((det[:,4]>.10).sum()),
               p_gt_001=int((det[:,4]>.01).sum()),images_with_p_gt_010=sum(int((det[start:stop,4]>.1).any()) for start,stop in zip(offsets[:-1],offsets[1:])),
               classes_with_effective=int((effective[:,0]>0).sum()),classes_with_tp50=int((hits[:,0]>0).sum()),
               ap5095=aps.mean(),ap50=aps[:,0].mean(),matrix_max_abs_diff=error)
    return rows,audit


def nested_check(group,selected):
    root=ROOT/'full_val'/group
    subpath=ROOT/'runs'/group/('repaired_subset' if group=='O2M' else 'validation')/'epoch_20.npz'
    with np.load(root/'epoch_20.npz') as full,np.load(subpath) as sub:
        ids,offsets,det=full['ids'],full['offsets'],full['detections']
        direct_ids,direct_offsets,direct=sub['ids'],sub['offsets'],sub['detections']
    lookup={name:slot for slot,name in enumerate(ids)}
    blocks=[det[offsets[lookup[name]]:offsets[lookup[name]+1]] for name in direct_ids]
    recovered=np.concatenate(blocks)
    bounds=np.r_[0,np.cumsum([len(piece) for piece in blocks])]
    ap,*_=recalc(direct_ids,bounds,recovered,selected)
    original=read_json(subpath.with_suffix('.json'))
    identical=sum(np.array_equal(piece,direct[left:right]) for piece,left,right in zip(blocks,direct_offsets[:-1],direct_offsets[1:]))
    return dict(group=group,shared_images=len(blocks),identical_image_prediction_arrays=identical,full_run_subset_AP5095=float(ap.mean()),
                direct_subset_AP5095=original['ap5095'],absolute_AP5095_difference=abs(float(ap.mean())-original['ap5095']))


def main():
    subsets=read_json(DATA/'manifests/val.json')
    complete=read_json(DATA/'manifests/full_val.json')
    audits,classes,full_classes=[],[],[]
    for group in GROUPS:
        for epoch in range(1,21):
            rows,out=compare(ROOT/'runs'/group/'validation'/f'epoch_{epoch}.npz',group,epoch,'historical250' if group=='O2M' else 'subset250',subsets)
            classes.extend(rows)
            audits.append(out)
        if group=='O2M':
            for epoch in (19,20):
                rows,out=compare(ROOT/'runs/O2M/repaired_subset'/f'epoch_{epoch}.npz',group,epoch,'subset250',subsets)
                classes.extend(rows)
                audits.append(out)
        rows,out=compare(ROOT/'full_val'/group/'epoch_20.npz',group,20,'official_val2510',complete)
        full_classes.extend(rows)
        audits.append(out)
        print('reconciled metrics',group,flush=True)
    write_csv(EVIDENCE/'class_metrics_subset250.csv',classes)
    write_csv(ROOT/'full_val/class_metrics.csv',full_classes)
    write_csv(EVIDENCE/'ap_reconciliation.csv',audits)
    write_csv(ROOT/'full_val/nested_subset_check.csv',[nested_check(group,subsets) for group in GROUPS])
    write_json(EVIDENCE/'metric_audit.json',dict(passed=True,subset_matrices=62,full_matrices=3,class_iou_cells=13000,
               max_abs_diff=max(row['matrix_max_abs_diff'] for row in audits),counts_all_classes_verified=True))


if __name__=='__main__':
    main()
