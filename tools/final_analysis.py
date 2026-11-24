"""Independent AP/assignment audit and scope-separated tables from saved predictions (no dataset inference)."""
import numpy as np
from audit import overlaps
from files import DATA,EVIDENCE,GROUPS,NAMES,ROOT,digest,read_json,write_csv,write_json
from review_metrics import main as verify_metrics,read_rows
from review_training import main as verify_training


def files_for(group,epoch):
    run=ROOT/'runs'/group
    yield ('historical250' if group=='O2M' else 'subset250'),run/'validation'/f'epoch_{epoch}.json'
    if group=='O2M' and epoch in (19,20):
        yield 'subset250',run/'repaired_subset'/f'epoch_{epoch}.json'
    if epoch==20:
        yield 'official_val2510',ROOT/'full_val'/group/'epoch_20.json'


def size_recall(file,records,scope,group,metrics):
    with np.load(file.with_suffix('.npz')) as archive:
        ids,starts,det=archive['ids'],archive['offsets'],archive['detections']
    assert ids.tolist()==[item['id'] for item in records]
    counts=np.zeros(3,int); hits=np.zeros((2,3),int)
    for rec,left,right in zip(records,starts,starts[1:]):
        objects=[obj for obj in rec['objects'] if not obj['difficult']]
        width,height=rec['width'],rec['height']; scale=640/max(width,height)
        sizes=[]
        for obj in objects:
            x1,y1,x2,y2=obj['xyxy']
            area=(x2-x1)*(y2-y1)*round(scale*width)/width*round(scale*height)/height
            sizes.append(int(area>=32**2)+int(area>=96**2))
        sizes=np.asarray(sizes,dtype=int); counts+=np.bincount(sizes,minlength=3)
        iou=overlaps(objects,det[left:right])
        for idx,threshold in enumerate((.5,.75)):
            targets,outputs=np.where(iou>=threshold)
            order=np.argsort(iou[targets,outputs])[::-1]
            targets,outputs=targets[order],outputs[order]
            targets=targets[np.unique(outputs,return_index=True)[1]]
            hits[idx]+=np.bincount(sizes[np.unique(targets)],minlength=3)
    assert counts.sum()==metrics['regular_gt']
    assert all(hits[idx].sum()==sum(row['tp_by_iou'][column] for row in metrics['per_class']) for idx,column in enumerate((0,5)))
    return [dict(scope=scope,group=group,size=label,gt=int(counts[idx]),tp50=int(hits[0,idx]),tp75=int(hits[1,idx]),
                 recall50=float(hits[0,idx]/counts[idx]),recall75=float(hits[1,idx]/counts[idx])) for idx,label in enumerate(('small','medium','large'))]


def main():
    verify_metrics(); verify_training()
    summary=[]; sizes=[]; duplicates=[]
    for group in GROUPS:
        for epoch in range(1,21):
            for scope,file in files_for(group,epoch):
                val=read_json(file); classes=val['per_class']; finite=val.get('all_finite_images','')
                if scope!='historical250':
                    assert finite==val['evaluated_images'] and val['checked_batches']==(finite+3)//4
                    precision=val['precision_evidence']
                    assert not any(precision['autocast_at_op'].values()) and len(precision['aten_calls'])==4
                    assert all(op['output']=='float32' and set(op['inputs'])=={'float32'} for op in precision['aten_calls'])
                tp=sum(row['tp_by_iou'][0] for row in classes); effective=sum(row['effective_by_iou'][0] for row in classes)
                record=dict(scope=scope,group=group,epoch=epoch,images=val['evaluated_images'],regular_gt=val['regular_gt'],
                    ap5095=val['ap5095'],ap50=val['ap50'],tp50=tp,fp50=effective-tp,fn50=val['regular_gt']-tp,
                    ignored50=val['ignored_by_iou'][0],effective50=effective,classes_with_tp50=sum(row['tp_by_iou'][0]>0 for row in classes),
                    raw_anchors=val['raw_anchor_positions'],raw_eligible_box_class=sum(val['before_filter_eligible_by_class']),
                    nms_kernel_input=val['nms']['candidates'] if group=='O2M' else '',nms_survivors_before_cap=val['nms']['survivors_before_cap'] if group=='O2M' else '',
                    topk_before_conf=val['evaluated_images']*300 if group!='O2M' else '',post_predictions=val['detections'],nms_calls=val['nms']['calls'],
                    val_loss_per_image=val.get('val_loss_per_image',''),finite_images=finite,checked_batches=val.get('checked_batches',''))
                assert effective+record['ignored50']==record['post_predictions']
                summary.append(record)
                if epoch==20 and scope!='historical250':
                    records=read_json(DATA/'manifests'/('full_val.json' if scope=='official_val2510' else 'val.json'))
                    sizes.extend(size_recall(file,records,scope,group,val))
                    for stage,key in [('final','duplicates_post'),('before_nms','duplicates_pre')]:
                        if val[key] is not None:
                            total,matched,excess=val[key]
                            duplicates.append(dict(scope=scope,group=group,stage=stage,threshold=.1,predictions=total,matched=matched,excess=excess,
                                                   excess_over_matched=excess/matched if matched else ''))
    write_csv(EVIDENCE/'validation_all.csv',summary);write_csv(EVIDENCE/'size_recall.csv',sizes);write_csv(EVIDENCE/'duplicate_counts.csv',duplicates)
    training=read_rows(EVIDENCE/'epochs_all.csv'); costs=[]
    for group in GROUPS:
        subset=[row for row in training if row['group']==group]
        cost=dict(group=group,historical_train=int(group=='O2M'),updates=sum(int(row['updates']) for row in subset),
            skips=sum(int(row['amp_skips']) for row in subset),max_alloc_mib=max(float(row['peak_alloc_mib']) for row in subset),
            max_reserved_mib=max(float(row['peak_reserved_mib']) for row in subset),run_wall_s=read_json(ROOT/'runs'/group/'status.json')['wall_s'])
        for key in ('train_s','val_s','epoch_wall_s'):
            values=[float(row[key]) for row in subset];cost[key+'_sum']=sum(values);cost[key+'_median']=float(np.median(values))
        costs.append(cost)
    write_csv(EVIDENCE/'training_costs.csv',costs)
    states=[]
    for group in ('O2O','Dual'):
        import torch
        run=ROOT/'runs'/group/'weights'
        for epoch in range(1,21):
            file=run/f'training_state_{epoch:02}.pt'; ckpt=torch.load(file,map_location='cpu')
            assert ckpt['epoch']==epoch and ckpt['optimizer'] and ckpt['scaler'] and ckpt['updates']==250*epoch
            assert digest(run/ckpt['model_file'])==ckpt['model_sha256'] and ckpt['group']==group
            states.append(dict(group=group,epoch=epoch,file=file.name,sha256=digest(file),keys=' '.join(sorted(ckpt))))
    write_csv(EVIDENCE/'training_state_index.csv',states)


if __name__=='__main__':
    main()
