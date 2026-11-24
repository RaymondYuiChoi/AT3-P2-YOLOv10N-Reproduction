"""Verify optimizer accounting, paired inputs, final ownership and gradient paths from original traces."""
from collections import Counter
import csv
import numpy as np
import runtime
import torch
from files import DATA,EVIDENCE,GROUPS,ROOT,digest,read_json,write_csv,write_json
from observations import grad_norms
from recheck import restore
from review_metrics import read_rows
from runtime import fingerprint,region,seed_all,split_params
from train import learning_rate
from voc_loader import FrozenVOC,loader


def joined(data):
    values=data.astype(np.int64)
    image=(values[:,0]-1)*4+values[:,1]
    location=image*8400+values[:,3]
    identity=location*65536+values[:,2]
    return image,location,identity


def audit_trace(group,branch,epoch,archive,steps,row,plan,gt_by_id,examples):
    pre,post=archive[branch+'_pre'],archive[branch+'_post']
    _,positions,pairs=joined(pre)
    image,final,owners=joined(post)
    ids=[token.split(':')[0] for token in plan]
    bounds=np.array([gt_by_id[key] for key in ids])
    assert len(set(ids))==1000 and len(np.unique(pairs))==len(pairs)
    for part in (pre,post):
        slots,_,_=joined(part)
        assert (part[:,0]>=1).all() and (part[:,0]<=250).all() and (part[:,1]>=0).all() and (part[:,1]<4).all()
        assert (part[:,2]>=0).all() and (part[:,2]<bounds[slots]).all() and (part[:,3]>=0).all() and (part[:,3]<8400).all()
    union,counts=np.unique(positions,return_counts=True)
    assert np.array_equal(np.sort(final),union) and len(final)==len(np.unique(final))
    contested=union[counts>1]
    outside=~np.isin(owners,pairs)
    assert np.isin(final[outside],contested).all()
    nweighted=post[:,4].sum()
    assert np.isin(post[:,4],[0,1]).all()
    rank=np.r_[0,np.cumsum(bounds)[:-1]]
    gt_counts=np.bincount(rank[image]+post[:,2],minlength=int(bounds.sum()))
    checks=dict(gt=int(bounds.sum()),candidates=len(pre),positives=len(post),positive_weight=int(nweighted),
                contested_anchors=len(contested),unmatched=int((gt_counts==0).sum()),max_per_gt=int(gt_counts.max()))
    for field,value in checks.items():
        assert value==int(row[branch+'_'+field]),(group,epoch,branch,field)
    for field,data in [('positives',post[:,0]),('candidates',pre[:,0]),('positive_weight',post[post[:,4]==1,0])]:
        assert np.array_equal(np.bincount(data,minlength=251)[1:],np.array([int(step[branch+'_'+field]) for step in steps]))
    selected=np.flatnonzero(outside)
    if not len(selected):
        selected=np.flatnonzero(np.isin(final,contested))
    if len(selected):
        choice=selected[0]
        value=post[choice]
        candidates=pre[positions==final[choice],2].tolist()
        examples.append(dict(group=group,epoch=epoch,branch=branch,image_id=ids[image[choice]],microbatch=int(value[0]),slot=int(value[1]),
                         anchor=int(value[3]),candidate_gt=candidates,final_gt=int(value[2]),positive_weight=int(value[4]),
                         winner_outside_candidates=bool(outside[choice])))
    return dict(group=group,epoch=epoch,branch=branch,**checks,final_outside_candidates=int(outside.sum()),
                **{f'gt_with_{count}_positives':int((gt_counts==count).sum()) for count in range(11)},
                gt_above_ten=int((gt_counts>10).sum()),
                positives_per_gt=len(post)/bounds.sum(),gt_zero_share=checks['unmatched']/bounds.sum(),
                weighted_share=nweighted/len(post)),(pairs,owners)


def isolated_final():
    seed_all()
    batch=next(iter(loader(FrozenVOC('train'))))
    images=batch['img'].cuda().float()/255
    out={}
    for group,branch in [('Dual','one2many'),('Dual','one2one'),('O2O','sole')]:
        model,epoch,_=restore(group,'last')
        model.train().zero_grad(set_to_none=True)
        criterion=model.init_criterion()
        preds=model(images)
        loss,_=(criterion(preds,batch) if branch=='sole' else getattr(criterion,branch)(preds[branch],batch))
        loss.backward()
        norms=grad_norms(split_params(model))
        assert all(np.isfinite(list(norms.values())))
        if branch=='one2one':
            assert norms['backbone']==norms['neck']==norms['head_main']==0 and norms['head_o2o']>0
        elif branch=='one2many':
            assert norms['backbone']>0 and norms['neck']>0 and norms['head_main']>0 and norms['head_o2o']==0
        else:
            assert min(norms.values())>0
        out[group+'_'+branch]=norms
    write_json(EVIDENCE/'isolated_final.json',dict(epoch=20,zero_optimizer_updates=True,precision='FP32',image_ids=[r['id'] for r in batch['records']],results=out))


def main():
    seed_all()
    per_group={group:read_rows(ROOT/'runs'/group/'epochs.csv') for group in GROUPS}
    steps_all={group:read_rows(ROOT/'runs'/group/'steps.csv') for group in GROUPS}
    indices={group:read_rows(ROOT/'runs'/group/'weights/index.csv') for group in GROUPS}
    frozen=read_json(EVIDENCE/'freeze.json')
    plans=read_json(EVIDENCE/'paired_plans.json')
    gt_by_id={rec['id']:sum(not obj['difficult'] for obj in rec['objects']) for rec in read_json(DATA/'manifests/train.json')}
    traces,overlaps,examples,regions,epochs=[],[],[],[],[]
    for epoch in range(1,21):
        tracks={}
        group_rows=[per_group[group][epoch-1] for group in GROUPS]
        for key in ('training_pixels_sha256','order_flip_sha256','train_list_sha256','val_list_sha256','lr_first','lr_last'):
            assert len({r[key] for r in group_rows})==1,(epoch,key)
        for group,row in zip(GROUPS,group_rows):
            root=ROOT/'runs'/group
            phase=steps_all[group][(epoch-1)*250:epoch*250]
            assert len(per_group[group])==20 and len(steps_all[group])==5000 and len(indices[group])==20
            assert list(range(1,251))==[int(step['batch']) for step in phase] and all(int(step['epoch'])==epoch for step in phase)
            assert sum(int(step['optimizer_update']) for step in phase)==int(row['updates'])
            assert sum(int(step['amp_skipped']) for step in phase)==int(row['amp_skips'])==250-int(row['updates'])
            assert all(int(step['finite_gradients']) or int(step['amp_skipped']) for step in phase)
            assert all(int(step['head_input_checks'])==(12 if group=='Dual' else 6) for step in phase)
            assert all(float(step['lr'])==learning_rate((epoch-1)*250+slot) for slot,step in enumerate(phase))
            assert [token for step in phase for token in step['ids_flips'].split()]==plans[str(epoch)]
            assert row['order_flip_sha256']==frozen['order_flip_sha256'][str(epoch)]
            arms=['many','one'] if group=='Dual' else ['many' if group=='O2M' else 'one']
            current={}
            with np.load(root/'assignments'/f'epoch_{epoch}.npz') as artifact:
                for arm in arms:
                    audited,current[arm]=audit_trace(group,arm,epoch,artifact,phase,row,plans[str(epoch)],gt_by_id,examples)
                    traces.append(audited)
                    for name in ('box','cls','dfl'):
                        assert np.isclose(float(row[arm+'_'+name]),np.mean([float(step[arm+'_'+name]) for step in phase]))
            for step in phase:
                assert np.isclose(4*sum(float(step[arm+'_'+name]) for arm in arms for name in ('box','cls','dfl')),
                                  float(step['objective']),rtol=2e-6)
                assert int(step['amp_skipped']) or min(float(step[key]) for key in ('grad_backbone','grad_neck','grad_head_main'))>0
            tracks[group]=current
            rank=indices[group][epoch-1]
            best=int(np.argmax([float(item['ap5095']) for item in per_group[group][:epoch]]))+1
            assert rank['best']==f'epoch_{best:02}.pt' and rank['last']==f'epoch_{epoch:02}.pt'
            assert rank['last_sha256']==digest(root/'weights'/rank['last']) and rank['best_sha256']==digest(root/'weights'/rank['best'])
            assert 0<float(row['val_s']) and 0<float(row['train_s']) and float(row['val_s'])+float(row['train_s'])<=float(row['epoch_wall_s'])
            assert int(row['gpu_samples'])>0 and float(row['peak_alloc_mib'])<=float(row['peak_reserved_mib'])
            status=read_json(root/'status.json')
            assert status['epochs']==20 and status['status']=='complete' and status['updates']==sum(int(rec['updates']) for rec in per_group[group])
            epochs.append(row)
        for context,one,many in [('within_Dual',tracks['Dual']['one'],tracks['Dual']['many']),('B_vs_A',tracks['O2O']['one'],tracks['O2M']['many'])]:
            wins=one[1]
            in_before=int(np.isin(wins,many[0]).sum())
            in_after=int(np.isin(wins,many[1]).sum())
            if context=='within_Dual':
                assert in_before==int(group_rows[-1]['one_winners_in_many_candidates']) and in_after==int(group_rows[-1]['one_winners_in_many_final'])
            overlaps.append(dict(context=context,epoch=epoch,winners=len(wins),in_many_candidates=in_before,in_many_final=in_after,
                                 candidate_rate=in_before/len(wins),final_rate=in_after/len(wins)))
        states=[torch.load(ROOT/'runs'/group/'weights'/f'epoch_{epoch:02}.pt',map_location='cpu')['model'] for group in ('O2M','Dual')]
        for part in ('backbone','neck','head_main'):
            keys=[key for key in states[0] if region(key)==part]
            hashes=[fingerprint((key,state[key]) for key in keys) for state in states]
            differences=[abs(float(a['grad_'+part])-float(c['grad_'+part])) for a,c in zip(steps_all['O2M'][(epoch-1)*250:epoch*250],steps_all['Dual'][(epoch-1)*250:epoch*250])]
            regions.append(dict(epoch=epoch,region=part,tensors=len(keys),equal=hashes[0]==hashes[1],A_sha256=hashes[0],C_sha256=hashes[1],
                                max_step_gradient_difference=max(differences)))
    epochs.sort(key=lambda item:(GROUPS.index(item['group']),int(item['epoch'])))
    keys=list(dict.fromkeys(key for item in epochs for key in item))
    write_csv(EVIDENCE/'epochs_all.csv',[{key:item.get(key,'') for key in keys} for item in epochs])
    write_csv(EVIDENCE/'assignment_verified.csv',traces)
    write_csv(EVIDENCE/'winner_overlap.csv',overlaps)
    write_csv(EVIDENCE/'paired_regions.csv',regions)
    write_json(EVIDENCE/'real_conflicts.json',examples)
    isolated_final()
    write_json(EVIDENCE/'training_audit.json',dict(passed=True,training_microbatches=15000,epochs=60,branch_epochs=80,
               post_ownership_arrays=80,fully_paired_epochs=20,updates={g:sum(int(r['updates']) for r in per_group[g]) for g in GROUPS},
               shared_region_checkpoints_equal=sum(r['equal'] for r in regions),shared_region_checkpoints_compared=len(regions)))


if __name__=='__main__':
    main()
