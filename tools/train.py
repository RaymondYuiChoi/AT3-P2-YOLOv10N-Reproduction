"""Paired 20-epoch run from the exact pilot initialization, never a trained checkpoint."""
import argparse
import csv
import hashlib
import math
import shutil
import subprocess
import threading
import time
import traceback
from collections import Counter
from datetime import datetime

import runtime
import numpy as np
import torch
from evaluation import validate
from files import DATA, EVIDENCE, GROUPS, NAMES, ROOT, digest, read_json, write_json
from observations import TrainingAssignments, grad_norms, make_optimizer
from runtime import COMMON, load_initial, save_tensors, seed_all, split_params
from voc_loader import FrozenVOC, epoch_plan, loader


def gate(group):
    assert group in ('O2O','Dual'), 'O2M training is prohibited'
    ready=read_json(EVIDENCE/'resume_gate.json')
    assert ready['passed'] and ready['dual_disposable_update']==1
    for path,expected in ready['locked_files'].items():
        assert digest(ROOT/path)==expected,path
    initial=read_json(EVIDENCE/'initialization.json')
    assert all(len({initial[g]['parts'][k] for g in GROUPS})==1 for k in ('backbone','neck','main_head','dfl'))
    assert digest(ROOT/'weights'/f'init_{group}.pt')==initial[group]['file_sha256']
    source=read_json(EVIDENCE/'dataset.json')
    assert read_json(DATA/'manifests/classes.json')==NAMES
    for s, n in [('train',1000),('val',250)]:
        stats=source['splits'][s]
        assert stats['selected']==n and digest(DATA/'manifests'/f'{s}.txt')==stats['list_sha256']
        assert digest(DATA/'manifests'/f'{s}.json')==stats['annotations_sha256']
    with (DATA/'manifests/file_hashes.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            assert digest(DATA/'JPEGImages'/r['split']/(r['image_id']+'.jpg'))==r['jpeg_sha256']


class GpuMonitor:
    def __init__(self,path):
        self.file=path.open('w',encoding='utf-8',newline='')
        self.writer=csv.writer(self.file)
        self.writer.writerow(['unix_s','util_gpu_pct','driver_mem_used_MiB','power_W'])
        self.rows=[]
        self.proc=subprocess.Popen(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw',
                '--format=csv,noheader,nounits','-l','1'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.thread=threading.Thread(target=self.collect,daemon=True)
        self.thread.start()

    def collect(self):
        for line in self.proc.stdout:
            fields=[v.strip() for v in line.split(',')]
            if len(fields)!=3 or not fields[0].isdigit():
                continue
            t=time.time()
            self.rows.append((t,float(fields[0])))
            self.writer.writerow([t,*fields])
            self.file.flush()

    def during(self,t0,t1):
        sample=[v for t,v in list(self.rows) if t0<=t<=t1]
        return (len(sample),float(np.mean(sample)),max(sample)) if sample else (0,None,None)

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.thread.join(timeout=3)
        self.file.close()


def learning_rate(step, steps=5000):
    warm=COMMON['warmup_updates']
    if step < warm:
        return COMMON['lr_start'] + (COMMON['lr_peak']-COMMON['lr_start'])*step/(warm-1)
    fraction=(step-warm)/(steps-warm-1)
    return COMMON['lr_end'] + (COMMON['lr_peak']-COMMON['lr_end'])*(1+math.cos(math.pi*fraction))/2


def add_csv(path,row):
    new=not path.exists()
    with path.open('a',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def checkpoint(model, group, epoch, target):
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    assert all(torch.isfinite(v).all() for v in state.values()), 'nonfinite checkpoint'
    save_tensors(target,dict(group=group,epoch=epoch,model=state))


def train(group):
    gate(group)
    seed_all()
    run=ROOT/'runs'/group
    run.mkdir(exist_ok=False)
    for folder in ('weights','validation','assignments'):
        (run/folder).mkdir()
    monitor=GpuMonitor(run/'gpu.csv')
    start=time.time()
    epoch=step=0
    try:
        train_ds,val_ds=FrozenVOC('train'),FrozenVOC('val')
        model=load_initial(group)
        parts=split_params(model)
        optimizer=make_optimizer(model,COMMON)
        scaler=torch.cuda.amp.GradScaler(init_scale=COMMON['amp_init_scale'],growth_interval=COMMON['amp_growth_interval'])
        criterion=model.init_criterion()
        observer=TrainingAssignments(group,criterion,model)
        pixels=read_json(EVIDENCE/'paired_image_pixels.json')
        frozen=read_json(EVIDENCE/'freeze.json')
        split_hash=frozen['train_list_sha256']
        plan_records=read_json(EVIDENCE/'paired_plans.json')
        pilot_rows=list(csv.DictReader((ROOT.parent/'AT3-P2-YOLOv10N-PILOT-01'/'runs'/group/'epochs.csv').open(encoding='utf-8-sig')))
        zero_aps=[]
        branches=['many','one'] if group=='Dual' else (['many'] if group=='O2M' else ['one'])
        best=-1.
        streak=0
        for epoch in range(1,COMMON['epochs']+1):
            torch.cuda.synchronize()
            t0=time.time()
            torch.cuda.reset_peak_memory_stats()
            model.train()
            train_ds.plan=epoch_plan(len(train_ds),epoch)
            content_sha=hashlib.sha256()
            plan_sha=hashlib.sha256()
            start_params={k:p[0].detach().clone() for k,p in parts.items()}
            totals={k:Counter() for k in branches}
            gradients={k:[] for k in parts}
            common_wins=common_pre=skips=max_streak=0
            trace={}
            losses=[]
            before_updates=optimizer.updates
            for step,batch in enumerate(loader(train_ds),1):
                images=batch['img']
                content_sha.update(memoryview(images.numpy()).cast('B'))
                records=[]
                for slot,(rec,f) in enumerate(zip(batch['records'],batch['flips'])):
                    plan_sha.update(f"{rec['id']} {int(f)}\n".encode())
                    token=rec['id']+':'+str(int(f))
                    assert token==plan_records[str(epoch)][(step-1)*4+slot], 'unpaired order or flip'
                    assert hashlib.sha256(memoryview(images[slot].numpy()).cast('B')).hexdigest()==pixels[token], 'input pixels changed'
                    records.append(token)
                images=images.cuda(non_blocking=True).float().div_(255)
                lr=learning_rate((epoch-1)*250+step-1)
                for pgroup in optimizer.param_groups:
                    pgroup['lr']=lr
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    observer.connections.clear()
                    output=model(images)
                    objective,components=criterion(output,batch)
                assert len(observer.connections)==(12 if group=='Dual' else 6), 'missing gradient connection probe'
                if not bool(torch.isfinite(objective)) or not torch.isfinite(components).all():
                    raise FloatingPointError('nonfinite detection loss')
                losses.append([float(objective),*components.detach().cpu().tolist()])
                scaler.scale(objective).backward()
                scaler.unscale_(optimizer)
                norms=grad_norms(parts)
                old_count=optimizer.updates
                if not all(math.isfinite(value) for value in norms.values()):
                    add_csv(run/'unsafe_gradient.csv',dict(epoch=epoch,batch=step,lr=lr,ids=' '.join(records),
                            updates_before=old_count,amp_scale=scaler.get_scale(),optimizer_step_called=False,**norms))
                    raise FloatingPointError('nonfinite unscaled gradient: stopped before optimizer.step; no batch skipped or retried')
                if not all(value>0 for value in norms.values()):
                    raise FloatingPointError('expected shared or active head gradient vanished before optimizer.step')
                scaler.step(optimizer)
                scaler.update()
                skipped=optimizer.updates==old_count
                finite=all(math.isfinite(x) for x in norms.values())
                assert finite or skipped
                skips+=int(skipped)
                streak=streak+1 if skipped else 0
                max_streak=max(max_streak,streak)
                row=dict(epoch=epoch,batch=step,lr=lr,objective=float(objective),optimizer_update=int(not skipped),
                         amp_scale=scaler.get_scale(),finite_gradients=int(finite),ids_flips=' '.join(records),
                         amp_skipped=int(skipped),head_input_checks=len(observer.connections))
                for k,v in norms.items():
                    row['grad_'+k]=v
                    if math.isfinite(v):
                        gradients[k].append(v)
                for branch_index,k in enumerate(branches):
                    stats=observer.results[k]
                    for offset,name in enumerate(('box','cls','dfl')):
                        row[k+'_'+name]=losses[-1][1+branch_index*3+offset]
                    for field,value in observer.trace(k,step,trace).items():
                        row[k+'_'+field]=value
                        totals[k][field]+=value
                    for field in ('gt','positives','unmatched','positive_weight'):
                        totals[k][field]+=stats[field]
                        row[k+'_'+field]=stats[field]
                    totals[k]['max_per_gt']=max(totals[k]['max_per_gt'],stats['max_per_gt'])
                if group=='Dual':
                    wins=observer.common_winners()
                    common_wins+=wins
                    row['one_winners_in_many_final']=wins
                    row['one_winners_in_many_candidates']=observer.common_candidates()
                    common_pre+=row['one_winners_in_many_candidates']
                add_csv(run/'steps.csv',row)
                assert skipped or all(norms[k]>0 for k in ('backbone','neck')), 'shared gradients vanished'
                if streak>=3:
                    raise FloatingPointError('three consecutive skipped optimizer updates')
            torch.cuda.synchronize()
            train_s=time.time()-t0
            assert step==250 and plan_sha.hexdigest()==frozen['order_flip_sha256'][str(epoch)]
            if epoch<=5:
                assert content_sha.hexdigest()==pilot_rows[epoch-1]['training_pixels_sha256']
            if group!='O2M':
                with (ROOT/'runs/O2M/epochs.csv').open(encoding='utf-8-sig') as previous:
                    assert content_sha.hexdigest()==list(csv.DictReader(previous))[epoch-1]['training_pixels_sha256']
            np.savez_compressed(run/'assignments'/f'epoch_{epoch}.npz',**{key:np.concatenate(rows) for key,rows in trace.items()})
            tval=time.perf_counter()
            val=validate(model,group,val_ds,run/'validation'/f'epoch_{epoch}.npz',extra_raw=group=='O2M')
            torch.cuda.synchronize()
            val_s=time.perf_counter()-tval
            assert val['evaluated_images']==250 and val['regular_gt']==660 and bool(val['nms']['calls'])==(group=='O2M')
            assert all(c['gt']>0 for c in val['per_class']) and val['detections']>0
            missing=[c['name'] for c in val['per_class'] if not c['effective_by_iou'][0]]
            zero_aps.append(val['ap5095'])
            if missing or (len(zero_aps)>=6 and max(zero_aps[-5:])<=zero_aps[-6]):
                from audit import independent_ap
                inspected,above=independent_ap(run/'validation'/f'epoch_{epoch}.npz',val_ds.records)
                diff=float(np.abs(inspected-np.array(val['ap_by_class_iou'])).max())
                details=[]
                for c in val['per_class']:
                    index=c['class_id']
                    if c['name'] in missing:
                        eligible=val['before_filter_eligible_by_class'][index]
                        reason='no raw score above floor' if not eligible else ('absent after ranking/cap/NMS' if not c['predictions'] else 'difficult ignore')
                        details.append(dict(name=c['name'],gt=c['gt'],raw_eligible=eligible,raw_max=val['before_filter_max_score_by_class'][index],
                                            post=c['predictions'],reason=reason))
                reliable=diff<1e-7 and above==val['duplicates_post'][0] and all(c['gt']>0 for c in val['per_class'])
                write_json(run/'diagnostics'/f'epoch_{epoch}.json',dict(paused_before_next_update=True,missing=details,
                           independent_ap_max_abs=diff,ap_history=zero_aps,verified=bool(reliable),
                           decision='retain frozen plan' if reliable else 'halt',reason='independent raw-box AP and labelled support reconciliation'))
                assert reliable,'class/metric diagnostic failed'
            if len(zero_aps)>=5 and max(zero_aps[-5:])<.0001 and val['duplicates_post'][0]<3:
                raise RuntimeError('sustained near-zero validation and sparse display; manual review required')
            current=val['ap5095']
            snapshot=run/'weights'/f'epoch_{epoch:02}.pt'
            checkpoint(model,group,epoch,snapshot)
            save_tensors(run/'weights'/f'training_state_{epoch:02}.pt',dict(epoch=epoch,
                         model_file=snapshot.name,model_sha256=digest(snapshot),group=group,
                         optimizer=optimizer.state_dict(),updates=optimizer.updates,scaler=scaler.state_dict(),
                         planned_microbatches=epoch*250,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                         numpy_rng=np.random.get_state(),python_rng=runtime.random.getstate()))
            shutil.copyfile(snapshot,run/'weights/last.pt')
            if current>best:
                best=current
                best_epoch=epoch
                shutil.copyfile(snapshot,run/'weights/best.pt')
            add_csv(run/'weights/index.csv',dict(epoch=epoch,last=snapshot.name,best=f'epoch_{best_epoch:02}.pt',
                    best_ap5095=best,last_sha256=digest(snapshot),best_sha256=digest(run/'weights/best.pt')))
            deltas={k:float((params[0]-start_params[k]).detach().abs().max()) for k,params in parts.items()}
            assert min(deltas.values())>0 and optimizer.updates-before_updates>0
            torch.cuda.synchronize()
            t1=time.time()
            samples,mean_util,max_util=monitor.during(t0,t1)
            av=np.mean(np.asarray(losses),axis=0)
            result=dict(group=group,epoch=epoch,train_loss=float(av[0]),ap5095=current,ap50=val['ap50'],
                   box_sum=float(av[1::3].sum()),cls_sum=float(av[2::3].sum()),dfl_sum=float(av[3::3].sum()),
                   train_images=len(train_ds),val_images=len(val_ds),batches=step,updates=optimizer.updates-before_updates,
                   amp_skips=skips,longest_skip_streak=max_streak,amp_scale=scaler.get_scale(),
                   lr_first=learning_rate((epoch-1)*250),lr_last=lr,train_s=train_s,val_s=val_s,epoch_wall_s=t1-t0,
                   best_epoch=best_epoch,train_list_sha256=split_hash,val_list_sha256=frozen['val_list_sha256'],
                   peak_alloc_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                   gpu_samples=samples,gpu_util_mean_pct=mean_util,gpu_util_max_pct=max_util,
                   val_detections=val['detections'],nms_calls=val['nms']['calls'],order_flip_sha256=plan_sha.hexdigest(),
                   training_pixels_sha256=content_sha.hexdigest(),started=datetime.fromtimestamp(t0).astimezone().isoformat(),
                   ended=datetime.fromtimestamp(t1).astimezone().isoformat())
            for i,k in enumerate(branches):
                for j,name in enumerate(('box','cls','dfl')):
                    result[k+'_'+name]=float(av[1+3*i+j])
                for field,value in totals[k].items():
                    result[k+'_'+field]=value
            if group=='Dual':
                result['one_winners_in_many_final']=common_wins
                result['one_winners_in_many_candidates']=common_pre
            for k,values in gradients.items():
                result['grad_'+k+'_mean']=float(np.mean(values))
                result['delta_'+k+'_max_probe']=deltas[k]
            add_csv(run/'epochs.csv',result)
            print(f"{group} {epoch}/20  AP50-95 {current:.5f}  AP50 {val['ap50']:.5f}  "
                  f"{result['updates']} updates, {skips} skips, {t1-t0:.1f} s",flush=True)
        observer.close()
        write_json(run/'status.json',dict(status='complete',epochs=COMMON['epochs'],updates=optimizer.updates,wall_s=time.time()-start))
    except BaseException:
        write_json(run/'failure.json',dict(epoch=epoch,step=step,traceback=traceback.format_exc(),unix_s=time.time()))
        if 'model' in locals() and 'optimizer' in locals():
            save_tensors(run/'failure_state_diagnostic_only.pt',dict(group=group,epoch=epoch,microbatch=step,
                         model=model.state_dict(),optimizer=optimizer.state_dict(),updates=optimizer.updates,
                         scaler=scaler.state_dict(),gradients={name:param.grad for name,param in model.named_parameters() if param.grad is not None},
                         ids_flips=records if 'records' in locals() else [],torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
        raise
    finally:
        monitor.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('group',choices=GROUPS)
    train(parser.parse_args().group)
