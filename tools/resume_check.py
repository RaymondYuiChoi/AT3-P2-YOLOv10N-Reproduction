import csv
import hashlib
import math
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import runtime
import numpy as np
import torch
import torchvision
from audit import check_voc, read_csv
from evaluation import count_nms, decoded, postprocess, validate
from files import COMMIT, DATA, EVIDENCE, GROUPS, NAMES, PRISTINE, REPO, ROOT, digest, read_json, write_csv, write_json
from observations import assigners, grad_norms, make_optimizer
from operator_trace import Trace
from preflight import metric_fixture
from recheck import restore
from runtime import COMMON, fingerprint, load_initial, region, seed_all, split_params
from train import learning_rate
from voc_loader import FrozenVOC, epoch_plan, loader

EXP=ROOT.parent/'AT3-P2-YOLOv10N-EXP20-01'
PILOT=ROOT.parent/'AT3-P2-YOLOv10N-PILOT-01'
NUM=ROOT.parent/'AT3-P2-YOLOv10N-NUMERIC-01'


def history():
    result={}
    reading=[]
    for directory in (EXP,PILOT,NUM):
        inventory=directory/'evidence/SHA256SUMS.txt'
        entries=inventory.read_text(encoding='utf-8-sig').splitlines()
        for entry in entries:
            checksum,relative=entry.split('  ',1)
            assert digest(directory/relative)==checksum,(directory.name,relative)
        result[directory.name]=dict(files_verified=len(entries),manifest_sha256=digest(inventory))
        for source in sorted(directory.glob('*.md')):
            text=source.read_text(encoding='utf-8-sig')
            reading.append(dict(source=str(source.relative_to(ROOT.parent)),sha256=digest(source),lines=len(text.splitlines()),headings=' | '.join(line.strip() for line in text.splitlines() if line.startswith('#'))))
    decision=next((NUM/'source_records').glob('*恢复建议*'))
    assert digest(decision)==digest(ROOT/'source_records'/decision.name)
    reading.append(dict(source=str(decision.relative_to(ROOT.parent)),sha256=digest(decision),lines=len(decision.read_text('utf-8').splitlines()),headings='研究端决策 V1.0'))
    write_csv(EVIDENCE/'sources_read.csv',reading)
    workbook=next(EXP.glob('*.xlsx'))
    with ZipFile(workbook) as archive:
        shared=[]
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared=[''.join(x.itertext()) for x in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        sheets={}
        rels={node.get('Id'):node.get('Target') for node in ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))}
        for sheet in ET.fromstring(archive.read('xl/workbook.xml')).find('{*}sheets'):
            target=rels[sheet.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')]
            rows=[]
            for row in ET.fromstring(archive.read(target.lstrip('/') if target.startswith('/') else 'xl/'+target)).find('{*}sheetData'):
                values=[]
                for cell in row:
                    value=cell.findtext('{*}v')
                    if cell.get('t')=='s' and value is not None: value=shared[int(value)]
                    if cell.get('t')=='inlineStr': value=''.join(cell.find('{*}is').itertext())
                    if value is not None: values.append(value)
                if values: rows.append(values)
            sheets[sheet.get('name')]=dict(nonempty_rows=len(rows),first=rows[0],last=rows[-1])
    prior=read_json(NUM/'evidence/provenance_after.json')
    assert {key:item['nonempty_rows'] for key,item in sheets.items()}==prior['exp20_workbook_nonempty_rows']
    assert digest(workbook)==prior['exp20_workbook_sha256']
    matrix=read_csv(NUM/'evidence/matrix.csv'); reconciliation=read_csv(NUM/'evidence/ap_reconciliation.csv')
    assert len(matrix)==12 and all(int(row['valid_images'])==250 for row in matrix if row['mode']=='psa32')
    assert len(reconciliation)==8 and not any(float(row['max_absolute_cell_difference']) for row in reconciliation)
    return dict(inventories=result,source_workbook=sheets,workbook_sha256=digest(workbook),numeric_matrix_rows=len(matrix),prior_reconciled_cells=1600,decision_sha256=digest(decision))


def git(directory,*args):
    return subprocess.check_output(['git','--no-optional-locks','-C',str(directory),*args],text=True,encoding='utf-8').strip()


def probe(group,batch):
    seed_all()
    model=load_initial(group); parts=split_params(model); criterion=model.init_criterion()
    initial=fingerprint(model.state_dict().items()); expected=read_json(EVIDENCE/'initialization.json')[group]
    assert initial==expected['tensor_sha256']
    tops={name:item.topk for name,item in assigners(group,criterion).items()}
    assert list(tops.values())==([10,1] if group=='Dual' else [10 if group=='O2M' else 1])
    assert hasattr(model.model[-1],'one2one_cv2')==(group=='Dual')
    images=batch['img'].cuda().float()/255
    train_core={}; eval_core={}; isolated={}; total_value=None; norms={}; deltas={}; updates=0
    if group!='O2M':
        model.train()
        watcher=Trace(model,'original')
        with torch.cuda.amp.autocast():
            output=model(images); total,components=criterion(output,batch)
            branch_losses={key:getattr(criterion,key)(output[key],batch)[0] for key in ('one2one','one2many')} if group=='Dual' else {}
        train_core=watcher.first_events; watcher.release()
        assert all(train_core['autocast_at_op'].values()) and [op['output'] for op in train_core['aten_calls'] if op['op']=='aten.bmm.default']==['float16','float16']
        assert torch.isfinite(total) and torch.isfinite(components).all()
        total_value=total.item()
        for key,loss in branch_losses.items():
            model.zero_grad(set_to_none=True); loss.backward(retain_graph=True); isolated[key]=grad_norms(parts)
        if branch_losses:
            assert sum(branch_losses.values()).item()==total_value
            assert all(isolated['one2one'][name]==0 for name in ('backbone','neck','head_main'))
            assert isolated['one2one']['head_o2o']>0 and isolated['one2many']['head_o2o']==0
            assert all(isolated['one2many'][name]>0 for name in ('backbone','neck','head_main'))
        model.zero_grad(set_to_none=True)
        optimizer=make_optimizer(model,COMMON)
        scaler=torch.cuda.amp.GradScaler(init_scale=32,growth_interval=2000)
        scaler.scale(total).backward(); scaler.unscale_(optimizer); norms=grad_norms(parts)
        assert all(math.isfinite(value) and value>0 for value in norms.values())
        if group=='Dual':
            before={name:params[0].detach().clone() for name,params in parts.items()}
            scaler.step(optimizer); scaler.update(); updates=optimizer.updates
            deltas={name:float((params[0].detach()-before[name]).abs().max()) for name,params in parts.items()}
            assert updates==1 and all(math.isfinite(value) and value>0 for value in deltas.values())
    model.eval()
    watcher=Trace(model,'psa32')
    with torch.no_grad(),torch.cuda.amp.autocast(): output=model(images)
    diagnostics=watcher.records(decoded(output),decoded(output['one2many']) if group=='Dual' else None)
    assert all(row['all_finite'] for row in diagnostics)
    eval_core=watcher.first_events; watcher.release()
    with count_nms() as calls: postprocess(group,output,.0001)
    assert (calls['calls']>0)==(group=='O2M')
    assert digest(ROOT/'weights'/f'init_{group}.pt')==expected['file_sha256']
    return dict(group=group,loaded_tensor_sha256=initial,topk=tops,heads=expected['heads'],ids=[rec['id'] for rec in batch['records']],train_core=train_core,eval_core=eval_core,eval_dtypes={key:value for key,value in diagnostics[0].items() if key.endswith('_dtype')},finite=True,loss=total_value,gradient_norms=norms,branch_isolation=isolated,parameter_max_change=deltas,disposable_updates=updates,nms_probe=calls)


def main():
    seed_all()
    prior=read_json(NUM/'evidence/provenance_after.json')
    assert Path(sys.executable).resolve()==Path(prior['executable']).resolve() and sys.version_info[:3]==(3,10,11)
    assert (torch.__version__,torchvision.__version__,torch.version.cuda)==('2.0.1+cu118','0.15.2+cu118','11.8')
    assert torch.cuda.get_device_name()==prior['gpu'] and sys.platform=='win32'
    for source in (PRISTINE,REPO): assert git(source,'rev-parse','HEAD')==COMMIT
    assert not git(PRISTINE,'status','--porcelain')
    assert git(REPO,'diff','--name-only')=='ultralytics/nn/modules/block.py' and git(REPO,'ls-files','--others','--exclude-standard')=='at3_variant.py'
    assert git(REPO,'diff')==git(NUM/'research/yolov10','diff')
    hist=history(); write_json(EVIDENCE/'historical_provenance.json',hist)
    frozen=read_json(EVIDENCE/'freeze.json'); init=read_json(EVIDENCE/'initialization.json')
    for relative,sha in frozen['locked_files'].items(): assert digest(EXP/relative)==sha,relative
    for relative in ('research/yolov10/at3_variant.py','tools/observations.py','tools/runtime.py','tools/voc_loader.py','configs/Dual.json','configs/O2M.json','configs/O2O.json','weights/init_O2O.pt','weights/init_Dual.pt','weights/init_O2M.pt','evidence/paired_plans.json','evidence/paired_image_pixels.json','evidence/initialization.json','evidence/dataset.json'):
        assert digest(ROOT/relative)==frozen['locked_files'][relative],relative
    for group in GROUPS:
        cfg=read_json(ROOT/'configs'/f'{group}.json'); assert all(cfg[key]==value for key,value in COMMON.items())
        assert init[group]['file_sha256']==digest(PILOT/'weights'/f'init_{group}.pt')
    weights={label:digest(ROOT/relative) for label,relative in [('A19','runs/O2M/weights/best.pt'),('A20','runs/O2M/weights/last.pt'),('O2O_init','weights/init_O2O.pt'),('Dual_init','weights/init_Dual.pt')]}
    assert weights['A19']==prior['weights']['A19']['sha256'] and weights['A20']==prior['weights']['A20']['sha256'] and weights['Dual_init']==prior['weights']['C0']['sha256']
    assert all(len({init[g]['parts'][part] for g in GROUPS})==1 for part in ('backbone','neck','main_head','dfl'))
    assert init['Dual']['parts']['main_head']==init['Dual']['parts']['one2one_head']
    assert check_voc()==1250 and read_json(DATA/'manifests/classes.json')==NAMES
    for item in (DATA/'manifests').iterdir(): assert digest(item)==digest(EXP/'datasets/VOC2007/manifests'/item.name)
    source=read_csv(NUM/'evidence/val250_inputs.csv'); ds=FrozenVOC('val')
    for idx,row in enumerate(source): assert row['image_id']==ds.records[idx]['id'] and hashlib.sha256(ds[idx][0].numpy().tobytes()).hexdigest()==row['letterbox_uint8_sha256']
    lrrows=read_csv(ROOT/'configs/lr_epochs.csv'); assert len(lrrows)==20
    for epoch,row in enumerate(lrrows,1): assert float(row['first'])==learning_rate((epoch-1)*250) and float(row['last'])==learning_rate(epoch*250-1)
    write_csv(ROOT/'configs/lr_5000.csv',[dict(step0=step,epoch=step//250+1,lr=learning_rate(step)) for step in range(5000)])
    train_ds=FrozenVOC('train'); train_ds.plan=epoch_plan(len(train_ds),1); batch=next(iter(loader(train_ds)))
    probes={group:probe(group,batch) for group in GROUPS}; tests=metric_fixture()
    write_json(EVIDENCE/'single_step.json',dict(passed=True,groups=probes,post_fixture=tests))
    checks=[]
    dest=ROOT/'runs/O2M/repaired_subset'; dest.mkdir(exist_ok=True)
    for state,tag,epoch in [('A19','best',19),('A20','last',20)]:
        model,loaded,sha=restore('O2M',tag); assert loaded==epoch and sha==weights[state]
        path=dest/f'epoch_{epoch}.npz'; result=validate(model,'O2M',ds,path,extra_raw=True)
        historical=NUM/'outputs'/state/'psa32'
        with np.load(path) as actual,np.load(historical/'detections.npz') as expected:
            assert all(np.array_equal(actual[key],expected[key]) for key in ('ids','offsets','detections'))
        assert np.array_equal(result['ap_by_class_iou'],read_json(historical/'metrics.json')['ap_by_class_iou'])
        checks.append(dict(state=state,epoch=epoch,checkpoint_sha256=sha,exact_predictions_and_AP=True,images=250,ap50=result['ap50'],ap5095=result['ap5095']))
        del model
    write_csv(EVIDENCE/'O2M_recheck.csv',checks)
    keys=['tools/'+name+'.py' for name in ('train','runtime','evaluation','observations','operator_trace','voc_loader','resume_check')]+['research/yolov10/ultralytics/nn/modules/block.py','research/yolov10/at3_variant.py','configs/lr_5000.csv','evidence/paired_image_pixels.json','evidence/paired_plans.json','evidence/freeze.json']+[f'configs/{group}.json' for group in GROUPS]
    smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    write_json(EVIDENCE/'environment.json',dict(executable=sys.executable,python=sys.version,os=platform.platform(),windows_build=sys.getwindowsversion().build,gpu=smi,torch=torch.__version__,torchvision=torchvision.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),numpy=np.__version__,tf32=False,seed=2007,commit=COMMIT,imported=runtime.ultralytics.__file__))
    write_json(EVIDENCE/'resume_gate.json',dict(passed=True,dual_disposable_update=1,formal_updates_at_gate=0,hashes=weights,images_checked=1250,paired_lr_steps=5000,official_commit=COMMIT,locked_files={key:digest(ROOT/key) for key in keys}))
    print('PASS: history, freeze, PSA operators, gradients, Dual disposable update, A19/A20 exact repair recheck',flush=True)


if __name__=='__main__':
    main()
