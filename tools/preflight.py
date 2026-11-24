"""Independent pristine-Dual reference and gated single-step/behavior checks."""
import argparse
import math
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import runtime
import torch
from ultralytics.models.yolov10.val import YOLOv10DetectionValidator

from evaluation import count_nms, decoded, matched, postprocess
from files import EVIDENCE, GROUPS, write_json
from observations import Assignments, assigners, grad_norms, make_optimizer
from runtime import COMMON, SOURCE, PRISTINE, fingerprint, load_initial, save_tensors, seed_all, split_params


def fixture(b):
    x = torch.rand((b,3,640,640),generator=torch.Generator().manual_seed(318)).cuda()
    # Coincident targets explicitly exercise conflict resolution; a tiny corner has no anchor centre.
    boxes = [[.33,.4,.27,.32],[.33,.4,.27,.32],[.72,.64,.19,.22],[.001,.001,.001,.001]]
    return x, dict(batch_idx=torch.arange(b).repeat_interleave(4).float().cuda(),
                   cls=torch.tensor([7,7,14,1],dtype=torch.float32).repeat(b).view(-1,1).cuda(),
                   bboxes=torch.tensor(boxes,dtype=torch.float32).repeat(b,1).cuda())


def flatten(value, key=''):
    if isinstance(value, dict):
        return {name:t for k,v in value.items() for name,t in flatten(v,key+'/'+k).items()}
    if isinstance(value,(tuple,list)):
        return {name:t for i,v in enumerate(value) for name,t in flatten(v,key+'/'+str(i)).items()}
    return {key:value.detach().cpu()}


def reference(save=False):
    seed_all()
    model = load_initial('Dual')
    x,batch=fixture(1)
    with torch.no_grad():
        out=model.train()(x)
        tensors=flatten(out,'train')
        tensors.update(flatten(model.loss(batch,preds=out),'loss'))
        ev=model.eval()(x)
        tensors.update(flatten(ev,'eval'))
        context=SimpleNamespace(nc=20,args=SimpleNamespace(max_det=300))
        tensors.update(flatten(YOLOv10DetectionValidator.postprocess(context,ev),'original_post'))
    filename=EVIDENCE/'pristine_dual_tensors.pt'
    if save:
        assert SOURCE==PRISTINE
        save_tensors(filename,tensors)
        write_json(EVIDENCE/'pristine_reference.json',dict(import_root=str(SOURCE), tensor_sha256=fingerprint(tensors.items())))
        return
    assert SOURCE!=PRISTINE
    prior=torch.load(filename,map_location='cpu')
    assert prior.keys()==tensors.keys()
    differences={key:float((prior[key]-val).abs().max()) for key,val in tensors.items()}
    assert max(differences.values())==0, differences
    return differences


def inspect_group(group):
    seed_all()
    x,batch=fixture(4)
    model=load_initial(group)
    criterion=model.init_criterion()
    observer=Assignments(group,criterion)
    candidates={}
    opt=make_optimizer(model,COMMON)
    scaler=torch.cuda.amp.GradScaler(init_scale=COMMON['amp_init_scale'],growth_interval=COMMON['amp_growth_interval'])
    parts=split_params(model)
    before={k:params[0].detach().clone() for k,params in parts.items()}
    with ExitStack() as stack:
        for branch, mod in assigners(group,criterion).items():
            original=mod.select_highest_overlaps
            def capture(mask,overlap,n,original=original,branch=branch):
                candidates[branch]=mask[0].detach().cpu().bool()
                return original(mask,overlap,n)
            stack.enter_context(patch.object(mod,'select_highest_overlaps',capture))
        with torch.cuda.amp.autocast():
            out=model(x)
            total, components=criterion(out,batch)
    details={}
    for branch, raw in candidates.items():
        fg,idx=observer.maps[branch]
        assigned=[torch.where(fg[0] & (idx[0]==i))[0].cpu().tolist() for i in range(4)]
        details[branch]=dict(per_gt=[len(s) for s in assigned],assigned_anchors=assigned,
                candidates=[torch.where(v)[0].tolist() for v in raw],
                unmatched_gt=[i for i,a in enumerate(assigned) if not a],
                conflict_anchors=torch.where(raw.sum(0)>1)[0].tolist())
    overlap=None
    if group=='Dual':
        pairs=[(i,a) for i, anchors in enumerate(details['one']['assigned_anchors']) for a in anchors]
        overlap=dict(winners=len(pairs),in_many_candidates=sum(bool(candidates['many'][i,a]) for i,a in pairs),
                     in_many_final=sum(a in details['many']['assigned_anchors'][i] for i,a in pairs))
    observer.close()
    scaler.scale(total).backward()
    scaler.unscale_(opt)
    norms=grad_norms(parts)
    scaler.step(opt)
    scaler.update()
    deltas={k:float((params[0]-before[k]).detach().abs().max()) for k,params in parts.items()}
    assert opt.updates==1 and torch.isfinite(total) and all(math.isfinite(v) and v>0 for v in (*norms.values(),*deltas.values()))
    assert all(len(d['conflict_anchors']) and 3 in d['unmatched_gt'] for d in details.values())
    result=dict(input=list(x.shape),train_output={k:list(v.shape) for k,v in flatten(out).items()},loss=float(total),
                components=components.tolist(),gradient_l2=norms,probe_parameter_max_delta=deltas,
                optimizer_updates=opt.updates,amp_skips=1-opt.updates,assignments_image0=details,overlap=overlap,
                assigners={k:dict(topk=v.topk,alpha=v.alpha,beta=v.beta,eps=v.eps) for k,v in assigners(group,criterion).items()})
    with torch.inference_mode(), count_nms() as calls:
        ev=model.eval()(x[:1])
        # A low floor exercises the NMS kernel even at initialization.
        pred=postprocess(group,ev,conf=.0001)
        result['postprocess']=dict(**calls, raw_anchors=decoded(ev).shape[-1], score_floor=.0001,
                    eligible_box_classes=int((decoded(ev)[0,4:]>.0001).sum()), topk_or_cap=300, returned=len(pred[0]),
                    input_branch='one2one' if group=='Dual' else 'sole',eval_output={k:list(v.shape) for k,v in flatten(ev).items()})
        if group=='Dual':
            changed={'one2many':None,'one2one':ev['one2one']}
            assert torch.equal(pred[0],postprocess(group,changed,conf=.0001)[0])
    assert (calls['calls']>0)==(group=='O2M')
    if group=='Dual':
        isolated={}
        for name in ('one2one','one2many'):
            model.train().zero_grad(set_to_none=True)
            raw=model(x[:1])
            _, b1=fixture(1)
            loss,_=getattr(criterion,name)(raw[name],b1)
            loss.backward()
            isolated[name]=grad_norms(parts)
        assert isolated['one2one']['backbone']==isolated['one2one']['neck']==isolated['one2one']['head_main']==0
        assert isolated['one2one']['head_o2o']>0 and isolated['one2many']['backbone']>0 and isolated['one2many']['head_o2o']==0
        result['isolated_branch_grad_l2']=isolated
    return result


def metric_fixture():
    record=dict(objects=[dict(xyxy=[0,0,20,20],cls=0,difficult=0),dict(xyxy=[50,50,70,70],cls=1,difficult=1)])
    rows=torch.tensor([[0,0,20,20,.9,0],[50,50,70,70,.8,1],[0,0,20,20,.7,2]],dtype=torch.float32)
    tp,ignore,*_=matched(rows,record)
    assert tp[0].all() and not tp[1:].any() and ignore[1].all() and not ignore[[0,2]].any()
    pred=torch.zeros(1,24,301,device='cuda')
    pred[0,:4,:3]=torch.tensor([100.,100.,30.,30.],device='cuda')[:,None]
    pred[0,4,0]=.9
    pred[0,4,1]=.8
    pred[0,5,2]=.7
    kept=postprocess('O2M',pred,conf=.1)[0]
    assert len(kept)==2 and set(kept[:,5].tolist())=={0,1}
    return dict(same_class_duplicate_suppressed=True,different_class_overlap_retained=True,difficult_ignore_verified=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pristine',action='store_true')
    args=parser.parse_args()
    if args.pristine:
        reference(True)
        return
    delta=reference()
    groups={g:inspect_group(g) for g in GROUPS}
    tests=metric_fixture()
    write_json(EVIDENCE/'single_step.json',dict(passed=True, pristine_vs_copy_max_abs=delta,groups=groups,fixtures=tests))
    print('PASS: independent pristine regression, three AMP updates, TAL, gradients and postprocessing.')


if __name__=='__main__':
    main()
