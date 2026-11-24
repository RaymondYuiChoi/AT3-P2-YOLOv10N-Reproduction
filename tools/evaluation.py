"""One metric implementation for all heads; difficult regions are threshold-specific ignores."""
from contextlib import contextmanager, ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import runtime
import numpy as np
import torch
import torchvision
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolov10.val import YOLOv10DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import ap_per_class, box_iou

from files import NAMES, write_json
from runtime import COMMON, save_tensors
from operator_trace import Trace
from voc_loader import loader, original_boxes

IOUV = torch.linspace(.5,.95,10)
MATCHER = SimpleNamespace(iouv=IOUV)


@contextmanager
def count_nms():
    counts = dict(calls=0, candidates=0, survivors_before_cap=0)
    actual = torchvision.ops.nms
    def observe(boxes, scores, iou):
        result = actual(boxes, scores, iou)
        counts['calls'] += 1
        counts['candidates'] += len(scores)
        counts['survivors_before_cap'] += len(result)
        return result
    with patch.object(torchvision.ops, 'nms', observe):
        yield counts


def decoded(predictions):
    if isinstance(predictions, dict):
        predictions = predictions['one2one']
    return predictions[0] if isinstance(predictions, (tuple,list)) else predictions


def postprocess(group, predictions, conf=COMMON['conf_ap']):
    scores = decoded(predictions).float()
    if group == 'O2M':
        return ops.non_max_suppression(scores, conf_thres=conf, iou_thres=COMMON['nms_iou'],
                   agnostic=False, multi_label=True, max_det=COMMON['max_det'], nc=20, in_place=False)
    context = SimpleNamespace(args=SimpleNamespace(max_det=COMMON['max_det']),nc=20)
    # Call the author's validator unchanged. Apply the shared confidence floor afterwards.
    rows = YOLOv10DetectionValidator.postprocess(context, scores)
    return [p[p[:,4] > conf] for p in rows]


def before_nms(scores, conf):
    xywh, classes = scores.float().T.split([4,20], dim=1)
    i, c = torch.where(classes > conf)
    return torch.cat((ops.xywh2xyxy(xywh[i]),classes[i,c,None],c.float()[:,None]),dim=1)


def gt(record, difficult):
    items = [o for o in record['objects'] if bool(o['difficult'])==difficult]
    return (torch.tensor([o['xyxy'] for o in items],dtype=torch.float32).reshape(-1,4),
            torch.tensor([o['cls'] for o in items],dtype=torch.float32))


def matched(rows, record):
    regular, classes = gt(record, False)
    difficult, diffclasses = gt(record, True)
    ious = box_iou(regular,rows[:,:4])
    correct = BaseValidator.match_predictions(MATCHER, rows[:,5], classes, ious)
    has_reg = (ious * (classes[:,None]==rows[:,5])).amax(0) if len(classes) else torch.zeros(len(rows))
    has_diff = ((box_iou(difficult, rows[:,:4]) * (diffclasses[:,None]==rows[:,5])).amax(0)
                if len(difficult) else torch.zeros(len(rows)))
    ignore = (has_diff[:,None]>=IOUV) & (has_reg[:,None]<IOUV)
    return correct.numpy(), ignore.numpy(), rows[:,4].numpy(), rows[:,5].numpy(), classes.numpy()


def duplicate_counts(rows, record, threshold=COMMON['conf_visual']):
    # An extra detection of the same annotated regular instance is counted once.
    candidates = rows[rows[:,4] > threshold]
    boxes, classes = gt(record, False)
    if not len(candidates) or not len(boxes):
        return [len(candidates),0,0]
    overlaps = box_iou(boxes, candidates[:,:4]) * (classes[:,None]==candidates[:,5])
    values, indices = overlaps.max(0)
    which = indices[values >= .5]
    return [len(candidates), len(which), len(which)-len(which.unique())]


def summarize(stats):
    tp, ig, conf, cls, targets = [np.concatenate(x) for x in zip(*stats)]
    aps = []
    for j in range(10):
        k = ~ig[:,j]
        metrics = ap_per_class(tp[k,j:j+1], conf[k], cls[k], targets, names=dict(enumerate(NAMES)))
        assert metrics[6].tolist() == list(range(20))
        aps.append(metrics[5][:,0])
    matrix = np.stack(aps,axis=1)
    per_class = []
    for index, name in enumerate(NAMES):
        select = cls == index
        per_class.append(dict(class_id=index, name=name, gt=int((targets==index).sum()), predictions=int(select.sum()),
            effective_by_iou=(~ig[select]).sum(0).tolist(), tp_by_iou=(tp[select] & ~ig[select]).sum(0).tolist(),
            above_010=int((select & (conf>.10)).sum()), above_001=int((select & (conf>.01)).sum())))
    assert np.isfinite(matrix).all() and (matrix>=0).all() and (matrix<=1).all()
    return dict(ap50=float(matrix[:,0].mean()), ap5095=float(matrix.mean()), ap_by_class_iou=matrix.tolist(), per_class=per_class,
                ignored_by_iou=ig.sum(0).tolist(), detections=len(conf), regular_gt=len(targets))


@torch.inference_mode()
def validate(model, group, val_ds, file, extra_raw=False):
    model.eval()
    stats, all_rows, ids, offsets = [], [], [], [0]
    dup_post, dup_pre = np.zeros(3,dtype=int), np.zeros(3,dtype=int)
    raw_eligible = torch.zeros(20,device='cuda',dtype=torch.long)
    max_score = torch.zeros(20,device='cuda')
    loss_by_batch=[]
    verify_batches=0
    with ExitStack() as stack:
        counts=stack.enter_context(count_nms())
        guard=Trace(model,'psa32')
        stack.callback(guard.release)
        criterion=model.init_criterion()
        for batch in loader(val_ds):
            images = batch['img'].cuda(non_blocking=True).float().div_(255)
            with torch.cuda.amp.autocast():
                output = model(images)
            auxiliary=decoded(output['one2many']) if group=='Dual' else None
            checked=guard.records(decoded(output),auxiliary)
            if not all(record['all_finite'] for record in checked):
                write_json(file.with_suffix('.failed_batch.json'),dict(ids=[item['id'] for item in batch['records']],checks=checked))
                save_tensors(file.with_suffix('.failed_batch.pt'),dict(ids=[item['id'] for item in batch['records']],attention=guard.last,decoded=decoded(output),aux=auxiliary))
                raise FloatingPointError('PSA or detector output nonfinite; no replacement or next batch')
            verify_batches+=1
            with torch.cuda.amp.autocast():
                objective,components=criterion(output,batch)
            assert torch.isfinite(objective) and torch.isfinite(components).all(), 'nonfinite validation loss'
            loss_by_batch.append(dict(images=len(images),objective=float(objective),components=components.tolist()))
            score=decoded(output)[:,4:,:].float()
            assert score.shape[1:]==(20,8400) and score.min()>=0 and score.max()<=1
            raw_eligible+=(score>COMMON['conf_ap']).sum((0,2))
            max_score=torch.maximum(max_score,score.amax((0,2)))
            pred = postprocess(group, output)
            for i,(p,r,geom) in enumerate(zip(pred,batch['records'],batch['geom'])):
                rows = original_boxes(p,geom,r)
                assert torch.isfinite(rows).all() and (not len(rows) or rows[:,4].min()>COMMON['conf_ap'])
                stats.append(matched(rows,r))
                all_rows.append(rows.numpy())
                ids.append(r['id'])
                offsets.append(offsets[-1]+len(rows))
                dup_post += duplicate_counts(rows,r)
                if group=='O2M' and extra_raw:
                    raw = before_nms(decoded(output)[i], COMMON['conf_visual'])
                    dup_pre += duplicate_counts(original_boxes(raw,geom,r),r)
    assert ids==[r['id'] for r in val_ds.records]
    dets = np.concatenate(all_rows)
    np.savez_compressed(file, ids=np.array(ids), offsets=np.array(offsets), detections=dets)
    result = summarize(stats)
    result.update(nms=counts, raw_anchor_positions=8400*len(ids), evaluated_images=len(ids),
                  precision='AMP; PSA core FP32, cast back; FP32 post',all_finite_images=len(ids),checked_batches=verify_batches,
                  precision_evidence=guard.first_events,val_loss_batches=loss_by_batch,
                  val_loss_per_image=sum(item['objective'] for item in loss_by_batch)/len(ids),
                  before_filter_eligible_by_class=raw_eligible.tolist(),before_filter_max_score_by_class=max_score.tolist(),
                  duplicates_post=dup_post.tolist(), duplicates_pre=dup_pre.tolist() if group=='O2M' and extra_raw else None)
    write_json(file.with_suffix('.json'),result)
    return result
