"""Fresh checkpoint validation, paired VOC examples and synchronized eager timing."""
import argparse
import time
from pathlib import Path

import runtime
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from evaluation import before_nms, decoded, duplicate_counts, postprocess, validate
from files import DATA,EVIDENCE,GROUPS,NAMES,ROOT,digest,read_json,write_csv,write_json
from runtime import COMMON,build,seed_all
from voc_loader import FrozenVOC,loader,original_boxes


def restore(group, tag):
    path=ROOT/'runs'/group/'weights'/(tag+'.pt')
    ckpt=torch.load(path,map_location='cpu')
    assert ckpt['group']==group
    model=build(group)
    model.load_state_dict(ckpt['model'],strict=True)
    return model.cuda().eval(), ckpt['epoch'],digest(path)


def independent(ds,groups=GROUPS):
    output={}
    for group in groups:
        dest=ROOT/'runs'/group/'independent'
        dest.mkdir(exist_ok=True)
        for tag in ('best','last'):
            model,epoch,sha=restore(group,tag)
            file=dest/(tag+'.npz')
            now=validate(model,group,ds,file)
            source=ROOT/'runs'/group/'validation'/f'epoch_{epoch}.npz'
            orig=read_json(source.with_suffix('.json'))
            with np.load(source) as a,np.load(file) as b:
                ids_equal=np.array_equal(a['ids'],b['ids']) and np.array_equal(a['offsets'],b['offsets'])
                maxabs=float(np.max(np.abs(a['detections']-b['detections']))) if ids_equal else None
            result=dict(group=group,weights=tag,epoch=epoch,sha256=sha,ap5095=now['ap5095'],ap50=now['ap50'],
                   ap5095_abs_diff=abs(orig['ap5095']-now['ap5095']),ap50_abs_diff=abs(orig['ap50']-now['ap50']),
                   same_ids_offsets=ids_equal,detections_max_abs_diff=maxabs,images=now['evaluated_images'])
            assert ids_equal and result['ap5095_abs_diff']<1e-9 and result['ap50_abs_diff']<1e-9 and maxabs<1e-4,result
            output[group+'_'+tag]=result
            print('Reload',group,tag,epoch,now['ap5095'],flush=True)
    write_json(EVIDENCE/'independent_validation.json',output)


def font(size,bold=False):
    path=Path('C:/Windows/Fonts')/('msyhbd.ttc' if bold else 'msyh.ttc')
    return ImageFont.truetype(str(path),size) if path.exists() else ImageFont.truetype('DejaVuSans.ttf',size)


def panel(image,rows,title,thresh):
    w,h=image.size
    tile=Image.new('RGB',(564,428),'#f4f5f2')
    draw=ImageDraw.Draw(tile)
    draw.text((14,7),title,font=font(19,True),fill='#172d3b')
    draw.text((390,10),f'{len(rows)} 框  ·  p > {thresh:.2f}',font=font(15),fill='#334f60')
    scale=min(540/w,376/h)
    iw,ih=round(scale*w),round(scale*h)
    left,top=(564-iw)//2,42+(376-ih)//2
    tile.paste(image.resize((iw,ih),Image.Resampling.LANCZOS),(left,top))
    palette=['#e55225','#118bc5','#a339b4','#39a659','#e9a112','#b43255','#26a4a4']
    ranked=sorted(rows,key=lambda r:r[4])
    for x,y,X,Y,p,cls in ranked:
        color=palette[int(cls)%len(palette)]
        xx,yy,XX,YY=left+x*scale,top+y*scale,left+X*scale,top+Y*scale
        draw.rectangle((xx,yy,XX,YY),outline=color,width=2)
    occupied=[]
    label_font=font(12)
    for x,y,X,Y,p,cls in reversed(ranked[-5:]):
        xx,yy,XX,YY=left+x*scale,top+y*scale,left+X*scale,top+Y*scale
        label=f'{NAMES[int(cls)]} {p:.3f}'
        width=draw.textbbox((0,0),label,font=label_font)[2]+5
        positions=[(xx,yy-17),(XX-width,yy-17),(xx,YY),(XX-width,YY)]
        positions.extend((xx, yy+18*row) for row in range(6))
        for lx,ly in positions:
            lx=max(left,min(lx,left+iw-width))
            ly=max(top,min(ly,top+ih-17))
            if all(lx+width<a or b<lx or ly+17<c or d<ly for a,c,b,d in occupied):
                break
        bounds=(lx,ly,lx+width,ly+16)
        occupied.append(bounds)
        draw.rectangle(bounds,fill=palette[int(cls)%len(palette)])
        draw.text((lx+2,ly-1),label,fill='white',font=label_font)
    return tile


@torch.inference_mode()
def cases(models,ds):
    kinds=read_json(DATA/'manifests/cases.json')
    want=set(kinds.values())
    answers={i:{} for i in want}
    totals={key:np.zeros(3,dtype=int) for key in ('O2M_pre','O2M_post','O2O','Dual')}
    branch_pairs=[]
    for batch in loader(ds):
        selected=[j for j,r in enumerate(batch['records']) if r['id'] in want]
        imgs=batch['img'].cuda().float()/255
        for group,model in models.items():
            with torch.cuda.amp.autocast():
                output=model(imgs)
            assert torch.isfinite(decoded(output)).all()
            if group=='Dual':
                many=decoded(output['one2many']).float(); one=decoded(output).float()
                assert torch.isfinite(many).all()
                many_score,many_cls=many[:,4:].max(1); one_score,one_cls=one[:,4:].max(1)
                many_xy,many_wh=many[:,:2],many[:,2:4]; one_xy,one_wh=one[:,:2],one[:,2:4]
                intersection=(torch.minimum(many_xy+many_wh/2,one_xy+one_wh/2)-torch.maximum(many_xy-many_wh/2,one_xy-one_wh/2)).clamp_min(0).prod(1)
                iou=intersection/(many_wh.prod(1)+one_wh.prod(1)-intersection+1e-7)
                for threshold in (.1,.01):
                    active=one_score>threshold; both=active&(many_score>threshold); same=both&(many_cls==one_cls)
                    for slot,record in enumerate(batch['records']):
                        branch_pairs.append(dict(image_id=record['id'],threshold=threshold,anchors=8400,active_one=int(active[slot].sum()),
                            active_many=int((many_score[slot]>threshold).sum()),both_active=int(both[slot].sum()),
                            same_class=int(same[slot].sum()),same_class_iou50=int((same[slot]&(iou[slot]>=.5)).sum())))
            # Use the same AP floor and 300 cap; only the displayed rows are filtered to 0.10.
            pred=postprocess(group,output)
            for j,r in enumerate(batch['records']):
                geom=batch['geom'][j]
                key='O2M_post' if group=='O2M' else group
                final=original_boxes(pred[j],geom,r)
                totals[key]+=duplicate_counts(final,r,.01)
                if j in selected:
                    answers[r['id']][key]=final.numpy()
                if group=='O2M':
                    raw=before_nms(decoded(output)[j],COMMON['conf_ap'] if j in selected else .01)
                    raw=original_boxes(raw,geom,r)
                    totals['O2M_pre']+=duplicate_counts(raw,r,.01)
                    if j in selected:
                        answers[r['id']]['O2M_pre']=raw.numpy()
    dest=ROOT/'figures'/'predictions'
    dest.mkdir(parents=True,exist_ok=True)
    labels={'ordinary':'普通场景','dense':'密集目标','small':'小目标','occlusion_candidate':'遮挡候选','duplicate_prone':'同类目标框重叠'}
    titles={'O2M_pre':'O2M · NMS 前','O2M_post':'O2M · NMS 后','O2O':'O2O · Top-K','Dual':'Dual · O2O Top-K'}
    table=[]
    for case,i in kinds.items():
        r=next(r for r in ds.records if r['id']==i)
        with Image.open(DATA/'JPEGImages/val'/(i+'.jpg')) as src:
            rgb=src.convert('RGB')
        for threshold,suffix in ((COMMON['conf_visual'],'primary'),(.01,'diagnostic')):
            plate=Image.new('RGB',(1144,932),'#e4ecea')
            draw=ImageDraw.Draw(plate)
            draw.text((14,6),f'VOC2007  /  {i}    {labels[case]}',font=font(23,True),fill='#172d3b')
            note='主结果 · 预设展示阈值' if suffix=='primary' else '低分诊断 · 不替代主结果'
            draw.text((14,43),f'第 20 轮最终权重  ·  {note}  ·  绘出全部过阈值框，标注最高 5 项的类别和分数',font=font(15),fill='#334f60')
            for j,(key,title) in enumerate(titles.items()):
                a=answers[i][key]
                rows=a[a[:,4]>threshold].tolist()
                plate.paste(panel(rgb,rows,title,threshold),(5+(j%2)*570,69+(j//2)*430))
                table.append(dict(case=case,image_id=i,output=key,threshold=threshold,shown_boxes=len(rows),
                                  regular_gt=sum(not o['difficult'] for o in r['objects'])))
            plate.save(dest/f'{case}_{i}_{suffix}.jpg',quality=92,subsampling=0)
    np.savez_compressed(dest/'case_predictions.npz',**{i+'__'+k:v for i,d in answers.items() for k,v in d.items()})
    write_json(dest/'schema.json',dict(columns=['xmin','ymin','xmax','ymax','score','class_id'],
               coords='original pixels, clipped 0-based half-open',stored_floor=COMMON['conf_ap'], max_det_post=300,
               primary=COMMON['conf_visual'],supplementary=.01,case_ids=kinds))
    write_json(EVIDENCE/'low_threshold_duplicates.json',dict(threshold=.01,images=250,
               columns=['predictions','same_class_regular_gt_iou_ge_0.5','excess_predictions_for_same_instance'],
               outputs={k:v.tolist() for k,v in totals.items()}))
    write_csv(dest/'case_counts.csv',table)
    write_csv(EVIDENCE/'dual_branch_decoded_anchor_agreement.csv',branch_pairs)


def deploy_view(dual):
    model=build('O2O')
    state=dual.state_dict()
    picked={}
    for k in model.state_dict():
        if k.startswith(('model.23.cv2.','model.23.cv3.')):
            old=k.replace('model.23.', 'model.23.one2one_',1)
        else:
            old=k
        picked[k]=state[old]
    model.load_state_dict(picked,strict=True)
    return model.cuda().eval()


@torch.inference_mode()
def benchmark(models,ds,destination=None):
    models=dict(models)
    models['Dual_single_inference']=deploy_view(models['Dual'])
    names=list(models)
    def predict(name,img):
        model=models[name]
        with torch.cuda.amp.autocast():
            outputs=model(img)
        return outputs
    for i in range(10):
        x=ds[i][0].unsqueeze(0).cuda().float()/255
        for name in names:
            postprocess(name,predict(name,x))
    rows=[]
    max_delta=0.
    for i in range(50):
        data=ds[i]
        x=data[0].unsqueeze(0).cuda().float()/255
        paired={}
        for j in range(4):
            name=names[(j+i)%4]
            torch.cuda.synchronize()
            t0=time.perf_counter_ns()
            output=predict(name,x)
            torch.cuda.synchronize()
            t1=time.perf_counter_ns()
            out=postprocess(name,output)
            torch.cuda.synchronize()
            t2=time.perf_counter_ns()
            assert torch.isfinite(decoded(output)).all() and torch.isfinite(out[0]).all()
            if name=='Dual':
                assert torch.isfinite(decoded(output['one2many'])).all()
            rows.append(dict(image_id=data[3]['id'], group=name, forward_ms=(t1-t0)/1e6,
                             post_ms=(t2-t1)/1e6, total_ms=(t2-t0)/1e6, boxes=len(out[0])))
            if name.startswith('Dual'):
                paired[name]=(decoded(output),out[0])
        for eager,single in zip(paired['Dual'],paired['Dual_single_inference']):
            assert eager.shape==single.shape
            if eager.numel():
                max_delta=max(max_delta,float((eager-single).abs().max()))
    assert max_delta==0
    summary={}
    for name in names:
        data=[r for r in rows if r['group']==name]
        line={'images':len(data)}
        for k in ('forward_ms','post_ms','total_ms'):
            line[k+'_p50'],line[k+'_p95']=map(float,np.percentile([r[k] for r in data],[50,95]))
        summary[name]=line
    destination=destination or EVIDENCE
    destination.mkdir(parents=True,exist_ok=True)
    write_csv(destination/'inference_timings.csv',rows)
    write_json(destination/'inference_timing.json',dict(settings=dict(batch=1,warmup_per_graph=10,measured_per_graph=50,
               precision='eager FP16 autocast with eval-only PSA core FP32 / FP32 post',copied_dual_max_abs=max_delta,threshold=COMMON['conf_ap'],
               boundaries='GPU tensor ready → decoded prediction → final GPU detections, CUDA synchronize at each boundary'),groups=summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('all','validation','figures','timing'),default='all')
    stage=parser.parse_args().stage
    seed_all()
    ds=FrozenVOC('val')
    if stage in ('all','validation'):
        independent(ds)
    if stage in ('all','figures','timing'):
        networks={g:restore(g,'last')[0] for g in GROUPS}
        if stage in ('all','figures'):
            cases(networks,ds)
        if stage in ('all','timing'):
            benchmark(networks,ds)
