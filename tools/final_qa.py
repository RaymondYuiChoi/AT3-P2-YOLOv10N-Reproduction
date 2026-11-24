"""Cross-read the serialized XLSX, original scalar tables, frozen case arrays and all 4,000 latency samples."""
import csv
import math
import re
from collections import defaultdict
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
from PIL import Image
from files import EVIDENCE,GROUPS,ROOT,digest,read_json,write_json


def rows(relative):
    with (ROOT/relative).open(encoding='utf-8-sig',newline='') as source:
        return list(csv.DictReader(source))


def close(actual,expected):
    assert math.isclose(float(actual),float(expected),rel_tol=2e-12,abs_tol=2e-12),(actual,expected)


def contents(path):
    sheets={}; formulas={}
    with ZipFile(path) as archive:
        assert archive.testzip() is None
        strings=[]
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings=[''.join(node.text or '' for node in entry.iter('{*}t')) for entry in ElementTree.fromstring(archive.read('xl/sharedStrings.xml'))]
        rels={node.get('Id'):node.get('Target') for node in ElementTree.fromstring(archive.read('xl/_rels/workbook.xml.rels'))}
        for sheet in ElementTree.fromstring(archive.read('xl/workbook.xml')).find('{*}sheets'):
            name=sheet.get('name'); saved={}; calc={}
            target=rels[sheet.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')]
            address=target.lstrip('/') if target.startswith('/') else 'xl/'+target
            for cell in ElementTree.fromstring(archive.read(address)).findall('.//{*}sheetData/{*}row/{*}c'):
                ref=cell.get('r');kind=cell.get('t');value=cell.findtext('{*}v');formula=cell.findtext('{*}f')
                assert kind!='e',(name,ref,value)
                if kind=='s' and value is not None:value=strings[int(value)]
                if kind=='inlineStr':value=''.join(node.text or '' for node in cell.iter('{*}t'))
                if value is not None and value!='':saved[ref]=value
                if formula is not None:calc[ref]=formula
            sheets[name]=saved;formulas[name]=calc
    return sheets,formulas


def workbook():
    data,formulas=contents(ROOT/'20Epoch三组实验总表.xlsx')
    counts={'终点评分':6,'逐轮训练':60,'逐轮验证':62,'终点逐类':120,'标签分配':80,'训练成本':3,'推理成本':4,'尺度召回':18}
    assert set(data)==set(counts)|{'结论速览'}
    for sheet,count in counts.items():
        assert [idx for idx in range(5,5+count+1) if 'A'+str(idx) in data[sheet]]==list(range(5,5+count)),sheet
    scope={'250 · 统一评估':'subset250','2,510 · 完整Val':'official_val2510','250 · A历史FP16':'historical250'}
    val={(row['group'],row['scope'],row['epoch']):row for row in rows('evidence/validation_all.csv')}
    epoch={(row['group'],row['epoch']):row for row in rows('evidence/epochs_all.csv')}
    class_rows=rows('evidence/class_metrics_subset250.csv')+rows('full_val/class_metrics.csv')
    classes={(row['group'],row['scope'],row['name']):row for row in class_rows if row['epoch']=='20'}
    assigned={(row['group'],row['branch'],row['epoch']):row for row in rows('evidence/assignment_verified.csv')}
    for name,count in counts.items():
        for idx in range(5,5+count):
            sheet=data[name]
            def value(col):return sheet.get(col+str(idx))
            def check(record,start,fields):
                for offset,key in enumerate(fields.split()):
                    cell=value(chr(ord(start)+offset));expected=record.get(key,'')
                    if expected=='':assert cell is None,(name,idx,key,cell)
                    else:close(cell,expected)
            if name=='终点评分':check(val[value('A'),scope[value('B')],'20'],'C','images regular_gt ap5095 ap50 tp50 fp50 fn50 ignored50 post_predictions')
            if name=='逐轮训练':check(epoch[value('A'),value('B')],'C','train_loss many_box many_cls many_dfl one_box one_cls one_dfl updates amp_skips train_s val_s peak_alloc_mib grad_backbone_mean grad_neck_mean grad_head_main_mean grad_head_o2o_mean')
            if name=='逐轮验证':check(val[value('A'),scope[value('B')],value('C')],'D','ap5095 ap50 val_loss_per_image effective50 tp50 fp50 fn50 raw_eligible_box_class raw_anchors post_predictions finite_images')
            if name=='终点逐类':check(classes[value('B'),scope[value('A')],value('C')],'D','gt ap5095 ap50 effective50 tp50 fp50 fn50')
            if name=='标签分配':check(assigned[value('A'),value('B'),value('C')],'D','gt candidates positives positives_per_gt unmatched contested_anchors positive_weight')
            if name=='训练成本':check(next(row for row in rows('evidence/training_costs.csv') if row['group']==value('A')),'B','updates skips train_s_sum val_s_sum run_wall_s train_s_median val_s_median max_alloc_mib max_reserved_mib')
            if name=='推理成本':
                check(next(row for row in rows('evidence/graph_costs.csv') if row['graph']==value('A')),'B','parameters profiler_supported_GFLOPs')
                check(next(row for row in rows('evidence/latency_epochs.csv') if row['epoch']=='20' and row['group']==value('A')),'D','forward_ms_p50 forward_ms_p95 post_ms_p50 post_ms_p95 total_ms_p50 total_ms_p95')
                check(next(row for row in rows('evidence/inference_memory.csv') if row['graph']==value('A')),'J','peak_alloc_mib')
            if name=='尺度召回':check(next(row for row in rows('evidence/size_recall.csv') if (row['scope'],row['group'],row['size'])==(scope[value('A')],value('B'),value('C'))),'D','gt tp50 recall50 tp75 recall75')
    assert len(formulas['结论速览'])==12
    for idx,group in enumerate(GROUPS,5):
        assert data['结论速览']['A'+str(idx)]==group
        for col,span,metric in [('E','subset250','ap5095'),('F','subset250','ap50'),('G','official_val2510','ap5095'),('H','official_val2510','ap50')]:
            ref=col+str(idx); formula=formulas['结论速览'][ref]
            target=re.fullmatch(r"'?终点评分'?!(\$?[EF]\$?\d+)",formula).group(1).replace('$','')
            close(data['结论速览'][ref],data['终点评分'][target]);close(data['结论速览'][ref],val[group,span,'20'][metric])
    return dict(sheets=len(data),rows_checked=counts,summary_formulas_verified=12)


def figures():
    case_dir=ROOT/'figures/predictions';table=rows('figures/predictions/case_counts.csv');cases=read_json(case_dir/'schema.json')['case_ids'];assert len(cases)==5 and len(table)==40
    with np.load(case_dir/'case_predictions.npz') as arrays:
        assert len(arrays.files)==20
        for row in table:
            pred=arrays[row['image_id']+'__'+row['output']];assert pred.ndim==2 and pred.shape[1]==6 and np.isfinite(pred).all()
            assert int((pred[:,4]>float(row['threshold'])).sum())==int(row['shown_boxes'])
    for case,ident in cases.items():
        for stage in ('primary','diagnostic'):
            with Image.open(case_dir/f'{case}_{ident}_{stage}.jpg') as img:img.load();assert img.size==(1144,932)
    for file in (ROOT/'figures').glob('*.png'):
        with Image.open(file) as img:img.load();assert img.width>500 and img.height>180 and np.asarray(img).std()>8
    assert len(list((ROOT/'figures').glob('*.png')))==4
    return dict(case_arrays=20,threshold_counts_verified=40,case_montages=10,charts_and_workbook_previews=4)


def timings():
    samples=rows('evidence/latency_samples.csv');summaries=rows('evidence/latency_epochs.csv');assert len(samples)==4000 and len(summaries)==80
    buckets=defaultdict(list)
    for row in samples:
        close(float(row['forward_ms'])+float(row['post_ms']),row['total_ms']);buckets[row['group'],row['epoch']].append(row)
    for summary in summaries:
        items=buckets[summary['group'],summary['epoch']];assert len(items)==len({row['image_id'] for row in items})==50
        if summary['group']=='Dual':
            assert read_json(EVIDENCE/'timing'/('epoch_'+summary['epoch'].zfill(2))/'inference_timing.json')['settings']['copied_dual_max_abs']==0
        for key in ('forward_ms','post_ms','total_ms'):
            for percentile in (50,95):close(np.percentile([float(item[key]) for item in items],percentile),summary[key+'_p'+str(percentile)])
    return dict(samples=4000,summaries=80,percentiles_reconciled=480,equivalent_egress_epochs=20)


def references():
    root_docs=sorted(ROOT.glob('*.md'));assert len(root_docs)==8
    count=0;deferred={'evidence/SHA256SUMS.txt','evidence/delivery_validation.json','evidence/provenance_after.json'}
    for doc in root_docs:
        text=doc.read_text('utf-8'); assert '\ufffd' not in text
        for url in re.findall(r'\[[^\]\n]*\]\(([^)\n]+)\)',text):
            if '://' not in url:
                if url in deferred:continue
                assert (doc.parent/url).exists(),(doc.name,url);count+=1
    study=next(ROOT.glob('06_*.md')).read_text('utf-8');assert [int(num) for num in re.findall(r'\*\*(\d+)｜',study)]==list(range(1,15))
    return dict(root_documents=len(root_docs),local_link_targets_checked=count,verified_by_final_seal=sorted(deferred),answered_questions=14)


def weights():
    full={row['group']:row for row in rows('full_val/results.csv')};pairs={}
    for group in GROUPS:
        folder=ROOT/'runs'/group/'weights'; indexes=rows('runs/'+group+'/weights/index.csv');assert len(indexes)==20
        last=indexes[-1];pairs[group]={'best':last['best'],'last':last['last']}
        for tag in ('best','last'):assert digest(folder/(tag+'.pt'))==digest(folder/last[tag])==last[tag+'_sha256']
        assert full[group]['checkpoint_sha256']==last['last_sha256']
    assert read_json(EVIDENCE/'metric_audit.json')['passed'] and read_json(EVIDENCE/'training_audit.json')['passed']
    assert len(rows('evidence/training_state_index.csv'))==40
    return pairs


if __name__=='__main__':
    result=dict(passed=True,workbook=workbook(),images=figures(),latency=timings(),weights=weights())
    destination=EVIDENCE/'delivery_validation.json'
    result['reports']=references();write_json(destination,result)
    print('PASS: XLSX/CSV, images, latency quantiles, model identities and research documents',flush=True)
