"""Render the research handoff directly from measured training, boxes, assignments and timings."""
from statistics import median
from files import EVIDENCE,GROUPS,NAMES,ROOT,read_json
from review_metrics import read_rows
from nested_differences import save_differences


def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join('---' for _ in head)+' |',*['| '+' | '.join(str(item) for item in row)+' |' for row in rows]])


def fmt(val,digits=3):return f'{float(val):,.{digits}f}'
def pct(val):return fmt(float(val)*100,3)+'%'
def integer(val):return f'{int(val):,}'
def save(name,parts): (ROOT/name).write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
def source(path):return read_rows(ROOT/path)
def select(rows,**keys):return [row for row in rows if all(str(value)==row[key] for key,value in keys.items())]
def get(rows,**keys):
    found=select(rows,**keys);assert len(found)==1,(keys,len(found));return found[0]
def txt_norm(values):return ' / '.join(f'{key} {fmt(value,2)}' for key,value in values.items())


def main():
    epochs=source('evidence/epochs_all.csv');validation=source('evidence/validation_all.csv');classes=source('evidence/class_metrics_subset250.csv')+source('full_val/class_metrics.csv');assign=source('evidence/assignment_verified.csv');timing=source('evidence/latency_epochs.csv');cost=source('evidence/training_costs.csv');scales=source('evidence/size_recall.csv');dups=source('evidence/duplicate_counts.csv');mem=source('evidence/inference_memory.csv');graphs=source('evidence/graph_costs.csv');shared=source('evidence/paired_regions.csv');overlap=source('evidence/winner_overlap.csv')
    subset='subset250';full='official_val2510';init=read_json(EVIDENCE/'single_step.json')['groups'];isolated=read_json(EVIDENCE/'isolated_final.json')['results'];audit=read_json(EVIDENCE/'metric_audit.json');ta=read_json(EVIDENCE/'training_audit.json');split=read_json(ROOT/'full_val/split.json');reloads=read_json(EVIDENCE/'independent_validation.json');cases=source('figures/predictions/case_counts.csv');low=read_json(EVIDENCE/'low_threshold_duplicates.json')['outputs'];eager='Dual';single='Dual_single_inference'
    def point(scope,group,epoch=20):return get(validation,scope=scope,group=group,epoch=epoch)
    def scoretable(scope):return table(['组别','AP50–95','AP50','TP50','FP50','FN50','有效框50','有 TP 的类别'],[[name,pct(row['ap5095']),pct(row['ap50']),*[integer(row[key]) for key in ('tp50','fp50','fn50','effective50')],row['classes_with_tp50']+'/20'] for name in GROUPS for row in [point(scope,name)]])
    def class_table(scope):
        rows=[]
        for name in NAMES:
            selections=[get(classes,scope=scope,group=group,epoch=20,name=name) for group in GROUPS]
            rows.append([name,selections[0]['gt'],*[pct(rec['ap5095']) for rec in selections],*[pct(rec['ap50']) for rec in selections]])
        return table(['类别','普通GT','A AP50–95','B AP50–95','C AP50–95','A AP50','B AP50','C AP50'],rows)
    def size_table(scope):return table(['640图尺度','普通GT','A R50 / R75','B R50 / R75','C R50 / R75'],[[label,get(scales,scope=scope,group='O2M',size=key)['gt'],*[pct(row['recall50'])+' / '+pct(row['recall75']) for name in GROUPS for row in [get(scales,scope=scope,group=name,size=key)]]] for key,label in [('small','小（<32²）'),('medium','中（32²–<96²）'),('large','大（≥96²）')]])
    def rank(scope,key='ap5095'):return ' > '.join(sorted(GROUPS,key=lambda group:-float(point(scope,group)[key])))
    def pp(scope,first,other):return fmt((float(point(scope,first)['ap5095'])-float(point(scope,other)['ap5095']))*100,3)
    def wline(group,tag):
        row=reloads[group+'_'+tag];return f"{tag}（E{row['epoch']}）：`{row['sha256']}`，重载 AP50–95 {pct(row['ap5095'])}、AP50 {pct(row['ap50'])}；逐框最大绝对差 {row['detections_max_abs_diff']:.3g}。"
    for group,filename,title,why in [('O2O','02_O2O训练过程与结果.md','B｜O2O-only：冻结初值重新训练','取消 detach 后，top-1 的检测损失可直接更新 backbone/neck；有梯度不代表有限预算下可达到同等精度。'),('Dual','03_Dual训练过程与结果.md','C｜Dual：首次正式训练','主干由 O2M 路监督。O2O 的输入在官方头内 detach，专属检测头仍然学习和更新；推理计分只取该头。')]:
        tr=select(epochs,group=group);c=get(cost,group=group);best=max(tr,key=lambda r:float(r['ap5095']));branches=('many','one') if group=='Dual' else ('one',)
        parts=['# '+title,why,f"**完成 20/20，正式更新 {integer(c['updates'])}，AMP 跳步 {c['skips']}。** 起点 `weights/init_{group}.pt`；不继承小批诊断的更新、优化器或 RNG。起止时间（含时区）：{tr[0]['started']} 至 {tr[-1]['ended']}。训练从第1轮到第20轮的目标均值 {fmt(tr[0]['train_loss'])}→{fmt(tr[-1]['train_loss'])}；250图 AP50–95 由 {pct(tr[0]['ap5095'])} 至 {pct(tr[-1]['ap5095'])}，最高为 E{best['epoch']} 的 {pct(best['ap5095'])}，不是单调曲线。",wline(group,'best'),wline(group,'last'),'## 逐轮主记录',table(['轮','Train loss','Val loss/图','AP50–95','AP50','更新 / 跳步','训练 / 验证 s','峰值分配 MiB'],[[row['epoch'],fmt(row['train_loss']),fmt(point(subset,group,row['epoch'])['val_loss_per_image']),pct(row['ap5095']),pct(row['ap50']),row['updates']+' / '+row['amp_skips'],fmt(row['train_s'],1)+' / '+fmt(row['val_s'],1),fmt(row['peak_alloc_mib'],1)] for row in tr]),'训练 loss 是按 B=4 计算的标量再取步均值；验证损失是冻结 val 批次 objective 之和除以总图数。二者尺度不同。box/cls/DFL 是已加权分量；C 的总损失是两路之和，不与单头总损失判优。','## 检测头与监督（E20）',table(['监督分支','Box','Cls','DFL','候选数','最终正样本','每GT','未匹配GT','冲突锚点'],[[arm,*[fmt(tr[-1][arm+'_'+item]) for item in ('box','cls','dfl')],*[integer(rec[item]) for item in ('candidates','positives')],fmt(rec['positives_per_gt']),rec['unmatched'],rec['contested_anchors']] for arm in branches for rec in [get(assign,group=group,branch=arm,epoch=20)]]),'准入时合计损失的非缩放梯度 L2：'+txt_norm(init[group]['gradient_norms'])+'。', 'E20 步均梯度 L2：'+txt_norm({key:tr[-1]['grad_'+key+'_mean'] for key in ('backbone','neck','head_main','head_o2o') if tr[-1].get('grad_'+key+'_mean')})+'。']
        if group=='Dual':parts+=['E20 独立分支反传（FP32 诊断、无 SGD）：O2M 路 '+txt_norm(isolated['Dual_one2many'])+'；O2O 路 '+txt_norm(isolated['Dual_one2one'])+'。',f"E20 的 O2O 赢者有 {pct(get(overlap,context='within_Dual',epoch=20)['candidate_rate'])} 在**相同 GT** 的 O2M 候选里；冲突消解后仍属于该 GT 的比例是 {pct(get(overlap,context='within_Dual',epoch=20)['final_rate'])}。这是训练归属，不是两头预测框的 IoU。"]
        else:parts+=['E20 在相同诊断批次反传（FP32，无 SGD）仍测得 '+txt_norm(isolated['O2O_sole'])+'。单头的 `head_main` 在本组指 O2O 分类/回归头。','B250 原始 FP16 评估有 24/250 图失效，那次 AP 无效。本轮从初值重训，E1 的修复协议 AP 与历史「诊断回放 + PSA32」相符；绝不将两次250步相加。']
        parts+=[f"**复核入口：** `runs/{group}/steps.csv` 含5,000步的顺序、分路损失、梯度与更新；`epochs.csv` 含20轮逐项均值。`assignments/` 保留每个 GT 与锚点在冲突前后的归属；`validation/` 包含每类 AP、有效框、预测 NPZ、逐批验证损失；`weights/` 有每轮权重和训练状态，`gpu.csv` 为原始监控。"]
        save(filename,parts)
    comparison=[]
    for epoch in range(1,21):
        old=point('historical250','O2M',epoch);patched=point(subset,'O2M',epoch) if epoch>=19 else None
        comparison.append([epoch,pct(old['ap5095']),pct(patched['ap5095']) if patched else '—',*[pct(point(subset,group,epoch)['ap5095']) for group in ('O2O','Dual')]])
    save('04_三组逐轮结果与250图比较.md',['# 04｜固定250张：曲线、重评与比较',f"同一 250 张、660 个普通 GT。**固定 E20 的 AP50–95 顺序：{rank(subset)}。** C 比 B 高 {pp(subset,'Dual','O2O')} 个百分点；A 比 C 高 {pp(subset,'O2M','Dual')} 个百分点。按置信度 >0.001，IoU=0.50，raw FP 包含大量低分预测，不与展示阈值 0.10 混用。",scoretable(subset),'![逐轮曲线](figures/01_逐轮曲线.png)','## 各轮 AP50–95',table(['Epoch','A 历史 FP16','A 当轮统一重评','B 统一协议','C 统一协议'],comparison),'“—”是没有统一重评，不是零。A 的 E19 是由最初20轮历史协议选出的 best；不声称它是统一协议下20轮的最优。B/C 的 best 在同一250张上选择，完整 Val 一律未使用这些 best。原样保存的历史 A 逐轮验证损失及精度探针缺失，不用插值回填。','## E20：逐类 AP',class_table(subset),'三组逐类分数、有效数、每轮结果和每类 TP/FP/FN 见 `evidence/class_metrics_subset250.csv`；`scope=historical250` 仅为 A 原指标，横向终点评分必须指定 `scope=subset250, epoch=20`。','## E20：目标尺度',size_table(subset),'按真实标注框在640输入上的面积分组，报告召回而不是尺寸 AP；计数和所有尺度的 TP 已与全体 TP 独立核对。完整验证集另见报告05。'])
    nested=source('full_val/nested_subset_check.csv');fullrows=source('full_val/results.csv');deltas=[]
    for name in GROUPS:
        row=point(full,name);sub=point(subset,name);deltas.append([name,pct(sub['ap5095']),pct(row['ap5095']),fmt((float(row['ap5095'])-float(sub['ap5095']))*100),pct(sub['ap50']),pct(row['ap50']),fmt((float(row['ap50'])-float(sub['ap50']))*100)])
    perclass=[]
    for name in NAMES:
        a,b,c=(get(classes,scope=full,group=group,epoch=20,name=name) for group in GROUPS)
        perclass.append((100*(float(c['ap5095'])-float(b['ap5095'])),name,a['gt']))
    def differences(items):return '、'.join(f'{name} {delta:+.2f} pp（GT {gt}）' for delta,name,gt in items)
    save('05_完整VOC2007验证结果.md',['# 05｜官方 VOC2007 Val：固定 E20 一次性验证',f"**全量 {integer(split['images'])} 张，普通 GT {integer(split['regular_gt'])}，difficult GT {integer(split['difficult_gt'])}。** 250 张子集在此集合内（逐图 JPEG 和标注一致）；与 1,000 训练图的 ID、JPEG 哈希重叠均为零。官方 `val.txt` SHA-256：`{split['official_list_sha256']}`。这不是 Test，也不是第二个独立测试集。",scoretable(full),'![两个验证范围](figures/02_两个验证范围.png)','## 子集能预测完整 Val 的排序吗？',table(['组别','250 AP50–95','全量 AP50–95','变化 pp','250 AP50','全量 AP50','变化 pp'],deltas),f"AP50–95 排序：250 图 {rank(subset)}；完整 Val {rank(full)}。AP50 排序分别为 {rank(subset,'ap50')} 和 {rank(full,'ap50')}。绝对值、各类样本比例及差距不要求相等。两个集合重叠，顺序一致不能当作独立重复，更不是跨种子显著性。",'## 逐类结果',class_table(full),'与 O2O-only 相比，Dual 完整 Val 的逐类 AP50–95 变化最大三项：'+differences(sorted(perclass,reverse=True)[:3])+'；最小三项：'+differences(sorted(perclass)[:3])+'。','## 尺度召回',size_table(full),'小/中/大按预处理后 GT 面积划分，阈值 32² / 96²；每组按同类真实 GT 去重。R50/R75 是两个 IoU 下的召回，不等价于 COCO 尺度 AP。','## 来源与一致性',table(['组','E20 last SHA-256','一次评估秒','嵌套子集逐框完全相同','嵌套子集 AP50–95 绝对差'],[[name,'`'+get(fullrows,group=name)['checkpoint_sha256']+'`',fmt(get(fullrows,group=name)['wall_s'],1),get(nested,group=name)['identical_image_prediction_arrays']+'/250',f"{float(get(nested,group=name)['absolute_AP5095_difference']):.9g}"] for name in GROUPS]),'嵌套检查仅从同一批已保存的完整预测中按 ID 抽取子集，不重复完整 Val 前向；直接子集与全量批次构成不同。`full_val/results.csv`、`class_metrics.csv`、`source_hashes.csv` 与三组 `epoch_20.npz/.json` 为原始证据。'])
    write_research(locals());write_handoff(locals())
    mismatches=save_differences();assert len(mismatches)==sum(250-int(row['identical_image_prediction_arrays']) for row in nested)
    assert {rec['image_id'] for rec in mismatches}=={'009780','009932'} and all((rec['subset_batch_size'],rec['full_batch_size'])==(2,4) for rec in mismatches)
    with (ROOT/'05_完整VOC2007验证结果.md').open('a',encoding='utf-8') as output:
        output.write('\n三组各有 248/250 张与直接子集评估逐框完全一致。差异集中在同两张 `009780`、`009932`：直接子集的最后一批大小为 2，它们在完整 Val 中的批大小为 4。最大子集 AP50–95 漂移仅 '+fmt(100*max(float(row['absolute_AP5095_difference']) for row in nested),6)+' 个百分点。这与混合精度对批形状的敏感性相符，但未做原因隔离实验。没有替换任何图的输出；见[逐图差异](full_val/nested_differences.csv)。\n')
    small=[get(scales,scope=full,group=name,size='small') for name in GROUPS]
    large=[get(scales,scope=full,group=name,size='large') for name in GROUPS]
    with (ROOT/'05_完整VOC2007验证结果.md').open('a',encoding='utf-8') as output:
        output.write('\n## 哪处没有随总排名一起变化？\n\n')
        output.write('完整 Val 的 '+small[0]['gt']+' 个小目标，在 IoU 0.50 下 A/B/C 分别只命中 '+' / '.join(row['tp50'] for row in small)+' 个；IoU 0.75 下三组都是零。不要用这组低基数命中数给小目标能力排稳定名次。')
        output.write('相比之下，'+large[0]['gt']+' 个大目标的 R75 分别为 '+' / '.join(pct(row['recall75']) for row in large)+'。Dual 在这个定位更严的单项高于 O2M，尽管总体 AP50–95 排名相反；R75 只数命中，不评价误报和置信度排序。\n')
    primary=select(cases,threshold='0.1');diagnostic=select(cases,threshold='0.01')
    with (ROOT/'06_三组受控比较与研究结论.md').open('a',encoding='utf-8') as output:
        output.write('\n## 读图示例：少框不等于更准\n\n')
        def boxset(items,case):return ' / '.join(get(items,case=case,output=key)['shown_boxes'] for key in ('O2M_pre','O2M_post','O2O','Dual'))
        output.write('按 A-NMS前 / A-NMS后 / B / C 顺序，普通图 002689 在主阈值 0.10 显示 '+boxset(primary,'ordinary')+' 个框；同一组数组在诊断阈值 0.01 显示 '+boxset(diagnostic,'ordinary')+'。看不到框并非模型没输出。\n\n')
        output.write('同类重叠图 006198 有 '+get(primary,case='duplicate_prone',output='Dual')['regular_gt']+' 个普通 GT，四路在 0.10 显示 '+boxset(primary,'duplicate_prone')+' 个框；密集图 004349 有 '+get(primary,case='dense',output='Dual')['regular_gt']+' 个普通 GT，只显示 '+boxset(primary,'dense')+'。NMS 的确减少候选，但单靠出框数不能区分正确去重和漏检；Top-K 也不保证每个实例只有一个框。判据是同类 GT 匹配、重复余量与整集 AP，而不是图面是否干净。\n')


def write_research(data):
    epochs,assign,cost,graphs,timing,mem,dups,low,shared,overlap,init,isolated,ta=(data[key] for key in ('epochs','assign','cost','graphs','timing','mem','dups','low','shared','overlap','init','isolated','ta'))
    point,rank,pp,subset,full,eager,single=(data[key] for key in ('point','rank','pp','subset','full','eager','single'))
    ar=[]
    for name in GROUPS:
        for arm in (('many','one') if name=='Dual' else (('many',) if name=='O2M' else ('one',))):
            rows=select(assign,group=name,branch=arm);row=rows[-1];ar.append([name,arm,row['gt'],integer(row['candidates']),integer(row['positives']),fmt(row['positives_per_gt']),row['unmatched'],row['contested_anchors'],row['final_outside_candidates'],row['max_per_gt']]);assert len(rows)==20
    joint=get(overlap,epoch=20,context='within_Dual');case=read_json(EVIDENCE/'real_conflicts.json')[0]
    agreement=select(source('evidence/dual_branch_decoded_anchor_agreement.csv'),threshold=0.1);agg={key:sum(int(row[key]) for row in agreement) for key in ('active_one','both_active','same_class','same_class_iou50')}
    dr=[]; pr=[]
    for split,title in ((subset,'250'),(full,'2510')):
        for group in GROUPS:
            row=point(split,group);pr.append([title,group,integer(row['raw_anchors']),integer(row['raw_eligible_box_class']),integer(row['nms_kernel_input']) if group=='O2M' else '—',integer(row['topk_before_conf']) if group!='O2M' else '—',integer(row['post_predictions']),row['nms_calls']]);
            for dtype in (('before_nms','final') if group=='O2M' else ('final',)):
                rec=get(dups,scope=split,group=group,stage=dtype);dr.append([title,group,dtype,*[integer(rec[key]) for key in ('predictions','matched','excess')],pct(rec['excess_over_matched']) if rec['excess_over_matched'] else '—'])
    latency_rows=[]
    for grp in (*GROUPS,single):
        t=get(timing,epoch=20,group=grp);g=get(graphs,graph=grp);m=get(mem,graph=grp);latency_rows.append([grp,integer(g['parameters']),fmt(g['profiler_supported_GFLOPs'],3),*[fmt(t[key+'_p50'])+' / '+fmt(t[key+'_p95']) for key in ('forward_ms','post_ms','total_ms')],fmt(m['peak_alloc_mib'],1)])
    te=get(timing,epoch=20,group=eager);ts=get(timing,epoch=20,group=single);tm=get(timing,epoch=20,group='O2M')
    def mult(name,other):return float(get(timing,epoch=20,group=name)['total_ms_p50'])/float(get(timing,epoch=20,group=other)['total_ms_p50'])
    case_names={'ordinary':'普通','dense':'密集','small':'小目标','occlusion_candidate':'遮挡候选','duplicate_prone':'同类重叠'}
    cr=[]
    for key,label in case_names.items():
        these=[row for row in data['cases'] if row['case']==key and float(row['threshold'])==.10];image=these[0]['image_id'];cr.append([f'[{label} · {image}](figures/predictions/{key}_{image}_primary.jpg)',these[0]['regular_gt'],*[get(these,output=group)['shown_boxes'] for group in ('O2M_pre','O2M_post','O2O','Dual')]])
    gradients=table(['组别 · E20','Backbone','Neck','主检测头','独立 O2O 头'],[[name,*[fmt(row['grad_'+part+'_mean'],2) if row.get('grad_'+part+'_mean') else '—' for part in ('backbone','neck','head_main','head_o2o')]] for name in GROUPS for row in [get(epochs,group=name,epoch=20)]])
    cost_table=table(['组别','训练合计 s','验证合计 s','训练 / 验证中位 s/轮','总经过 s','训练峰值 alloc / reserved MiB'],[[row['group'],fmt(row['train_s_sum'],1),fmt(row['val_s_sum'],1),fmt(row['train_s_median'],1)+' / '+fmt(row['val_s_median'],1),fmt(row['run_wall_s'],1),fmt(row['max_alloc_mib'],1)+' / '+fmt(row['max_reserved_mib'],1)] for row in cost]);
    case_gate=table(['固定图（点开四联图）','普通 GT','A NMS 前','A NMS 后','B Top-K','C O2O Top-K'],cr)
    eq=ta['shared_region_checkpoints_equal'];total=ta['shared_region_checkpoints_compared'];summ=[]
    for part in ('backbone','neck','head_main'):
        s=select(shared,region=part);summ.append(f"{part} {sum(row['equal']=='True' for row in s)}/20")
    e19=get(data['epochs'],group='O2M',epoch=19);bx=get(data['epochs'],group='O2O',epoch=20);cx=get(data['epochs'],group='Dual',epoch=20)
    rr=[('1｜训练是否完成？','A 为历史20/20，本轮不重训；B 自初值重训20/20；C 自初值首次正式训练20/20。每组5,000次实际 SGD。'),('2｜初始化与预算？','A/B 文件逐字节相同；C 的共享层与两头的对应初始张量相同，双头本身使整模型文件不同。20轮输入像素、顺序/翻转与学习率逐步配对；C 多头所以算力预算不是相同 FLOPs。'),('3｜AP 与 loss 如何变化？',f"曲线有回落，非逐轮单调。子集 AP50–95：A 历史 best E19 为 {pct(e19['ap5095'])}，统一重评 {pct(point(subset,'O2M',19)['ap5095'])}；B best E{bx['best_epoch']}、C best E{cx['best_epoch']}。两单头及 C 的分支损失见图01与报告02–04；不横比双头合计损失。"),('4｜B 能训练骨干吗？','能。准入、正式步梯度以及 E20 独立反向均非零；因此 B 较低 AP 不能归因于主干被 detach。'),('5｜C 保留 detach 吗？','是。同一批 E20 隔离反传时 O2O 损失在 backbone、neck、O2M 头均为零而其专属头非零；O2M 路方向相反。每个正式微批均核对头输入的 require_grad。'),('6｜每 GT 监督差异？','O2M top-10 提供约十个最终正锚点/GT，O2O top-1 约一个。冲突会减少候选，计入前景不保证有正的目标权重；逐 GT 直方图在分配 CSV。'),('7｜Dual 保留了什么优势？',f"结构上保留密集 O2M 共享监督；A/C 的主干、颈部与主头共 {eq}/{total} 个区域×轮次张量哈希一致。统一 AP 上 C 超过 B，但此组合实验不能单独识别改善的唯一原因。"),('8｜NMS 和 Top-K 如何处理重复？','A 显式按类抑制重叠框；B/C 排序不做重叠抑制。无 NMS 不等于无重复框；还须结合命中数与漏检看重复率，见下表。'),('9｜两套 Val 一致吗？',f"E20 的 AP50–95 排序在250/2510上分别为 {rank(subset)} / {rank(full)}，绝对值与逐类结果不完全相同。"),('10｜PSA 修复稳定吗？','在已执行的修复协议批次中，QK、缩放、softmax、V聚合以及解码均有限；验证器同时检查 C 的未计分分支。没有逐图切换数值精度。未证明在其他数据、训练长度或设备上总能稳定。'),('11｜成本与效率？',f"双头前向训练成本更高；E20 的实际双头 eager / 提取单头的端到端 P50 为 {fmt(te['total_ms_p50'])} / {fmt(ts['total_ms_p50'])} ms。A 的对应值 {fmt(tm['total_ms_p50'])} ms。完整分段与环境限制如下。"),('12｜哪些结果与论文主张相符？','双路训练、单路推理实际可行；C 在两套 Val 均优于只有 top-1 的 B，并实测推理可不调用 NMS。不能由此宣称双头在所有任务胜过 O2M；本实验甚至未在 E20 超过 A。亦未测原文的 COCO AP 与 TensorRT 延迟。'),('13｜哪些只是本实验观察？','AP 排名、差距、逐类与尺度分布、重复框率和延迟数值只对应 N 型、VOC 1,000 张训练、seed 2007、20轮、本机和所定义阈值；低分框多、部分小类别 GT 少。'),('14｜是否需要更多种子？','若要判断排序对随机性的稳健程度，需要经另行批准的多组配对种子，报告配对差及其离散性。本次只报告描述性差异，未追加种子、Epoch 或其他模型。')]
    save('06_三组受控比较与研究结论.md',['# 06｜检测精度、监督、重复框与延迟：受控比较',f"**主要观察：** 完整 Val 固定 E20，Dual 比 O2O-only 的 AP50–95 高 {pp(full,'Dual','O2O')} 个百分点，但低于 O2M + NMS {pp(full,'O2M','Dual')} 个百分点。更密集的共享监督与无 NMS 推理可以共存；在这套预算下，两者并没有同时胜过所有对照。",'## 14个研究问题',*['**'+question+'** '+answer for question,answer in rr],'## 监督与梯度：先看证据再解释',table(['组','分支','GT','冲突前候选','最终正锚点','每GT','未匹配GT','争用锚点','归属在原候选外','单GT最大'],ar),f"表为 E20，分母是本轮访问的2,508个非 difficult 训练 GT。官方 TAL 对争用锚点按所有 GT 的重叠量取最大值，不等于无条件保留分数最高候选。实例如 E{case['epoch']} {case['group']}/{case['branch']} 的图 {case['image_id']}，锚点 {case['anchor']}，候选 GT {case['candidate_gt']}，最终 GT {case['final_gt']}（图内零起编号）。候选交集和最终归属不能互换：E20 C 的 O2O 赢者落入同 GT 的 O2M 候选 {pct(joint['candidate_rate'])}，最终仍在同 GT 为 {pct(joint['final_rate'])}。",gradients,'同 E20 隔离 B：'+txt_norm(isolated['O2O_sole'])+'；C O2O：'+txt_norm(isolated['Dual_one2one'])+'；C O2M：'+txt_norm(isolated['Dual_one2many'])+'。训练轨迹中 A/C 同区域完全一致的轮数：'+'；'.join(summ)+'。该张量比对包括 BN 等状态，不能由参数形状或梯度范数相近来替代。',f"两头预测另按同一 anchor 对齐（非训练标签、非两独立检测集合最优匹配）：250 图、score>0.10，O2O 激活 {integer(agg['active_one'])}，双路均激活 {integer(agg['both_active'])}，其中最高分类相同 {integer(agg['same_class'])}，且解码框 IoU≥0.5 有 {integer(agg['same_class_iou50'])}。逐图两种阈值见 `evidence/dual_branch_decoded_anchor_agreement.csv`。",'## 后处理和重复框',table(['图数','组','解码位置','>0.001 框类对','送 NMS 内核','Top-K 阈值前','最终框','NMS 调用'],pr),'一图三层总计8,400位置，每位置20类。A 按类别展开、NMS 并截至300；B/C 先取300个位置、再选300个框类对，按共同置信度过滤。不同阶段的数量不能直接当作误报数。',table(['图数','组','阶段','>0.10 框数','同类真值命中框','多余命中框','多余/命中'],dr),'同类 regular GT 且 IoU≥0.50 的每个显示框归到其最高 IoU 实例；超过每实例第一个的计为多余，不包含普通误报。AP 仍使用0.001。子集阈值降至0.01的对照（输出数 / 命中数 / 多余数）：'+'；'.join(f"{name} {' / '.join(map(str,nums))}" for name,nums in low.items())+'。','## 同图、同尺度可视化',case_gate,'同图第一行有无 NMS 的差别可用 `case_counts.csv` 与 `case_predictions.npz` 核查；零框只表示未过展示阈值。附录诊断四联图用0.01，未替代0.10主图。遮挡为预先固定的候选例，不宣称 VOC 提供了遮挡掩码。','## 训练及推理成本',cost_table,'训练/验证计时含相应观测开销；原历史 A 的评估没有本轮额外的精度与逐批损失记录。系统负载、笔记本功耗/频率未锁定，故训练墙钟不能只归因于结构。原始 GPU 利用率与显存见各组 `gpu.csv` 和 `epochs.csv`。',table(['推理图','参数量','支持的 GFLOPs','前向 P50 / P95 ms','后处理 P50 / P95 ms','GPU端到端 P50 / P95 ms','单图路径峰值 MiB'],latency_rows),f"每个 epoch 的每个图均预热10次、计时同一组50张，轮换组顺序，CUDA 同步分段。端到端从已到 GPU 的图像张量至最终 GPU 检测框，不含读盘、预处理或 H2D；百分位分别计算，不能相加。E20 双头 eager/单分支 P50 比 {mult(eager,single):.2f}×，A/该单分支为 {mult('O2M',single):.2f}×。四图20轮逐图原始计时见 `latency_samples.csv`。提取图与 C 原图在每轮各50张的 O2O 解码和最终框最大绝对差均为0；计时是本机 eager，不是 TensorRT/ONNX 部署产品。CPU FP32 B1 profiler 仅统计支持运算的 FLOPs；显存为单图路径 CUDA 分配峰值，不是显卡全部占用。",'**原论文对照：** Wang 等，*YOLOv10: Real-Time End-to-End Object Detection*，arXiv:2405.14458（2024），双分配与端到端推理的主张；对应程序见固定 THU-MIG 提交。此处不援用论文硬件或延迟作为本机测量。'])


def write_handoff(data):
    point,rank,pp,subset,full=(data[key] for key in ('point','rank','pp','subset','full'));audit=data['audit'];shared=data['ta'];split=data['split']
    endpoints=table(['E20 last','250 AP50–95','250 AP50','2,510 AP50–95','2,510 AP50'],[[group,*[pct(point(scope,group)[key]) for scope in (subset,full) for key in ('ap5095','ap50')]] for group in GROUPS]);tim=data['timing']
    save('07_阶段执行摘要.md',['# 07｜执行摘要：三组 20 Epoch 最终对照','**已完成，等待研究讨论端审查；不自动启动后续实验。** A 是历史训练及本轮重评；B 自冻结初值重新训练；C 自双头冻结初值首次正式训练。B、C 各20轮、各5,000次实际 SGD，均无 AMP 跳步。',endpoints,f"两种验证范围的 AP50–95 排序分别为 **{rank(subset)}** 和 **{rank(full)}**。完整 Val 上 Dual 比 O2O-only 高 {pp(full,'Dual','O2O')} 个百分点，但比 O2M + NMS 低 {pp(full,'O2M','Dual')} 个百分点。2,510 张包含那250张，不能当作独立重复；这些结果只支持单种子的描述性比较。",'## 证据链',f"准入：历史三个目录清单、A19/A20 权重、B/C 初值及实际运算 dtype 均核验。A E19/E20 统一重评与已验收结果逐框一致。结尾从保存框独立核算 {audit['class_iou_cells']:,} 个 类×IoU AP 格，包括历史 A、统一子集与完整 Val；最大绝对误差 {audit['max_abs_diff']:.3g}。",f"机制：核对 {shared['training_microbatches']:,} 个正式微批与 {shared['branch_epochs']} 个分支轮的归属。B 的共享骨干有梯度，C 的 O2O 分支隔离反传时在共享层为零。完整 Val 的 {integer(split['regular_gt'])} 个普通 GT 与 difficult 处理可由逐类原始框对账。",'成本：E20 GPU 端到端 P50（单图、ms）：'+'；'.join(name+' '+fmt(get(tim,epoch=20,group=name)['total_ms_p50']) for name in (*GROUPS,'Dual_single_inference'))+'。双头 eager 不作为最终单路部署开销；分段 P50/P95、显存和测量边界见报告06。','## 使用建议','课程作业可引用上表的统一 E20 终点，并附图01、图02及至少一份同图可视化。不要把 best 与 last 混列、把 AP 百分数当小数、把 FP16 历史 A 曲线称为统一重训，或把结果写成 COCO / VOC Test / 统计显著。若要估计跨种子稳定性，请先由研究端审查本阶段证据，另行设计。'])


if __name__=='__main__':
    main()
