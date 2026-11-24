import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const lib=process.env.AT3_ARTIFACT_MODULE;
const {Workbook,SpreadsheetFile}=await import(lib?pathToFileURL(lib).href:'@oai/artifact-tool');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
async function csv(relative){
  const text=(await fs.readFile(path.join(root,relative),'utf8')).replace(/^\uFEFF/,'').trimEnd();
  const lines=text.split(/\r?\n/),keys=lines.shift().split(',');
  return lines.map(line=>{const vals=line.split(',');if(line.includes('"')||vals.length!==keys.length)throw new Error(relative);return Object.fromEntries(keys.map((key,index)=>[key,vals[index]]));});
}
const groups=['O2M','O2O','Dual'];
const scope={historical250:'250 · A历史FP16',subset250:'250 · 统一评估',official_val2510:'2,510 · 完整Val'};
const [epochs,valid,classes,fullclasses,cost,time,flops,mem,assigned,sizes]=await Promise.all(['evidence/epochs_all.csv','evidence/validation_all.csv','evidence/class_metrics_subset250.csv','full_val/class_metrics.csv','evidence/training_costs.csv','evidence/latency_epochs.csv','evidence/graph_costs.csv','evidence/inference_memory.csv','evidence/assignment_verified.csv','evidence/size_recall.csv'].map(csv));
const wb=Workbook.create(); const label=index=>String.fromCharCode(65+index);
const num=value=>value===''||value==null?null:Number(value);
const values=(row,...keys)=>keys.map(key=>num(row[key]));
const main=wb.worksheets.add('结论速览');
function band(sheet,row,end,text,fill,color,size){sheet.mergeCells(`A${row}:${end}${row}`);const target=sheet.getRange(`A${row}:${end}${row}`);sheet.getRange(`A${row}`).values=[[text]];target.format.fill=fill;target.format.font={name:'Microsoft YaHei',size,bold:size>=12,color};target.format.rowHeight=size>=16?37:26;}
function table(sheet,title,note,headers,data,widths,pct=[],decimal=[]){
  const end=label(headers.length-1),last=4+data.length;sheet.showGridLines=false;
  const entire=sheet.getRange(`A1:${end}${last}`);entire.format.font={name:'Microsoft YaHei',size:10,color:'#163442'};entire.format.rowHeight=21;entire.format.verticalAlignment='center';
  widths.forEach((width,index)=>{sheet.getRange(`${label(index)}:${label(index)}`).format.columnWidth=width;});
  band(sheet,1,end,title,'#183845','#FFFFFF',16);band(sheet,2,end,note,'#E4F0ED','#385963',10);
  const head=sheet.getRange(`A4:${end}4`);head.values=[headers];head.format.fill='#126F70';head.format.font={name:'Microsoft YaHei',bold:true,color:'#FFFFFF',size:10};head.format.rowHeight=28;
  sheet.getRange(`A5:${end}${last}`).values=data;
  for(let row=6;row<=last;row+=2)sheet.getRange(`A${row}:${end}${row}`).format.fill='#F0F5F4';
  for(const key of pct)sheet.getRange(`${key}5:${key}${last}`).setNumberFormat('0.000%');
  for(const key of decimal)sheet.getRange(`${key}5:${key}${last}`).setNumberFormat('0.000');
  sheet.freezePanes.freezeRows(4);return sheet;
}
const endpoints=valid.filter(row=>row.epoch==='20'&&row.scope!=='historical250').sort((left,right)=>Object.keys(scope).indexOf(left.scope)-Object.keys(scope).indexOf(right.scope));
if(epochs.length!==60||endpoints.length!==6||fullclasses.length!==60)throw new Error('Incomplete sources');
table(main,'AT3-P2   /   最终受控实验','固定 E20，种子 2007。AP 按百分数显示。官方 Val 含 250 张子集；不是 Test，不在完整 Val 选模。',['组别','训练来源','完成轮数','SGD 更新','250 · AP50–95','250 · AP50','2,510 · AP50–95','2,510 · AP50'],groups.map((group,index)=>[group,index===0?'历史训练 · 本轮重评':'本轮冻结初值',20,5000,null,null,null,null]),[13,22,14,15,20,19,22,19],['E','F','G','H']);
const endpointSheet=wb.worksheets.add('终点评分');
table(endpointSheet,'两套验证集  /  E20 同协议','FP / FN 按置信度 >0.001、IoU=0.50。被 difficult 规则忽略者不计 TP、FP。',['组别','数据范围','图数','普通GT','AP50–95','AP50','TP50','FP50','FN50','忽略框50','输出框数'],endpoints.map(row=>[row.group,scope[row.scope],...values(row,'images','regular_gt','ap5095','ap50','tp50','fp50','fn50','ignored50','post_predictions')]),[15,23,12,12,17,17,13,13,13,15,16],['E','F']);
for (let index=0;index<3;index++){
  const row=5+index;const subset=5+endpoints.findIndex(item=>item.group===groups[index]&&item.scope==='subset250');const full=5+endpoints.findIndex(item=>item.group===groups[index]&&item.scope==='official_val2510');
  main.getRange(`E${row}:H${row}`).formulas=[[[subset,'E'],[subset,'F'],[full,'E'],[full,'F']].map(([src,col])=>`='终点评分'!${col}${src}`)];
}
band(main,9,'H','如何读：AP 是 20 类宏平均；TP/FP/FN 是计数，两种汇总不等价。','#E4F0ED','#264B4D',12);
band(main,10,'H','A 的前 20 个曲线点保留历史 FP16 口径；E19/E20 只在「逐轮验证」另列统一修复重评。','#F3F6F5','#385963',10);
band(main,11,'H','B/C 未接续诊断权重。Dual 的总 loss 为两支之和，不宜和单头直接比高低。','#F3F6F5','#385963',10);
band(main,12,'H','四图延迟区分前向、后处理和 GPU 端到端。Dual 单分支不是 TensorRT 或 ONNX 测时。','#F3F6F5','#385963',10);
const add=(name,...args)=>table(wb.worksheets.add(name),...args);
add('逐轮训练','训练  /  60 个组轮','A 来自历史训练。分支分项为已加权 loss；同组比较趋势，不将 Dual 总损失直接与单头比较。',['组别','轮','总损失','O2M Box','O2M Cls','O2M DFL','O2O Box','O2O Cls','O2O DFL','更新','跳步','训练秒','验证秒','峰值 MiB','骨干梯度','颈部梯度','主头梯度','独立O2O头'],epochs.map(row=>[row.group,...values(row,'epoch','train_loss','many_box','many_cls','many_dfl','one_box','one_cls','one_dfl','updates','amp_skips','train_s','val_s','peak_alloc_mib','grad_backbone_mean','grad_neck_mean','grad_head_main_mean','grad_head_o2o_mean')]),[13,8,13,13,13,13,13,13,13,10,10,13,13,15,14,14,14,15],[],['C','D','E','F','G','H','I','L','M','N','O','P','Q','R']);
add('逐轮验证','固定 250 图  /  每轮与 A 重评','历史 A 的验证损失及逐图有限性无对应原始观测，留空而非填零。合资格候选按 anchor × class；不同于 NMS 内核实际入框数。',['组别','评估口径','轮','AP50–95','AP50','验证损失/图','有效框50','TP50','FP50','FN50','候选框类对','解码锚点','最终框数','有限图数'],valid.filter(row=>row.scope!=='official_val2510').map(row=>[row.group,scope[row.scope],...values(row,'epoch','ap5095','ap50','val_loss_per_image','effective50','tp50','fp50','fn50','raw_eligible_box_class','raw_anchors','post_predictions','finite_images')]),[14,24,9,16,15,17,14,12,12,12,18,16,16,15],['D','E'],['F']);
const byClass=[...classes.filter(row=>row.epoch==='20'&&row.scope==='subset250'),...fullclasses];
add('终点逐类','两套验证集  /  20 类','每行一个类别和组；各 IoU 的 AP、有效数，以及所有轮次的同口径明细存于原始 CSV。',['范围','组别','类别','GT','AP50–95','AP50','有效框50','TP50','FP50','FN50'],byClass.map(row=>[scope[row.scope],row.group,row.name,...values(row,'gt','ap5095','ap50','effective50','tp50','fp50','fn50')]),[23,14,19,12,16,16,16,13,13,13],['E','F']);
add('标签分配','监督机制  /  80 个分支轮','候选与最终正样本分开；有效权重和未匹配 GT 均按冻结的原始归属轨迹复算。',['组别','分支','轮','GT','候选','最终正样本','正样本/GT','未匹配GT','冲突锚点','正权重数'],assigned.map(row=>[row.group,row.branch,...values(row,'epoch','gt','candidates','positives','positives_per_gt','unmatched','contested_anchors','positive_weight')]),[14,14,9,12,16,17,17,16,16,16],[],['G']);
add('训练成本','训练成本  /  三组运行日志','训练、验证时长为20轮之和；E2E运行含其他开销。A 是历史运行；本轮验证多了有限性与损失监测。',['组别','更新','跳步','训练合计 s','验证合计 s','运行合计 s','训练中位 s','验证中位 s','最大分配 MiB','最大预留 MiB'],cost.map(row=>[row.group,...values(row,'updates','skips','train_s_sum','val_s_sum','run_wall_s','train_s_median','val_s_median','max_alloc_mib','max_reserved_mib')]),[14,12,10,17,17,17,17,17,20,20],[],['D','E','F','G','H','I','J']);
add('推理成本','E20 · B1  /  四条计算路径','10 图预热，50 张同图轮换；单位 ms。GPU 张量到输出框，不含图片IO和H2D；峰值显存另按单图独占测量。',['图','参数','GFLOPs¹','前向 P50','前向 P95','后处理 P50','后处理 P95','合计 P50','合计 P95','最大分配 MiB'],time.filter(row=>row.epoch==='20').map(row=>{const cost=flops.find(item=>item.graph===row.group),vram=mem.find(item=>item.graph===row.group);return [row.group,...values(cost,'parameters','profiler_supported_GFLOPs'),...values(row,'forward_ms_p50','forward_ms_p95','post_ms_p50','post_ms_p95','total_ms_p50','total_ms_p95'),num(vram.peak_alloc_mib)];}),[30,17,14,15,15,16,16,15,15,20],[],['C','D','E','F','G','H','I','J']);
add('尺度召回','按 GT 尺度  /  固定 E20','640 letterbox 后面积，小 <32²、中 <96²、大 ≥96²。按尺寸分的普通 GT 召回，不是 COCO APs/APm/APl。',['范围','组别','尺度','GT','TP50','召回50','TP75','召回75'],sizes.map(row=>[scope[row.scope],row.group,row.size,...values(row,'gt','tp50','recall50','tp75','recall75')]),[24,15,15,12,13,16,13,16],['F','H']);
wb.recalculate();
if(Math.abs(main.getRange('G5').values[0][0]-num(endpoints.find(item=>item.scope==='official_val2510'&&item.group==='O2M').ap5095))>1e-10)throw new Error('Formula reconciliation');
for(const [sheetName,range,name] of [['结论速览','A1:H12','summary'],['推理成本','A1:J8','latency']]){const image=await wb.render({sheetName,range,scale:1.12,format:'png'});await fs.writeFile(path.join(root,'figures',`workbook_${name}.png`),new Uint8Array(await image.arrayBuffer()));}
const exportFile=await SpreadsheetFile.exportXlsx(wb);await exportFile.save(path.join(root,'20Epoch三组实验总表.xlsx'));
