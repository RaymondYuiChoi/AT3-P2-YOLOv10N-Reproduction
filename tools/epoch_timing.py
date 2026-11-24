"""Benchmark four inference graphs on identical inputs after all training, without extra updates."""
import csv
import runtime
import torch
from files import EVIDENCE,GROUPS,ROOT,digest,read_json,write_csv,write_json
from recheck import benchmark,deploy_view,restore
from runtime import save_tensors,seed_all
from voc_loader import FrozenVOC


def costs(networks):
    models=dict(networks,Dual_single_inference=deploy_view(networks['Dual']))
    path=ROOT/'runs/Dual/weights/epoch_20_o2o_deploy.pt'
    save_tensors(path,dict(source_sha256=digest(ROOT/'runs/Dual/weights/epoch_20.pt'),group='Dual_single_inference',epoch=20,
                          model={key:value.cpu() for key,value in models['Dual_single_inference'].state_dict().items()}))
    rows,operators=[],[]
    for name,model in models.items():
        model=model.cpu().float().eval()
        image=torch.zeros(1,3,640,640)
        with torch.inference_mode():
            model(image)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True,with_flops=True) as profiler:
                model(image)
        counted=[dict(graph=name,operator=entry.key,invocations=entry.count,flops=entry.flops) for entry in profiler.key_averages() if entry.flops]
        operators.extend(counted)
        flops=sum(entry['flops'] for entry in counted)
        assert flops>1e9 and any('conv2d' in item['operator'] for item in counted)
        rows.append(dict(graph=name,input='1x3x640x640',parameters=sum(param.numel() for param in model.parameters()),
                    trainable=sum(param.numel() for param in model.parameters() if param.requires_grad),
                    profiler_supported_GFLOPs=flops/1e9))
    write_csv(EVIDENCE/'graph_costs.csv',rows)
    write_csv(EVIDENCE/'flops_by_operator.csv',operators)


def main():
    seed_all()
    rows,samples=[],[]
    dataset=FrozenVOC('val')
    for epoch in range(1,21):
        folder=EVIDENCE/'timing'/f'epoch_{epoch:02}'
        networks={group:restore(group,f'epoch_{epoch:02}')[0] for group in GROUPS}
        benchmark(networks,dataset,folder)
        info=read_json(folder/'inference_timing.json')
        assert info['settings']['copied_dual_max_abs']==0
        rows.extend(dict(epoch=epoch,group=group,**record) for group,record in info['groups'].items())
        with (folder/'inference_timings.csv').open(encoding='utf-8-sig') as raw:
            samples.extend(dict(epoch=epoch,**record) for record in csv.DictReader(raw))
        if epoch==20:
            costs(networks)
        print('timing',epoch,flush=True)
    write_csv(EVIDENCE/'latency_epochs.csv',rows)
    write_csv(EVIDENCE/'latency_samples.csv',samples)


if __name__=='__main__':
    main()
