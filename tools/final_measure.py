"""E20-only reload, paired qualitative panels and isolated inference VRAM. No formal SGD or full-Val pass."""
import gc
import runtime
import torch
from files import EVIDENCE,GROUPS,ROOT,read_json,write_csv
from epoch_timing import main as time_all
from evaluation import decoded,postprocess
from recheck import cases,deploy_view,independent,restore
from runtime import seed_all
from voc_loader import FrozenVOC


@torch.inference_mode()
def memory(dataset):
    rows=[]
    for name in (*GROUPS,'Dual_single_inference'):
        model=restore('Dual' if name=='Dual_single_inference' else name,'last')[0]
        if name=='Dual_single_inference':
            single=deploy_view(model)
            del model
            model=single
            del single
        gc.collect(); torch.cuda.empty_cache()
        for slot in range(60):
            image=dataset[slot if slot<10 else slot-10][0].unsqueeze(0).cuda().float()/255
            if slot==10:
                torch.cuda.synchronize(); torch.cuda.empty_cache()
                base=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
            with torch.cuda.amp.autocast():
                prediction=model(image)
            boxes=postprocess(name,prediction)
            assert torch.isfinite(decoded(prediction)).all() and torch.isfinite(boxes[0]).all()
            if name=='Dual':
                assert torch.isfinite(decoded(prediction['one2many'])).all()
            del prediction,boxes,image
        torch.cuda.synchronize()
        allocated=torch.cuda.max_memory_allocated()
        rows.append(dict(graph=name,images=50,batch=1,baseline_alloc_mib=base/2**20,peak_alloc_mib=allocated/2**20,
                         peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,increment_alloc_mib=(allocated-base)/2**20))
        del model
        gc.collect(); torch.cuda.empty_cache()
    write_csv(EVIDENCE/'inference_memory.csv',rows)


if __name__=='__main__':
    assert all(read_json(ROOT/'runs'/group/'status.json')['epochs']==20 for group in GROUPS)
    assert (ROOT/'full_val/results.csv').is_file()
    seed_all()
    dataset=FrozenVOC('val')
    independent(dataset,('O2O','Dual'))
    cases({name:restore(name,'last')[0] for name in GROUPS},dataset)
    gc.collect(); torch.cuda.empty_cache()
    memory(dataset)
    time_all()
