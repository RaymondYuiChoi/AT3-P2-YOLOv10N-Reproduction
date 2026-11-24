"""Read-only TAL and gradient observations; optimizer.step counts actual AMP updates."""
import numpy as np
import torch


class CountingSGD(torch.optim.SGD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.updates = 0

    def step(self, closure=None):
        result = super().step(closure)
        self.updates += 1
        return result


def make_optimizer(model, cfg):
    decay, plain = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.ndim > 1 else plain).append(p)
    return CountingSGD([{'params':decay,'weight_decay':cfg['weight_decay']},
                        {'params':plain,'weight_decay':0.}], lr=cfg['lr_start'],
                       momentum=cfg['momentum'],nesterov=cfg['nesterov'])


def grad_norms(parts):
    result = {}
    for name, params in parts.items():
        grads = [p.grad.detach() for p in params if p.grad is not None]
        result[name] = torch.stack(torch._foreach_norm(grads,2)).norm().item() if grads else 0.
    return result


def assigners(group, loss):
    if group == 'Dual':
        return {'many':loss.one2many.assigner, 'one':loss.one2one.assigner}
    return {'many' if group=='O2M' else 'one':loss.assigner}


class Assignments:
    def __init__(self, group, criterion):
        self.results, self.maps, self.handles = {}, {}, []
        for branch, assigner in assigners(group, criterion).items():
            self.handles.append(assigner.register_forward_hook(self.hook(branch)))

    def hook(self, branch):
        def observe(module, args, output):
            mask = args[-1].squeeze(-1).bool()
            fg, idx = output[3].bool(), output[4].long()
            b, width = mask.shape
            batch = torch.arange(b,device=idx.device)[:,None]
            flat = (idx + batch * width)[fg]
            counts = torch.bincount(flat.cpu(),minlength=b*width).reshape(b,width)[mask.cpu()]
            count_list = counts.tolist()
            self.results[branch] = dict(gt=len(count_list), positives=sum(count_list), unmatched=count_list.count(0),
                                        max_per_gt=max(count_list,default=0), positive_weight=int((output[2].sum(-1)>0).sum()),
                                        counts=count_list)
            self.maps[branch] = (fg.detach(),idx.detach())
        return observe

    def common_winners(self):
        many, mi = self.maps['many']
        one, oi = self.maps['one']
        return int((many & one & (mi==oi)).sum())

    def close(self):
        for handle in self.handles:
            handle.remove()


class TrainingAssignments(Assignments):
    def __init__(self, group, criterion, model):
        super().__init__(group, criterion)
        self.pre, self.post, self.original, self.connections = {}, {}, {}, []
        for branch, assigner in assigners(group, criterion).items():
            self.original[branch] = (assigner, assigner.get_pos_mask)
            def read_candidates(*args, branch=branch, original=assigner.get_pos_mask):
                result = original(*args)
                self.pre[branch] = result[0].nonzero().to(torch.int16).cpu().numpy()
                return result
            assigner.get_pos_mask = read_candidates
            self.handles.append(assigner.register_forward_hook(self.final_capture(branch)))
        head = model.model[-1]
        for name in (('cv2','cv3','one2one_cv2','one2one_cv3') if group=='Dual' else ('cv2','cv3')):
            for layer in getattr(head,name):
                def check_connection(module, args, name=name):
                    if module.training:
                        actual = args[0].requires_grad
                        assert actual == (not name.startswith('one2one_')), 'unexpected head input gradient: '+name
                        self.connections.append((name,actual))
                self.handles.append(layer.register_forward_pre_hook(check_connection))

    def final_capture(self, branch):
        def capture(module, args, output):
            slots, anchors = output[3].bool().nonzero(as_tuple=True)
            owner = output[4][slots,anchors].long()
            weighted = (output[2][slots,anchors].sum(-1)>0).long()
            self.post[branch] = torch.stack((slots,owner,anchors,weighted),1).to(torch.int16).cpu().numpy()
        return capture

    def trace(self, branch, step, accumulator):
        pre, post = self.pre[branch], self.post[branch]
        for role, data in [('pre',pre),('post',post)]:
            accumulator.setdefault(branch+'_'+role,[]).append(np.column_stack((np.full(len(data),step,np.int16),data)))
        keys = pre[:,0].astype(np.int32)*8400 + pre[:,2]
        _, counts = np.unique(keys,return_counts=True)
        assert len(post)==self.results[branch]['positives'] and post[:,3].sum()==self.results[branch]['positive_weight']
        return dict(candidates=len(pre),contested_anchors=int((counts>1).sum()))

    def common_candidates(self):
        def keys(data):
            return ((data[:,0].astype(np.int64)*65536 + data[:,1])*8400 + data[:,2])
        return int(np.isin(keys(self.post['one']),keys(self.pre['many'])).sum())

    def close(self):
        for assigner, original in self.original.values():
            assigner.get_pos_mask = original
        super().close()
