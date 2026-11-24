"""Observe tensors returned by the executed attention operations, not substitute outputs."""
from functools import wraps

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode


def dtype(tensor):
    return str(tensor.dtype).removeprefix('torch.')


class NativeOps(TorchDispatchMode):
    def __init__(self):
        self.events = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        if func in (torch.ops.aten.bmm.default,torch.ops.aten.matmul.default,torch.ops.aten.softmax.int,
                    torch.ops.aten._softmax.default,torch.ops.aten.mul.Tensor):
            self.events.append(dict(op=str(func), inputs=[dtype(item) for item in args if isinstance(item,torch.Tensor)],
                                    output=dtype(output),shape=list(output.shape)))
        return output


class PythonOps(TorchFunctionMode):
    def __init__(self):
        self.tensors = {}
        self.contexts = {}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        name = func.__name__
        if name == 'split' and isinstance(output, tuple) and len(output) == 3:
            self.tensors.update(zip(('q','k','v'),output))
        if name in ('matmul','__matmul__'):
            label = 'dot' if 'dot' not in self.tensors else 'aggregate'
        elif name in ('mul','__mul__') and 'dot' in self.tensors and 'softmax' not in self.tensors:
            label = 'scaled'
        elif name == 'softmax':
            label = 'softmax'
        else:
            return output
        self.tensors[label] = output
        self.contexts[label] = torch.is_autocast_enabled()
        return output


class Trace:
    def __init__(self, model, mode):
        self.mode = mode
        self.attention = model.model[10].attn
        self.psa = model.model[10]
        self.original = self.attention.forward
        self.batch = 0
        self.last = {}
        self.first_events = None
        self.handle = self.psa.register_forward_hook(self.finish_psa)

        @wraps(self.original)
        def observed(*args, **kwargs):
            python = PythonOps()
            if self.first_events is None:
                native = NativeOps()
                with native, python:
                    out = self.original(*args, **kwargs)
                self.first_events = dict(aten_calls=native.events, autocast_at_op=python.contexts)
            else:
                with python:
                    out = self.original(*args, **kwargs)
            self.last = {**python.tensors, 'attention_out':out}
            self.batch += 1
            return out

        self.attention.forward = observed

    def finish_psa(self, module, inputs, output):
        self.last['psa_out'] = output

    def release(self):
        self.handle.remove()
        del self.attention.forward

    def records(self, raw, auxiliary=None):
        with torch.cuda.amp.autocast(enabled=False):
            tensors = self.last
            assert set(tensors) == {'q','k','v','dot','scaled','softmax','aggregate','attention_out','psa_out'}, set(tensors)
            value_type = 'float32' if self.mode=='full32' else 'float16'
            core_type = 'float16' if self.mode=='original' else 'float32'
            assert all(dtype(tensors[key]) == value_type for key in ('q','k','v','attention_out','psa_out'))
            assert all(dtype(tensors[key]) == core_type for key in ('dot','scaled','aggregate'))
            assert dtype(tensors['softmax']) == 'float32'
            assert all(value == (self.mode=='original') for value in self.first_events['autocast_at_op'].values())
            kernels = [event for event in self.first_events['aten_calls'] if event['op'] in ('aten.bmm.default','aten.matmul.default')]
            assert len(kernels)==2 and all(event['inputs']==[core_type,core_type] and event['output']==core_type for event in kernels), kernels
            softmax = [event for event in self.first_events['aten_calls'] if event['op'] in ('aten.softmax.int','aten._softmax.default')]
            assert len(softmax)==1 and softmax[0]['output']=='float32'
            assert raw.shape[1:] == (24,8400)
            tensors = {**tensors, 'decoded_coords':raw[:,:4], 'decoded_scores':raw[:,4:]}
            if auxiliary is not None:
                tensors.update(aux_coords=auxiliary[:,:4],aux_scores=auxiliary[:,4:])
            columns, values, common = [], [], {}
            for key, tensor in tensors.items():
                common[key+'_dtype'] = dtype(tensor)
                flat = tensor.flatten(1)
                columns.extend((key+'_nan',key+'_inf'))
                values.extend((flat.isnan().sum(1),flat.isinf().sum(1)))
            reference_dot = tensors['q'].float().transpose(-2,-1) @ tensors['k'].float()
            columns.extend(('same_qk_fp32_absmax','bad_anchors','finite_score_max'))
            values.extend((reference_dot.abs().flatten(1).amax(1), (~raw.isfinite().all(1)).sum(1),
                           raw[:,4:].masked_fill(~raw[:,4:].isfinite(), -1).flatten(1).amax(1)))
            table = torch.stack([value.double() for value in values],dim=1).cpu().tolist()
            answers = []
            for line in table:
                rec = {**common,**{key:(float(val) if key in ('same_qk_fp32_absmax','finite_score_max') else int(val)) for key,val in zip(columns,line)}}
                for key in tensors:
                    rec[key+'_finite'] = not (rec[key+'_nan'] or rec[key+'_inf'])
                rec['all_finite'] = all(rec[key+'_finite'] for key in tensors)
                answers.append(rec)
            return answers
