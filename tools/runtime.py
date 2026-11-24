"""Build the fixed detector, isolated package paths and reproducible initial state."""
import hashlib
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from files import COMMIT, EVIDENCE, GROUPS, NAMES, PRISTINE, REPO, ROOT, SEED, digest, write_json

SOURCE = PRISTINE if os.getenv('AT3_PRISTINE') == '1' else REPO
sys.path.insert(0, str(SOURCE))
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'runtime' / 'yolo')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'runtime' / 'mpl')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['NUMEXPR_MAX_THREADS'] = '4'
for key in ('YOLO_CONFIG_DIR', 'MPLCONFIGDIR'):
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
import cv2
import numpy as np
import torch
import torchvision
import ultralytics
from ultralytics.nn.tasks import YOLOv10DetectionModel

assert Path(ultralytics.__file__).resolve().is_relative_to(SOURCE)
assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
YAML = SOURCE / 'ultralytics/cfg/models/v10/yolov10n.yaml'
HEAD = 'model.23.'
PRETRAINED = ROOT / 'weights' / 'yolov10n-coco.pt'
COMMON = dict(seed=SEED, epochs=20, imgsz=640, batch=4, workers=0, optimizer='SGD', momentum=.9,
              nesterov=True, weight_decay=.0005, lr_peak=.002, lr_start=.0001, lr_end=.0002,
              warmup_updates=100, schedule='linear warmup then cosine by planned minibatch',
              amp=True, amp_init_scale=32, amp_growth_interval=2000, accumulation=1, clip=None, ema=False,
              fliplr=.5, augment='paired horizontal flip only; centered letterbox 114; linear resize',
              box=7.5, cls=.5, dfl=1.5, conf_ap=.001, conf_visual=.10, nms_iou=.65,
              max_det=300, classes=NAMES, size_train=1000, size_val=250, device='cuda:0')


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def build(group):
    if group == 'Dual':
        model = YOLOv10DetectionModel(str(YAML), nc=20, verbose=False)
    else:
        from at3_variant import SingleV10Model
        model = SingleV10Model(str(YAML), group)
    model.args = SimpleNamespace(**COMMON)
    model.names = dict(enumerate(NAMES))
    return model


def load_initial(group, device='cuda'):
    model = build(group)
    model.load_state_dict(torch.load(ROOT / 'weights' / f'init_{group}.pt', map_location='cpu'), strict=True)
    return model.to(device)


def save_tensors(path, state):
    # torch 2.0.1's native Win32 zip writer fails on some Unicode absolute paths.
    temporary = path.with_suffix(path.suffix + '.part')
    with temporary.open('wb') as stream:
        torch.save(state, stream)
    temporary.replace(path)


def fingerprint(items):
    h = hashlib.sha256()
    for name, t in sorted(items):
        a = t.detach().cpu().contiguous().numpy()
        h.update((name + '|' + str(a.dtype) + '|' + repr(a.shape) + '\n').encode())
        h.update(a.tobytes())
    return h.hexdigest()


def region(name):
    layer = int(name.split('.')[1])
    if layer < 11:
        return 'backbone'
    if layer < 23:
        return 'neck'
    return 'head_o2o' if 'one2one_' in name else 'head_main'


def split_params(model):
    parts = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            parts.setdefault(region(name), []).append(p)
    return parts


def initialize():
    seed_all()
    base = build('Dual')
    assert digest(PRETRAINED) == '61b91ffc99b284792dca49bf40216945833cc2a515e1a742954e6e9327cfc19e'
    archive = torch.load(PRETRAINED, map_location='cpu')
    pretrained = (archive.get('ema') or archive['model']).float().state_dict()
    state = base.state_dict()
    loaded, retained, unused = [], [], []
    for key, value in state.items():
        if key.startswith(HEAD):
            retained.append(key)
        else:
            assert key in pretrained and pretrained[key].shape == value.shape, key
            value.copy_(pretrained[key])
            loaded.append(key)
    unused = [k for k in pretrained if k not in loaded]
    write_json(EVIDENCE/'weight_transfer.json', dict(source_url='https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt',
               sha256=digest(PRETRAINED), loaded_shared=loaded, initialized_20class_head=retained, unused_source=unused))
    output = {}
    for group in GROUPS:
        model = build(group)
        for key in model.state_dict():
            assert key in state
        model.load_state_dict({k: state[k] for k in model.state_dict()}, strict=True)
        own = model.state_dict()
        file = ROOT / 'weights' / f'init_{group}.pt'
        save_tensors(file, own)
        (EVIDENCE/f'structure_{group}.txt').write_text(str(model), encoding='utf-8')
        parts = {r: fingerprint((k,v) for k,v in own.items() if region(k)==r) for r in ('backbone','neck')}
        h = model.model[-1].state_dict()
        def branch(prefix):
            # Canonical local names omit branch aliases; shared fixed DFL is identical.
            return fingerprint((k.removeprefix(prefix),v) for k,v in h.items() if k.startswith((prefix+'cv2.',prefix+'cv3.')))
        parts['main_head'] = branch('')
        if group == 'Dual':
            parts['one2one_head'] = branch('one2one_')
            assert parts['main_head'] == parts['one2one_head']
        parts['dfl'] = fingerprint((k,v) for k,v in h.items() if k.startswith('dfl.'))
        output[group] = dict(file_sha256=digest(file), tensor_sha256=fingerprint(own.items()), parts=parts,
                             parameters=sum(p.numel() for p in model.parameters()),
                             trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                             trainable_by_region={k:sum(p.numel() for p in v) for k,v in split_params(model).items()},
                             heads=2 if group=='Dual' else 1)
        write_json(ROOT/'configs'/f'{group}.json', dict(**COMMON, group=group, input_weights=file.name,
            assignment_topk=([10,1] if group=='Dual' else [10 if group=='O2M' else 1]),
            alpha=.5,beta=6,detach_o2o=group=='Dual',post='class-aware NMS' if group=='O2M' else 'official v10 top-k'))
    for key in ('backbone', 'neck', 'main_head', 'dfl'):
        assert len({x['parts'][key] for x in output.values()}) == 1
    write_json(EVIDENCE/'initialization.json',output)
    info = dict(python=sys.version, executable=sys.executable, torch=torch.__version__, torchvision=torchvision.__version__,
                ultralytics=ultralytics.__version__, module=str(ultralytics.__file__), commit=COMMIT, gpu=torch.cuda.get_device_name(),
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), numpy=np.__version__, opencv=cv2.__version__,
                driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version,memory.total','--format=csv,noheader'],text=True).strip())
    write_json(EVIDENCE/'environment.json', info)
    print({g: (o['parameters'], o['trainable']) for g,o in output.items()})


if __name__ == '__main__':
    initialize()
