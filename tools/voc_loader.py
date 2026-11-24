"""Paired augmentation plans and invertible, square letterboxing for VOC."""
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from files import DATA, SEED, read_json


def epoch_plan(n, epoch):
    rng = np.random.default_rng(SEED + 1000 + epoch)
    return [(int(i), bool(f)) for i,f in zip(rng.permutation(n), rng.random(n) < .5)]


class FrozenVOC(Dataset):
    def __init__(self, split):
        self.split = split
        self.records = read_json(DATA/'manifests'/f'{split}.json')
        assert [r['id'] for r in self.records] == (DATA/'manifests'/f'{split}.txt').read_text().split()
        self.plan = [(i, False) for i in range(len(self.records))]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, slot):
        index, flip = self.plan[slot]
        r = self.records[index]
        path = DATA/'JPEGImages'/self.split/(r['id']+'.jpg')
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        h, w = image.shape[:2]
        assert (w, h) == (r['width'], r['height'])
        scale = 640/max(h,w)
        neww, newh = round(w*scale), round(h*scale)
        left, top = (640-neww)//2, (640-newh)//2
        rgb = cv2.cvtColor(cv2.resize(image,(neww,newh),interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        canvas = np.full((640,640,3),114,np.uint8)
        canvas[top:top+newh,left:left+neww] = rgb
        regular = [o for o in r['objects'] if not o['difficult']]
        labels = torch.tensor([o['cls'] for o in regular], dtype=torch.float32).reshape(-1,1)
        xyxy = np.asarray([o['xyxy'] for o in regular], dtype=np.float32).reshape(-1,4)
        gx, gy = neww/w, newh/h
        xyxy[:,[0,2]] = xyxy[:,[0,2]]*gx+left
        xyxy[:,[1,3]] = xyxy[:,[1,3]]*gy+top
        xywh = np.column_stack(((xyxy[:,0]+xyxy[:,2])/2,(xyxy[:,1]+xyxy[:,3])/2,
                                xyxy[:,2]-xyxy[:,0],xyxy[:,3]-xyxy[:,1])) / 640
        if flip:
            canvas = canvas[:,::-1]
            xywh[:,0] = 1-xywh[:,0]
        return (torch.from_numpy(np.ascontiguousarray(canvas.transpose(2,0,1))), labels,
                torch.from_numpy(xywh), r, (gx,gy,left,top), flip)


def collate(items):
    imgs, classes, boxes, records, geom, flips = zip(*items)
    return dict(img=torch.stack(imgs), cls=torch.cat(classes), bboxes=torch.cat(boxes),
                batch_idx=torch.cat([torch.full((len(c),),i,dtype=torch.float32) for i,c in enumerate(classes)]),
                records=records, geom=geom, flips=flips)


def loader(ds, batch=4):
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate)


def original_boxes(rows, geom, record):
    result = rows.detach().float().cpu().clone()
    gx, gy, left, top = geom
    result[:,[0,2]] = ((result[:,[0,2]]-left)/gx).clamp(0, record['width'])
    result[:,[1,3]] = ((result[:,[1,3]]-top)/gy).clamp(0, record['height'])
    return result
