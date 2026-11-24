"""VOC pilot single-head derivatives of THU-MIG YOLOv10 v1.1."""
from torch import nn

from ultralytics.nn.modules.head import Detect, v10Detect
from ultralytics.nn.tasks import YOLOv10DetectionModel
from ultralytics.utils.loss import v8DetectionLoss


class SingleV10Head(v10Detect):
    """Keep one original v10 regression/classification pair and its DFL decoder."""

    def __init__(self, source):
        nn.Module.__init__(self)
        for name in ('nc', 'nl', 'reg_max', 'no', 'stride', 'i', 'f', 'type'):
            setattr(self, name, getattr(source, name))
        self.cv2, self.cv3, self.dfl = source.cv2, source.cv3, source.dfl
        self.np = sum(p.numel() for p in self.parameters())

    def forward(self, features):
        # Both single heads must receive connected backbone/neck features.
        return Detect.forward(self, features)


class SingleV10Model(YOLOv10DetectionModel):
    def __init__(self, cfg, group, nc=20):
        if group not in ('O2M', 'O2O'):
            raise ValueError(group)
        super().__init__(cfg, nc=nc, verbose=False)
        self.model[-1] = SingleV10Head(self.model[-1])
        self.topk = 10 if group == 'O2M' else 1

    def init_criterion(self):
        return v8DetectionLoss(self, tal_topk=self.topk)
