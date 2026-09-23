"""LPRNet-style fully convolutional CTC network, versioned with its checkpoints.

RGB 48x192 -> 48 temporal steps. No recurrent layers or character boxes.
"""

import numpy as np
import torch
from PIL import Image
from torch import nn

from .alphabet import TOKENS
from .enhancement import prepare_image

ARCHITECTURE = "lprnet-fa-v1"
PREPROCESS = {
    "width": 192,
    "height": 48,
    "color": "RGB",
    "padding": 127,
    "normalization": "x/127.5-1",
    "resize": "bilinear",
    "align": "center",
}


def preprocess(image: Image.Image, *, enhancement="none") -> torch.Tensor:
    image = prepare_image(image, enhancement)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def conv(cin, cout, kernel=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SmallBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.layers = nn.Sequential(
            conv(cin, cout // 4, 1, padding=0),
            conv(cout // 4, cout // 4, (3, 1), padding=(1, 0)),
            conv(cout // 4, cout // 4, (1, 3), padding=(0, 1)),
            conv(cout // 4, cout, 1, padding=0),
        )

    def forward(self, x):
        return self.layers(x)


class LPRNet(nn.Module):
    def __init__(self, classes=None):
        super().__init__()
        self.features = nn.Sequential(
            conv(3, 64, stride=2),
            SmallBlock(64, 128),
            nn.MaxPool2d(2, 2),
            SmallBlock(128, 256),
            SmallBlock(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Dropout(0.1),
            conv(256, 256, (1, 5), padding=(0, 2)),
        )
        self.classifier = nn.Conv1d(256, classes or len(TOKENS), 1)

    def forward(self, x):
        return self.classifier(self.features(x).mean(dim=2)).transpose(1, 2)
