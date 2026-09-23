import json
import random
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from app.lpr.alphabet import encode
from app.lpr.network import preprocess


def augment(image):
    if random.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.65, 1.35))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.15, 1.1)))
    if random.random() < 0.35:
        w, h = image.size
        source = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        offsets = np.float32(
            [[random.uniform(-0.035, 0.035) * w, random.uniform(-0.08, 0.08) * h] for _ in range(4)]
        )
        array = cv2.warpPerspective(
            np.asarray(image),
            cv2.getPerspectiveTransform(source, source + offsets),
            (w, h),
            borderMode=cv2.BORDER_REPLICATE,
        )
        image = Image.fromarray(array)
    if random.random() < 0.25:
        w, h = image.size
        dx, dy = int(w * 0.025), int(h * 0.04)
        image = image.crop(
            (
                random.randint(0, dx),
                random.randint(0, dy),
                w - random.randint(0, dx),
                h - random.randint(0, dy),
            )
        )
    if random.random() < 0.3:
        buffer = BytesIO()
        image.save(buffer, "JPEG", quality=random.randint(45, 95))
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            image = compressed.convert("RGB")
    return image


class PlateDataset(Dataset):
    def __init__(self, manifest, split, *, augment_images=False, limit=0):
        self.manifest = json.loads(Path(manifest).read_text())
        self.root = Path(self.manifest["root"])
        self.rows = [
            row
            for row in self.manifest["samples"]
            if row["dataset"] == "LPR" and row["split"] == split
        ]
        if limit:
            self.rows = self.rows[:limit]
        self.augment_images = augment_images

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["image"]) as source:
            image = source.convert("RGB")
        if self.augment_images:
            image = augment(image)
        text = row["labels"][0]
        return preprocess(image), torch.tensor(encode(text), dtype=torch.long), text


def collate(samples):
    images, targets, texts = zip(*samples, strict=True)
    return torch.stack(images), torch.cat(targets), torch.tensor(list(map(len, targets))), texts
