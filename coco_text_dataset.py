#!/usr/bin/env python3
"""
COCO 2017 Text-Image dataset loader

Expects a JSONL file (e.g., /data/chenxiaoyu/coco2017/captions_train2017.jsonl)
with records like: {"image": "/abs/path/to/train2017/000000000009.jpg", "text": "a caption ..."}

Returns image tensors and raw text strings. Tokenization and text encoding
are handled in the training script to keep flexibility.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class CocoCaptionDataset(Dataset):
    def __init__(self,
                 jsonl_path: str | Path,
                 image_size: int = 128,
                 train: bool = True) -> None:
        self.jsonl_path = Path(jsonl_path)
        assert self.jsonl_path.exists(), f"JSONL not found: {self.jsonl_path}"

        self.records: List[dict] = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "image" in rec and "text" in rec:
                        self.records.append(rec)
                except Exception:
                    continue
        if len(self.records) == 0:
            raise RuntimeError(f"No valid records found in {self.jsonl_path}")

        if train:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        rec = self.records[idx]
        img_path = Path(rec["image"])  # absolute path recommended
        text = rec["text"]
        with Image.open(img_path) as im:
            image = im.convert("RGB")
        image_t = self.transform(image)
        return image_t, text


def create_coco_dataloader(base_dir: str | Path,
                           jsonl_name: str = "captions_train2017.jsonl",
                           batch_size: int = 32,
                           image_size: int = 128,
                           train: bool = True,
                           num_workers: int = 4,
                           sampler = None) -> DataLoader:
    base = Path(base_dir)
    jsonl_path = base / jsonl_name
    ds = CocoCaptionDataset(jsonl_path, image_size=image_size, train=train)
    dl = DataLoader(ds,
                    batch_size=batch_size,
                    shuffle=(train and sampler is None),
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True,
                    sampler=sampler)
    return dl
