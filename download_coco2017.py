#!/usr/bin/env python3
"""
Download MSCOCO 2017 train images and captions, extract to a target directory,
and build JSONL files mapping image paths to captions.

Usage:
  python download_coco2017.py --output_dir /data/chenxiaoyu

This will create:
  /data/chenxiaoyu/coco2017/
    train2017/                         # images
    annotations/                       # original COCO annotations
      captions_train2017.json
      instances_train2017.json
      ...
    captions_train2017.jsonl           # one entry per (image, caption)
    captions_train2017_agg.jsonl       # one entry per image with list of captions

Requirements: Python 3.8+, tqdm, requests (optional; falls back to urllib if missing)
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional

# Optional imports with fallbacks
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

import urllib.request

COCO_BASE_URL = "http://images.cocodataset.org"
TRAIN_IMAGES_URL = f"{COCO_BASE_URL}/zips/train2017.zip"
ANNOTS_URL = f"{COCO_BASE_URL}/annotations/annotations_trainval2017.zip"


def human_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"Skip (exists): {dst} ({human_size(dst.stat().st_size)})")
        return

    print(f"Downloading: {url}\n  -> {dst}")

    if requests is not None and tqdm is not None:
        with requests.get(url, stream=True, timeout=60) as r:  # type: ignore
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dst, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:  # type: ignore
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    else:  # fallback to urllib
        with urllib.request.urlopen(url) as response, open(dst, "wb") as out_file:
            data = response.read()
            out_file.write(data)

    print(f"Finished: {dst} ({human_size(dst.stat().st_size)})")


def unzip(zip_path: Path, out_dir: Path, members: Optional[list[str]] = None) -> None:
    print(f"Extracting: {zip_path} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        if members is None:
            zf.extractall(out_dir)
        else:
            for m in members:
                zf.extract(m, out_dir)
    print("Done extracting.")


def build_jsonl(ann_path: Path, images_dir: Path, out_jsonl: Path, out_agg_jsonl: Path) -> None:
    print(f"Building JSONL from {ann_path}")
    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    # id -> file_name
    id_to_file = {img["id"]: img["file_name"] for img in ann.get("images", [])}

    # Gather captions per image id
    caps_by_img: dict[int, list[str]] = {}
    for c in ann.get("annotations", []):
        img_id = c["image_id"]
        txt = c["caption"].strip()
        if not txt:
            continue
        caps_by_img.setdefault(img_id, []).append(txt)

    # One entry per (image, caption)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    num_pairs = 0
    with open(out_jsonl, "w", encoding="utf-8") as fw:
        for img_id, file_name in id_to_file.items():
            rel_path = Path("train2017") / file_name
            abs_path = (images_dir / file_name).resolve()
            for caption in caps_by_img.get(img_id, []):
                rec = {"image": str(abs_path), "relative_image": str(rel_path), "text": caption}
                fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                num_pairs += 1

    # One entry per image with all captions
    num_images = 0
    with open(out_agg_jsonl, "w", encoding="utf-8") as fw:
        for img_id, file_name in id_to_file.items():
            rel_path = Path("train2017") / file_name
            abs_path = (images_dir / file_name).resolve()
            rec = {
                "image": str(abs_path),
                "relative_image": str(rel_path),
                "captions": caps_by_img.get(img_id, []),
            }
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            num_images += 1

    print(f"Wrote {num_pairs} image-caption pairs to {out_jsonl}")
    print(f"Wrote {num_images} images with captions to {out_agg_jsonl}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Base output dir, e.g., /data/chenxiaoyu")
    args = parser.parse_args()

    base = Path(args.output_dir).expanduser().resolve()
    out_root = base / "coco2017"
    zips_dir = out_root / "_zips"

    train_zip = zips_dir / "train2017.zip"
    ann_zip = zips_dir / "annotations_trainval2017.zip"

    # 1) Download
    download(TRAIN_IMAGES_URL, train_zip)
    download(ANNOTS_URL, ann_zip)

    # 2) Extract
    # train images -> out_root/train2017/
    unzip(train_zip, out_root)
    # annotations -> out_root/annotations/
    unzip(ann_zip, out_root)

    # 3) Build JSONL
    ann_dir = out_root / "annotations"
    images_dir = out_root / "train2017"
    captions_ann = ann_dir / "captions_train2017.json"
    if not captions_ann.exists():
        print(f"ERROR: {captions_ann} not found after extraction.")
        sys.exit(1)

    jsonl_pairs = out_root / "captions_train2017.jsonl"
    jsonl_agg = out_root / "captions_train2017_agg.jsonl"
    build_jsonl(captions_ann, images_dir, jsonl_pairs, jsonl_agg)

    print("\nAll done.")
    print(f"Images: {images_dir}")
    print(f"Annotations: {ann_dir}")
    print(f"Pairs JSONL: {jsonl_pairs}")
    print(f"Agg JSONL: {jsonl_agg}")


if __name__ == "__main__":
    main()
