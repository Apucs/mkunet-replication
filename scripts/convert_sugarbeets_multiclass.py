"""Convert SugarBeets2016 annotated data to a 3-class segmentation layout
(target/<name>/{train,val,test}/{images,masks}) for the MK-UNet pipeline.

Class mapping, following the official labelMap in the dataset's bundled
visualizeRandomSample.py:
    soil  {0, 1, 97}              -> 0 (background)
    crop  {10000, 10001, 10002}   -> 1
    weed  2 (weed), 20000-20011 (dycot), 20100-20105 (grass) -> 2

A full scan of all 12,340 iMapCleaned files found only codes 0, 1, 2, 10000,
20000-20009 and 20100-20105 in practice (code 1: 12 files / 30,196 px).

Masks are saved as uint8 PNGs holding literal class indices {0, 1, 2}.

Frame selection (seed, shuffle, vegetation filter, split sizes) is kept
identical to convert_sugarbeets.py, so the train/val/test splits contain
exactly the same frames as the binary SugarBeets dataset.

Usage:
    python convert_sugarbeets_multiclass.py <sugarbeets_annotations_root> <output_root>

<sugarbeets_annotations_root> is the folder containing the CKA_* session dirs.
Output lands in <output_root>/target/SugarBeets3C/{train,val,test}/{images,masks}.
"""
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 42
N_TOTAL = 2500          # subsample size, ~ISIC18 scale used in the MK-UNet paper
SPLIT = (0.8, 0.1, 0.1)  # matches the paper's 80:10:10 protocol
MIN_FG_PIXELS = 100     # skip frames with essentially no vegetation

# Official code groups from the dataset's visualizeRandomSample.py labelMap
SOIL_CODES = np.array([0, 1, 97])
CROP_CODES = np.array([10000, 10001, 10002])


def collect_pairs(root: Path):
    """Return (rgb_path, imap_path, unique_name) for every annotated frame."""
    pairs = []
    for session in sorted(root.iterdir()):
        imap_dir = session / "annotations" / "dlp" / "iMapCleaned"
        rgb_dir = session / "images" / "rgb"
        if not imap_dir.is_dir() or not rgb_dir.is_dir():
            continue
        for imap in sorted(imap_dir.glob("*.png")):
            rgb = rgb_dir / imap.name
            if rgb.exists():
                pairs.append((rgb, imap, f"{session.name}_{imap.name}"))
    return pairs


def to_class_map(arr: np.ndarray) -> np.ndarray:
    """Map raw iMap codes to class indices 0 (soil), 1 (crop), 2 (weed)."""
    classes = np.full(arr.shape, 2, dtype=np.uint8)  # default: weed
    classes[np.isin(arr, SOIL_CODES)] = 0
    classes[np.isin(arr, CROP_CODES)] = 1
    return classes


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) / "target" / "SugarBeets3C"

    pairs = collect_pairs(src)
    print(f"annotated rgb/iMap pairs found: {len(pairs)}")

    rng = random.Random(SEED)
    rng.shuffle(pairs)

    # keep the first N_TOTAL frames that contain actual vegetation
    # (same filter as the binary script, so the kept set is identical)
    kept, skipped_empty = [], 0
    for rgb, imap, name in pairs:
        arr = np.array(Image.open(imap))
        fg = arr > 0
        if fg.sum() < MIN_FG_PIXELS:
            skipped_empty += 1
            continue
        kept.append((rgb, imap, name, to_class_map(arr)))
        if len(kept) == N_TOTAL:
            break
    print(f"kept {len(kept)} frames (skipped {skipped_empty} near-empty masks)")

    n_train = int(len(kept) * SPLIT[0])
    n_val = int(len(kept) * SPLIT[1])
    splits = {
        "train": kept[:n_train],
        "val": kept[n_train:n_train + n_val],
        "test": kept[n_train + n_val:],
    }

    class_px = np.zeros(3, dtype=np.int64)
    frames_with = {"crop": 0, "weed": 0, "both": 0}
    for split, items in splits.items():
        img_dir = out / split / "images"
        msk_dir = out / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)
        for rgb, imap, name, classes in items:
            shutil.copy(rgb, img_dir / name)
            Image.fromarray(classes, mode="L").save(msk_dir / name)
            class_px += np.bincount(classes.ravel(), minlength=3)
            has_crop, has_weed = (classes == 1).any(), (classes == 2).any()
            frames_with["crop"] += has_crop
            frames_with["weed"] += has_weed
            frames_with["both"] += has_crop and has_weed
        print(f"{split}: {len(items)} images -> {img_dir.parent}")

    total_px = class_px.sum()
    for idx, label in enumerate(["soil", "crop", "weed"]):
        print(f"class {idx} ({label}): {class_px[idx]} px ({class_px[idx] / total_px:.4%})")
    print(f"frames with crop: {frames_with['crop']}, weed: {frames_with['weed']}, "
          f"both: {frames_with['both']} (of {len(kept)})")
    print(f"seed={SEED}, split={SPLIT}, min_fg={MIN_FG_PIXELS}")


if __name__ == "__main__":
    main()
