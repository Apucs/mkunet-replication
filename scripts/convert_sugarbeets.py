"""Convert SugarBeets2016 annotated data to the binary segmentation layout
used by the MK-UNet polyp pipeline (target/<name>/{train,val,test}/{images,masks}).

Classes are merged: soil (0) -> background 0; crop (10000), weed (2) and
weed-species codes (20000+) -> foreground 255 ("vegetation vs. soil").

Usage:
    python convert_sugarbeets.py <sugarbeets_annotations_root> <output_root>

<sugarbeets_annotations_root> is the folder containing the CKA_* session dirs.
Output lands in <output_root>/target/SugarBeets/{train,val,test}/{images,masks}.
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


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) / "target" / "SugarBeets"

    pairs = collect_pairs(src)
    print(f"annotated rgb/iMap pairs found: {len(pairs)}")

    rng = random.Random(SEED)
    rng.shuffle(pairs)

    # keep the first N_TOTAL frames that contain actual vegetation
    kept, skipped_empty = [], 0
    for rgb, imap, name in pairs:
        arr = np.array(Image.open(imap))
        fg = arr > 0
        if fg.sum() < MIN_FG_PIXELS:
            skipped_empty += 1
            continue
        kept.append((rgb, imap, name, fg))
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

    fg_fractions = []
    for split, items in splits.items():
        img_dir = out / split / "images"
        msk_dir = out / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)
        for rgb, imap, name, fg in items:
            shutil.copy(rgb, img_dir / name)
            Image.fromarray((fg * 255).astype(np.uint8), mode="L").save(msk_dir / name)
            fg_fractions.append(fg.mean())
        print(f"{split}: {len(items)} images -> {img_dir.parent}")

    print(f"mean vegetation fraction: {np.mean(fg_fractions):.4f}")
    print(f"seed={SEED}, split={SPLIT}, min_fg={MIN_FG_PIXELS}")


if __name__ == "__main__":
    main()
