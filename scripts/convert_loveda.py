"""Reorganize LoveDA into the MK-UNet layout (target/<name>/{train,val,test}/{images,masks}).

No pixel transformation: images and masks are copied verbatim. Masks already
hold class indices 0-7 (0 = no-data, to be IGNORED in training/eval; 1-7 =
background, building, road, water, barren, forest, agriculture, per the
official repo: github.com/Junjue-Wang/LoveDA).

Split mapping: LoveDA Train -> train; LoveDA Val -> our val/test (seeded 50/50
per domain; LoveDA Test has no public labels). Urban/Rural kept via filename
prefix (urban_/rural_) for later domain-transfer experiments.

Usage:
    python convert_loveda.py <loveda_root> <output_root>

<loveda_root> contains Train/ and Val/, each with {Urban,Rural}/{images_png,masks_png}.
Output lands in <output_root>/target/LoveDA/.
"""
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 42
IGNORE = 0
NAMES = {0: 'ignore', 1: 'background', 2: 'building', 3: 'road',
         4: 'water', 5: 'barren', 6: 'forest', 7: 'agriculture'}


def collect(split_dir: Path):
    """Return (img_path, mask_path, unique_name) for every pair in a LoveDA split."""
    pairs = []
    for domain in ['Urban', 'Rural']:
        img_dir = split_dir / domain / 'images_png'
        msk_dir = split_dir / domain / 'masks_png'
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.glob('*.png')):
            msk = msk_dir / img.name
            if msk.exists():
                pairs.append((img, msk, f'{domain.lower()}_{img.name}'))
    return pairs


def write(pairs, out_split: Path, class_px: Counter):
    img_dir, msk_dir = out_split / 'images', out_split / 'masks'
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)
    for img, msk, name in pairs:
        # validate BEFORE copying so a bad mask never lands in the output
        v, c = np.unique(np.array(Image.open(msk)), return_counts=True)
        assert set(v.tolist()) <= set(NAMES), f'unexpected label in {msk}: {v}'
        shutil.copy(img, img_dir / name)
        shutil.copy(msk, msk_dir / name)
        for vi, ci in zip(v.tolist(), c.tolist()):
            class_px[vi] += ci


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) / 'target' / 'LoveDA'
    assert not out.exists(), \
        f'{out} already exists; remove it first (reruns must not mix old and new files)'

    train_pairs = collect(src / 'Train')
    val_pairs = collect(src / 'Val')
    print(f'LoveDA Train pairs: {len(train_pairs)}  Val pairs: {len(val_pairs)}')

    # our val/test: seeded 50/50 split of LoveDA Val, per domain
    rng = random.Random(SEED)
    our_val, our_test = [], []
    for domain in ['urban_', 'rural_']:
        dom = [p for p in val_pairs if p[2].startswith(domain)]
        rng.shuffle(dom)
        half = len(dom) // 2
        our_val += dom[:half]
        our_test += dom[half:]

    class_px = Counter()
    for split, pairs in [('train', train_pairs), ('val', our_val), ('test', our_test)]:
        write(pairs, out / split, class_px)
        n_u = sum(p[2].startswith('urban_') for p in pairs)
        print(f'{split}: {len(pairs)} images (urban {n_u}, rural {len(pairs) - n_u}) -> {out / split}')

    total = sum(class_px.values())
    for k in sorted(class_px):
        print(f'class {k} ({NAMES[k]:11s}): {class_px[k] / total:.4%}')
    print(f'seed={SEED}, ignore_index={IGNORE} (pass --ignore_index 0 when training)')


if __name__ == '__main__':
    main()
