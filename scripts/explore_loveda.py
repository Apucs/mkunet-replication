import os
import numpy as np
from PIL import Image

# resolve the dataset relative to THIS FILE, so it works from any working directory
HERE = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(HERE, '..', '..', 'datasets', 'loveda')
assert os.path.isdir(root), f'dataset not found at: {os.path.abspath(root)}'

NAMES = {0: 'ignore', 1: 'background', 2: 'building', 3: 'road',
         4: 'water', 5: 'barren', 6: 'forest', 7: 'agriculture'}

# 1. counts per split/domain (works with whichever splits are downloaded)
for split in ['Train', 'Val', 'Test']:
    for domain in ['Urban', 'Rural']:
        img_dir = os.path.join(root, split, domain, 'images_png')
        msk_dir = os.path.join(root, split, domain, 'masks_png')
        n_img = len(os.listdir(img_dir)) if os.path.isdir(img_dir) else 0
        n_msk = len(os.listdir(msk_dir)) if os.path.isdir(msk_dir) else 0
        if n_img or n_msk:
            print(f'{split}/{domain}: images={n_img}  masks={n_msk}')

# 2. one sample pair: shapes, dtypes, mask values
sd = os.path.join(root, 'Val', 'Rural')
f = sorted(os.listdir(os.path.join(sd, 'images_png')))[0]
im = np.array(Image.open(os.path.join(sd, 'images_png', f)))
mk = np.array(Image.open(os.path.join(sd, 'masks_png', f)))
print(f'\nsample {f}: image {im.shape} {im.dtype} | mask {mk.shape} {mk.dtype} unique {np.unique(mk)}')

# 3. per-class pixel share, urban vs rural (the dataset's whole premise)
limit = 200   # masks per domain; raise to scan more
print(f'\nclass share over first {limit} Val masks per domain:')
print(f'{"class":22s} {"urban":>8s} {"rural":>8s}')
share = {}
for domain in ['Urban', 'Rural']:
    msk_dir = os.path.join(root, 'Val', domain, 'masks_png')
    counts = np.zeros(8, dtype=np.int64)
    for f in sorted(os.listdir(msk_dir))[:limit]:
        v, c = np.unique(np.array(Image.open(os.path.join(msk_dir, f))), return_counts=True)
        counts[v] += c
    share[domain] = counts / counts.sum()
for k in range(8):
    print(f'{k} {NAMES[k]:20s} {share["Urban"][k]:8.4f} {share["Rural"][k]:8.4f}')

# 4. per-image class areas (what SAS sees): median px per class where present
print(f'\nmedian per-image class area in px (first {limit} Rural masks, native 1024^2):')
areas = {k: [] for k in range(1, 8)}
msk_dir = os.path.join(root, 'Val', 'Rural', 'masks_png')
for f in sorted(os.listdir(msk_dir))[:limit]:
    a = np.array(Image.open(os.path.join(msk_dir, f)))
    for k in range(1, 8):
        n = int((a == k).sum())
        if n > 0:
            areas[k].append(n)
for k in range(1, 8):
    if areas[k]:
        print(f'{k} {NAMES[k]:20s} present in {len(areas[k]):3d} imgs, median {int(np.median(areas[k])):7d} px')
